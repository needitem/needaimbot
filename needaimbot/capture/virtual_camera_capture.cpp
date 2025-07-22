#include "virtual_camera_capture.h"
#include "AppContext.h"
#include <iostream>
#include <vector>
#include <dshow.h>
#include <uuids.h>

#pragma comment(lib, "strmiids.lib")

VirtualCameraCapture::VirtualCameraCapture(int width, int height, int device_id)
    : width_(width), height_(height), device_id_(device_id), target_fps_(60.0)
{
    std::cout << "[VirtualCamera] Initializing virtual camera capture..." << std::endl;
    std::cout << "[VirtualCamera] Target resolution: " << width_ << "x" << height_ << std::endl;
    std::cout << "[VirtualCamera] Device ID: " << device_id_ << std::endl;
    
    // Create CUDA event
    if (cudaEventCreate(&capture_event_) != cudaSuccess) {
        std::cerr << "[VirtualCamera] Failed to create CUDA event" << std::endl;
        return;
    }
    
    if (initializeCamera()) {
        initialized_ = true;
        std::cout << "[VirtualCamera] Virtual camera initialized successfully" << std::endl;
        
        // List available devices for debugging
        auto devices = ListAvailableDevices();
        std::cout << "[VirtualCamera] Available devices:" << std::endl;
        for (size_t i = 0; i < devices.size(); ++i) {
            std::cout << "[VirtualCamera] Device " << i << ": " << devices[i] << std::endl;
        }
    }
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    should_stop_ = true;
    cleanupResources();
    
    if (capture_event_) {
        cudaEventDestroy(capture_event_);
    }
    
    std::cout << "[VirtualCamera] Virtual camera capture destroyed" << std::endl;
}

bool VirtualCameraCapture::initializeCamera()
{
    try {
        // Open the camera with DirectShow backend for better virtual camera support
        capture_.open(device_id_, cv::CAP_DSHOW);
        
        if (!capture_.isOpened()) {
            std::cerr << "[VirtualCamera] Failed to open camera device " << device_id_ << std::endl;
            return false;
        }
        
        // Set resolution
        if (!SetResolution(width_, height_)) {
            std::cerr << "[VirtualCamera] Failed to set resolution" << std::endl;
            return false;
        }
        
        // Set frame rate
        SetFrameRate(target_fps_);
        
        // Set buffer size to reduce latency
        capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        // Test capture
        cv::Mat test_frame;
        if (!capture_.read(test_frame)) {
            std::cerr << "[VirtualCamera] Failed to read test frame" << std::endl;
            return false;
        }
        
        std::cout << "[VirtualCamera] Actual resolution: " << test_frame.cols << "x" << test_frame.rows << std::endl;
        
        // Pre-allocate GPU memory
        gpu_frame_.create(height_, width_, CV_8UC3);
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[VirtualCamera] Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

void VirtualCameraCapture::cleanupResources()
{
    if (capture_.isOpened()) {
        capture_.release();
    }
    
    gpu_frame_.release();
    cpu_frame_.release();
}

bool VirtualCameraCapture::SetResolution(int width, int height)
{
    if (!capture_.isOpened()) {
        return false;
    }
    
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    
    // Verify the resolution was set
    int actual_width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "[VirtualCamera] Requested: " << width << "x" << height 
              << ", Got: " << actual_width << "x" << actual_height << std::endl;
    
    width_ = actual_width;
    height_ = actual_height;
    
    return (actual_width > 0 && actual_height > 0);
}

void VirtualCameraCapture::SetFrameRate(double fps)
{
    if (capture_.isOpened()) {
        capture_.set(cv::CAP_PROP_FPS, fps);
        target_fps_ = fps;
        
        double actual_fps = capture_.get(cv::CAP_PROP_FPS);
        std::cout << "[VirtualCamera] Target FPS: " << fps << ", Actual FPS: " << actual_fps << std::endl;
    }
}

std::vector<std::string> VirtualCameraCapture::ListAvailableDevices() const
{
    std::vector<std::string> devices;
    
    // Try to enumerate DirectShow video capture devices
    CoInitialize(NULL);
    
    ICreateDevEnum* pCreateDevEnum = NULL;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
                                  IID_ICreateDevEnum, (void**)&pCreateDevEnum);
    
    if (SUCCEEDED(hr)) {
        IEnumMoniker* pEnumMoniker = NULL;
        hr = pCreateDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnumMoniker, 0);
        
        if (SUCCEEDED(hr) && pEnumMoniker != NULL) {
            IMoniker* pMoniker = NULL;
            int deviceIndex = 0;
            
            while (pEnumMoniker->Next(1, &pMoniker, NULL) == S_OK) {
                IPropertyBag* pPropertyBag = NULL;
                hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&pPropertyBag);
                
                if (SUCCEEDED(hr)) {
                    VARIANT varName;
                    VariantInit(&varName);
                    hr = pPropertyBag->Read(L"FriendlyName", &varName, 0);
                    
                    if (SUCCEEDED(hr)) {
                        char deviceName[256];
                        WideCharToMultiByte(CP_UTF8, 0, varName.bstrVal, -1, deviceName, 256, NULL, NULL);
                        devices.push_back(std::string(deviceName));
                    }
                    
                    VariantClear(&varName);
                    pPropertyBag->Release();
                }
                
                pMoniker->Release();
                deviceIndex++;
            }
            
            pEnumMoniker->Release();
        }
        
        pCreateDevEnum->Release();
    }
    
    CoUninitialize();
    
    return devices;
}

cv::cuda::GpuMat VirtualCameraCapture::GetNextFrameGpu()
{
    if (!initialized_ || should_stop_ || !capture_.isOpened()) {
        return cv::cuda::GpuMat();
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    try {
        cv::Mat frame;
        if (!capture_.read(frame)) {
            std::cerr << "[VirtualCamera] Failed to read frame from camera" << std::endl;
            return cv::cuda::GpuMat();
        }
        
        if (frame.empty()) {
            return cv::cuda::GpuMat();
        }
        
        // Process frame if needed (resize, format conversion, etc.)
        if (!processFrame(frame)) {
            return cv::cuda::GpuMat();
        }
        
        // Upload to GPU
        gpu_frame_.upload(frame);
        
        // Record CUDA event
        cudaEventRecord(capture_event_);
        
        // Update FPS counter
        auto now = std::chrono::steady_clock::now();
        if (last_frame_time_.time_since_epoch().count() > 0) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time_).count();
            if (duration > 0) {
                current_fps_ = 1000000.0f / duration;
            }
        }
        last_frame_time_ = now;
        
        return gpu_frame_;
    }
    catch (const std::exception& e) {
        std::cerr << "[VirtualCamera] Exception in GetNextFrameGpu: " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!initialized_ || should_stop_ || !capture_.isOpened()) {
        return cv::Mat();
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    try {
        cv::Mat frame;
        if (!capture_.read(frame)) {
            return cv::Mat();
        }
        
        if (frame.empty()) {
            return cv::Mat();
        }
        
        // Process frame
        if (!processFrame(frame)) {
            return cv::Mat();
        }
        
        cpu_frame_ = frame.clone();
        return cpu_frame_;
    }
    catch (const std::exception& e) {
        std::cerr << "[VirtualCamera] Exception in GetNextFrameCpu: " << e.what() << std::endl;
        return cv::Mat();
    }
}

cudaEvent_t VirtualCameraCapture::GetCaptureDoneEvent() const
{
    return capture_event_;
}

bool VirtualCameraCapture::processFrame(cv::Mat& frame)
{
    try {
        // Resize if necessary
        if (frame.cols != width_ || frame.rows != height_) {
            cv::resize(frame, frame, cv::Size(width_, height_));
        }
        
        // Convert to BGR if necessary (most capture devices use BGR)
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        }
        else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[VirtualCamera] Frame processing error: " << e.what() << std::endl;
        return false;
    }
}