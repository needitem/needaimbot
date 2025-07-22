#include "ndi_capture.h"
#include "AppContext.h"
#include <iostream>
#include <chrono>

NDICapture::NDICapture(int width, int height, const std::string& source_name)
    : width_(width), height_(height), source_name_(source_name), low_latency_mode_(true)
#ifdef NDI_SDK_AVAILABLE
    , ndi_receiver_(nullptr), ndi_finder_(nullptr)
#endif
{
    std::cout << "[NDI] Initializing NDI capture..." << std::endl;
    std::cout << "[NDI] Target resolution: " << width_ << "x" << height_ << std::endl;
    
    if (!source_name_.empty()) {
        std::cout << "[NDI] Target source: " << source_name_ << std::endl;
    }
    
    // Create CUDA event
    if (cudaEventCreate(&capture_event_) != cudaSuccess) {
        std::cerr << "[NDI] Failed to create CUDA event" << std::endl;
        return;
    }
    
#ifdef NDI_SDK_AVAILABLE
    if (initializeNDI()) {
        initialized_ = true;
        std::cout << "[NDI] NDI capture initialized successfully" << std::endl;
    } else {
        std::cout << "[NDI] NDI initialization failed, trying network stream fallback" << std::endl;
        use_network_fallback_ = true;
    }
#else
    std::cout << "[NDI] NDI SDK not available, using network stream fallback" << std::endl;
    use_network_fallback_ = true;
#endif
    
    if (use_network_fallback_) {
        if (initializeNetworkStream()) {
            initialized_ = true;
            std::cout << "[NDI] Network stream fallback initialized successfully" << std::endl;
        }
    }
    
    if (initialized_) {
        // Start capture thread
        capture_thread_ = std::thread(&NDICapture::captureLoop, this);
    }
}

NDICapture::~NDICapture()
{
    should_stop_ = true;
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    cleanupResources();
    
    if (capture_event_) {
        cudaEventDestroy(capture_event_);
    }
    
    std::cout << "[NDI] NDI capture destroyed" << std::endl;
}

#ifdef NDI_SDK_AVAILABLE
bool NDICapture::initializeNDI()
{
    try {
        // Initialize NDI
        if (!NDIlib_initialize()) {
            std::cerr << "[NDI] Failed to initialize NDI runtime" << std::endl;
            return false;
        }
        
        // Create finder
        NDIlib_find_create_t find_create = {};
        find_create.show_local_sources = true;
        find_create.p_groups = nullptr;
        find_create.p_extra_ips = nullptr;
        
        ndi_finder_ = NDIlib_find_create_v2(&find_create);
        if (!ndi_finder_) {
            std::cerr << "[NDI] Failed to create NDI finder" << std::endl;
            return false;
        }
        
        // Wait for sources
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Find sources
        uint32_t num_sources = 0;
        const NDIlib_source_t* sources = NDIlib_find_get_current_sources(ndi_finder_, &num_sources);
        
        std::cout << "[NDI] Found " << num_sources << " NDI sources:" << std::endl;
        for (uint32_t i = 0; i < num_sources; i++) {
            std::cout << "[NDI] Source " << i << ": " << sources[i].p_ndi_name << std::endl;
            ndi_sources_.push_back(sources[i]);
        }
        
        // Connect to specified source or first available
        NDIlib_source_t target_source = {};
        bool source_found = false;
        
        if (!source_name_.empty()) {
            for (const auto& source : ndi_sources_) {
                if (std::string(source.p_ndi_name).find(source_name_) != std::string::npos) {
                    target_source = source;
                    source_found = true;
                    break;
                }
            }
        } else if (!ndi_sources_.empty()) {
            target_source = ndi_sources_[0];
            source_found = true;
        }
        
        if (!source_found) {
            std::cerr << "[NDI] No suitable NDI source found" << std::endl;
            return false;
        }
        
        // Create receiver
        NDIlib_recv_create_v3_t recv_create = {};
        recv_create.source_to_connect_to = target_source;
        recv_create.color_format = NDIlib_recv_color_format_BGRX_BGRA;
        recv_create.bandwidth = low_latency_mode_ ? NDIlib_recv_bandwidth_highest : NDIlib_recv_bandwidth_lowest;
        recv_create.allow_video_fields = false;
        
        ndi_receiver_ = NDIlib_recv_create_v3(&recv_create);
        if (!ndi_receiver_) {
            std::cerr << "[NDI] Failed to create NDI receiver" << std::endl;
            return false;
        }
        
        std::cout << "[NDI] Connected to source: " << target_source.p_ndi_name << std::endl;
        
        // Pre-allocate GPU memory
        gpu_frame_.create(height_, width_, CV_8UC3);
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[NDI] Exception during NDI initialization: " << e.what() << std::endl;
        return false;
    }
}
#else
bool NDICapture::initializeNDI()
{
    return false; // NDI SDK not available
}
#endif

bool NDICapture::initializeNetworkStream()
{
    try {
        // Common network stream URLs for OBS/streaming software
        std::vector<std::string> possible_urls = {
            "http://localhost:8080/video.mjpg",      // OBS Browser Source
            "http://127.0.0.1:8080/video.mjpg",
            "rtmp://localhost:1935/live/stream",      // RTMP server
            "udp://127.0.0.1:1234",                  // UDP stream
            network_url_  // Custom URL if set
        };
        
        for (const auto& url : possible_urls) {
            if (url.empty()) continue;
            
            std::cout << "[NDI] Trying to connect to: " << url << std::endl;
            
            network_capture_.open(url);
            if (network_capture_.isOpened()) {
                network_url_ = url;
                
                // Set properties
                network_capture_.set(cv::CAP_PROP_BUFFERSIZE, 1); // Reduce latency
                network_capture_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
                network_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
                
                // Test capture
                cv::Mat test_frame;
                if (network_capture_.read(test_frame) && !test_frame.empty()) {
                    std::cout << "[NDI] Successfully connected to: " << url << std::endl;
                    std::cout << "[NDI] Stream resolution: " << test_frame.cols << "x" << test_frame.rows << std::endl;
                    
                    // Pre-allocate GPU memory
                    gpu_frame_.create(height_, width_, CV_8UC3);
                    
                    return true;
                }
                
                network_capture_.release();
            }
        }
        
        std::cerr << "[NDI] Failed to connect to any network stream" << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "[NDI] Exception during network stream initialization: " << e.what() << std::endl;
        return false;
    }
}

void NDICapture::cleanupResources()
{
#ifdef NDI_SDK_AVAILABLE
    if (ndi_receiver_) {
        NDIlib_recv_destroy(ndi_receiver_);
        ndi_receiver_ = nullptr;
    }
    
    if (ndi_finder_) {
        NDIlib_find_destroy(ndi_finder_);
        ndi_finder_ = nullptr;
    }
    
    NDIlib_destroy();
#endif
    
    if (network_capture_.isOpened()) {
        network_capture_.release();
    }
    
    gpu_frame_.release();
    cpu_frame_.release();
}

std::vector<std::string> NDICapture::FindNDISources() const
{
    std::vector<std::string> source_names;
    
#ifdef NDI_SDK_AVAILABLE
    if (ndi_finder_) {
        uint32_t num_sources = 0;
        const NDIlib_source_t* sources = NDIlib_find_get_current_sources(ndi_finder_, &num_sources);
        
        for (uint32_t i = 0; i < num_sources; i++) {
            source_names.push_back(std::string(sources[i].p_ndi_name));
        }
    }
#endif
    
    return source_names;
}

bool NDICapture::ConnectToNetworkStream(const std::string& url)
{
    network_url_ = url;
    use_network_fallback_ = true;
    return initializeNetworkStream();
}

void NDICapture::SetLowLatencyMode(bool enable)
{
    low_latency_mode_ = enable;
    
#ifdef NDI_SDK_AVAILABLE
    if (ndi_receiver_) {
        // Recreate receiver with new bandwidth setting
        // This would require reconnection in a real implementation
    }
#endif
}

void NDICapture::captureLoop()
{
    std::cout << "[NDI] Capture loop started" << std::endl;
    
    while (!should_stop_) {
        try {
            cv::Mat frame;
            bool frame_captured = false;
            
            if (use_network_fallback_ && network_capture_.isOpened()) {
                frame_captured = network_capture_.read(frame);
            }
#ifdef NDI_SDK_AVAILABLE
            else if (ndi_receiver_) {
                NDIlib_video_frame_v2_t video_frame = {};
                
                switch (NDIlib_recv_capture_v2(ndi_receiver_, &video_frame, nullptr, nullptr, 1000)) {
                case NDIlib_frame_type_video:
                    {
                        // Convert NDI frame to OpenCV Mat
                        int cv_type = (video_frame.FourCC == NDIlib_FourCC_type_BGRA) ? CV_8UC4 : CV_8UC3;
                        cv::Mat ndi_mat(video_frame.yres, video_frame.xres, cv_type, video_frame.p_data, video_frame.line_stride_in_bytes);
                        
                        frame = ndi_mat.clone();
                        frame_captured = true;
                        
                        // Update bandwidth info
                        bandwidth_mbps_ = static_cast<float>(video_frame.line_stride_in_bytes * video_frame.yres * current_fps_) / (1024 * 1024);
                    }
                    
                    NDIlib_recv_free_video_v2(ndi_receiver_, &video_frame);
                    break;
                    
                case NDIlib_frame_type_none:
                    // No frame available, continue
                    break;
                    
                default:
                    // Error or other frame type
                    break;
                }
            }
#endif
            
            if (frame_captured && !frame.empty()) {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                
                if (processFrame(frame)) {
                    cpu_frame_ = frame.clone();
                    
                    // Update FPS
                    auto now = std::chrono::steady_clock::now();
                    if (last_frame_time_.time_since_epoch().count() > 0) {
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time_).count();
                        if (duration > 0) {
                            current_fps_ = 1000000.0f / duration;
                        }
                    }
                    last_frame_time_ = now;
                }
            } else {
                // Avoid busy waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[NDI] Exception in capture loop: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    std::cout << "[NDI] Capture loop ended" << std::endl;
}

cv::cuda::GpuMat NDICapture::GetNextFrameGpu()
{
    if (!initialized_ || should_stop_) {
        return cv::cuda::GpuMat();
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    if (cpu_frame_.empty()) {
        return cv::cuda::GpuMat();
    }
    
    try {
        // Upload to GPU
        gpu_frame_.upload(cpu_frame_);
        
        // Record CUDA event
        cudaEventRecord(capture_event_);
        
        return gpu_frame_;
    }
    catch (const std::exception& e) {
        std::cerr << "[NDI] Exception in GetNextFrameGpu: " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}

cv::Mat NDICapture::GetNextFrameCpu()
{
    if (!initialized_ || should_stop_) {
        return cv::Mat();
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return cpu_frame_.clone();
}

cudaEvent_t NDICapture::GetCaptureDoneEvent() const
{
    return capture_event_;
}

bool NDICapture::processFrame(cv::Mat& frame)
{
    try {
        // Resize if necessary
        if (frame.cols != width_ || frame.rows != height_) {
            cv::resize(frame, frame, cv::Size(width_, height_));
        }
        
        // Convert to BGR if necessary
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        }
        else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[NDI] Frame processing error: " << e.what() << std::endl;
        return false;
    }
}

float NDICapture::GetCurrentBandwidth() const
{
    return bandwidth_mbps_;
}