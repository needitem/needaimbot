#include "zero_copy_capture.h"
#include "../cuda/cuda_error_check.h"
#include <iostream>
#include <chrono>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

ZeroCopyCapture::ZeroCopyCapture(int width, int height) 
    : width_(width), height_(height) {
    
    // Create CUDA stream for capture operations
    CUDA_CHECK(cudaStreamCreateWithFlags(&captureStream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&captureDoneEvent_, 
        cudaEventDisableTiming | cudaEventBlockingSync));
    
    // Initialize GPU frame buffer
    gpuFrame_.create(height_, width_, 4); // BGRA format
    
    // Initialize D3D11 and CUDA interop
    if (InitializeD3D11() && InitializeDuplication() && InitializeCudaInterop()) {
        initialized_ = true;
        
        // Check for NVLink support
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 1) {
            int canAccessPeer;
            cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
            useNvLink_ = (canAccessPeer == 1);
            if (useNvLink_) {
                std::cout << "[ZeroCopyCapture] NVLink detected - enabling optimized transfers" << std::endl;
            }
        }
        
        std::cout << "[ZeroCopyCapture] Initialized successfully with zero-copy capture" << std::endl;
    } else {
        std::cerr << "[ZeroCopyCapture] Failed to initialize" << std::endl;
        ReleaseResources();
    }
}

ZeroCopyCapture::~ZeroCopyCapture() {
    ReleaseResources();
    
    if (captureStream_) {
        cudaStreamDestroy(captureStream_);
    }
    if (captureDoneEvent_) {
        cudaEventDestroy(captureDoneEvent_);
    }
    
    gpuFrame_.release();
}

bool ZeroCopyCapture::InitializeD3D11() {
    HRESULT hr;
    
    // Create D3D11 device with CUDA interop support
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;
    
    UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createFlags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &d3dDevice_,
        &featureLevel,
        &d3dContext_
    );
    
    if (FAILED(hr)) {
        std::cerr << "[ZeroCopyCapture] Failed to create D3D11 device: " << std::hex << hr << std::endl;
        return false;
    }
    
    // Set CUDA device to match D3D11 adapter
    IDXGIDevice* dxgiDevice;
    d3dDevice_->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    
    IDXGIAdapter* adapter;
    dxgiDevice->GetAdapter(&adapter);
    
    // Find matching CUDA device
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    
    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        // Match by LUID if available
        int d3dDevice;
        if (cudaD3D11GetDevice(&d3dDevice, adapter) == cudaSuccess) {
            cudaSetDevice(d3dDevice);
            std::cout << "[ZeroCopyCapture] Matched CUDA device " << d3dDevice 
                      << " with D3D11 adapter" << std::endl;
            break;
        }
    }
    
    adapter->Release();
    dxgiDevice->Release();
    
    return true;
}

bool ZeroCopyCapture::InitializeDuplication() {
    HRESULT hr;
    
    // Get DXGI device
    IDXGIDevice* dxgiDevice;
    hr = d3dDevice_->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    if (FAILED(hr)) return false;
    
    // Get adapter
    IDXGIAdapter* adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    dxgiDevice->Release();
    if (FAILED(hr)) return false;
    
    // Get output (monitor)
    IDXGIOutput* output;
    hr = adapter->EnumOutputs(0, &output);
    adapter->Release();
    if (FAILED(hr)) return false;
    
    // Get output1 interface for duplication
    IDXGIOutput1* output1;
    hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&output1);
    output->Release();
    if (FAILED(hr)) return false;
    
    // Create desktop duplication
    hr = output1->DuplicateOutput(d3dDevice_, &duplication_);
    output1->Release();
    
    if (FAILED(hr)) {
        std::cerr << "[ZeroCopyCapture] Failed to create desktop duplication: " 
                  << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}

bool ZeroCopyCapture::InitializeCudaInterop() {
    // Create staging texture for CUDA interop
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width_;
    desc.Height = height_;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    HRESULT hr = d3dDevice_->CreateTexture2D(&desc, nullptr, &stagingTexture_);
    if (FAILED(hr)) {
        std::cerr << "[ZeroCopyCapture] Failed to create staging texture" << std::endl;
        return false;
    }
    
    // Register texture with CUDA for zero-copy access
    cudaError_t cudaErr = cudaGraphicsD3D11RegisterResource(
        &cudaResource_,
        stagingTexture_,
        cudaGraphicsRegisterFlagsNone
    );
    
    if (cudaErr != cudaSuccess) {
        std::cerr << "[ZeroCopyCapture] Failed to register D3D11 resource with CUDA: " 
                  << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    
    std::cout << "[ZeroCopyCapture] CUDA-D3D11 interop initialized successfully" << std::endl;
    return true;
}

SimpleCudaMat ZeroCopyCapture::GetNextFrameGpu() {
    if (!initialized_) {
        return SimpleCudaMat();
    }
    
    if (CaptureFrameDirect()) {
        // Map D3D11 resource for CUDA access
        cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_, captureStream_);
        if (err != cudaSuccess) {
            std::cerr << "[ZeroCopyCapture] Failed to map resources: " 
                      << cudaGetErrorString(err) << std::endl;
            return SimpleCudaMat();
        }
        
        // Get CUDA array from mapped resource
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource_, 0, 0);
        if (err != cudaSuccess) {
            cudaGraphicsUnmapResources(1, &cudaResource_, captureStream_);
            return SimpleCudaMat();
        }
        
        // Direct memory copy from D3D11 texture to CUDA memory (zero-copy)
        cudaMemcpy2DFromArrayAsync(
            gpuFrame_.ptr(),
            gpuFrame_.step(),
            cuArray,
            0, 0,
            width_ * 4,  // BGRA format
            height_,
            cudaMemcpyDeviceToDevice,
            captureStream_
        );
        
        // Unmap resources
        cudaGraphicsUnmapResources(1, &cudaResource_, captureStream_);
        
        // Record completion event
        cudaEventRecord(captureDoneEvent_, captureStream_);
        
        return gpuFrame_;
    }
    
    return SimpleCudaMat();
}

bool ZeroCopyCapture::CaptureFrameDirect() {
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    IDXGIResource* desktopResource = nullptr;
    
    // Acquire next frame with minimal timeout for low latency
    HRESULT hr = duplication_->AcquireNextFrame(0, &frameInfo, &desktopResource);
    
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return false; // No new frame available
    }
    
    if (FAILED(hr)) {
        // Recreate duplication if lost
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            duplication_->Release();
            InitializeDuplication();
        }
        return false;
    }
    
    // Get texture interface
    ID3D11Texture2D* desktopTexture;
    hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), 
                                         (void**)&desktopTexture);
    desktopResource->Release();
    
    if (SUCCEEDED(hr)) {
        // Copy to staging texture with region of interest
        D3D11_BOX box;
        box.left = static_cast<UINT>(offsetX_);
        box.top = static_cast<UINT>(offsetY_);
        box.right = box.left + width_;
        box.bottom = box.top + height_;
        box.front = 0;
        box.back = 1;
        
        // Direct GPU-to-GPU copy (no CPU involvement)
        d3dContext_->CopySubresourceRegion(
            stagingTexture_, 0, 0, 0, 0,
            desktopTexture, 0, &box
        );
        
        desktopTexture->Release();
    }
    
    // Release frame
    duplication_->ReleaseFrame();
    
    return SUCCEEDED(hr);
}

void ZeroCopyCapture::UpdateCaptureRegion(float offsetX, float offsetY) {
    offsetX_ = offsetX;
    offsetY_ = offsetY;
}

void ZeroCopyCapture::ReleaseResources() {
    initialized_ = false;
    
    if (cudaResource_) {
        cudaGraphicsUnregisterResource(cudaResource_);
        cudaResource_ = nullptr;
    }
    
    if (stagingTexture_) {
        stagingTexture_->Release();
        stagingTexture_ = nullptr;
    }
    
    if (duplication_) {
        duplication_->Release();
        duplication_ = nullptr;
    }
    
    if (d3dContext_) {
        d3dContext_->Release();
        d3dContext_ = nullptr;
    }
    
    if (d3dDevice_) {
        d3dDevice_->Release();
        d3dDevice_ = nullptr;
    }
}