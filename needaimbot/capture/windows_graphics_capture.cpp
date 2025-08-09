#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "windows_graphics_capture.h"
#include "frame_buffer_pool.h"
#include "../AppContext.h"
#include "../cuda/cuda_image_processing.h"
#include "../cuda/cuda_error_check.h"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <ppl.h>

#ifndef __INTELLISENSE__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#endif

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// Simplified implementation using optimized Desktop Duplication API
// This avoids the complex WinRT APIs while achieving the same performance goals

class WindowsGraphicsCaptureImpl
{
public:
    ID3D11Device* m_device = nullptr;
    ID3D11DeviceContext* m_context = nullptr;
    IDXGIOutputDuplication* m_duplication = nullptr;
    ID3D11Texture2D* m_regionTexture = nullptr;  // Region-sized texture
    cudaGraphicsResource_t m_cudaResource = nullptr;  // CUDA interop resource
    
    int m_captureWidth;
    int m_captureHeight;
    int m_screenWidth;
    int m_screenHeight;
    RECT m_captureRegion;
    
    bool m_initialized = false;
};

WindowsGraphicsCapture::WindowsGraphicsCapture(int captureWidth, int captureHeight)
    : m_captureWidth(captureWidth), m_captureHeight(captureHeight)
{
    auto impl = new WindowsGraphicsCaptureImpl();
    m_captureItem = impl;
    
    
    // Get screen dimensions
    impl->m_captureWidth = captureWidth;
    impl->m_captureHeight = captureHeight;
    impl->m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
    impl->m_screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    
    // Calculate center region
    impl->m_captureRegion.left = (impl->m_screenWidth - captureWidth) / 2;
    impl->m_captureRegion.top = (impl->m_screenHeight - captureHeight) / 2;
    impl->m_captureRegion.right = impl->m_captureRegion.left + captureWidth;
    impl->m_captureRegion.bottom = impl->m_captureRegion.top + captureHeight;
    
    if (!InitializeCapture()) {
        std::cerr << "[WinGraphicsCapture] Failed to initialize capture" << std::endl;
        delete impl;
        m_captureItem = nullptr;
        return;
    }
    
    // No need for pinned memory with direct GPU interop
    
    // Create CUDA resources with high priority
    int leastPriority, greatestPriority;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    if (err != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to get stream priority range: " << cudaGetErrorString(err) << std::endl;
        delete impl;
        m_captureItem = nullptr;
        return;
    }
    
    err = cudaStreamCreateWithPriority(&m_cudaStream, cudaStreamNonBlocking, greatestPriority);
    if (err != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        delete impl;
        m_captureItem = nullptr;
        return;
    }
    
    err = cudaEventCreateWithFlags(&m_captureDoneEvent, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to create CUDA event: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(m_cudaStream);
        delete impl;
        m_captureItem = nullptr;
        return;
    }
    
    
    m_initialized = true;
    impl->m_initialized = true;
}

void WindowsGraphicsCapture::UpdateCaptureRegion(float offsetX, float offsetY)
{
    auto impl = reinterpret_cast<WindowsGraphicsCaptureImpl*>(m_captureItem);
    if (!impl) return;
    
    // Calculate new capture region with offset
    // Note: When user moves crosshair right (positive offsetX), capture region should move right too
    // When user moves crosshair down (positive offsetY), capture region should move down too
    impl->m_captureRegion.left = (impl->m_screenWidth - impl->m_captureWidth) / 2 + static_cast<int>(offsetX);
    impl->m_captureRegion.top = (impl->m_screenHeight - impl->m_captureHeight) / 2 + static_cast<int>(offsetY);
    impl->m_captureRegion.right = impl->m_captureRegion.left + impl->m_captureWidth;
    impl->m_captureRegion.bottom = impl->m_captureRegion.top + impl->m_captureHeight;
    
    // Clamp to screen boundaries
    if (impl->m_captureRegion.left < 0) {
        impl->m_captureRegion.left = 0;
        impl->m_captureRegion.right = impl->m_captureWidth;
    }
    if (impl->m_captureRegion.top < 0) {
        impl->m_captureRegion.top = 0;
        impl->m_captureRegion.bottom = impl->m_captureHeight;
    }
    if (impl->m_captureRegion.right > impl->m_screenWidth) {
        impl->m_captureRegion.right = impl->m_screenWidth;
        impl->m_captureRegion.left = impl->m_screenWidth - impl->m_captureWidth;
    }
    if (impl->m_captureRegion.bottom > impl->m_screenHeight) {
        impl->m_captureRegion.bottom = impl->m_screenHeight;
        impl->m_captureRegion.top = impl->m_screenHeight - impl->m_captureHeight;
    }
}

WindowsGraphicsCapture::~WindowsGraphicsCapture()
{
    auto impl = reinterpret_cast<WindowsGraphicsCaptureImpl*>(m_captureItem);
    if (!impl) return;
    
    // Cleanup CUDA resources
    if (impl->m_cudaResource) {
        cudaGraphicsUnregisterResource(impl->m_cudaResource);
    }
    
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
    
    if (m_captureDoneEvent) {
        cudaEventDestroy(m_captureDoneEvent);
    }
    
    if (impl->m_duplication) {
        impl->m_duplication->Release();
    }
    
    if (impl->m_regionTexture) {
        impl->m_regionTexture->Release();
    }
    
    
    if (impl->m_context) {
        impl->m_context->Release();
    }
    
    if (impl->m_device) {
        impl->m_device->Release();
    }
    
    delete impl;
}

bool WindowsGraphicsCapture::InitializeCapture()
{
    auto impl = reinterpret_cast<WindowsGraphicsCaptureImpl*>(m_captureItem);
    
    // Create D3D11 device
    HRESULT hr;
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    
    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &impl->m_device,
        nullptr,
        &impl->m_context
    );
    
    if (FAILED(hr)) {
        std::cerr << "[WinGraphicsCapture] Failed to create D3D11 device. HRESULT: 0x" 
                  << std::hex << hr << std::dec << std::endl;
        return false;
    }
    
    // Get DXGI device
    IDXGIDevice* dxgiDevice = nullptr;
    hr = impl->m_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    if (FAILED(hr)) {
        return false;
    }
    
    // Get adapter
    IDXGIAdapter* adapter = nullptr;
    hr = dxgiDevice->GetAdapter(&adapter);
    dxgiDevice->Release();
    if (FAILED(hr)) {
        return false;
    }
    
    // Get output
    IDXGIOutput* output = nullptr;
    hr = adapter->EnumOutputs(0, &output);
    adapter->Release();
    if (FAILED(hr)) {
        return false;
    }
    
    // Get output1
    IDXGIOutput1* output1 = nullptr;
    hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&output1);
    output->Release();
    if (FAILED(hr)) {
        return false;
    }
    
    // Create desktop duplication
    hr = output1->DuplicateOutput(impl->m_device, &impl->m_duplication);
    output1->Release();
    if (FAILED(hr)) {
        std::cerr << "[WinGraphicsCapture] Failed to create desktop duplication. HRESULT: 0x" 
                  << std::hex << hr << std::dec << std::endl;
        if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE) {
            std::cerr << "[WinGraphicsCapture] Screen is likely locked or another application is using Desktop Duplication" << std::endl;
        } else if (hr == E_ACCESSDENIED) {
            std::cerr << "[WinGraphicsCapture] Access denied - may need to run as administrator" << std::endl;
        }
        return false;
    }
    
    // Create region-sized texture (not full screen!)
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = impl->m_captureWidth;
    desc.Height = impl->m_captureHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    hr = impl->m_device->CreateTexture2D(&desc, nullptr, &impl->m_regionTexture);
    if (FAILED(hr)) {
        return false;
    }
    
    // Register texture with CUDA for direct GPU access
    cudaError_t cudaErr = cudaGraphicsD3D11RegisterResource(
        &impl->m_cudaResource,
        impl->m_regionTexture,
        cudaGraphicsRegisterFlagsNone
    );
    if (cudaErr != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to register D3D11 texture with CUDA: " 
                  << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    
    return true;
}

bool WindowsGraphicsCapture::CreateCaptureItemForMonitor(int monitorIndex)
{
    // Not needed for this implementation
    return true;
}

void WindowsGraphicsCapture::OnFrameArrived()
{
    // Not needed for this implementation
}

SimpleCudaMat WindowsGraphicsCapture::GetNextFrameGpu()
{
    auto impl = reinterpret_cast<WindowsGraphicsCaptureImpl*>(m_captureItem);
    if (!impl || !impl->m_initialized) {
        std::cerr << "[WinGraphicsCapture] GetNextFrameGpu: Not initialized" << std::endl;
        return SimpleCudaMat();
    }
    
    // Acquire frame
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    IDXGIResource* resource = nullptr;
    
    HRESULT hr = impl->m_duplication->AcquireNextFrame(0, &frameInfo, &resource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        // This is normal - no new frame available
        return SimpleCudaMat();
    }
    
    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            std::cerr << "[WinGraphicsCapture] Desktop duplication access lost - need to recreate" << std::endl;
        } else {
            std::cerr << "[WinGraphicsCapture] AcquireNextFrame failed: 0x" << std::hex << hr << std::dec << std::endl;
        }
        impl->m_duplication->ReleaseFrame();
        return SimpleCudaMat();
    }
    
    // Get texture
    ID3D11Texture2D* texture = nullptr;
    hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&texture);
    resource->Release();
    
    if (FAILED(hr)) {
        impl->m_duplication->ReleaseFrame();
        return SimpleCudaMat();
    }
    
    // Copy only the center region to our smaller texture
    D3D11_BOX box;
    box.left = impl->m_captureRegion.left;
    box.top = impl->m_captureRegion.top;
    box.right = impl->m_captureRegion.right;
    box.bottom = impl->m_captureRegion.bottom;
    box.front = 0;
    box.back = 1;
    
    impl->m_context->CopySubresourceRegion(
        impl->m_regionTexture, 0,
        0, 0, 0,
        texture, 0,
        &box
    );
    
    texture->Release();
    impl->m_duplication->ReleaseFrame();
    
    // Map CUDA resource directly - no CPU copy needed!
    cudaError_t err = cudaGraphicsMapResources(1, &impl->m_cudaResource, m_cudaStream);
    if (err != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to map CUDA resource: " << cudaGetErrorString(err) << std::endl;
        return SimpleCudaMat();
    }
    
    // Get CUDA array from the mapped resource
    cudaArray_t cudaArray;
    err = cudaGraphicsSubResourceGetMappedArray(&cudaArray, impl->m_cudaResource, 0, 0);
    if (err != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to get mapped array: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &impl->m_cudaResource, m_cudaStream);
        return SimpleCudaMat();
    }
    
    // Get GPU buffer
    if (!g_frameBufferPool) {
        g_frameBufferPool = std::make_unique<FrameBufferPool>(10);
    }
    
    // Keep BGRA format (4 channels) for direct use without conversion
    SimpleCudaMat gpuFrame = g_frameBufferPool->acquireGpuBuffer(
        impl->m_captureHeight, impl->m_captureWidth, 4  // BGRA
    );
    
    // Copy from CUDA array to linear memory (GPU to GPU, very fast!)
    err = cudaMemcpy2DFromArrayAsync(
        gpuFrame.data(), gpuFrame.step(),
        cudaArray, 0, 0,
        impl->m_captureWidth * 4, impl->m_captureHeight,
        cudaMemcpyDeviceToDevice, m_cudaStream
    );
    
    // Unmap the resource
    cudaGraphicsUnmapResources(1, &impl->m_cudaResource, m_cudaStream);
    
    if (err != cudaSuccess) {
        std::cerr << "[WinGraphicsCapture] Failed to copy from CUDA array: " << cudaGetErrorString(err) << std::endl;
        g_frameBufferPool->releaseGpuBuffer(std::move(gpuFrame));
        return SimpleCudaMat();
    }
    
    cudaEventRecord(m_captureDoneEvent, m_cudaStream);
    return gpuFrame;
}

