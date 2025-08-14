// Unified GPU Capture Implementation
// Combines gpu_only_capture and gpu_capture_manager into a single file

#include "../core/windows_headers.h"
#include <d3d11_4.h>
#include <dxgi1_5.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>

#include "../AppContext.h"
#include "../cuda/simple_cuda_mat.h"
#include "../cuda/unified_graph_pipeline.h"
using Microsoft::WRL::ComPtr;

class GPUCapture {
private:
    // DXGI Desktop Duplication
    ComPtr<IDXGIOutputDuplication> m_duplication;
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<ID3D11Texture2D> m_stagingTexture;
    
    // GPU synchronization
    ComPtr<ID3D11Fence> m_fence;
    ComPtr<ID3D11Device5> m_device5;
    ComPtr<ID3D11DeviceContext4> m_context4;
    HANDLE m_fenceEvent;
    UINT64 m_fenceValue;
    
    // CUDA interop
    cudaGraphicsResource_t m_cudaResource;
    cudaStream_t m_captureStream;
    cudaEvent_t m_frameReadyEvent;
    
    // Capture dimensions
    int m_width;
    int m_height;
    
    // State
    std::atomic<bool> m_isCapturing;
    bool m_useVSync;

public:
    GPUCapture(int width, int height) 
        : m_width(width), m_height(height), 
          m_cudaResource(nullptr), m_captureStream(nullptr),
          m_frameReadyEvent(nullptr), m_fenceEvent(nullptr),
          m_fenceValue(0), m_isCapturing(false), m_useVSync(false) {
    }
    
    ~GPUCapture() {
        StopCapture();
        
        if (m_frameReadyEvent) {
            cudaEventDestroy(m_frameReadyEvent);
        }
        if (m_captureStream) {
            cudaStreamDestroy(m_captureStream);
        }
        if (m_cudaResource) {
            cudaGraphicsUnregisterResource(m_cudaResource);
        }
        if (m_fenceEvent) {
            CloseHandle(m_fenceEvent);
        }
        
        m_duplication.Reset();
        m_stagingTexture.Reset();
        m_context.Reset();
        m_device.Reset();
    }
    
    bool Initialize() {
        // 1. Initialize DXGI and create device
        if (!InitializeDXGI()) {
            return false;
        }
        
        // 2. Setup Desktop Duplication
        ComPtr<IDXGIDevice> dxgiDevice;
        m_device->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
        
        ComPtr<IDXGIAdapter> adapter;
        dxgiDevice->GetAdapter(&adapter);
        
        ComPtr<IDXGIOutput> output;
        adapter->EnumOutputs(0, &output);
        
        ComPtr<IDXGIOutput1> output1;
        output->QueryInterface(IID_PPV_ARGS(&output1));
        
        HRESULT hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
        if (FAILED(hr)) {
            std::cerr << "[GPUCapture] Failed to duplicate output: " << std::hex << hr << std::endl;
            return false;
        }
        
        // 3. Create staging texture for CUDA interop
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = m_width;
        desc.Height = m_height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        
        hr = m_device->CreateTexture2D(&desc, nullptr, &m_stagingTexture);
        if (FAILED(hr)) {
            std::cerr << "[GPUCapture] Failed to create staging texture" << std::endl;
            return false;
        }
        
        // 4. Initialize CUDA interop
        if (!InitializeCUDAInterop()) {
            return false;
        }
        
        std::cout << "[GPUCapture] Initialized successfully" << std::endl;
        return true;
    }
    
    void StartCapture() {
        m_isCapturing = true;
    }
    
    void StopCapture() {
        m_isCapturing = false;
    }
    
    bool WaitForNextFrame() {
        if (!m_isCapturing || !m_duplication) {
            return false;
        }
        
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        ComPtr<IDXGIResource> desktopResource;
        
        // Release previous frame if any
        m_duplication->ReleaseFrame();
        
        // Try to acquire next frame (with timeout)
        HRESULT hr = m_duplication->AcquireNextFrame(16, &frameInfo, &desktopResource);
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            return false;  // No new frame available
        }
        
        if (FAILED(hr)) {
            // Recreate duplication on error
            m_duplication.Reset();
            
            ComPtr<IDXGIDevice> dxgiDevice;
            m_device->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
            ComPtr<IDXGIAdapter> adapter;
            dxgiDevice->GetAdapter(&adapter);
            ComPtr<IDXGIOutput> output;
            adapter->EnumOutputs(0, &output);
            ComPtr<IDXGIOutput1> output1;
            output->QueryInterface(IID_PPV_ARGS(&output1));
            output1->DuplicateOutput(m_device.Get(), &m_duplication);
            
            return false;
        }
        
        // Get the desktop texture
        ComPtr<ID3D11Texture2D> desktopTexture;
        desktopResource->QueryInterface(IID_PPV_ARGS(&desktopTexture));
        
        // Copy to our staging texture (cropped to capture area)
        D3D11_BOX sourceBox = {};
        sourceBox.left = (1920 - m_width) / 2;    // Center crop
        sourceBox.right = sourceBox.left + m_width;
        sourceBox.top = (1080 - m_height) / 2;
        sourceBox.bottom = sourceBox.top + m_height;
        sourceBox.front = 0;
        sourceBox.back = 1;
        
        m_context->CopySubresourceRegion(
            m_stagingTexture.Get(), 0, 0, 0, 0,
            desktopTexture.Get(), 0, &sourceBox
        );
        
        // Signal fence for GPU synchronization
        if (m_fence && m_context4) {
            m_context4->Signal(m_fence.Get(), ++m_fenceValue);
        }
        
        return true;
    }
    
    cudaGraphicsResource_t GetCudaResource() const { 
        return m_cudaResource; 
    }
    
private:
    bool InitializeDXGI() {
        // Create D3D11 device
        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };
        
        UINT createDeviceFlags = 0;
#ifdef _DEBUG
        createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        
        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            createDeviceFlags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            &m_device,
            &featureLevel,
            &m_context
        );
        
        if (FAILED(hr)) {
            std::cerr << "[GPUCapture] Failed to create D3D11 device" << std::endl;
            return false;
        }
        
        // Try to get ID3D11Device5 for fence support
        hr = m_device->QueryInterface(IID_PPV_ARGS(&m_device5));
        if (SUCCEEDED(hr)) {
            m_context->QueryInterface(IID_PPV_ARGS(&m_context4));
            
            // Create fence for GPU sync
            hr = m_device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
            if (SUCCEEDED(hr)) {
                m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            }
        }
        
        return true;
    }
    
    bool InitializeCUDAInterop() {
        // Create CUDA stream for capture operations
        cudaError_t err = cudaStreamCreate(&m_captureStream);
        if (err != cudaSuccess) {
            std::cerr << "[GPUCapture] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Register D3D11 texture with CUDA
        err = cudaGraphicsD3D11RegisterResource(
            &m_cudaResource,
            m_stagingTexture.Get(),
            cudaGraphicsRegisterFlagsNone
        );
        
        if (err != cudaSuccess) {
            std::cerr << "[GPUCapture] Failed to register D3D11 resource with CUDA: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Create event for frame ready signaling
        err = cudaEventCreateWithFlags(&m_frameReadyEvent, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            std::cerr << "[GPUCapture] Failed to create CUDA event: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        return true;
    }
};

// GPU capture thread function
void gpuOnlyCaptureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT) {
    auto& ctx = AppContext::getInstance();
    
    // Initialize GPU capture
    GPUCapture gpuCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
    if (!gpuCapture.Initialize()) {
        std::cerr << "[GPUCapture] Failed to initialize" << std::endl;
        return;
    }
    
    // Set thread priority
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
    
    // FPS counter
    int frameCount = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    gpuCapture.StartCapture();
    
    // Get pipeline instance
    auto& pipelineManager = needaimbot::PipelineManager::getInstance();
    auto* pipeline = pipelineManager.getPipeline();
    
    if (!pipeline) {
        std::cerr << "[GPUCapture] Pipeline not initialized" << std::endl;
        gpuCapture.StopCapture();
        return;
    }
    
    // Set CUDA resource in pipeline
    pipeline->setInputTexture(gpuCapture.GetCudaResource());
    
    // Main capture loop
    while (!ctx.should_exit) {
        // Wait for next frame
        bool frameAvailable = gpuCapture.WaitForNextFrame();
        
        if (frameAvailable) {
            frameCount++;
            
            // Measure frame interval
            auto currentTime = std::chrono::steady_clock::now();
            auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            
            // Execute pipeline (detection, tracking, mouse movement)
            if (ctx.use_cuda_graph && pipeline && pipeline->isGraphReady()) {
                pipeline->executeGraph();
            } else if (pipeline) {
                pipeline->executeDirect();
            }
            
            // Calculate FPS
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
            if (elapsed >= 1.0f) {
                float fps = frameCount / elapsed;
                ctx.g_current_capture_fps.store(fps);
                frameCount = 0;
                lastFpsTime = now;
            }
        }
        
        // Check for configuration changes
        if (ctx.capture_method_changed.load()) {
            ctx.capture_method_changed.store(false);
        }
        
        // Check exit signal
        if (ctx.should_exit) {
            break;
        }
    }
    
    // Cleanup
    gpuCapture.StopCapture();
}