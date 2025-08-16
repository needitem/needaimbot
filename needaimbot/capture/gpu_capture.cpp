// Unified GPU Capture Implementation
// Combines gpu_only_capture and gpu_capture_manager into a single file

#include "gpu_capture.h"
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
#include <algorithm>

#include "../AppContext.h"
#include "../cuda/simple_cuda_mat.h"
#include "../cuda/unified_graph_pipeline.h"
#include "../core/performance_monitor.h"

// Include capture headers
#include "game_capture.h"  // OBS hook-based game capture

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

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
        std::cout << "[GPUCapture::Initialize] Starting initialization..." << std::endl;
        
        // 1. Initialize DXGI and create device
        std::cout << "[GPUCapture::Initialize] Initializing DXGI..." << std::endl;
        if (!InitializeDXGI()) {
            std::cerr << "[GPUCapture::Initialize] ERROR: Failed to initialize DXGI!" << std::endl;
            return false;
        }
        std::cout << "[GPUCapture::Initialize] DXGI initialized successfully" << std::endl;
        
        // 2. Setup Desktop Duplication
        std::cout << "[GPUCapture::Initialize] Setting up Desktop Duplication..." << std::endl;
        ComPtr<IDXGIDevice> dxgiDevice;
        m_device->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
        
        ComPtr<IDXGIAdapter> adapter;
        dxgiDevice->GetAdapter(&adapter);
        
        ComPtr<IDXGIOutput> output;
        HRESULT hr = adapter->EnumOutputs(0, &output);
        if (FAILED(hr)) {
            std::cerr << "[GPUCapture::Initialize] ERROR: Failed to enumerate outputs: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        ComPtr<IDXGIOutput1> output1;
        output->QueryInterface(IID_PPV_ARGS(&output1));
        
        hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
        if (FAILED(hr)) {
            std::cerr << "[GPUCapture::Initialize] ERROR: Failed to duplicate output: 0x" << std::hex << hr << std::endl;
            std::cerr << "[GPUCapture::Initialize] Common error codes:" << std::endl;
            std::cerr << "  E_ACCESSDENIED (0x80070005): Another process already using Desktop Duplication" << std::endl;
            std::cerr << "  DXGI_ERROR_NOT_CURRENTLY_AVAILABLE (0x887A0022): Desktop Duplication not available" << std::endl;
            std::cerr << "  DXGI_ERROR_UNSUPPORTED (0x887A0004): Feature not supported" << std::endl;
            return false;
        }
        std::cout << "[GPUCapture::Initialize] Desktop Duplication created successfully" << std::endl;
        
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
    
    bool RecreateDesktopDuplication() {
        try {
            // Release existing duplication
            m_duplication.Reset();
            
            // Wait a bit before recreating
            Sleep(100);
            
            // Get DXGI device, adapter and output
            ComPtr<IDXGIDevice> dxgiDevice;
            HRESULT hr = m_device->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
            if (FAILED(hr)) {
                std::cerr << "[GPUCapture] Failed to get DXGI device: 0x" << std::hex << hr << std::dec << std::endl;
                return false;
            }
            
            ComPtr<IDXGIAdapter> adapter;
            hr = dxgiDevice->GetAdapter(&adapter);
            if (FAILED(hr)) {
                std::cerr << "[GPUCapture] Failed to get adapter: 0x" << std::hex << hr << std::dec << std::endl;
                return false;
            }
            
            ComPtr<IDXGIOutput> output;
            hr = adapter->EnumOutputs(0, &output);
            if (FAILED(hr)) {
                std::cerr << "[GPUCapture] Failed to enumerate outputs: 0x" << std::hex << hr << std::dec << std::endl;
                return false;
            }
            
            ComPtr<IDXGIOutput1> output1;
            hr = output->QueryInterface(IID_PPV_ARGS(&output1));
            if (FAILED(hr)) {
                std::cerr << "[GPUCapture] Failed to get IDXGIOutput1: 0x" << std::hex << hr << std::dec << std::endl;
                return false;
            }
            
            hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
            if (FAILED(hr)) {
                std::cerr << "[GPUCapture] Failed to duplicate output: 0x" << std::hex << hr << std::dec << std::endl;
                return false;
            }
            
            std::cout << "[GPUCapture] Desktop duplication recreated successfully" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[GPUCapture] Exception in RecreateDesktopDuplication: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool WaitForNextFrame() {
        static int callCount = 0;
        static bool frameAcquired = false;  // Track if we have an unreleased frame
        static int timeoutCount = 0;
        static int successCount = 0;
        callCount++;
        
        try {
            if (!m_isCapturing || !m_duplication) {
                return false;
            }
            
            // Debug logging disabled - running stable now
            /*
            if (callCount % 500 == 0) {
                std::cout << "[GPUCapture] Stats - Calls: " << callCount 
                         << ", Success: " << successCount 
                         << ", Timeouts: " << timeoutCount 
                         << ", FenceValue: " << m_fenceValue << std::endl;
            }
            */
            
            // Check for fence value overflow (prevent wrap around)
            if (m_fence && m_fenceValue > UINT64_MAX - 1000) {
                std::cout << "[GPUCapture] WARNING: Fence value approaching overflow, resetting..." << std::endl;
                m_fenceValue = 0;
                // Recreate fence to reset
                if (m_device5) {
                    m_fence.Reset();
                    HRESULT hr = m_device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
                    if (FAILED(hr)) {
                        std::cout << "[GPUCapture] WARNING: Failed to recreate fence, continuing without it" << std::endl;
                        m_fence.Reset();
                    }
                }
            }
            
            DXGI_OUTDUPL_FRAME_INFO frameInfo;
            ComPtr<IDXGIResource> desktopResource;
        
        // Only release if we previously acquired a frame
        if (frameAcquired) {
            HRESULT releaseHr = m_duplication->ReleaseFrame();
            if (FAILED(releaseHr)) {
                std::cerr << "[GPUCapture] WARNING: ReleaseFrame failed: 0x" << std::hex << releaseHr << std::dec << std::endl;
            }
            frameAcquired = false;
        }
        
        // Use 0 timeout to immediately return - don't wait for new frames
        HRESULT hr = m_duplication->AcquireNextFrame(0, &frameInfo, &desktopResource);
        
        // Handle timeout - this is normal when no new frame is available
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            timeoutCount++;
            // For stability, don't reuse frames indefinitely
            // Return false to skip this iteration and try again next time
            return false;
        }
        
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            std::cerr << "[GPUCapture] Access lost at frame " << callCount << ", recreating..." << std::endl;
            // Try to recreate duplication
            if (!RecreateDesktopDuplication()) {
                std::cerr << "[GPUCapture] Failed to recreate duplication" << std::endl;
                return false;
            }
            return false; // Skip this frame
        } else if (hr == DXGI_ERROR_INVALID_CALL) {
            std::cerr << "[GPUCapture] Invalid call at frame " << callCount << ", recreating..." << std::endl;
            if (!RecreateDesktopDuplication()) {
                std::cerr << "[GPUCapture] Failed to recreate after invalid call" << std::endl;
                return false;
            }
            return false;
        } else if (hr == E_INVALIDARG) {
            std::cerr << "[GPUCapture] Invalid argument at frame " << callCount << ", recreating..." << std::endl;
            if (!RecreateDesktopDuplication()) {
                std::cerr << "[GPUCapture] Failed to recreate after invalid argument" << std::endl;
                return false;
            }
            return false;
        } else if (FAILED(hr)) {
            std::cerr << "[GPUCapture] AcquireNextFrame failed at frame " << callCount 
                     << ": 0x" << std::hex << hr << std::dec << std::endl;
            
            // Try to recreate duplication with delay
            Sleep(500); // Give system time to stabilize
            if (!RecreateDesktopDuplication()) {
                std::cerr << "[GPUCapture] Failed to recreate duplication after error" << std::endl;
                // Don't crash, just return false to continue
            }
            
            return false;
        }
        
        // Successfully acquired a new frame
        frameAcquired = true;
        successCount++;
        
        // Get the desktop texture
        ComPtr<ID3D11Texture2D> desktopTexture;
        hr = desktopResource->QueryInterface(IID_PPV_ARGS(&desktopTexture));
        if (FAILED(hr) || !desktopTexture) {
            std::cerr << "[GPUCapture] Failed to get desktop texture at frame " << callCount 
                     << ": 0x" << std::hex << hr << std::dec << std::endl;
            // Release the frame we just acquired
            if (frameAcquired) {
                m_duplication->ReleaseFrame();
                frameAcquired = false;
            }
            return false;
        }
        
        // Get actual screen resolution
        D3D11_TEXTURE2D_DESC fullDesc;
        desktopTexture->GetDesc(&fullDesc);
        
        // Calculate capture area (center + offset)
        auto& ctx = AppContext::getInstance();
        int centerX = fullDesc.Width / 2;
        int centerY = fullDesc.Height / 2;
        
        // Apply offset from config
        bool useAimShootOffset = ctx.aiming && ctx.shooting;
        int offsetX = useAimShootOffset ? 
                      static_cast<int>(ctx.config.aim_shoot_offset_x) : 
                      static_cast<int>(ctx.config.crosshair_offset_x);
        int offsetY = useAimShootOffset ? 
                      static_cast<int>(ctx.config.aim_shoot_offset_y) : 
                      static_cast<int>(ctx.config.crosshair_offset_y);
        
        // Calculate top-left corner of capture area
        int srcX = centerX + offsetX - (m_width / 2);
        int srcY = centerY + offsetY - (m_height / 2);
        
        // Clamp to screen bounds
        srcX = (std::max)(0, (std::min)(srcX, (int)fullDesc.Width - m_width));
        srcY = (std::max)(0, (std::min)(srcY, (int)fullDesc.Height - m_height));
        
        // Copy to our staging texture (cropped to capture area)
        D3D11_BOX sourceBox = {};
        sourceBox.left = srcX;
        sourceBox.right = srcX + m_width;
        sourceBox.top = srcY;
        sourceBox.bottom = srcY + m_height;
        sourceBox.front = 0;
        sourceBox.back = 1;
        
        m_context->CopySubresourceRegion(
            m_stagingTexture.Get(), 0, 0, 0, 0,
            desktopTexture.Get(), 0, &sourceBox
        );
        
        // Signal fence for GPU synchronization with new value only for new frames
        if (m_fence && m_fence.Get() && m_context4) {
            try {
                m_context4->Signal(m_fence.Get(), ++m_fenceValue);
            } catch (...) {
                // Fence signaling failed, continue without it
            }
        }
        
        return true;
        
        } catch (const std::exception& e) {
            std::cerr << "[GPUCapture] Exception at frame " << callCount << ": " << e.what() << std::endl;
            
            // Release frame if we have one
            if (frameAcquired) {
                try {
                    m_duplication->ReleaseFrame();
                } catch (...) {}
                frameAcquired = false;
            }
            
            // Try to recover by recreating duplication
            if (!RecreateDesktopDuplication()) {
                std::cerr << "[GPUCapture] Failed to recover from exception" << std::endl;
            }
            return false;
        } catch (...) {
            std::cerr << "[GPUCapture] Unknown exception at frame " << callCount << std::endl;
            
            // Release frame if we have one
            if (frameAcquired) {
                try {
                    m_duplication->ReleaseFrame();
                } catch (...) {}
                frameAcquired = false;
            }
            
            // Try to recover
            if (!RecreateDesktopDuplication()) {
                std::cerr << "[GPUCapture] Failed to recover from unknown exception" << std::endl;
            }
            return false;
        }
    }
    
    cudaGraphicsResource_t GetCudaResource() const { 
        return m_cudaResource; 
    }
    
    // Public method to reinitialize CUDA resources (for periodic refresh)
    bool ReinitializeCudaResource() {
        // Unregister existing resource if any
        if (m_cudaResource) {
            cudaGraphicsUnregisterResource(m_cudaResource);
            m_cudaResource = nullptr;
        }
        
        // Re-register the D3D11 texture with CUDA
        if (m_stagingTexture) {
            cudaError_t err = cudaGraphicsD3D11RegisterResource(
                &m_cudaResource,
                m_stagingTexture.Get(),
                cudaGraphicsRegisterFlagsNone
            );
            
            if (err != cudaSuccess) {
                std::cerr << "[GPUCapture] Failed to re-register D3D11 resource with CUDA: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            return true;
        }
        return false;
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
    try {
        auto& ctx = AppContext::getInstance();
        
        std::cout << "[Capture] Starting capture thread with resolution: " << CAPTURE_WIDTH << "x" << CAPTURE_HEIGHT << std::endl;
        std::cout << "[Capture] Capture method selected: " << ctx.capture_method.load() 
                  << " (0=Desktop Duplication, 1=Virtual Camera, 2=OBS Hook, 3=Game Capture)" << std::endl;
    
    // Check if game name is configured
    std::string gameName = ctx.config.game_window_name;
    if (gameName.empty()) {
        gameName = "Apex Legends";  // Default game name
        std::cout << "[Capture] No game window name configured, using default: " << gameName << std::endl;
    }
    
    // Get screen resolution
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // Try Game Capture first (OBS hook method)
    if (ctx.capture_method.load() == 3 || ctx.capture_method.load() == 2) {
        std::cout << "[Capture] Using Game Capture (OBS Hook) for: " << gameName << std::endl;
        
        GameCapture* gameCapture = nullptr;
        try {
            gameCapture = new GameCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT, screenWidth, screenHeight, gameName);
            
            if (gameCapture->StartCapture()) {
                std::cout << "[GameCapture] Capture started successfully" << std::endl;
                
                // Get pipeline instance
                auto& pipelineManager = needaimbot::PipelineManager::getInstance();
                auto* pipeline = pipelineManager.getPipeline();
                
                if (!pipeline) {
                    std::cerr << "[GameCapture] ERROR: Pipeline not initialized!" << std::endl;
                    gameCapture->StopCapture();
                    delete gameCapture;
                    return;
                }
                
                // Set CUDA resource in pipeline
                pipeline->setInputTexture(gameCapture->GetCudaResource());
                std::cout << "[GameCapture] CUDA resource set in pipeline" << std::endl;
                
                // Main capture loop
                int frameCount = 0;
                auto lastFpsTime = std::chrono::steady_clock::now();
                auto lastFrameTime = std::chrono::steady_clock::now();
                
                std::cout << "[GameCapture] Starting main capture loop..." << std::endl;
                
                while (!ctx.should_exit) {
                    // Check for capture method change
                    if (ctx.capture_method_changed.load()) {
                        std::cout << "[GameCapture] Capture method changed, exiting Game Capture mode..." << std::endl;
                        ctx.capture_method_changed.store(false);
                        break;
                    }
                    
                    // Wait for next frame
                    bool frameAvailable = gameCapture->WaitForNextFrame();
                    
                    if (frameAvailable) {
                        frameCount++;
                        
                        // Measure frame interval
                        auto currentTime = std::chrono::steady_clock::now();
                        auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
                        lastFrameTime = currentTime;
                        
                        // Execute pipeline (detection, tracking, mouse movement)
                        {
                            PERF_TIMER("Pipeline_Total");
                            bool pipelineSuccess = false;
                            if (ctx.use_cuda_graph && pipeline && pipeline->isGraphReady()) {
                                PERF_TIMER("Pipeline_Graph");
                                pipelineSuccess = pipeline->executeGraph();
                            } else if (pipeline) {
                                PERF_TIMER("Pipeline_Direct");
                                pipelineSuccess = pipeline->executeDirect();
                            }
                            
                            if (!pipelineSuccess) {
                                std::cerr << "[GameCapture] Pipeline execution failed at frame #" << frameCount << std::endl;
                                // Continue with next frame instead of crashing
                            }
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
                    
                    // No sleep needed - WaitForNextFrame handles timing
                }
                
                gameCapture->StopCapture();
            } else {
                std::cerr << "[GameCapture] Failed to start capture, falling back to Desktop Duplication" << std::endl;
                ctx.capture_method.store(0);  // Fall back to Desktop Duplication
            }
            
            delete gameCapture;
        } catch (const std::exception& e) {
            std::cerr << "[GameCapture] Exception: " << e.what() << ", falling back to Desktop Duplication" << std::endl;
            if (gameCapture) delete gameCapture;
            ctx.capture_method.store(0);  // Fall back to Desktop Duplication
        }
        
        // If we exited due to capture method change or error, check if we should continue
        if (ctx.capture_method.load() != 0 || ctx.should_exit) {
            return;
        }
    }
    
    // Use Desktop Duplication as fallback or if selected
    if (ctx.capture_method.load() == 0 || ctx.capture_method.load() == 1) {
        std::cout << "[Capture] Using Desktop Duplication API..." << std::endl;
        GPUCapture gpuCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        if (!gpuCapture.Initialize()) {
            std::cerr << "[GPUCapture] Failed to initialize Desktop Duplication" << std::endl;
            return;
        }
        std::cout << "[GPUCapture] Desktop Duplication initialized successfully" << std::endl;
        
        // Set thread priority
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
        
        // FPS counter
        int frameCount = 0;
        auto lastFpsTime = std::chrono::steady_clock::now();
        auto lastFrameTime = std::chrono::steady_clock::now();
        float totalProcessTime = 0.0f;
        int processedFrames = 0;
        
        // START CAPTURE
        gpuCapture.StartCapture();
        std::cout << "[GPUCapture] Capture started!" << std::endl;
        
        // Get pipeline instance
        auto& pipelineManager = needaimbot::PipelineManager::getInstance();
        auto* pipeline = pipelineManager.getPipeline();
        
        if (!pipeline) {
            std::cerr << "[GPUCapture] ERROR: Pipeline not initialized!" << std::endl;
            gpuCapture.StopCapture();
            return;
        }
        std::cout << "[GPUCapture] Pipeline found and ready" << std::endl;
        
        // Set CUDA resource in pipeline
        pipeline->setInputTexture(gpuCapture.GetCudaResource());
        std::cout << "[GPUCapture] CUDA resource set in pipeline" << std::endl;
        
        std::cout << "[GPUCapture] Starting main capture loop..." << std::endl;
        
        // Frame counter for debugging
        int totalFrameCount = 0;
        
        // Main capture loop
        int consecutiveFailures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 100;
        
        while (!ctx.should_exit) {
            // Debug logging disabled - running stable now
            /*
            static int waitCount = 0;
            waitCount++;
            if (waitCount % 1000 == 0) {
                std::cout << "[GPUCapture] About to call WaitForNextFrame (call #" << waitCount << ")" << std::endl;
            }
            */
            
            // Wait for next frame with timeout check
            auto waitStart = std::chrono::steady_clock::now();
            bool frameAvailable = gpuCapture.WaitForNextFrame();
            auto waitEnd = std::chrono::steady_clock::now();
            auto waitTime = std::chrono::duration<float, std::milli>(waitEnd - waitStart).count();
            
            // Warn if wait took too long
            if (waitTime > 1000.0f) {  // More than 1 second
                std::cerr << "[GPUCapture] WARNING: WaitForNextFrame took " << waitTime << "ms!" << std::endl;
            }
            
            if (frameAvailable) {
                consecutiveFailures = 0;  // Reset failure counter
                frameCount++;
                totalFrameCount++;
                
                // Comment out periodic refresh for now - need to find root cause
                // The crash happens even without refresh, so this isn't the solution
                /*
                if (totalFrameCount % 10000 == 0 && totalFrameCount > 0) {
                    std::cout << "[GPUCapture] Would refresh at frame " << totalFrameCount << " (disabled)" << std::endl;
                }
                */
                
                // Measure frame interval
                auto currentTime = std::chrono::steady_clock::now();
                auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
                lastFrameTime = currentTime;
                
                // Measure pipeline execution time
                auto pipelineStart = std::chrono::high_resolution_clock::now();
                
                // Execute pipeline (detection, tracking, mouse movement)
                {
                    PERF_TIMER("Pipeline_Total");
                    
                    
                    auto pipelineStartTime = std::chrono::steady_clock::now();
                    bool pipelineSuccess = false;
                    
                    try {
                        // Verify pipeline and CUDA resource are valid before execution
                        if (!pipeline) {
                            std::cerr << "[GPUCapture] ERROR: Pipeline is NULL at frame #" << totalFrameCount << std::endl;
                            pipelineSuccess = false;
                        } else if (!gpuCapture.GetCudaResource()) {
                            std::cerr << "[GPUCapture] ERROR: CUDA resource is NULL at frame #" << totalFrameCount << std::endl;
                            pipelineSuccess = false;
                        } else if (ctx.use_cuda_graph && pipeline->isGraphReady()) {
                            PERF_TIMER("Pipeline_Graph");
                            pipelineSuccess = pipeline->executeGraph();
                        } else {
                            PERF_TIMER("Pipeline_Direct");
                            pipelineSuccess = pipeline->executeDirect();
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[GPUCapture] Pipeline exception at frame #" << totalFrameCount 
                                  << ": " << e.what() << std::endl;
                        pipelineSuccess = false;
                    } catch (...) {
                        std::cerr << "[GPUCapture] Unknown pipeline exception at frame #" << totalFrameCount << std::endl;
                        pipelineSuccess = false;
                    }
                    
                    auto pipelineEndTime = std::chrono::steady_clock::now();
                    auto pipelineExecTime = std::chrono::duration<float, std::milli>(pipelineEndTime - pipelineStartTime).count();
                    
                    // Warn if pipeline took too long
                    if (pipelineExecTime > 100.0f) {  // More than 100ms
                        std::cerr << "[GPUCapture] WARNING: Pipeline execution took " << pipelineExecTime 
                                  << "ms at frame #" << totalFrameCount << std::endl;
                    }
                    
                    if (!pipelineSuccess) {
                        std::cerr << "[GPUCapture] Pipeline execution failed at frame #" << totalFrameCount << std::endl;
                        // Continue with next frame instead of crashing
                    }
                }
                
                auto pipelineEnd = std::chrono::high_resolution_clock::now();
                float pipelineTime = std::chrono::duration<float, std::milli>(pipelineEnd - pipelineStart).count();
                totalProcessTime += pipelineTime;
                processedFrames++;
                
                // Calculate FPS and average processing time
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
                if (elapsed >= 1.0f) {
                    float fps = frameCount / elapsed;
                    float avgProcessTime = processedFrames > 0 ? totalProcessTime / processedFrames : 0.0f;
                    float maxPossibleFps = avgProcessTime > 0 ? 1000.0f / avgProcessTime : 0.0f;
                    
                    ctx.g_current_capture_fps.store(fps);
                    
                    // Always output FPS stats when calculated
                    float actualFrameTime = elapsed * 1000.0f / frameCount;  // Actual ms per frame
                    float captureOverhead = actualFrameTime - avgProcessTime;  // Time spent waiting for frames
                    
                    std::cout << "[Desktop Duplication] FPS: " << fps 
                              << " | Pipeline: " << avgProcessTime << "ms"
                              << " | Frame Time: " << actualFrameTime << "ms"
                              << " | Display Sync Delay: " << captureOverhead << "ms"
                              << " (Limited by " << (captureOverhead > 15 ? "60Hz monitor" : "capture rate") << ")"
                              << " | Frame #" << totalFrameCount << std::endl;
                    
                    frameCount = 0;
                    totalProcessTime = 0.0f;
                    processedFrames = 0;
                    lastFpsTime = now;
                }
            } else {
                // Frame capture failed (error already handled in WaitForNextFrame)
                consecutiveFailures++;
                
                // Don't fallback to Game Capture - just continue trying
                // This prevents switching capture methods when frames are not available
                if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                    // Reset counter to prevent overflow, but don't switch modes
                    consecutiveFailures = 0;
                    // Don't log - this is normal operation
                    // Frames are only available when screen content changes
                }
            }
            
            // Check for configuration changes
            if (ctx.capture_method_changed.load()) {
                std::cout << "[GPUCapture] Capture method changed detected! New method: " << ctx.capture_method.load() << std::endl;
                ctx.capture_method_changed.store(false);
                
                // If changed to Game Capture, need to exit this loop and restart
                if (ctx.capture_method.load() == 2 || ctx.capture_method.load() == 3) {
                    std::cout << "[GPUCapture] Switching to Game Capture mode, exiting Desktop Duplication loop..." << std::endl;
                    break;  // Exit the loop to restart with new capture method
                }
            }
            
            // Check exit signal
            if (ctx.should_exit) {
                break;
            }
        }
        
        // Cleanup
        gpuCapture.StopCapture();
        
        // Check if we need to restart with different capture method
        if ((ctx.capture_method.load() == 2 || ctx.capture_method.load() == 3) && !ctx.should_exit) {
            std::cout << "[GPUCapture] Restarting with Game Capture mode..." << std::endl;
            
            // Recursively call ourselves to restart with new capture method
            gpuOnlyCaptureThread(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        }
    }
    } catch (const std::exception& e) {
        std::cerr << "[Capture] FATAL ERROR: " << e.what() << std::endl;
        std::cerr << "[Capture] Thread terminated unexpectedly" << std::endl;
    } catch (...) {
        std::cerr << "[Capture] FATAL ERROR: Unknown exception in capture thread" << std::endl;
        std::cerr << "[Capture] Thread terminated unexpectedly" << std::endl;
    }
}