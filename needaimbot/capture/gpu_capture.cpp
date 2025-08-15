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
// #include "virtual_camera_capture.h"  // Commented out - incomplete implementation
// #include "obs_game_capture.h"        // Commented out - incomplete implementation

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
    
    bool WaitForNextFrame() {
        if (!m_isCapturing || !m_duplication) {
            return false;
        }
        
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        ComPtr<IDXGIResource> desktopResource;
        
        // Release previous frame if any
        HRESULT releaseHr = m_duplication->ReleaseFrame();
        
        // Try to acquire next frame (100ms timeout to wait for changes)
        HRESULT hr = m_duplication->AcquireNextFrame(100, &frameInfo, &desktopResource);
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            // Log occasionally to debug
            static int timeoutCount = 0;
            timeoutCount++;
            if (timeoutCount % 1000 == 0) {
                std::cout << "[GPUCapture] Timeout waiting for frame (count: " << timeoutCount << ")" << std::endl;
            }
            return false;  // No new frame available
        }
        
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            std::cerr << "[GPUCapture::WaitForNextFrame] Access lost, need to recreate duplication" << std::endl;
        } else if (FAILED(hr)) {
            std::cerr << "[GPUCapture::WaitForNextFrame] AcquireNextFrame failed: 0x" << std::hex << hr << std::dec << std::endl;
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

// Removed RegionCapture - using Virtual Camera or OBS Hook instead
/*
void runRegionCaptureLoop(RegionCapture* regionCapture, int CAPTURE_WIDTH, int CAPTURE_HEIGHT, bool deleteOnExit = true) {
    auto& ctx = AppContext::getInstance();
    
    std::cout << "[RegionCapture] Starting capture loop with resolution: " << CAPTURE_WIDTH << "x" << CAPTURE_HEIGHT << std::endl;
    
    // Set thread priority
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
    
    // FPS counter
    int frameCount = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    // Get pipeline instance
    auto& pipelineManager = needaimbot::PipelineManager::getInstance();
    auto* pipeline = pipelineManager.getPipeline();
    
    if (!pipeline) {
        std::cerr << "[RegionCapture] ERROR: Pipeline not initialized!" << std::endl;
        regionCapture->StopCapture();
        return;
    }
    
    // Set CUDA resource in pipeline
    pipeline->setInputTexture(regionCapture->GetCudaResource());
    
    // Main capture loop
    while (!ctx.should_exit) {
        // Check for resolution changes
        if (ctx.detection_resolution_changed.load()) {
            std::cout << "[RegionCapture] Resolution changed, restarting capture..." << std::endl;
            regionCapture->StopCapture();
            
            // Update with new resolution
            int newResolution = ctx.config.detection_resolution;
            delete regionCapture;
            regionCapture = new RegionCapture(newResolution, newResolution);
            
            if (!regionCapture->Initialize() || !regionCapture->StartCapture()) {
                std::cerr << "[RegionCapture] Failed to restart with new resolution" << std::endl;
                break;
            }
            
            pipeline->setInputTexture(regionCapture->GetCudaResource());
            ctx.detection_resolution_changed.store(false);
        }
        
        // Check for offset changes
        if (ctx.crosshair_offset_changed.load()) {
            std::cout << "[RegionCapture] Offset changed, restarting capture..." << std::endl;
            regionCapture->StopCapture();
            
            // Restart capture with new offset
            if (!regionCapture->StartCapture()) {
                std::cerr << "[RegionCapture] Failed to restart with new offset" << std::endl;
                break;
            }
            
            ctx.crosshair_offset_changed.store(false);
        }
        
        // Wait for next frame
        bool frameAvailable = regionCapture->WaitForNextFrame();
        
        if (frameAvailable) {
            frameCount++;
            
            // Log every 100 frames
            if (frameCount % 100 == 0) {
                std::cout << "[RegionCapture] Captured frame #" << frameCount << std::endl;
            }
            
            // Measure frame interval
            auto currentTime = std::chrono::steady_clock::now();
            auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            
            // Execute pipeline (detection, tracking, mouse movement)
            {
                PERF_TIMER("Pipeline_Total");
                if (ctx.use_cuda_graph && pipeline && pipeline->isGraphReady()) {
                    PERF_TIMER("Pipeline_Graph");
                    std::cout << "[RegionCapture] Executing CUDA Graph pipeline" << std::endl;
                    pipeline->executeGraph();
                } else if (pipeline) {
                    PERF_TIMER("Pipeline_Direct");
                    if (frameCount <= 10) {
                        std::cout << "[RegionCapture] Executing Direct pipeline for frame #" << frameCount << std::endl;
                    }
                    pipeline->executeDirect();
                }
            }
        } else {
            // Log when no frame is available
            static int noFrameCount = 0;
            noFrameCount++;
            if (noFrameCount % 100 == 0) {
                std::cout << "[RegionCapture] No frame available (count: " << noFrameCount << ")" << std::endl;
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
            std::cout << "[RegionCapture] Capture method changed detected! New method: " << ctx.capture_method.load() << std::endl;
            ctx.capture_method_changed.store(false);
            
            if (ctx.capture_method.load() != 1) {
                // User switched to Desktop Duplication
                std::cout << "[RegionCapture] Switching to Desktop Duplication mode, exiting Region Capture loop..." << std::endl;
                break;
            }
        }
        
        // Check exit signal
        if (ctx.should_exit) {
            break;
        }
    }
    
    // Cleanup
    regionCapture->StopCapture();
    
    // Check if we need to restart with Desktop Duplication
    if (ctx.capture_method.load() == 0 && !ctx.should_exit) {
        std::cout << "[RegionCapture] Restarting with Desktop Duplication mode..." << std::endl;
        
        // Delete is handled in the calling function if deleteOnExit is false
        if (deleteOnExit) {
            delete regionCapture;
        }
        
        // Recursively call ourselves to restart with new capture method
        gpuOnlyCaptureThread(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        return;
    }
    
    if (deleteOnExit) {
        delete regionCapture;
    }
}
*/

// GPU capture thread function
void gpuOnlyCaptureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT) {
    auto& ctx = AppContext::getInstance();
    
    std::cout << "[Capture] Starting capture thread with resolution: " << CAPTURE_WIDTH << "x" << CAPTURE_HEIGHT << std::endl;
    std::cout << "[Capture] Capture method selected: " << ctx.capture_method.load() 
              << " (0=Desktop Duplication, 1=Virtual Camera, 2=OBS Hook)" << std::endl;
    
    // Choose capture method based on user selection
    if (ctx.capture_method.load() == 1) {
        // Virtual Camera capture - currently disabled (incomplete implementation)
        std::cerr << "[Capture] Virtual Camera capture is not available in this build" << std::endl;
        return;  // Exit without fallback
        /*
        // Use Virtual Camera (OBS Virtual Camera or similar)
        std::cout << "[Capture] Using Virtual Camera capture..." << std::endl;
        
        VirtualCameraCapture* virtualCapture = new VirtualCameraCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        
        // Try different virtual camera names
        std::vector<std::string> cameraNames = {
            "OBS Virtual Camera",
            "OBS-Camera",
            "OBS",
            "Virtual Camera",
            "XSplit VCam",
            "ManyCam"
        };
        
        bool initialized = false;
        for (const auto& cameraName : cameraNames) {
            std::cout << "[Capture] Trying to connect to: " << cameraName << std::endl;
            if (virtualCapture->Initialize(cameraName)) {
                initialized = true;
                std::cout << "[Capture] Successfully connected to: " << cameraName << std::endl;
                break;
            }
        }
        
        if (!initialized) {
            std::cerr << "[Capture] Failed to initialize any virtual camera, falling back to Desktop Duplication" << std::endl;
            delete virtualCapture;
            ctx.capture_method.store(0);  // Fall back to Desktop Duplication
        } else {
            // Start virtual camera capture
            if (virtualCapture->StartCapture()) {
                std::cout << "[VirtualCamera] Capture started, entering main loop..." << std::endl;
                
                // Get pipeline instance
                auto& pipelineManager = needaimbot::PipelineManager::getInstance();
                auto* pipeline = pipelineManager.getPipeline();
                
                if (!pipeline) {
                    std::cerr << "[VirtualCamera] Pipeline not initialized" << std::endl;
                    virtualCapture->StopCapture();
                    delete virtualCapture;
                    return;
                }
                
                // Set CUDA resource in pipeline
                pipeline->setInputTexture(virtualCapture->GetCudaResource());
                
                // Main capture loop
                int frameCount = 0;
                auto lastFpsTime = std::chrono::steady_clock::now();
                
                while (!ctx.should_exit) {
                    // Check for capture method change
                    if (ctx.capture_method_changed.load()) {
                        std::cout << "[VirtualCamera] Capture method changed, exiting virtual camera mode..." << std::endl;
                        ctx.capture_method_changed.store(false);
                        break;
                    }
                    
                    // Wait for next frame
                    bool frameAvailable = virtualCapture->WaitForNextFrame();
                    
                    if (frameAvailable) {
                        frameCount++;
                        
                        // Execute pipeline
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
                            std::cout << "[VirtualCamera] FPS: " << fps << std::endl;
                            frameCount = 0;
                            lastFpsTime = now;
                        }
                    }
                    
                    // Small sleep to prevent busy waiting
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                
                virtualCapture->StopCapture();
            } else {
                std::cerr << "[VirtualCamera] Failed to start capture" << std::endl;
            }
            delete virtualCapture;
            return;
        }
        */
    }
    
    // Check for OBS Hook mode
    if (ctx.capture_method.load() == 2) {
        // OBS Game Capture Hook - currently disabled (incomplete implementation)
        std::cerr << "[Capture] OBS Game Capture Hook is not available in this build" << std::endl;
        return;  // Exit without fallback
        /*
        // Use OBS Game Capture Hook
        std::cout << "[Capture] Using OBS Game Capture Hook..." << std::endl;
        
        // Get game name from config (you'll need to add this to config)
        std::string gameName = ctx.config.game_window_name;
        if (gameName.empty()) {
            gameName = "Apex Legends";  // Default game name
        }
        
        OBSGameCapture* obsCapture = new OBSGameCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT, gameName);
        
        if (!obsCapture->Initialize()) {
            std::cerr << "[Capture] Failed to initialize OBS hook, falling back to Desktop Duplication" << std::endl;
            delete obsCapture;
            ctx.capture_method.store(0);
        } else {
            // Start OBS hook capture
            if (obsCapture->StartCapture()) {
                std::cout << "[OBSGameCapture] Hook started, entering main loop..." << std::endl;
                
                // Get pipeline instance
                auto& pipelineManager = needaimbot::PipelineManager::getInstance();
                auto* pipeline = pipelineManager.getPipeline();
                
                if (!pipeline) {
                    std::cerr << "[OBSGameCapture] Pipeline not initialized" << std::endl;
                    obsCapture->StopCapture();
                    delete obsCapture;
                    return;
                }
                
                // Set CUDA resource in pipeline
                pipeline->setInputTexture(obsCapture->GetCudaResource());
                
                // Main capture loop
                int frameCount = 0;
                auto lastFpsTime = std::chrono::steady_clock::now();
                
                while (!ctx.should_exit) {
                    // Check for capture method change
                    if (ctx.capture_method_changed.load()) {
                        std::cout << "[OBSGameCapture] Capture method changed, exiting OBS hook mode..." << std::endl;
                        ctx.capture_method_changed.store(false);
                        break;
                    }
                    
                    // Wait for next frame
                    bool frameAvailable = obsCapture->WaitForNextFrame();
                    
                    if (frameAvailable) {
                        frameCount++;
                        
                        // Execute pipeline
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
                            std::cout << "[OBSGameCapture] FPS: " << fps << std::endl;
                            frameCount = 0;
                            lastFpsTime = now;
                        }
                    }
                    
                    // Small sleep to prevent busy waiting
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                
                obsCapture->StopCapture();
            } else {
                std::cerr << "[OBSGameCapture] Failed to start capture" << std::endl;
            }
            delete obsCapture;
            return;
        }
        */
    }
    
    // Use Desktop Duplication (default)
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
    
    // START CAPTURE - THIS WAS MISSING!
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
    
    // Debug counter
    int debugCounter = 0;
    int noFrameCounter = 0;
    int totalFrameCount = 0;  // Total frame count for debugging
    
    // Main capture loop
    while (!ctx.should_exit) {
        // Wait for next frame
        bool frameAvailable = gpuCapture.WaitForNextFrame();
        
        if (frameAvailable) {
            frameCount++;
            totalFrameCount++;
            
            // Log every 100th frame
            // Removed frame count logging for cleaner output
            
            // Measure frame interval
            auto currentTime = std::chrono::steady_clock::now();
            auto frameDelta = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            
            // Execute pipeline (detection, tracking, mouse movement)
            {
                PERF_TIMER("Pipeline_Total");
                if (ctx.use_cuda_graph && pipeline && pipeline->isGraphReady()) {
                    PERF_TIMER("Pipeline_Graph");
                    pipeline->executeGraph();
                } else if (pipeline) {
                    PERF_TIMER("Pipeline_Direct");
                    pipeline->executeDirect();
                }
            }
            
            // Calculate FPS (internal tracking only)
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
            if (elapsed >= 1.0f) {
                float fps = frameCount / elapsed;
                ctx.g_current_capture_fps.store(fps);
                // Removed FPS logging for cleaner output
                frameCount = 0;
                lastFpsTime = now;
            }
        } else {
            noFrameCounter++;
            // Log every 1000 attempts without frame
            if (noFrameCounter % 1000 == 0) {
                std::cout << "[GPUCapture] No frame available (attempt #" << noFrameCounter << ")" << std::endl;
            }
        }
        
        // Check for configuration changes
        if (ctx.capture_method_changed.load()) {
            std::cout << "[GPUCapture] Capture method changed detected! New method: " << ctx.capture_method.load() << std::endl;
            ctx.capture_method_changed.store(false);
            
            // If changed to Region Capture, need to exit this loop and restart
            if (ctx.capture_method.load() == 1) {
                std::cout << "[GPUCapture] Switching to Region Capture mode, exiting Desktop Duplication loop..." << std::endl;
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
    if (ctx.capture_method.load() == 1 && !ctx.should_exit) {
        std::cout << "[GPUCapture] Restarting with Region Capture mode..." << std::endl;
        
        // Recursively call ourselves to restart with new capture method
        gpuOnlyCaptureThread(CAPTURE_WIDTH, CAPTURE_HEIGHT);
    }
}