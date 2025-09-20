#include "core/windows_headers.h"

// OpenCV removed - using custom CUDA image processing
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <DbgHelp.h>
#pragma comment(lib, "dbghelp.lib")

#include "AppContext.h"
#include "core/constants.h"
#include "cuda/unified_graph_pipeline.h"
#include "cuda/cuda_resource_manager.h"
#include "mouse/mouse.h"
#include "mouse/recoil_control_thread.h"
#include "needaimbot.h"
#include "keyboard/keyboard_listener.h"
#include "overlay/overlay.h"
#include "mouse/input_drivers/SerialConnection.h"
#include "mouse/input_drivers/MakcuConnection.h"
#include "mouse/input_drivers/ghub.h"
#include "mouse/input_drivers/InputMethod.h"
#include "mouse/input_drivers/kmboxNet.h"
#include "include/other_tools.h"
#include "core/thread_manager.h"
#include "core/error_manager.h"
#include "capture/nvfbc_capture.h"


#ifndef __INTELLISENSE__
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#endif
#include <iomanip> 
#include <csignal>
#include <random>
#include <tlhelp32.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// #include "mouse/aimbot_components/AimbotTarget.h" - Removed, using core/Target.h
#include <algorithm>

// Global variable definitions
std::atomic<bool> should_exit{false};
std::mutex configMutex;
std::atomic<bool> detection_resolution_changed{false};
std::atomic<bool> capture_borders_changed{false};
std::atomic<bool> capture_cursor_changed{false};
std::atomic<bool> show_window_changed{false};

// Forward declarations
bool initializeScreenCapture(needaimbot::UnifiedGraphPipeline* pipeline);

// Combined UI thread function for keyboard + overlay
void combinedUIThread() {
    auto& ctx = AppContext::getInstance();
    
    // Launch keyboard and overlay in alternating pattern
    std::thread keyboardThread(keyboardListener);
    std::thread overlayThread(OverlayThread);
    
    // Set thread names for debugging
    #ifdef _WIN32
    SetThreadDescription(keyboardThread.native_handle(), L"KeyboardListener");
    SetThreadDescription(overlayThread.native_handle(), L"OverlayRenderer");
    #endif
    
    // Wait for both to complete
    keyboardThread.join();
    overlayThread.join();
}

std::unique_ptr<InputMethod> initializeInputMethod() {
    auto& ctx = AppContext::getInstance();

    if (ctx.config.input_method == "ARDUINO") {
        try {
            auto arduinoSerial = std::make_unique<SerialConnection>(ctx.config.arduino_port, ctx.config.arduino_baudrate);
            if (arduinoSerial->isOpen()) {
                auto method = std::make_unique<SerialInputMethod>(arduinoSerial.release());
                setGlobalInputMethod(std::move(method));
                return nullptr;  // Don't return local copy, use global only
            }
            std::cerr << "[Mouse] Failed to open Arduino serial port " << ctx.config.arduino_port << ". Defaulting to Win32." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Mouse] Arduino initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
        }
    } else if (ctx.config.input_method == "GHUB") {
        auto gHub = std::make_unique<GhubMouse>();
        if (gHub->mouse_xy(0, 0)) {
            auto method = std::make_unique<GHubInputMethod>(gHub.release());
            setGlobalInputMethod(std::move(method));
            return nullptr;  // Don't return local copy, use global only
        }
        std::cerr << "[Mouse] Failed to initialize GHub mouse driver. Defaulting to Win32." << std::endl;
    } else if (ctx.config.input_method == "KMBOX") {
        char ip[256], port[256], mac[256];
        strncpy(ip, ctx.config.kmbox_ip.c_str(), sizeof(ip) - 1);
        ip[sizeof(ip) - 1] = '\0';
        strncpy(port, ctx.config.kmbox_port.c_str(), sizeof(port) - 1);
        port[sizeof(port) - 1] = '\0';
        strncpy(mac, ctx.config.kmbox_mac.c_str(), sizeof(mac) - 1);
        mac[sizeof(mac) - 1] = '\0';

        int rc = kmNet_init(ip, port, mac);
        if (rc == 0) {
            return std::make_unique<KmboxInputMethod>();
        }
        std::cerr << "[kmboxNet] init failed, code=" << rc << ". Defaulting to Win32.\n";
    } else if (ctx.config.input_method == "MAKCU") {
        try {
            auto makcuConnection = std::make_unique<MakcuConnection>(ctx.config.makcu_port, ctx.config.makcu_baudrate);
            if (makcuConnection->isOpen()) {
                auto method = std::make_unique<MakcuInputMethod>(makcuConnection.release());
                setGlobalInputMethod(std::move(method));
                return nullptr;  // Don't return local copy, use global only
            }
            std::cerr << "[Mouse] Failed to open MAKCU port " << ctx.config.makcu_port << ". Defaulting to Win32." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Mouse] MAKCU initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
        }
    } else if (ctx.config.input_method == "RAZER") {
        try {
            return std::make_unique<RZInputMethod>();
        } catch (const std::exception& e) {
            std::cerr << "[Mouse] Razer initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
        }
    }

    // Set Win32 as global
    auto method = std::make_unique<Win32InputMethod>();
    setGlobalInputMethod(std::move(method));
    return nullptr;  // Don't return local copy, use global only
}



// Mouse thread function removed - GPU now handles mouse control directly

bool loadAndValidateModel(std::string& modelName, const std::vector<std::string>& availableModels) {
    auto& ctx = AppContext::getInstance();
    if (modelName.empty() && !availableModels.empty()) {
        modelName = availableModels[0];
        ctx.config.saveConfig();
        return true;
    }

    std::string modelPath = "models/" + modelName;
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

        if (!availableModels.empty()) {
            modelName = availableModels[0];
            ctx.config.saveConfig();
            return true;
        } else {
            std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
            return false;
        }
    }

    return true;
}

// Initialize screen capture for the pipeline
bool initializeScreenCapture(needaimbot::UnifiedGraphPipeline* pipeline) {
    // Create D3D11 device for screen capture
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel;
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &device,
        &featureLevel,
        &context
    );
    
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to create D3D11 device for screen capture" << std::endl;
        return false;
    }
    
    // Get DXGI adapter and output for desktop duplication
    ComPtr<IDXGIDevice> dxgiDevice;
    hr = device.As(&dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to get DXGI device" << std::endl;
        return false;
    }
    
    ComPtr<IDXGIAdapter> dxgiAdapter;
    hr = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to get DXGI adapter" << std::endl;
        return false;
    }
    
    ComPtr<IDXGIOutput> dxgiOutput;
    hr = dxgiAdapter->EnumOutputs(0, &dxgiOutput);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to enumerate DXGI outputs" << std::endl;
        return false;
    }
    
    ComPtr<IDXGIOutput1> dxgiOutput1;
    hr = dxgiOutput.As(&dxgiOutput1);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to get DXGI Output1 interface" << std::endl;
        return false;
    }
    
    // Get desktop dimensions
    DXGI_OUTPUT_DESC outputDesc;
    hr = dxgiOutput->GetDesc(&outputDesc);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to get output description" << std::endl;
        return false;
    }
    
    int width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    int height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
    
    
    // Create texture for screen capture (only detection resolution size)
    auto& ctx = AppContext::getInstance();
    int captureSize = ctx.config.detection_resolution;  // 320x320
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = captureSize;
    textureDesc.Height = captureSize;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    ComPtr<ID3D11Texture2D> captureTexture;
    hr = device->CreateTexture2D(&textureDesc, nullptr, &captureTexture);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to create capture texture" << std::endl;
        return false;
    }
    
    // Register texture with CUDA
    cudaGraphicsResource_t cudaResource;
    cudaError_t cudaErr = cudaGraphicsD3D11RegisterResource(
        &cudaResource,
        captureTexture.Get(),
        cudaGraphicsRegisterFlagsNone
    );
    
    if (cudaErr != cudaSuccess) {
        std::cerr << "[CAPTURE] Failed to register D3D11 texture with CUDA: " 
                  << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    
    // Set the CUDA resource in the pipeline
    pipeline->setInputTexture(cudaResource);
    
    // Create Desktop Duplication interface and store in pipeline
    ComPtr<IDXGIOutputDuplication> desktopDuplication;
    hr = dxgiOutput1->DuplicateOutput(device.Get(), &desktopDuplication);
    if (FAILED(hr)) {
        std::cerr << "[CAPTURE] Failed to create Desktop Duplication interface" << std::endl;
        return false;
    }
    
    // Store references to prevent destruction
    static ComPtr<IDXGIOutputDuplication> s_desktopDuplication = desktopDuplication;
    static ComPtr<ID3D11Device> s_device = device;
    static ComPtr<ID3D11DeviceContext> s_context = context;
    static ComPtr<ID3D11Texture2D> s_captureTexture = captureTexture;
    
    // Pass Desktop Duplication interfaces to pipeline
    pipeline->setDesktopDuplication(
        s_desktopDuplication.Get(),
        s_device.Get(), 
        s_context.Get(),
        s_captureTexture.Get()
    );
    
    
    return true;
}

// Signal handler for clean shutdown  
static void signalHandler(int sig) {
    auto& ctx = AppContext::getInstance();
    ctx.should_exit = true;
    ctx.frame_cv.notify_all();  // Wake up main thread
    ctx.aiming_cv.notify_all();  // Wake up pipeline thread if waiting
    
    // Clean up CUDA resources on signal
    CudaResourceManager::Shutdown();
}

// Console control handler for Windows
static BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT || signal == CTRL_CLOSE_EVENT) {
        auto& ctx = AppContext::getInstance();
        ctx.should_exit = true;
        ctx.frame_cv.notify_all();  // Wake up main thread
        ctx.aiming_cv.notify_all();  // Wake up pipeline thread if waiting
        
        // Clean up CUDA resources on console event (selective cleanup, no device reset)
        CudaResourceManager::Shutdown();
        
        // Give a short time for cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return TRUE;
    }
    return FALSE;
}

// Alternative entry point for Windows subsystem (no console window)
#ifdef _WINDOWS
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    return main();
}
#endif

// Administrator privilege check removed - not required for normal operation

// Crash dump handler
LONG WINAPI UnhandledExceptionHandler(EXCEPTION_POINTERS* pExceptionPointers) {
    std::cerr << "\n[CRASH] Unhandled exception occurred!" << std::endl;
    std::cerr << "[CRASH] Exception code: 0x" << std::hex << pExceptionPointers->ExceptionRecord->ExceptionCode << std::dec << std::endl;
    std::cerr << "[CRASH] Exception address: 0x" << std::hex << pExceptionPointers->ExceptionRecord->ExceptionAddress << std::dec << std::endl;
    
    // Try to get more information about the crash location
    if (pExceptionPointers->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        std::cerr << "[CRASH] Access violation - ";
        if (pExceptionPointers->ExceptionRecord->NumberParameters >= 2) {
            if (pExceptionPointers->ExceptionRecord->ExceptionInformation[0] == 0) {
                std::cerr << "attempted to read from address: 0x" << std::hex 
                         << pExceptionPointers->ExceptionRecord->ExceptionInformation[1] << std::dec << std::endl;
            } else {
                std::cerr << "attempted to write to address: 0x" << std::hex 
                         << pExceptionPointers->ExceptionRecord->ExceptionInformation[1] << std::dec << std::endl;
            }
        }
    }
    
    // Print stack trace
    std::cerr << "\n[CRASH] Stack Trace:" << std::endl;
    void* stack[100];
    WORD frames = CaptureStackBackTrace(0, 100, stack, NULL);
    
    // Get process handle for symbol resolution
    HANDLE process = GetCurrentProcess();
    SymInitialize(process, NULL, TRUE);
    
    for (WORD i = 0; i < frames; i++) {
        DWORD64 address = (DWORD64)(stack[i]);
        
        // Get symbol name
        char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
        PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = MAX_SYM_NAME;
        
        DWORD64 displacement = 0;
        if (SymFromAddr(process, address, &displacement, symbol)) {
            // Get line info
            IMAGEHLP_LINE64 line;
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD lineDisplacement = 0;
            
            if (SymGetLineFromAddr64(process, address, &lineDisplacement, &line)) {
                std::cerr << "  [" << i << "] " << symbol->Name 
                         << " + 0x" << std::hex << displacement << std::dec
                         << " (" << line.FileName << ":" << line.LineNumber << ")" << std::endl;
            } else {
                std::cerr << "  [" << i << "] " << symbol->Name 
                         << " + 0x" << std::hex << displacement << std::dec
                         << " (0x" << std::hex << address << std::dec << ")" << std::endl;
            }
        } else {
            std::cerr << "  [" << i << "] 0x" << std::hex << address << std::dec << std::endl;
        }
    }
    
    SymCleanup(process);
    
    // Clean up CUDA resources before crash
    std::cerr << "\n[CRASH] Attempting to clean up CUDA resources..." << std::endl;
    try {
        CudaResourceManager::Shutdown();
        std::cerr << "[CRASH] CUDA resources cleaned up successfully." << std::endl;
    } catch (...) {
        std::cerr << "[CRASH] Failed to clean up CUDA resources." << std::endl;
    }
    
    // Flush output
    std::cerr.flush();
    std::cout.flush();
    
    // Return EXCEPTION_CONTINUE_SEARCH to allow Windows Error Reporting to handle it
    return EXCEPTION_CONTINUE_SEARCH;
}

int main()
{
    // Install crash handler
    SetUnhandledExceptionFilter(UnhandledExceptionHandler);
    
    // Enable crash dumps
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG);
    
    // Timer resolution modification removed - using event-driven architecture instead
    
    // Initialize Gaming Performance Analyzer
    
    // Administrator privileges not required - application runs fine without elevation
    // This improves user experience and reduces security prompts
    
    // Try to set high priority (doesn't require admin)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    
    
    // Process priority removed for better system stability
    
    // Set application title
    SetConsoleTitle(L"Gaming Performance Analyzer - Monitor & Optimize Gaming Performance");
    
    // Single instance check to prevent conflicts
    HANDLE hMutex = CreateMutex(NULL, TRUE, L"Global\\GamePerformanceAnalyzer_SingleInstance");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        MessageBox(NULL, L"Gaming Performance Analyzer is already running.\n\nPlease close the existing instance before starting a new one.", 
                   L"Gaming Performance Analyzer", MB_OK | MB_ICONINFORMATION);
        CloseHandle(hMutex);
        return 0;
    }
    
    // Initialize error handling for better user experience
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    
    // Log system information for performance analysis
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    if (GetVersionEx((OSVERSIONINFO*)&osvi)) {
    }
    
    // NVFBC Capture System
    std::cout << "\n=== NVFBC Capture System ===" << std::endl;
    if (NVFBCCapture::IsNVFBCAvailable()) {
        std::cout << "✓ NVFBC Hardware Available" << std::endl;

        NVFBCCapture capture;
        if (capture.Initialize()) {
            std::cout << "✓ NVFBC Capture Initialized" << std::endl;
            std::cout << "  - Full Screen: " << capture.GetWidth() << "x" << capture.GetHeight() << std::endl;

            // Test partial capture (center quarter of screen)
            int centerX = capture.GetWidth() / 4;
            int centerY = capture.GetHeight() / 4;
            int centerW = capture.GetWidth() / 2;
            int centerH = capture.GetHeight() / 2;

            capture.SetCaptureRegion(centerX, centerY, centerW, centerH);
            std::cout << "  - Partial Region: " << centerW << "x" << centerH << " at (" << centerX << "," << centerY << ")" << std::endl;
            std::cout << "  - Status: Ready for efficient region capture" << std::endl;

            capture.Shutdown();
        } else {
            std::cout << "✗ NVFBC Initialization Failed" << std::endl;
        }
    } else {
        std::cout << "✗ NVFBC Hardware Not Available" << std::endl;
        std::cout << "  - Requires RTX GPU with NVFBC support" << std::endl;
    }
    std::cout << "=============================\n" << std::endl;

    auto& ctx = AppContext::getInstance();

    // Set up error handling
    ErrorManager::getInstance().setCriticalHandler([](const ErrorManager::ErrorEntry& error) {
        std::cerr << "[CRITICAL ERROR] " << error.component << ": " << error.message << std::endl;
        // Could trigger emergency shutdown or recovery here
    });
    
    try {
        // Set up signal handlers for clean shutdown
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        SetConsoleCtrlHandler(consoleHandler, TRUE);
        
        if (!ctx.config.loadConfig())
        {
            std::cerr << "[Config] Error loading config! Check config.ini." << std::endl;
            std::cin.get();
            return -1;
        }
        
        // Initialize preview flag based on config
        ctx.preview_enabled = ctx.config.show_window;
        


        // Initialize CUDA context directly (Phase 1 integration)
        int cuda_devices = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_devices);

        if (err != cudaSuccess)
        {
            std::cerr << "[MAIN] No GPU devices with CUDA support available." << std::endl;
            std::cin.get();
            return -1;
        }
        
        // Initialize CUDA device
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "[MAIN] Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            std::cin.get();
            return -1;
        }
        

        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cerr << "[MAIN] Error with screenshot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        // MouseThread removed - GPU handles mouse control directly
        // Initialize InputMethod for recoil control only
        auto inputMethod = initializeInputMethod();
        
        // Initialize and start Recoil Control Thread
        RecoilControlThread recoilThread;
        // Note: RecoilThread should share the same input method or use a different approach
        // For now, we'll let it use Win32 fallback since Arduino doesn't support multiple connections
        recoilThread.setInputMethod(std::make_unique<Win32InputMethod>());
        recoilThread.setEnabled(true);
        recoilThread.start();

        std::vector<std::string> availableModels = getAvailableModels();
        if (!loadAndValidateModel(ctx.config.ai_model, availableModels)) {
            std::cin.get();
            return -1;
        }

        // Initialize TensorRT Integrated Pipeline (Phase 1)
        auto& pipelineManager = needaimbot::PipelineManager::getInstance();
        
        needaimbot::UnifiedPipelineConfig pipelineConfig;
        pipelineConfig.modelPath = "models/" + ctx.config.ai_model;  // Set model path for TensorRT
        pipelineConfig.enableCapture = true;
        pipelineConfig.enableDetection = true;
        pipelineConfig.useGraphOptimization = true;  // Enable CUDA Graph
        pipelineConfig.detectionWidth = ctx.config.detection_resolution;   // Use config value
        pipelineConfig.detectionHeight = ctx.config.detection_resolution;  // Use config value
        pipelineConfig.allowGraphUpdate = true;
        pipelineConfig.enableProfiling = false;  // Set to true for performance metrics
        
        if (!pipelineManager.initializePipeline(pipelineConfig)) {
            std::cerr << "[MAIN] Failed to initialize TensorRT Integrated Pipeline" << std::endl;
            std::cin.get();
            return -1;
        } else {
            
            // Pipeline is now fully integrated - no need to set detector reference
            auto* pipeline = pipelineManager.getPipeline();
            if (pipeline) {
                
                // Initialize screen capture for the pipeline
                if (!initializeScreenCapture(pipeline)) {
                    std::cerr << "[MAIN] Failed to initialize screen capture" << std::endl;
                    std::cin.get();
                    return -1;
                }
            }
        }

        // Get number of CPU cores and set affinity dynamically
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        DWORD numCores = sysInfo.dwNumberOfProcessors;
        
        // Use last core for main thread if available, otherwise use core 0
        DWORD_PTR mask = numCores > 1 ? (1ULL << (numCores - 1)) : 1;
        SetThreadAffinityMask(GetCurrentThread(), mask);
        
        // TensorRT pipeline starts automatically when initialized

        // OPTIMIZATION: Create thread managers with optimized core affinity for better cache locality
        // GPU-intensive pipeline gets dedicated first core (best GPU driver performance)
        ThreadManager pipelineThreadMgr("UnifiedPipelineThread", 
            [&]() { pipelineManager.runMainLoop(); }, 0);
        
        // OPTIMIZATION: Combined UI thread (keyboard + overlay) to reduce thread overhead
        ThreadManager uiThreadMgr("CombinedUIThread", 
            combinedUIThread, numCores > 2 ? 1 : -1);
        
        // Mouse thread removed - GPU handles mouse control directly
        // ThreadManager mouseThreadMgr removed
        
        // Start all threads
        pipelineThreadMgr.start();
        uiThreadMgr.start();
        // mouseThreadMgr.start(); - removed, GPU handles mouse directly

        welcome_message();
        

        // Wait for exit signal using condition variable instead of polling
        {
            std::unique_lock<std::mutex> lock(ctx.configMutex);
            ctx.frame_cv.wait(lock, [&ctx]() { return ctx.should_exit.load(); });
        }

        // Optimized shutdown sequence
        
        // Signal main waiting thread to exit (only necessary notify)
        ctx.frame_cv.notify_all();
        
        // Stop pipeline first
        pipelineManager.stopMainLoop();
        
        // Clean up input method
        executeMouseClick(false); // Release any pressed mouse button
        
        // ThreadManager destructors will automatically stop and join threads
        // Pipeline cleanup
        pipelineManager.shutdownPipeline();
        
        // Reset Windows mouse state
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP | MOUSEEVENTF_RIGHTUP | MOUSEEVENTF_MIDDLEUP;
        SendInput(1, &input, sizeof(INPUT));
        
        // Log final statistics
        
        // Clean up CUDA resources
        CudaResourceManager::Shutdown();
        
        std::exit(0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        
        // Clean up CUDA resources even on error
        std::cerr << "[MAIN] Cleaning up CUDA resources after error..." << std::endl;
        CudaResourceManager::Shutdown();
        
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}

