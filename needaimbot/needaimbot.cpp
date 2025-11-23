#include "core/windows_headers.h"
#include <timeapi.h>
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
#include "mouse/input_drivers/ghub.h"
#include "mouse/input_drivers/InputMethod.h"
#include "mouse/input_drivers/kmboxNet.h"
#include "include/other_tools.h"
#include "core/thread_manager.h"
#include "core/error_manager.h"
#include "capture/dda_capture.h"
#include "capture/capture_interface.h"
#include "capture/dda_capture_adapter.h"


#ifndef __INTELLISENSE__
#include <cuda_runtime_api.h>
#endif
#include <iomanip>
#include <csignal>
#include <random>
#include <array>
#include <cstdio>
#include <string_view>
#include <memory>
#include <filesystem>
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
bool initializeInputMethod();

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

namespace {
    void logInputMethodFallback(std::string_view method, std::string_view reason) {
        std::cerr << "[Mouse] " << method << " initialization failed: " << reason
                  << ". Defaulting to Win32." << std::endl;
    }

    template <std::size_t N>
    std::array<char, N> copyToBuffer(const std::string& value) {
        std::array<char, N> buffer{};
        if constexpr (N > 0) {
            std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
        }
        return buffer;
    }

    // Generic input method initializer with automatic error handling
    template<typename InitFunc>
    bool tryInitMethod(const char* name, InitFunc&& init) {
        try {
            if (auto method = init()) {
                setGlobalInputMethod(std::move(method));
                return true;
            }
        } catch (const std::exception& e) {
            logInputMethodFallback(name, e.what());
        }
        return false;
    }
}

bool initializeInputMethod() {
    auto& ctx = AppContext::getInstance();

    if (ctx.config.input_method == "ARDUINO") {
        return tryInitMethod("Arduino", [&]() -> std::unique_ptr<InputMethod> {
            auto serial = std::make_unique<SerialConnection>(ctx.config.arduino_port, ctx.config.arduino_baudrate);
            if (!serial->isOpen()) {
                logInputMethodFallback("Arduino", "failed to open serial port " + ctx.config.arduino_port);
                return nullptr;
            }
            return std::make_unique<SerialInputMethod>(serial.release());
        });
    }

    if (ctx.config.input_method == "GHUB") {
        return tryInitMethod("GHub", [&]() -> std::unique_ptr<InputMethod> {
            auto gHub = std::make_unique<GhubMouse>();
            if (!gHub->mouse_xy(0, 0)) {
                logInputMethodFallback("GHub", "failed to initialize mouse driver");
                return nullptr;
            }
            return std::make_unique<GHubInputMethod>(gHub.release());
        });
    }

    if (ctx.config.input_method == "KMBOX") {
        return tryInitMethod("kmboxNet", [&]() -> std::unique_ptr<InputMethod> {
            auto ip = copyToBuffer<256>(ctx.config.kmbox_ip);
            auto port = copyToBuffer<256>(ctx.config.kmbox_port);
            auto mac = copyToBuffer<256>(ctx.config.kmbox_mac);
            if (kmNet_init(ip.data(), port.data(), mac.data()) != 0) {
                return nullptr;
            }
            return std::make_unique<KmboxInputMethod>();
        });
    }

    if (ctx.config.input_method == "MAKCU") {
        return tryInitMethod("MAKCU", [&]() -> std::unique_ptr<InputMethod> {
            const auto& ip = ctx.config.makcu_remote_ip;
            int port = ctx.config.makcu_remote_port;
            if (ip.empty() || port <= 0) {
                logInputMethodFallback("MAKCU", "invalid remote IP/port configuration");
                return nullptr;
            }
            auto method = std::make_unique<MakcuInputMethod>(ip, port);
            if (!method->isValid()) {
                logInputMethodFallback("MAKCU", "failed to initialize UDP client to " + ip + ":" + std::to_string(port));
                return nullptr;
            }
            return method;
        });
    }

    if (ctx.config.input_method == "RAZER") {
        return tryInitMethod("Razer", []() -> std::unique_ptr<InputMethod> {
            return std::make_unique<RZInputMethod>();
        });
    }

    setGlobalInputMethod(std::make_unique<Win32InputMethod>());
    return false;
}



// Mouse thread function removed - GPU now handles mouse control directly

bool loadAndValidateModel(std::string& modelName, const std::vector<std::string>& availableModels) {
    auto& ctx = AppContext::getInstance();
    bool model_changed = false;

    if (modelName.empty() && !availableModels.empty()) {
        modelName = availableModels[0];
        model_changed = true;
    }

    std::string modelPath = "models/" + modelName;
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

        if (!availableModels.empty()) {
            modelName = availableModels[0];
            model_changed = true;
        } else {
            std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
            return false;
        }
    }

    // Save config only if model was changed
    if (model_changed) {
        ctx.config.saveConfig();
    }

    return true;
}

// Initialize screen capture for the pipeline
bool initializeScreenCapture(needaimbot::UnifiedGraphPipeline* pipeline) {
    auto& ctx = AppContext::getInstance();

    if (!DDACapture::IsDDACaptureAvailable()) {
        std::cerr << "[CAPTURE] Desktop Duplication is not available on this system" << std::endl;
        return false;
    }

    static DDACapture s_ddaCapture;
    static DDACaptureAdapter s_ddaAdapter(&s_ddaCapture);

    if (!s_ddaCapture.Initialize()) {
        std::cerr << "[CAPTURE] Failed to initialize Desktop Duplication capture" << std::endl;
        return false;
    }

    int screenW = s_ddaCapture.GetScreenWidth();
    int screenH = s_ddaCapture.GetScreenHeight();
    if (screenW <= 0 || screenH <= 0) {
        std::cerr << "[CAPTURE] Desktop Duplication reported invalid screen dimensions" << std::endl;
        return false;
    }

    int detectionRes = std::max(1, ctx.config.detection_resolution);
    int captureSize = std::min(detectionRes, std::min(screenW, screenH));

    int centerX = screenW / 2 + static_cast<int>(ctx.config.crosshair_offset_x);
    int centerY = screenH / 2 + static_cast<int>(ctx.config.crosshair_offset_y);

    int maxLeft = std::max(0, screenW - captureSize);
    int maxTop = std::max(0, screenH - captureSize);

    int left = std::clamp(centerX - captureSize / 2, 0, maxLeft);
    int top = std::clamp(centerY - captureSize / 2, 0, maxTop);

    if (!s_ddaCapture.SetCaptureRegion(left, top, captureSize, captureSize)) {
        std::cerr << "[CAPTURE] Failed to configure Desktop Duplication capture region" << std::endl;
        return false;
    }

    if (!s_ddaCapture.IsCapturing() && !s_ddaCapture.StartCapture()) {
        std::cerr << "[CAPTURE] Failed to start Desktop Duplication capture thread" << std::endl;
        return false;
    }

    pipeline->setCapture(&s_ddaAdapter);
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

    // Initialize Gaming Performance Analyzer

    // Ensure console prints UTF-8 to avoid '??' for non-ASCII
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // Administrator privileges not required
    
    // Set application title
    SetConsoleTitle(L"Gaming Performance Analyzer - Monitor & Optimize Gaming Performance");
    
    // Single instance check (local scope to avoid anti-cheat detection)
    HANDLE hMutex = CreateMutex(NULL, TRUE, L"Local\\GamePerformanceAnalyzer_SingleInstance");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        MessageBox(NULL, L"Gaming Performance Analyzer is already running.\n\nPlease close the existing instance before starting a new one.",
                   L"Gaming Performance Analyzer", MB_OK | MB_ICONINFORMATION);
        CloseHandle(hMutex);
        return 0;
    }
    
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    if (GetVersionEx((OSVERSIONINFO*)&osvi)) {
    }
    
    std::cout << "\n=== Desktop Duplication Capture ===" << std::endl;
    if (DDACapture::IsDDACaptureAvailable()) {
        std::cout << "✓ Desktop Duplication Available" << std::endl;

        DDACapture capture;
        if (capture.Initialize()) {
            std::cout << "✓ DDA Capture Initialized" << std::endl;
            std::cout << "  - Full Screen: " << capture.GetWidth() << "x" << capture.GetHeight() << std::endl;

            int centerX = capture.GetWidth() / 4;
            int centerY = capture.GetHeight() / 4;
            int centerW = capture.GetWidth() / 2;
            int centerH = capture.GetHeight() / 2;

            if (capture.SetCaptureRegion(centerX, centerY, centerW, centerH)) {
                std::cout << "  - Partial Region: " << centerW << "x" << centerH
                          << " at (" << centerX << "," << centerY << ")" << std::endl;
                std::cout << "  - Status: Ready for efficient region capture" << std::endl;
            } else {
                std::cout << "  - Failed to configure partial capture region" << std::endl;
            }

            capture.Shutdown();
        } else {
            std::cout << "✗ DDA Initialization Failed" << std::endl;
        }
    } else {
        std::cout << "✗ Desktop Duplication Unavailable" << std::endl;
        std::cout << "  - Requires Windows 8+ with WDDM 1.2 drivers" << std::endl;
    }
    std::cout << "=============================\n" << std::endl;

    auto& ctx = AppContext::getInstance();

    ErrorManager::getInstance().setCriticalHandler([](const ErrorManager::ErrorEntry& error) {
        std::cerr << "[CRITICAL ERROR] " << error.component << ": " << error.message << std::endl;
    });
    
    try {
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        SetConsoleCtrlHandler(consoleHandler, TRUE);
        
        if (!ctx.config.loadConfig())
        {
            std::cerr << "[Config] Error loading config! Check config.ini." << std::endl;
            std::cin.get();
            return -1;
        }
        
        ctx.preview_enabled = ctx.config.show_window;
        
        int cuda_devices = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_devices);

        if (err != cudaSuccess)
        {
            std::cerr << "[MAIN] No GPU devices with CUDA support available." << std::endl;
            std::cin.get();
            return -1;
        }
        
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

        (void)initializeInputMethod();

        startMouseConsumer();
        
        RecoilControlThread recoilThread;
        recoilThread.setInputMethod(std::make_unique<Win32InputMethod>());
        recoilThread.setEnabled(true);
        recoilThread.start();

        std::vector<std::string> availableModels = getAvailableModels();
        if (!loadAndValidateModel(ctx.config.ai_model, availableModels)) {
            std::cin.get();
            return -1;
        }

        auto& pipelineManager = needaimbot::PipelineManager::getInstance();
        
        needaimbot::UnifiedPipelineConfig pipelineConfig;
        pipelineConfig.modelPath = "models/" + ctx.config.ai_model;
        pipelineConfig.enableCapture = true;
        pipelineConfig.enableDetection = true;
        pipelineConfig.useGraphOptimization = true;
        pipelineConfig.detectionWidth = ctx.config.detection_resolution;
        pipelineConfig.detectionHeight = ctx.config.detection_resolution;
        pipelineConfig.allowGraphUpdate = true;
        pipelineConfig.enableProfiling = false;
        
        if (!pipelineManager.initializePipeline(pipelineConfig)) {
            std::cerr << "[MAIN] Failed to initialize TensorRT Integrated Pipeline" << std::endl;
            std::cin.get();
            return -1;
        } else {
            
            auto* pipeline = pipelineManager.getPipeline();
            if (pipeline) {
                if (!initializeScreenCapture(pipeline)) {
                    std::cerr << "[MAIN] Failed to initialize screen capture" << std::endl;
                    std::cin.get();
                    return -1;
                }
            }
        }

        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        DWORD numCores = sysInfo.dwNumberOfProcessors;
        
        ThreadManager pipelineThreadMgr("UnifiedPipelineThread",
            [&]() { pipelineManager.runMainLoop(); },
            -1,
            ThreadManager::Priority::NORMAL);

        ThreadManager uiThreadMgr("CombinedUIThread",
            combinedUIThread,
            -1,
            ThreadManager::Priority::NORMAL);

        pipelineThreadMgr.start();
        uiThreadMgr.start();

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
        
        // Stop mouse consumer thread
        stopMouseConsumer();

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
