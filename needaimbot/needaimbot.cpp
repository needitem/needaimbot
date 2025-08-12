#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <shellapi.h>  // For ShellExecuteEx
#include <timeapi.h>   // For timeBeginPeriod

#pragma comment(lib, "winmm.lib")

// OpenCV removed - using custom CUDA image processing
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "AppContext.h"
#include "capture/capture.h"
#include "core/constants.h"
#include "utils/constants.h"
#include "detector/detector.h"
#include "cuda/unified_graph_pipeline.h"
#include "mouse/mouse.h"
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
#include "core/sync_manager.h"
#include "core/error_manager.h"
#include "core/performance_monitor.h"


#ifndef __INTELLISENSE__
#include <cuda_runtime_api.h>
#endif
#include <iomanip> 
#include <csignal>
#include <random>
#include <tlhelp32.h>

#include "mouse/aimbot_components/AimbotTarget.h"
#include <algorithm>

// Global variable definitions
std::atomic<bool> should_exit{false};
std::mutex configMutex;
std::atomic<bool> detector_model_changed{false};
std::atomic<bool> capture_fps_changed{false};
std::atomic<bool> detection_resolution_changed{false};
std::atomic<bool> capture_borders_changed{false};
std::atomic<bool> capture_cursor_changed{false};
std::atomic<bool> show_window_changed{false};

// add_to_history function removed - use AppContext::getInstance().add_to_history() instead


std::unique_ptr<InputMethod> initializeInputMethod() {
    auto& ctx = AppContext::getInstance();

    if (ctx.config.input_method == "ARDUINO") {
        try {
            auto arduinoSerial = std::make_unique<SerialConnection>(ctx.config.arduino_port, ctx.config.arduino_baudrate);
            if (arduinoSerial->isOpen()) {
                return std::make_unique<SerialInputMethod>(arduinoSerial.release());
            }
            std::cerr << "[Mouse] Failed to open Arduino serial port " << ctx.config.arduino_port << ". Defaulting to Win32." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Mouse] Arduino initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
        }
    } else if (ctx.config.input_method == "GHUB") {
        auto gHub = std::make_unique<GhubMouse>();
        if (gHub->mouse_xy(0, 0)) {
            return std::make_unique<GHubInputMethod>(gHub.release());
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
                return std::make_unique<MakcuInputMethod>(makcuConnection.release());
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

    return std::make_unique<Win32InputMethod>();
}



// Event-based mouse thread function
void mouseThreadFunctionEventBased(MouseThread& mouseThread);

void mouseThreadFunction_OLD(MouseThread &mouseThread)
{
    auto& ctx = AppContext::getInstance();
    
    // Reduce thread priority to prevent excessive CPU usage
    SetThreadPriority(GetCurrentThread(), Constants::MOUSE_THREAD_PRIORITY);
    // Remove CPU core affinity to allow OS scheduling
    // SetThreadAffinityMask(GetCurrentThread(), 1 << 1);
    
    static int loop_count = 0;
    static bool had_target_before = false;
    
    // Cache for keyboard states to reduce system calls
    struct KeyStateCache {
        bool left_mouse = false;
        bool right_mouse = false;
        
        void update() {
            left_mouse = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
            right_mouse = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
        }
    } key_cache;
    
    while (!ctx.should_exit)
    {
        // Measure total cycle time
        auto cycle_start = std::chrono::high_resolution_clock::now();
        
        // Clear state at the beginning of each cycle to ensure no stale data
        bool current_aiming = ctx.aiming;
        bool current_has_target = false;
        Target current_target = {};  // Always start with clean state
        
        // Check for detection update without waiting
        static int last_detection_version = 0;
        bool has_new_detection = false;
        
        // Try to get new detection without waiting
        {
            std::lock_guard<std::mutex> lock(ctx.detector->detectionMutex);
            // Check if there's a new detection
            if (ctx.detector && ctx.detector->detectionVersion != last_detection_version) {
                current_has_target = ctx.detector->m_hasBestTarget;
                if (current_has_target) {
                    current_target = ctx.detector->m_bestTargetHost;
                } else {
                    current_target = {}; // Zero-initialize the struct
                }
                last_detection_version = ctx.detector->detectionVersion;
                has_new_detection = true;
            }
        }
        
        // Skip this iteration if no new detection
        if (!has_new_detection) {
            continue;
        }

        // Update AppContext for Overlay to read
        {
            std::lock_guard<std::mutex> lock(ctx.overlay_target_mutex);
            ctx.overlay_has_target.store(current_has_target);
            if (current_has_target) {
                ctx.overlay_target_info = current_target;
            } else {
                ctx.overlay_target_info = {}; // Explicitly clear if no target
            }
        }

        if (ctx.should_exit) break;

        if (ctx.input_method_changed.load()) {
            mouseThread.setInputMethod(initializeInputMethod());
            ctx.input_method_changed.store(false);
        }

        
        if (current_has_target) {
            // Convert Detection to AimbotTarget using copied data
            AimbotTarget target(
                current_target.x,
                current_target.y,
                current_target.width,
                current_target.height,
                current_target.confidence,
                current_target.classId
            );
            
            // Move mouse to target if aimbot is enabled AND aiming key is pressed
            if (current_aiming && ctx.config.enable_aimbot) {
                PERF_TIMER("MouseThread.MoveMouse");
                mouseThread.moveMouse(target);
            }
            
            // Auto-shoot if triggerbot is enabled (works independently of aiming)
            // Check if auto_shoot button is configured and pressed, or if no button is configured (always on)
            bool auto_shoot_active = ctx.config.button_auto_shoot.empty() || 
                                   ctx.config.button_auto_shoot[0] == "None" ||
                                   isAnyKeyPressed(ctx.config.button_auto_shoot);
            
            // Use cached mouse state
            bool manual_mouse_down = key_cache.left_mouse;
            
            if (ctx.config.enable_triggerbot && auto_shoot_active && !manual_mouse_down) {
                mouseThread.pressMouse(target);
            } else if (!manual_mouse_down) {
                // Only release if user isn't manually holding mouse
                mouseThread.releaseMouse();
            }
            
        } else {
            // Release mouse if no target
            mouseThread.releaseMouse();
            
            // Only reset once when transitioning from target to no target
            if (had_target_before) {
                mouseThread.resetAccumulatedStates();
                had_target_before = false;
            }
        }
        
        // Track aiming state changes for PID reset
        static bool last_aiming_state = false;
        if (last_aiming_state && !current_aiming) {
            // Aiming was just disabled - reset PID controller
            mouseThread.resetAccumulatedStates();
        }
        last_aiming_state = current_aiming;
        
        // Update the flag when we have a target
        if (current_aiming && current_has_target) {
            had_target_before = true;
        }
        
        // Update key cache periodically (moved before the continue check)
        key_cache.update();
        
        // Apply recoil compensation when both mouse buttons are pressed (ADS + Shooting)
        // This should work even without a target
        if (ctx.config.easynorecoil) {
            static bool was_recoil_active = false;
            bool recoil_active = key_cache.left_mouse && key_cache.right_mouse;
            
            // Check if crouch key (Left Control) is pressed for recoil reduction
            bool is_crouching = ctx.config.crouch_recoil_enabled && (GetAsyncKeyState(VK_LCONTROL) & 0x8000);
            
            was_recoil_active = recoil_active;
            
            if (recoil_active) {
                // Calculate recoil compensation strength with crouch modification
                float recoil_multiplier = 1.0f;
                if (is_crouching) {
                    // Apply crouch modification to recoil compensation
                    // -50% = compensate only 50% of recoil (0.5x multiplier)
                    // 0% = no change (1.0x multiplier)
                    // +50% = compensate 150% of recoil (1.5x multiplier)
                    recoil_multiplier = 1.0f + (ctx.config.crouch_recoil_reduction / 100.0f);
                    recoil_multiplier = (std::max)(0.0f, recoil_multiplier); // Prevent negative multiplier
                }
                
                // Check if we have an active weapon profile
                if (ctx.config.active_weapon_profile_index >= 0 && 
                    ctx.config.active_weapon_profile_index < ctx.config.weapon_profiles.size()) {
                    
                    WeaponRecoilProfile profile = ctx.config.weapon_profiles[ctx.config.active_weapon_profile_index];
                    // Apply crouch multiplier to the profile strength
                    profile.base_strength *= recoil_multiplier;
                    mouseThread.applyWeaponRecoilCompensation(&profile, ctx.config.active_scope_magnification);
                } else {
                    // Use simple recoil compensation with crouch multiplier
                    float adjusted_strength = ctx.config.easynorecoilstrength * recoil_multiplier;
                    mouseThread.applyRecoilCompensation(adjusted_strength);
                }
            }
        }
        
        // Measure end of cycle
        auto cycle_end = std::chrono::high_resolution_clock::now();
        float cycle_time_ms = std::chrono::duration<float, std::milli>(cycle_end - cycle_start).count();
        
        // Store the total cycle time
        ctx.g_current_total_cycle_time_ms.store(cycle_time_ms);
        ctx.add_to_history(ctx.g_total_cycle_time_history, cycle_time_ms, ctx.g_total_cycle_history_mutex);
    }

    mouseThread.releaseMouse();
    LOG_INFO("MouseThread", "Mouse thread exiting");
}

bool loadAndValidateModel(std::string& modelName, const std::vector<std::string>& availableModels) {
    auto& ctx = AppContext::getInstance();
    if (modelName.empty() && !availableModels.empty()) {
        modelName = availableModels[0];
        ctx.config.saveConfig();
        std::cout << "[MAIN] No AI model specified in config. Loaded first available model: " << modelName << std::endl;
        return true;
    }
    
    std::string modelPath = "models/" + modelName;
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

        if (!availableModels.empty()) {
            modelName = availableModels[0];
            ctx.config.saveConfig();
            std::cout << "[MAIN] Loaded first available model: " << modelName << std::endl;
            return true;
        } else {
            std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
            return false;
        }
    }
    
    return true;
}

// Signal handler for clean shutdown
static void signalHandler(int sig) {
    std::cout << "\n[MAIN] Received signal " << sig << ", initiating clean shutdown..." << std::endl;
    AppContext::getInstance().should_exit = true;
}

// Console control handler for Windows
static BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT || signal == CTRL_CLOSE_EVENT) {
        std::cout << "\n[MAIN] Console control event received, initiating clean shutdown..." << std::endl;
        AppContext::getInstance().should_exit = true;
        
        // Give the main thread time to clean up properly
        Sleep(2000);
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

// Check if running with administrator privileges
bool IsRunAsAdministrator()
{
    BOOL isAdmin = FALSE;
    HANDLE hToken = NULL;
    
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
        TOKEN_ELEVATION elevation;
        DWORD cbSize = sizeof(TOKEN_ELEVATION);
        if (GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &cbSize)) {
            isAdmin = elevation.TokenIsElevated;
        }
        CloseHandle(hToken);
    }
    
    return isAdmin;
}

// Restart the application with administrator privileges
bool RestartAsAdministrator()
{
    wchar_t szPath[MAX_PATH];
    if (GetModuleFileNameW(NULL, szPath, ARRAYSIZE(szPath))) {
        SHELLEXECUTEINFOW sei = { sizeof(sei) };
        sei.lpVerb = L"runas";
        sei.lpFile = szPath;
        sei.hwnd = NULL;
        sei.nShow = SW_NORMAL;
        
        if (!ShellExecuteExW(&sei)) {
            DWORD dwError = GetLastError();
            if (dwError == ERROR_CANCELLED) {
                // User refused the elevation
                return false;
            }
        }
        return true;
    }
    return false;
}

int main()
{
    // Set global timer resolution to 1ms for precise sleep timing
    timeBeginPeriod(1);
    
    // Initialize Gaming Performance Analyzer
    std::cout << "[INFO] Starting Gaming Performance Analyzer v1.0.0" << std::endl;
    
    // Check for administrator privileges
    if (!IsRunAsAdministrator()) {
        std::cout << "[INFO] Administrator privileges required for optimal performance." << std::endl;
        std::cout << "[INFO] Requesting administrator privileges..." << std::endl;
        
        // Ask user if they want to restart with admin privileges
        int result = MessageBoxW(NULL, 
            L"Gaming Performance Analyzer requires administrator privileges for:\n\n"
            L"• High process priority (better performance)\n"
            L"• Access to system performance counters\n"
            L"• Optimal GPU scheduling\n\n"
            L"Would you like to restart with administrator privileges?",
            L"Administrator Privileges Required", 
            MB_YESNO | MB_ICONQUESTION);
        
        if (result == IDYES) {
            if (RestartAsAdministrator()) {
                // Exit current process as we're restarting with admin
                return 0;
            } else {
                std::cout << "[WARNING] Failed to restart with administrator privileges." << std::endl;
                std::cout << "[WARNING] Running with limited performance capabilities." << std::endl;
            }
        } else {
            std::cout << "[WARNING] Running without administrator privileges." << std::endl;
            std::cout << "[WARNING] Performance may be limited in some scenarios." << std::endl;
        }
    } else {
        std::cout << "[INFO] Running with administrator privileges." << std::endl;
    }
    
    std::cout << "[INFO] Initializing performance monitoring systems..." << std::endl;
    
    // Set high process priority for better performance (requires admin)
    if (IsRunAsAdministrator()) {
        if (SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
            std::cout << "[INFO] Process priority set to HIGH for optimal performance." << std::endl;
        } else {
            std::cout << "[WARNING] Failed to set high process priority." << std::endl;
            SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
        }
    } else {
        // Without admin, we can only set up to ABOVE_NORMAL
        SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
    }
    
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
        std::cout << "[INFO] Operating System: Windows " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion << std::endl;
    }
    
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
        


        ctx.detector = new Detector();
        
        // CUDA 초기화 및 검증
        if (!ctx.detector->initializeCudaContext()) {
            std::cerr << "[MAIN] CUDA context initialization failed. Cannot continue." << std::endl;
            delete ctx.detector;
            ctx.detector = nullptr;
            std::cin.get();
            return -1;
        }

        int cuda_devices = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_devices);

        if (err != cudaSuccess)
        {
            std::cout << "[MAIN] No GPU devices with CUDA support available." << std::endl;
            delete ctx.detector;
            ctx.detector = nullptr;
            std::cin.get();
            return -1;
        }

        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with screenshoot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        MouseThread mouseThread(
            ctx.config.detection_resolution,
            static_cast<float>(ctx.config.kp_x),
            static_cast<float>(ctx.config.ki_x),
            static_cast<float>(ctx.config.kd_x),
            static_cast<float>(ctx.config.kp_y),
            static_cast<float>(ctx.config.ki_y),
            static_cast<float>(ctx.config.kd_y),
            ctx.config.bScope_multiplier,
            ctx.config.norecoil_ms,
            nullptr,
            nullptr);

        ctx.global_mouse_thread = &mouseThread;
        ctx.global_mouse_thread->setInputMethod(initializeInputMethod());

        std::vector<std::string> availableModels = getAvailableModels();
        if (!loadAndValidateModel(ctx.config.ai_model, availableModels)) {
            std::cin.get();
            return -1;
        }

        ctx.detector->initialize("models/" + ctx.config.ai_model);
        
        // Initialize CUDA Graph Pipeline for optimized GPU execution
        std::cout << "[MAIN] Initializing CUDA Graph Pipeline..." << std::endl;
        auto& pipelineManager = needaimbot::PipelineManager::getInstance();
        
        needaimbot::UnifiedPipelineConfig pipelineConfig;
        pipelineConfig.enableCapture = true;
        pipelineConfig.enableDetection = true;
        pipelineConfig.enableTracking = ctx.config.enable_tracking;
        pipelineConfig.enablePIDControl = true;
        pipelineConfig.useGraphOptimization = true;  // Enable CUDA Graph
        pipelineConfig.allowGraphUpdate = true;
        pipelineConfig.enableProfiling = false;  // Set to true for performance metrics
        
        if (!pipelineManager.initializePipeline(pipelineConfig)) {
            std::cerr << "[MAIN] Failed to initialize CUDA Graph Pipeline" << std::endl;
            std::cerr << "[MAIN] Falling back to standard pipeline" << std::endl;
            // Continue with standard pipeline
        } else {
            std::cout << "[MAIN] CUDA Graph Pipeline initialized successfully" << std::endl;
            
            // Set component references for the pipeline
            auto* pipeline = pipelineManager.getPipeline();
            if (pipeline) {
                pipeline->setDetector(ctx.detector);
                // TODO: Set tracker and PID controller when available
                // pipeline->setTracker(gpuKalmanTracker);
                // pipeline->setPIDController(gpuPidController);
                
                // Capture the graph on first execution
                cudaStream_t graphStream = nullptr;
                cudaStreamCreate(&graphStream);
                
                if (pipeline->captureGraph(graphStream)) {
                    std::cout << "[MAIN] CUDA Graph captured successfully" << std::endl;
                    ctx.use_cuda_graph = true;  // Flag to use graph in capture thread
                } else {
                    std::cout << "[MAIN] CUDA Graph capture failed, using direct execution" << std::endl;
                    ctx.use_cuda_graph = false;
                }
                
                cudaStreamDestroy(graphStream);
            }
        }

        SetThreadAffinityMask(GetCurrentThread(), 1 << 3);
        
        ctx.detector->start();

        // Create thread managers for better resource management
        ThreadManager captureThreadMgr("CaptureThread", 
            [&]() { captureThread(ctx.config.detection_resolution, ctx.config.detection_resolution); },
            THREAD_PRIORITY_ABOVE_NORMAL);
        
        ThreadManager keyThreadMgr("KeyboardThread", 
            keyboardListener,
            THREAD_PRIORITY_NORMAL);
        
        ThreadManager mouseThreadMgr("MouseThread", 
            [&mouseThread]() { mouseThreadFunctionEventBased(mouseThread); },
            THREAD_PRIORITY_TIME_CRITICAL);
        
        ThreadManager overlayThreadMgr("OverlayThread", 
            OverlayThread,
            THREAD_PRIORITY_BELOW_NORMAL);
        
        // Start all threads
        captureThreadMgr.start();
        keyThreadMgr.start();
        mouseThreadMgr.start();
        overlayThreadMgr.start();

        welcome_message();
        
        // Start performance monitoring thread
        ThreadManager perfMonitorMgr("PerformanceMonitor", []() {
            while (!AppContext::getInstance().should_exit) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                
                // Log system metrics
                auto sysMetrics = PerformanceMonitor::getInstance().getSystemMetrics();
                if (sysMetrics.cpu_usage_percent > 80.0f) {
                    LOG_WARNING("Performance", "High CPU usage: " + std::to_string(sysMetrics.cpu_usage_percent) + "%");
                }
                if (sysMetrics.memory_usage_mb > 1024) {
                    LOG_WARNING("Performance", "High memory usage: " + std::to_string(sysMetrics.memory_usage_mb) + "MB");
                }
                
                // Log slow operations
                PerformanceMonitor::getInstance().logSlowOperations();
            }
        }, THREAD_PRIORITY_LOWEST);
        perfMonitorMgr.start();

        // Wait for keyboard thread to signal exit with efficient polling
        while (keyThreadMgr.isRunning() && !ctx.should_exit) {
            // Sleep to reduce CPU usage while waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 안전한 종료 시퀀스
        std::cout << "[MAIN] Initiating safe shutdown..." << std::endl;
        
        // 1. 입력 메서드 명시적 정리 (특히 시리얼 연결)
        if (ctx.global_mouse_thread) {
            std::cout << "[MAIN] Cleaning up input method..." << std::endl;
            // 마우스 릴리스 확인
            ctx.global_mouse_thread->releaseMouse();
            // 입력 메서드 안전하게 종료
            ctx.global_mouse_thread->setInputMethod(nullptr);
        }
        
        // 2. 검출기 중지
        if (ctx.detector) {
            ctx.detector->stop();
        }

        // 3. 스레드들을 안전하게 종료 (ThreadManager가 자동으로 처리)
        std::cout << "[MAIN] Waiting for threads to finish..." << std::endl;
        
        // ThreadManager destructors will automatically stop and join threads
        // This happens in reverse order of construction (LIFO)

        // 4. 자원 정리
        if (ctx.detector) {
            delete ctx.detector;
            ctx.detector = nullptr;
            std::cout << "[MAIN] Detector resources cleaned up." << std::endl;
        }
        
        // 5. Windows 마우스 상태 정리
        std::cout << "[MAIN] Resetting Windows mouse state..." << std::endl;
        // 마우스 버튼이 눌려있을 수 있으므로 강제로 릴리스
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP | MOUSEEVENTF_RIGHTUP | MOUSEEVENTF_MIDDLEUP;
        SendInput(1, &input, sizeof(INPUT));
        
        // 마우스 위치를 현재 위치로 초기화 (움직임 없음)
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = 0;
        input.mi.dy = 0;
        SendInput(1, &input, sizeof(INPUT));
        
        // Log final statistics
        std::cout << "\n[MAIN] Final Statistics:" << std::endl;
        std::cout << "  Warnings: " << ErrorManager::getInstance().getWarningCount() << std::endl;
        std::cout << "  Errors: " << ErrorManager::getInstance().getErrorCount() << std::endl;
        std::cout << "  Critical Errors: " << ErrorManager::getInstance().getCriticalCount() << std::endl;
        
        std::cout << "\n[MAIN] Performance Summary:" << std::endl;
        auto allMetrics = PerformanceMonitor::getInstance().getAllMetrics();
        for (const auto& [name, metrics] : allMetrics) {
            if (metrics.sample_count > 0) {
                std::cout << "  " << name << ": avg=" << metrics.avg_time_ms 
                          << "ms, min=" << metrics.min_time_ms 
                          << "ms, max=" << metrics.max_time_ms << "ms" << std::endl;
            }
        }
        
        std::cout << "\n[MAIN] Safe shutdown completed." << std::endl;
        timeEndPeriod(1);  // Restore system timer
        std::exit(0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        timeEndPeriod(1);  // Restore system timer
        return -1;
    }
}

