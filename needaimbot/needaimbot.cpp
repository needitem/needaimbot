#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "AppContext.h"
#include "detector/capture.h"
#include "visuals.h"
#include "utils/constants.h"
#include "core/constants.h"
#include "detector.h"
#include "mouse.h"
#include "needaimbot.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "mouse/input_drivers/SerialConnection.h"
#include "mouse/input_drivers/ghub.h"
#include "other_tools.h"
#include "mouse/input_drivers/InputMethod.h"
#include "detector/detector.h"
#include "mouse/input_drivers/kmboxNet.h"
// #include "detector/optical_flow.h" // Optical flow removed


#ifndef __INTELLISENSE__
#include <cuda_runtime_api.h>
#endif
#include <iomanip> 





#include "needaimbot.h"
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

// Pre-allocated thread-local buffers for performance
thread_local struct {
    std::vector<float> scoreBuffer;
    bool initialized = false;
} tlBuffers;

struct alignas(64) DetectionData {
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    int version;
    
    DetectionData() : version(0) {
        boxes.reserve(Constants::DEFAULT_DETECTION_RESERVE);
        classes.reserve(Constants::DEFAULT_DETECTION_RESERVE);
    }
};

namespace optimized {
    inline void processBatchedBoxes(const std::vector<cv::Rect>& boxes, 
                                   const std::vector<int>& classes,
                                   std::vector<float>& scores,
                                   int resolution_x, 
                                   int resolution_y,
                                   bool disable_headshot) {
        const size_t count = boxes.size();
        if (count == 0) return;
        
        // Initialize thread-local buffer on first use
        if (!tlBuffers.initialized) {
            tlBuffers.scoreBuffer.reserve(Constants::THREAD_LOCAL_BUFFER_RESERVE);
            tlBuffers.initialized = true;
        }
        
        // Use pre-allocated buffer instead of resizing scores directly
        if (tlBuffers.scoreBuffer.capacity() < count) {
            tlBuffers.scoreBuffer.reserve(count * 2); // Double for future growth
        }
        scores.resize(count);
        
        const float half_res_x = resolution_x * 0.5f;
        const float half_res_y = resolution_y * 0.5f;
        const int class_head = 0; 
        
        for (size_t i = 0; i < count; i++) {
            const cv::Rect& box = boxes[i];
            const float center_x = box.x + box.width * 0.5f;
            const float center_y = box.y + box.height * 0.5f;
            
            const float diff_x = center_x - half_res_x;
            const float diff_y = center_y - half_res_y;
            
            const float squared_distance = diff_x * diff_x + diff_y * diff_y;
            
            float distance_score;
            if (squared_distance < Constants::MIN_SQUARED_DISTANCE) {
                distance_score = 1.0f;
            } else if (squared_distance > Constants::MAX_SQUARED_DISTANCE) {
                distance_score = 0.0001f;
            } else {
                float sqrtd = sqrtf(squared_distance);
                distance_score = 1.0f / (1.0f + sqrtd);
            }
            
            int head_mask = (!disable_headshot && classes[i] == class_head);
            float class_score = 1.0f + 0.5f * head_mask;
            
            scores[i] = distance_score * class_score;
        }
    }
}

std::unique_ptr<InputMethod> initializeInputMethod() {
    auto& ctx = AppContext::getInstance();

    if (ctx.config.input_method == "ARDUINO") {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
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
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        auto gHub = std::make_unique<GhubMouse>();
        if (gHub->mouse_xy(0, 0)) {
            return std::make_unique<GHubInputMethod>(gHub.release());
        }
        std::cerr << "[Mouse] Failed to initialize GHub mouse driver. Defaulting to Win32." << std::endl;
    } else if (ctx.config.input_method == "KMBOX") {
        std::cout << "[Mouse] Using kmboxNet method input.\n";
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
    } else if (ctx.config.input_method == "RAZER") {
        std::cout << "[Mouse] Using Razer method input." << std::endl;
        try {
            return std::make_unique<RZInputMethod>();
        } catch (const std::exception& e) {
            std::cerr << "[Mouse] Razer initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
        }
    }

    std::cout << "[Mouse] Using default Win32 method input." << std::endl;
    return std::make_unique<Win32InputMethod>();
}

inline void handleEasyNoRecoil(MouseThread &mouseThread)
{
    auto& ctx = AppContext::getInstance();
    // Check for left click (shooting) and right click (zooming) directly
    bool leftClicked = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    bool rightClicked = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    
    if (ctx.config.easynorecoil && leftClicked && rightClicked)
    {
        mouseThread.applyRecoilCompensation(ctx.config.easynorecoilstrength);
    }
}


void mouseThreadFunction(MouseThread &mouseThread)
{
    auto& ctx = AppContext::getInstance();
    std::cout << "Mouse thread started." << std::endl;
    
    // Reduce thread priority to prevent excessive CPU usage
    SetThreadPriority(GetCurrentThread(), Constants::MOUSE_THREAD_PRIORITY);
    // Remove CPU core affinity to allow OS scheduling
    // SetThreadAffinityMask(GetCurrentThread(), 1 << 1);
    
    static int loop_count = 0;
    
    // Cache for keyboard states to reduce system calls
    struct KeyStateCache {
        bool left_mouse = false;
        bool right_mouse = false;
        std::chrono::steady_clock::time_point last_update;
        const std::chrono::milliseconds update_interval{16}; // Update every 16ms (~60Hz)
        
        void update() {
            auto now = std::chrono::steady_clock::now();
            if (now - last_update >= update_interval) {
                left_mouse = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
                right_mouse = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
                last_update = now;
            }
        }
    } key_cache;
    
    // Dynamic wait time based on target availability
    std::chrono::milliseconds wait_timeout(Constants::ACTIVE_WAIT_TIMEOUT_MS);
    const std::chrono::milliseconds active_timeout(Constants::ACTIVE_WAIT_TIMEOUT_MS);
    const std::chrono::milliseconds idle_timeout(Constants::IDLE_WAIT_TIMEOUT_MS);
    
    while (!ctx.should_exit)
    {
        // Wait for detection update or exit signal
        static int last_detection_version = 0;
        bool current_aiming = ctx.aiming;
        bool current_has_target = false;
        Detection current_target = {};
        
        {
            std::unique_lock<std::mutex> lock(ctx.detector->detectionMutex);
            // Wait with dynamic timeout
            bool wait_result = ctx.detector->detectionCV.wait_for(lock, wait_timeout, [&]() { 
                return ctx.should_exit || ctx.input_method_changed.load() || 
                       ctx.detector->detectionVersion != last_detection_version; 
            });
            
            // Copy target data while mutex is held
            if (ctx.detector) {
                // If timeout occurred, treat as no target to prevent stale data usage
                if (!wait_result) {
                    current_has_target = false;
                    current_target = {};
                } else {
                    // LOG: About to copy data from Detector.
                    if (ctx.config.verbose) std::cout << "[Mouse] Reading from Detector. Detector state: hasBestTarget=" << (ctx.detector->m_hasBestTarget ? "true" : "false") << std::endl;

                    current_has_target = ctx.detector->m_hasBestTarget;
                    if (current_has_target) {
                        current_target = ctx.detector->m_bestTargetHost;
                    } else {
                        current_target = {}; // Zero-initialize the struct
                    }
                    last_detection_version = ctx.detector->detectionVersion;

                    // LOG: Data has been copied.
                    if (ctx.config.verbose) std::cout << "[Mouse] Copied to local. Local state: current_has_target=" << (current_has_target ? "true" : "false") << ", Pos: (" << current_target.x << ", " << current_target.y << ")" << std::endl;
                }
            }
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
            // Validate target is within screen bounds
            bool target_valid = (current_target.x >= 0 && 
                                current_target.y >= 0 &&
                                current_target.width > 0 && 
                                current_target.height > 0 &&
                                current_target.x + current_target.width <= ctx.config.detection_resolution &&
                                current_target.y + current_target.height <= ctx.config.detection_resolution &&
                                current_target.confidence > 0.0f);
            
            if (target_valid) {
                // Convert Detection to AimbotTarget using copied data
                AimbotTarget target(
                    current_target.x,
                    current_target.y,
                    current_target.width,
                    current_target.height,
                    current_target.classId
                );
                
                // Move mouse to target if aimbot is enabled AND aiming key is pressed
                if (current_aiming && ctx.config.enable_aimbot) {
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
                // Invalid target - release mouse
                mouseThread.releaseMouse();
            }
            
        } else {
            // Release mouse if no target
            mouseThread.releaseMouse();
            
            // Only reset once when transitioning from target to no target
            static bool had_target_before = false;
            if (had_target_before) {
                mouseThread.resetAccumulatedStates();
                had_target_before = false;
            }
        }
        
        // Update the flag when we have a target
        if (current_aiming && current_has_target) {
            static bool& had_target_before = *[]() { static bool flag = false; return &flag; }();
            had_target_before = true;
        }
        
        // Update dynamic wait timeout based on target presence
        wait_timeout = current_has_target ? active_timeout : idle_timeout;
        
        // Skip expensive operations when no target and not aiming
        if (!current_has_target && !current_aiming) {
            continue;
        }
        
        // Update key cache periodically
        key_cache.update();
        
        // Apply recoil compensation when both mouse buttons are pressed (ADS + Shooting)
        if (ctx.config.easynorecoil) {
            static bool was_recoil_active = false;
            bool recoil_active = key_cache.left_mouse && key_cache.right_mouse;
            
            // Track recoil state changes silently
            was_recoil_active = recoil_active;
            
            if (recoil_active) {
                // Check if we have an active weapon profile
                if (ctx.config.active_weapon_profile_index >= 0 && 
                    ctx.config.active_weapon_profile_index < ctx.config.weapon_profiles.size()) {
                    
                    const WeaponRecoilProfile& profile = ctx.config.weapon_profiles[ctx.config.active_weapon_profile_index];
                    mouseThread.applyWeaponRecoilCompensation(&profile, ctx.config.active_scope_magnification);
                } else {
                    // Use simple recoil compensation
                    mouseThread.applyRecoilCompensation(ctx.config.easynorecoilstrength);
                }
            }
        }
    }

    mouseThread.releaseMouse();
    std::cout << "Mouse thread exiting." << std::endl;
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

int main()
{
    auto& ctx = AppContext::getInstance();
    try {
        if (!ctx.config.loadConfig())
        {
            std::cerr << "[Config] Error loading config! Check config.ini." << std::endl;
            std::cin.get();
            return -1;
        }
        

        if (ctx.config.verbose) {
            std::cout << "--- Dependency Versions ---" << std::endl;

            int runtimeVersion = 0;
            cudaError_t cuda_err = cudaRuntimeGetVersion(&runtimeVersion);
            if (cuda_err == cudaSuccess) {
                int major = runtimeVersion / 1000;
                int minor = (runtimeVersion % 1000) / 10;
                std::cout << std::left << std::setw(20) << "CUDA Runtime:" << major << "." << minor << std::endl;
            } else {
                std::cerr << std::left << std::setw(20) << "CUDA Runtime:" << "Error getting version - " << cudaGetErrorString(cuda_err) << std::endl;
            }

            std::cout << std::left << std::setw(20) << "OpenCV Build Info:" << std::endl << cv::getBuildInformation() << std::endl;
            std::cout << "---------------------------" << std::endl << std::endl;
        }

        ctx.detector = new Detector();
        ctx.detector->initializeCudaContext();

        int cuda_devices = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_devices);

        if (err != cudaSuccess)
        {
            std::cout << "[MAIN] No GPU devices with CUDA support available." << std::endl;
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
            ctx.config.kp_x,
            ctx.config.ki_x,
            ctx.config.kd_x,
            ctx.config.kp_y,
            ctx.config.ki_y,
            ctx.config.kd_y,
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

        SetThreadAffinityMask(GetCurrentThread(), 1 << 3);
        
        ctx.detector->start();

        // Start capture thread
        std::thread captureThrd(captureThread, ctx.config.detection_resolution, ctx.config.detection_resolution);

        std::thread keyThread(keyboardListener);
        std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
        std::thread overlayThread(OverlayThread);
        
        SetThreadPriority(keyThread.native_handle(), THREAD_PRIORITY_NORMAL);
        // Detector thread priorities managed internally
        SetThreadPriority(mouseMovThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
        SetThreadPriority(overlayThread.native_handle(), THREAD_PRIORITY_BELOW_NORMAL);

        welcome_message();

        if (keyThread.joinable()) {
            keyThread.join();
        }

        ctx.detector->stop();

        if (captureThrd.joinable()) {
            captureThrd.detach();
        }

        if (mouseMovThread.joinable()) {
            mouseMovThread.detach();
        }

        if (overlayThread.joinable()) {
            overlayThread.detach();
        }

        delete ctx.detector;
        ctx.detector = nullptr;
        
        std::exit(0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}

