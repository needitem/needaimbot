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

#include "capture.h"
#include "visuals.h"
#include "detector.h"
#include "mouse.h"
#include "needaimbot.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "mouse/input_drivers/SerialConnection.h"
#include "mouse/input_drivers/ghub.h"
#include "other_tools.h"
#include "mouse/input_drivers/InputMethod.h"
#include "config.h"
#include "detector/detector.h"
#include "mouse/input_drivers/kmboxNet.h"
#include "capture/optical_flow.h"

// Include headers for version checking
#ifndef __INTELLISENSE__
#include <cuda_runtime_api.h>
#endif
#include <iomanip> // For std::setw

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

Config config;
OpticalFlow g_opticalFlow;

Detector detector;
MouseThread *globalMouseThread = nullptr;

GhubMouse *gHub = nullptr;
SerialConnection *arduinoSerial = nullptr;

std::atomic<bool> optimizing(false);

std::atomic<bool> detection_resolution_changed(false);
std::atomic<bool> capture_method_changed(false);
std::atomic<bool> capture_cursor_changed(false);
std::atomic<bool> capture_borders_changed(false);
std::atomic<bool> capture_fps_changed(false);
std::atomic<bool> capture_window_changed(false);
std::atomic<bool> detector_model_changed(false);
std::atomic<bool> show_window_changed(false);
std::atomic<bool> input_method_changed(false);
std::atomic<bool> prediction_settings_changed(false);
std::atomic<bool> capture_timeout_changed(false);

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);
std::atomic<bool> auto_shoot_active(false);
std::atomic<bool> silent_aim_trigger(false);
std::atomic<bool> config_optical_flow_changed(false);

// Recoil delay state
std::chrono::steady_clock::time_point shooting_key_press_time;
std::atomic<bool> recoil_active(false);
std::atomic<bool> was_shooting(false);
std::atomic<bool> start_delay_pending(false);

// Stats variables definitions
std::atomic<float> g_current_inference_time_ms(0.0f);
std::vector<float> g_inference_time_history;
std::mutex g_inference_history_mutex;

std::atomic<float> g_current_capture_fps(0.0f);
std::vector<float> g_capture_fps_history;
std::mutex g_capture_history_mutex;

// Detector Loop Cycle Time definitions
std::atomic<float> g_current_detector_cycle_time_ms(0.0f);
std::vector<float> g_detector_cycle_time_history;
std::mutex g_detector_cycle_history_mutex;

// Frame Acquisition Time definitions
std::atomic<float> g_current_frame_acquisition_time_ms(0.0f);
std::vector<float> g_frame_acquisition_time_history;
std::mutex g_frame_acquisition_history_mutex;

// PID Calculation Time definitions
std::atomic<float> g_current_pid_calc_time_ms(0.0f);
std::vector<float> g_pid_calc_time_history;
std::mutex g_pid_calc_history_mutex;

// Predictor Calculation Time definitions
std::atomic<float> g_current_predictor_calc_time_ms(0.0f);
std::vector<float> g_predictor_calc_time_history;
std::mutex g_predictor_calc_history_mutex;

// Input Send Time definitions
std::atomic<float> g_current_input_send_time_ms(0.0f);
std::vector<float> g_input_send_time_history;
std::mutex g_input_send_history_mutex;

std::atomic<bool> config_changed_flag(false);

struct alignas(64) DetectionData {
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    int version;
    
    DetectionData() : version(0) {
        boxes.reserve(64);
        classes.reserve(64);
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
            if (squared_distance < 100.0f) {
                distance_score = 1.0f;
            } else if (squared_distance > 500000.0f) {
                distance_score = 0.0001f;
            } else {
                distance_score = 1.0f / (1.0f + std::sqrt(squared_distance));
            }
            
            float class_score = (!disable_headshot && classes[i] == class_head) ? 1.5f : 1.0f;
            
            scores[i] = distance_score * class_score;
        }
    }
}

void initializeInputMethod()
{
    // 1. Release/destroy the old InputMethod within MouseThread FIRST.
    //    This ensures its destructor runs while arduinoSerial/gHub are still valid.
    if (globalMouseThread) { 
        globalMouseThread->setInputMethod(nullptr); 
    }

    // 2. Now it's safe to delete the global resources.
    if (arduinoSerial)
    {
        delete arduinoSerial;
        arduinoSerial = nullptr;
    }

    if (gHub)
    {
        // Assuming mouse_close() is safe to call even if not fully open or already closed.
        // If gHub can be non-null but not successfully initialized, mouse_close might need a check.
        gHub->mouse_close(); 
        delete gHub;
        gHub = nullptr;
    }

    // 3. Create new resources and the new InputMethod
    std::unique_ptr<InputMethod> new_input_method_instance; 

    if (config.input_method == "ARDUINO")
    {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
        arduinoSerial = new SerialConnection(config.arduino_port, config.arduino_baudrate);

        if (arduinoSerial && arduinoSerial->isOpen())
        {
            new_input_method_instance = std::make_unique<SerialInputMethod>(arduinoSerial);
        } else {
            std::cerr << "[Mouse] Failed to open Arduino serial port " << config.arduino_port << ". Defaulting to Win32." << std::endl;
            if(arduinoSerial) { // Cleanup if new failed
                delete arduinoSerial;
                arduinoSerial = nullptr;
            }
        }
    }
    else if (config.input_method == "GHUB")
    {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        gHub = new GhubMouse();
        if (gHub && gHub->mouse_xy(0, 0)) // mouse_xy for initialization check
        {
            new_input_method_instance = std::make_unique<GHubInputMethod>(gHub);
        }
        else
        {
            std::cerr << "[Ghub] Error with opening Ghub mouse. Defaulting to Win32." << std::endl;
            if (gHub) { // Cleanup if new failed
                 gHub->mouse_close(); // Attempt to close if partially initialized
                 delete gHub;
                 gHub = nullptr;
            }
        }
    }
    else if (config.input_method == "KMBOX")
    {
        std::cout << "[Mouse] Using kmboxNet method input.\n";
        char ip[256], port[256], mac[256];
        strncpy(ip, config.kmbox_ip.c_str(), sizeof(ip) -1);
        ip[sizeof(ip) - 1] = '\0';
        strncpy(port, config.kmbox_port.c_str(), sizeof(port) - 1);
        port[sizeof(port) - 1] = '\0';
        strncpy(mac, config.kmbox_mac.c_str(), sizeof(mac) -1);
        mac[sizeof(mac) - 1] = '\0';

        int rc = kmNet_init(ip, port, mac);
        if (rc == 0)
        {
            new_input_method_instance = std::make_unique<KmboxInputMethod>();
        }
        else
        {
            std::cerr << "[kmboxNet] init failed, code=" << rc << ". Defaulting to Win32.\n";
        }
    }
    else if (config.input_method == "RAZER")
    {
        std::cout << "[Mouse] Using Razer method input." << std::endl;
        try {
            new_input_method_instance = std::make_unique<RZInputMethod>();
            std::cout << "[Mouse] Razer input initialized successfully." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[Mouse] Razer initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
            // Additional error messages from original code can be kept if desired.
        }
    }
    
    // 4. If no specific input method was successfully created, or if config specified Win32 or an unknown method.
    if (!new_input_method_instance)
    {
        // This also catches the case where config.input_method was "WIN32" or an unrecognized value from the start.
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
        new_input_method_instance = std::make_unique<Win32InputMethod>();
    }

    // 5. Set the new input method in MouseThread.
    if (globalMouseThread) {
        globalMouseThread->setInputMethod(std::move(new_input_method_instance));
    }
}

inline void handleEasyNoRecoil(MouseThread &mouseThread)
{
    if (config.easynorecoil && shooting.load() && zooming.load())
    {
        mouseThread.applyRecoilCompensation(config.easynorecoilstrength);
    }
}

void mouseThreadFunction(MouseThread &mouseThread)
{
    std::cout << "Mouse thread started." << std::endl;
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    float target_fps = config.target_fps > 0 ? config.target_fps : 60.0f; // Ensure target_fps is positive
    float target_frame_time_ms = 1000.0f / target_fps; 

    int lastDetectionVersion = -1;
    const void* last_target_identifier = nullptr;
    std::chrono::steady_clock::time_point last_successful_target_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_mouse_move_time = std::chrono::steady_clock::now();
    const std::chrono::milliseconds active_timeout(15); 
    const std::chrono::milliseconds idle_timeout(1);    

    while (!shouldExit)
    {
        auto loop_start_time = std::chrono::high_resolution_clock::now();

        if (config_optical_flow_changed.load()) {
            if (config.enable_optical_flow) {
                // Check if thread is already running; a simple way is to see if it's joinable
                // This is not perfect, a dedicated is_running flag in OpticalFlow class would be better
                std::cout << "[Config Change] Enabling optical flow. Starting thread if not already running." << std::endl;
                g_opticalFlow.startOpticalFlowThread(); // startOpticalFlowThread handles if it's already running
            } else {
                std::cout << "[Config Change] Disabling optical flow. Stopping thread." << std::endl;
                g_opticalFlow.stopOpticalFlowThread();
            }
            config_optical_flow_changed = false; // Reset flag
        }

        auto timeout = aiming.load() ? active_timeout : idle_timeout;
        
        bool is_aiming = aiming.load();
        bool hotkey_pressed_for_trigger = auto_shoot_active.load();

        if (input_method_changed.load()) {
            initializeInputMethod();
            input_method_changed.store(false);
        }

        // Handle Easy No Recoil with delays
        bool current_shooting_state = shooting.load();
        bool currently_zooming = zooming.load();

        if (config.easynorecoil) {
            if (current_shooting_state && !was_shooting.load()) { // Key just pressed
                shooting_key_press_time = std::chrono::steady_clock::now();
                if (config.easynorecoil_start_delay_ms == 0) {
                    recoil_active = true;
                    start_delay_pending = false;
                } else {
                    recoil_active = false; // Ensure recoil is off until delay is met
                    start_delay_pending = true;
                }
            } else if (!current_shooting_state && was_shooting.load()) { // Key just released
                recoil_active = false;
                start_delay_pending = false; // Clear pending delay on release
            }

            if (start_delay_pending.load() && current_shooting_state) {
                auto elapsed_since_press = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - shooting_key_press_time
                ).count();
                if (elapsed_since_press >= config.easynorecoil_start_delay_ms) {
                    recoil_active = true;
                    start_delay_pending = false;
                }
            }

            was_shooting = current_shooting_state; // Update state AFTER all logic for the current tick

            if (recoil_active.load() && currently_zooming) {
                 mouseThread.applyRecoilCompensation(config.easynorecoilstrength);
            }
        } else {
            recoil_active = false; // Ensure recoil is off if easynorecoil is disabled
            start_delay_pending = false;
        }

        // Check for silent aim trigger first
        bool silent_trigger_active = silent_aim_trigger.load(std::memory_order_acquire);
        if (silent_trigger_active) {
            silent_aim_trigger.store(false, std::memory_order_release); // Consume trigger

            Detection best_target_for_silent_aim;
            bool has_target_for_silent_aim = false;
            {
                std::unique_lock<std::mutex> lock(detector.detectionMutex);
                // Use the latest available detection data, no need to wait if already processed.
                if (detector.detectionVersion > lastDetectionVersion) {
                    lastDetectionVersion = detector.detectionVersion; // Update version if we use this data
                }
                has_target_for_silent_aim = detector.m_hasBestTarget;
                if (has_target_for_silent_aim) {
                    best_target_for_silent_aim = detector.m_bestTargetHost;
                }
            }

            if (has_target_for_silent_aim) {
                AimbotTarget silent_aim_target(
                    best_target_for_silent_aim.box.x,
                    best_target_for_silent_aim.box.y,
                    best_target_for_silent_aim.box.width,
                    best_target_for_silent_aim.box.height,
                    best_target_for_silent_aim.classId
                );
                mouseThread.executeSilentAim(silent_aim_target);
                 // After silent aim, we might not want to immediately process normal aim for this frame
                // or we might. For now, let it fall through or continue to next iteration.
            }
        }
        else {
            // Original logic for normal aiming or idle
            if (config.easynorecoil && shooting.load() && zooming.load())
            {
                // This is now handled by the new delay logic above
                // mouseThread.applyRecoilCompensation(config.easynorecoilstrength); 
            }

            bool newFrameAvailable = false;
            bool has_target_from_detector = false;
            Detection best_target_from_detector;

            {
                std::unique_lock<std::mutex> lock(detector.detectionMutex);
                detector.detectionCV.wait_for(lock, timeout, [&]() {
                    return detector.detectionVersion > lastDetectionVersion || shouldExit;
                });

                if (shouldExit)
                    break;

                if (detector.detectionVersion > lastDetectionVersion) {
                    newFrameAvailable = true;
                    lastDetectionVersion = detector.detectionVersion;

                    has_target_from_detector = detector.m_hasBestTarget;
                    if (has_target_from_detector) {
                        best_target_from_detector = detector.m_bestTargetHost;
                    }
                }
            }

            if (newFrameAvailable)
            {
                if (is_aiming && has_target_from_detector)
                {
                    const void* current_target_identifier = &best_target_from_detector; // Using address of the object as identifier

                    if (current_target_identifier != last_target_identifier)
                    {
                        if (mouseThread.hasActivePredictor()) {
                            // std::cout << "[Mouse] New target acquired or target changed, resetting predictor." << std::endl;
                            mouseThread.resetPredictor();
                        }
                        last_target_identifier = current_target_identifier;
                    }

                    AimbotTarget best_target(
                        best_target_from_detector.box.x, 
                        best_target_from_detector.box.y, 
                        best_target_from_detector.box.width, 
                        best_target_from_detector.box.height,
                        best_target_from_detector.classId
                    );
                    mouseThread.moveMouse(best_target);

                    if (hotkey_pressed_for_trigger)
                    {
                        mouseThread.pressMouse(best_target);
                    }
                    else
                    {
                        mouseThread.releaseMouse();
                    }
                }
                else
                {
                    mouseThread.releaseMouse();
                    if (last_target_identifier != nullptr) {
                        if (mouseThread.hasActivePredictor()) {
                            // std::cout << "[Mouse] Target lost, resetting predictor." << std::endl;
                            mouseThread.resetPredictor();
                        }
                        last_target_identifier = nullptr;
                    }
                }
            }
            else
            {
                if (!hotkey_pressed_for_trigger || !is_aiming) {
                    mouseThread.releaseMouse();
                }
            }
        }

        // Frame capture logic using ring buffer
        cv::cuda::GpuMat currentGpuFrame;
        bool new_frame_for_detection = false;
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            if (frameCV.wait_for(lock, timeout, []{ return newFrameAvailable.load() || shouldExit.load(); })) {
                if (shouldExit.load()) break;
            }
        }
        if (newFrameAvailable.load(std::memory_order_acquire)) {
            int idx = captureGpuWriteIdx.load(std::memory_order_acquire);
            currentGpuFrame = captureGpuBuffer[idx]; // shallow copy
            newFrameAvailable.store(false, std::memory_order_release);
            new_frame_for_detection = true;
        }

        if (shouldExit.load()) break;

        if (new_frame_for_detection && !currentGpuFrame.empty()) {
            // Enqueue for optical flow if enabled
            if (config.enable_optical_flow && g_opticalFlow.isThreadRunning()) { // Check if thread is intended to be running
                g_opticalFlow.enqueueFrame(currentGpuFrame);
            }
            
            // Detection is handled by the detector's own thread (detThread).
            // The main loop (here) gets results via detector.detectionCV.
            // So, no explicit call to detector.run_detection() here.
            // auto detection_start_time = std::chrono::high_resolution_clock::now();
            // detector.run_detection(currentGpuFrame); // REMOVED
            // auto detection_end_time = std::chrono::high_resolution_clock::now();
            // g_current_inference_time_ms = std::chrono::duration<float, std::milli>(detection_end_time - detection_start_time).count();
            // add_to_history(g_inference_time_history, g_current_inference_time_ms.load(), g_inference_history_mutex);
            // The timing for inference should be done within the Detector's inferenceThread.

            // MouseThread already processes the best target derived from Detector's output earlier in this loop.
            // if (!detectionPaused.load())
            // {
            //    mouseThread.processDetections(detector.detectedBoxes, detector.detectedClasses, detector.detectedConfidences); // REMOVED
            // }
        }

        // Aiming / Firing logic 
        auto loop_end_time = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration<float, std::milli>(loop_end_time - loop_start_time).count();
        
        target_fps = config.target_fps > 0 ? config.target_fps : 60.0f;
        target_frame_time_ms = 1000.0f / target_fps;
        if (loop_duration < target_frame_time_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(target_frame_time_ms - loop_duration)));
        }
    }

    mouseThread.releaseMouse();
    std::cout << "Mouse thread exiting." << std::endl;
}

bool loadAndValidateModel(std::string& modelName, const std::vector<std::string>& availableModels) {
    if (modelName.empty() && !availableModels.empty()) {
        modelName = availableModels[0];
        config.saveConfig();
        std::cout << "[MAIN] No AI model specified in config. Loaded first available model: " << modelName << std::endl;
        return true;
    }
    
    std::string modelPath = "models/" + modelName;
    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

        if (!availableModels.empty()) {
            modelName = availableModels[0];
            config.saveConfig();
            std::cout << "[MAIN] Loaded first available model: " << modelName << std::endl;
            return true;
        } else {
            std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
            return false;
        }
    }
    
    return true;
}

void add_to_history(std::vector<float>& history, float value, std::mutex& mtx, int max_size) {
    std::lock_guard<std::mutex> lock(mtx);
    history.push_back(value);
    if (history.size() > static_cast<size_t>(max_size)) {
        history.erase(history.begin());
    }
}

int main()
{
    // Load config first to check the verbose flag
    if (!config.loadConfig())
    {
        std::cerr << "[Config] Error loading config! Check config.ini." << std::endl;
        std::cin.get();
        return -1;
    }

    // --- Version Logging Start ---
    if (config.verbose) {
        std::cout << "--- Dependency Versions ---" << std::endl;

        // CUDA Version
        int runtimeVersion = 0;
        cudaError_t cuda_err = cudaRuntimeGetVersion(&runtimeVersion);
        if (cuda_err == cudaSuccess) {
            int major = runtimeVersion / 1000;
            int minor = (runtimeVersion % 1000) / 10;
            std::cout << std::left << std::setw(20) << "CUDA Runtime:" << major << "." << minor << std::endl;
        } else {
            std::cerr << std::left << std::setw(20) << "CUDA Runtime:" << "Error getting version - " << cudaGetErrorString(cuda_err) << std::endl;
        }

        // OpenCV Version
        std::cout << std::left << std::setw(20) << "OpenCV Build Info:" << std::endl << cv::getBuildInformation() << std::endl;
        std::cout << "---------------------------" << std::endl << std::endl;
    }
    // --- Version Logging End ---

    // Initialize the CUDA context for the detector after loading config
    detector.initializeCudaContext();

    try
    {
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

        if (config.input_method == "ARDUINO")
        {
            try {
                arduinoSerial = new SerialConnection(config.arduino_port, config.arduino_baudrate);
                if (arduinoSerial->isOpen()) {
                    std::cout << "Arduino connected on port " << config.arduino_port << "." << std::endl;
                    
                } else {
                    std::cerr << "Error: Failed to open Arduino serial port " << config.arduino_port << ".";
                    std::cerr << " Falling back to Win32 input method." << std::endl;
                    delete arduinoSerial;
                    arduinoSerial = nullptr;
                    config.input_method = "WIN32"; // Fallback
                }
            } catch (const std::exception& e) {
                std::cerr << "Error initializing Arduino serial connection: " << e.what() << ".";
                std::cerr << " Falling back to Win32 input method." << std::endl;
                if (arduinoSerial) {
                    delete arduinoSerial;
                    arduinoSerial = nullptr;
                }
                config.input_method = "WIN32"; // Fallback
            }
        }
        else if (config.input_method == "GHUB")
        {
            gHub = new GhubMouse();
            if (!gHub || !gHub->mouse_xy(0, 0)) {
                std::cerr << "Error: Failed to initialize G HUB. Check if G HUB is running.";
                std::cerr << " Falling back to Win32 input method." << std::endl;
                delete gHub;
                gHub = nullptr;
                config.input_method = "WIN32"; // Fallback
            }
        }

        MouseThread mouseThread(
            config.detection_resolution,
            config.kp_x,
            config.ki_x,
            config.kd_x,
            config.kp_y,
            config.ki_y,
            config.kd_y,
            config.bScope_multiplier,
            config.norecoil_ms,
            arduinoSerial,
            gHub);

        globalMouseThread = &mouseThread;

        std::vector<std::string> availableModels = getAvailableModels();
        if (!loadAndValidateModel(config.ai_model, availableModels)) {
            std::cin.get();
            return -1;
        }

        // Call detector initialize AFTER initializing its CUDA context
        detector.initialize("models/" + config.ai_model);

        initializeInputMethod();

        std::thread keyThread(keyboardListener);
        std::thread capThread(captureThread, config.detection_resolution, config.detection_resolution);
        std::thread detThread(&Detector::inferenceThread, &detector);
        std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
        std::thread overlayThread(OverlayThread);

        welcome_message();

        displayThread();

        keyThread.join();
        std::cout << "Keyboard listener thread joined." << std::endl;
        capThread.join();
        std::cout << "Capture thread joined." << std::endl;
        detThread.join();
        std::cout << "Detection thread joined." << std::endl;
        mouseMovThread.join();
        std::cout << "Mouse movement thread joined." << std::endl;
        overlayThread.join();
        std::cout << "Overlay thread joined." << std::endl;

        if (arduinoSerial)
        {
            delete arduinoSerial;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
        }

        // Stop Optical Flow thread before exiting
        if (config.enable_optical_flow) { // Or more robustly, check if it was ever started
            std::cout << "Shutting down: Stopping optical flow thread." << std::endl;
            g_opticalFlow.stopOpticalFlowThread();
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}
