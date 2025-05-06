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
#include "sunone_aimbot_cpp.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "other_tools.h"
#include "mouse/InputMethod.h"
#include "config.h"
#include "detector/detector.h"
#include "kmboxNet.h"

// Include headers for version checking
#include <cuda_runtime_api.h>
#include <iomanip> // For std::setw

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

Config config;

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

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);
std::atomic<bool> auto_shoot_active(false);

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
    {
        std::lock_guard<std::mutex> lock(globalMouseThread->input_method_mutex);

        if (arduinoSerial)
        {
            delete arduinoSerial;
            arduinoSerial = nullptr;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
            gHub = nullptr;
        }
    }

    std::unique_ptr<InputMethod> input_method;

    if (config.input_method == "ARDUINO")
    {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
        arduinoSerial = new SerialConnection(config.arduino_port, config.arduino_baudrate);

        if (arduinoSerial && arduinoSerial->isOpen())
        {
            input_method = std::make_unique<SerialInputMethod>(arduinoSerial);
        }
    }
    else if (config.input_method == "GHUB")
    {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;

        gHub = new GhubMouse();
        if (!gHub->mouse_xy(0, 0))
        {
            std::cerr << "[Ghub] Error with opening mouse." << std::endl;
            delete gHub;
            gHub = nullptr;
        }
        else
        {
            input_method = std::make_unique<GHubInputMethod>(gHub);
        }
    }
    else if (config.input_method == "KMBOX")
    {
        std::cout << "[Mouse] Using kmboxNet method input.\n";
        char ip[256], port[256], mac[256];
        strncpy(ip, config.kmbox_ip.c_str(), sizeof(ip));
        strncpy(port, config.kmbox_port.c_str(), sizeof(port));
        strncpy(mac, config.kmbox_mac.c_str(), sizeof(mac));

        // Ensure null-termination
        ip[sizeof(ip) - 1] = '\0';
        port[sizeof(port) - 1] = '\0';
        mac[sizeof(mac) - 1] = '\0';

        int rc = kmNet_init(ip, port, mac);

        if (rc == 0)
        {
            input_method = std::make_unique<KmboxInputMethod>();
        }
        else
        {
            std::cerr << "[kmboxNet] init failed, code=" << rc << "\n";
        }
    }
    else if (config.input_method == "RAZER")
    {
        std::cout << "[Mouse] Using Razer method input." << std::endl;
        try {
            input_method = std::make_unique<RZInputMethod>();
            std::cout << "[Mouse] Razer input initialized successfully." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[Mouse] Razer initialization failed: " << e.what() << std::endl;
        }
    }
    
    if (!input_method)
    {
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
        input_method = std::make_unique<Win32InputMethod>();
    }

    globalMouseThread->setInputMethod(std::move(input_method));
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
    int lastDetectionVersion = -1;
    const void* last_target_identifier = nullptr;
    
    constexpr auto idle_timeout = std::chrono::milliseconds(30);
    constexpr auto active_timeout = std::chrono::milliseconds(5);
    
    while (!shouldExit)
    {
        auto timeout = aiming.load() ? active_timeout : idle_timeout;
        
        bool is_aiming = aiming.load();
        bool hotkey_pressed_for_trigger = auto_shoot_active.load();

        if (input_method_changed.load()) {
            initializeInputMethod();
            input_method_changed.store(false);
        }

        // Check if prediction settings need updating
        if (prediction_settings_changed.load()) {
            std::cout << "[Mouse Thread] Prediction settings changed, updating predictor." << std::endl;
            // Lock config mutex while reading prediction algorithm
            {
                std::lock_guard<std::mutex> lock(configMutex);
                mouseThread.setPredictor(config.prediction_algorithm);
            }
            prediction_settings_changed.store(false); // Reset the flag
        }

        if (config.easynorecoil && shooting.load() && zooming.load())
        {
            mouseThread.applyRecoilCompensation(config.easynorecoilstrength);
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
                const void* current_target_identifier = &best_target_from_detector;

                if (current_target_identifier != last_target_identifier)
                {
                    std::cout << "[Mouse] New target acquired or target changed, resetting predictor." << std::endl;
                    mouseThread.resetPredictor();
                    last_target_identifier = current_target_identifier;
                }

                // Convert Detection to AimbotTarget for mouse functions
                // Construct AimbotTarget directly with parameters
                AimbotTarget best_target(
                    best_target_from_detector.box.x, 
                    best_target_from_detector.box.y, 
                    best_target_from_detector.box.width, 
                    best_target_from_detector.box.height,
                    best_target_from_detector.classId
                );
                // Add other necessary fields if AimbotTarget constructor requires them

                // Now move the mouse towards the target (uses predicted X, raw Y internally)
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
                    std::cout << "[Mouse] Target lost, resetting predictor." << std::endl;
                    mouseThread.resetPredictor();
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
        capThread.join();
        detThread.join();
        mouseMovThread.join();
        overlayThread.join();

        if (arduinoSerial)
        {
            delete arduinoSerial;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
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
