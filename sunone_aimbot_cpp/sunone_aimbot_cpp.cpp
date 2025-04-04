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

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

Detector detector;
MouseThread *globalMouseThread = nullptr;
Config config;

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

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);

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
    
    constexpr auto idle_timeout = std::chrono::milliseconds(30);
    constexpr auto active_timeout = std::chrono::milliseconds(5);
    
    bool is_active = false;
    auto last_active_time = std::chrono::steady_clock::now();
    
    DetectionData detectionData;
    std::vector<float> target_scores;
    target_scores.reserve(64);
    
    const int FRAMES_BETWEEN_TIME_CHECK = 10;
    int frame_counter = 0;
    
    bool is_shooting = shooting.load();
    bool is_zooming = zooming.load();
    
    while (!shouldExit)
    {
        auto timeout = is_active ? active_timeout : idle_timeout;
        
        bool is_aiming = aiming.load();
        bool auto_shooting = config.auto_shoot;
        
        if (frame_counter % 10 == 0) {
            is_shooting = shooting.load();
            is_zooming = zooming.load();
        }
        
        bool newFrameAvailable = false;
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
                
                detectionData.boxes = std::move(detector.detectedBoxes);
                detectionData.classes = std::move(detector.detectedClasses);
                detectionData.version = detector.detectionVersion;
            }
        } 
        
        if (!newFrameAvailable) {
            if (is_active && (++frame_counter >= FRAMES_BETWEEN_TIME_CHECK)) {
                frame_counter = 0;
                if (std::chrono::steady_clock::now() - last_active_time > std::chrono::milliseconds(300)) {
                    is_active = false;
                }
            }
            continue;
        }
        
        if (config.easynorecoil && is_shooting && is_zooming) {
            mouseThread.applyRecoilCompensation(config.easynorecoilstrength);
        }
        
        if (input_method_changed.load()) {
            initializeInputMethod();
            input_method_changed.store(false);
        }

        if (detection_resolution_changed.load()) {
            {
                std::lock_guard<std::mutex> lock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution,
                    config.dpi,
                    config.fovX,
                    config.fovY,
                    config.kp_x,
                    config.ki_x,
                    config.kd_x,
                    config.kp_y,
                    config.ki_y,
                    config.kd_y,
                    config.process_noise_q,
                    config.measurement_noise_r,
                    config.auto_shoot,
                    config.bScope_multiplier);
            }
            detection_resolution_changed.store(false);
        }

        optimized::processBatchedBoxes(
            detectionData.boxes,
            detectionData.classes,
            target_scores,
            config.detection_resolution,
            config.detection_resolution,
            config.disable_headshot
        );

        // Find the best target using the pre-calculated scores
        AimbotTarget* target = findBestTarget(detectionData.boxes, 
                                            detectionData.classes, 
                                            target_scores, // Pass the calculated scores
                                            config.disable_headshot);
       
        bool has_target = (target != nullptr);
        
        if (is_aiming && has_target) {
            is_active = true;
            
            if (++frame_counter >= FRAMES_BETWEEN_TIME_CHECK) {
                frame_counter = 0;
                last_active_time = std::chrono::steady_clock::now();
            }
            
            mouseThread.moveMouse(*target);
            
            if (auto_shooting) {
                mouseThread.pressMouse(*target);
            }
        } else if (auto_shooting) {
            mouseThread.releaseMouse();
        }
        
        delete target;
    }
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

        if (!config.loadConfig())
        {
            std::cerr << "[Config] Error with loading config!" << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.input_method == "ARDUINO")
        {
            arduinoSerial = new SerialConnection(config.arduino_port, config.arduino_baudrate);
        }
        else if (config.input_method == "GHUB")
        {
            gHub = new GhubMouse();
            if (!gHub->mouse_xy(0, 0))
            {
                std::cerr << "[Ghub] Error with opening mouse." << std::endl;
                delete gHub;
                gHub = nullptr;
            }
        }

        MouseThread mouseThread(
            config.detection_resolution,
            config.dpi,
            config.fovX,
            config.fovY,
            config.kp_x,
            config.ki_x,
            config.kd_x,
            config.kp_y,
            config.ki_y,
            config.kd_y,
            config.process_noise_q,
            config.measurement_noise_r,
            config.auto_shoot,
            config.bScope_multiplier,
            arduinoSerial,
            gHub);

        globalMouseThread = &mouseThread;

        std::vector<std::string> availableModels = getAvailableModels();
        if (!loadAndValidateModel(config.ai_model, availableModels)) {
            std::cin.get();
            return -1;
        }

        detector.initialize("models/" + config.ai_model);
        detection_resolution_changed.store(true);

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