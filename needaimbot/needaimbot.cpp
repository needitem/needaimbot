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
#include <algorithm>

void add_to_history(std::vector<float>& history, float value, std::mutex& mtx, int max_size) {
    std::lock_guard<std::mutex> lock(mtx);
    history.push_back(value);
    if (history.size() > max_size)
    {
        history.erase(history.begin());
    }
}

struct alignas(64) DetectionData {
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    int version;
    
    DetectionData() : version(0) {
        boxes.reserve(512);  // Increased from 64 to 512
        classes.reserve(512);  // Increased from 64 to 512
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
        
        if (scores.capacity() < count) {
            scores.reserve(std::max(count, static_cast<size_t>(512)));
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
            if (squared_distance < 100.0f) {
                distance_score = 1.0f;
            } else if (squared_distance > 500000.0f) {
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
            if (arduinoSerial && arduinoSerial->isOpen()) {
                return std::make_unique<SerialInputMethod>(arduinoSerial.release());
            }
            std::cerr << "[Mouse] Failed to open Arduino serial port " << ctx.config.arduino_port << ". Defaulting to Win32." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Mouse] Arduino initialization failed: " << e.what() << ". Defaulting to Win32." << std::endl;
        }
    } else if (ctx.config.input_method == "GHUB") {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        auto gHub = std::make_unique<GhubMouse>();
        if (gHub && gHub->mouse_xy(0, 0)) {
            return std::make_unique<GHubInputMethod>(gHub.release());
        }
        std::cerr << "[Ghub] Error with opening Ghub mouse. Defaulting to Win32." << std::endl;
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
    if (ctx.config.easynorecoil && ctx.shooting.load() && ctx.zooming.load())
    {
        mouseThread.applyRecoilCompensation(ctx.config.easynorecoilstrength);
    }
    
    // Optical flow recoil compensation removed"
}

// #include "constants.h" // File removed
// #include "mouse_logic.h" // File removed

void mouseThreadFunction(MouseThread &mouseThread)
{
    auto& ctx = AppContext::getInstance();
    std::cout << "Mouse thread started." << std::endl;
    
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    SetThreadAffinityMask(GetCurrentThread(), 1 << 1);
    
    while (!ctx.shouldExit)
    {
        // Optical flow configuration removed

        if (ctx.input_method_changed.load()) {
            mouseThread.setInputMethod(initializeInputMethod());
            ctx.input_method_changed.store(false);
        }

        // MouseLogic functions removed - functionality integrated into MouseThread"
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 1ms idle timeout
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

// Duplicate add_to_history function removed (already defined above)

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

        ctx.globalMouseThread = &mouseThread;
        ctx.globalMouseThread->setInputMethod(initializeInputMethod());

        std::vector<std::string> availableModels = getAvailableModels();
        if (!loadAndValidateModel(ctx.config.ai_model, availableModels)) {
            std::cin.get();
            return -1;
        }

        ctx.detector->initialize("models/" + ctx.config.ai_model);

        SetThreadAffinityMask(GetCurrentThread(), 1 << 3);
        
        ctx.detector->start();

        std::thread keyThread(keyboardListener);
        std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
        std::thread overlayThread(OverlayThread);
        
        SetThreadPriority(keyThread.native_handle(), THREAD_PRIORITY_NORMAL);
        // Detector thread priorities managed internally
        SetThreadPriority(mouseMovThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
        SetThreadPriority(overlayThread.native_handle(), THREAD_PRIORITY_BELOW_NORMAL);

        welcome_message();

        keyThread.join();
        std::cout << "Keyboard listener thread joined." << std::endl;

        ctx.detector->stop();
        std::cout << "Capture and detection threads joined." << std::endl;

        mouseMovThread.join();
        std::cout << "Mouse movement thread joined." << std::endl;
        overlayThread.join();
        std::cout << "Overlay thread joined." << std::endl;

        // Optical flow shutdown code removed

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

