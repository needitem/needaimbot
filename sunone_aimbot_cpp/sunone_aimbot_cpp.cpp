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

// 노리코일 처리를 위한 함수 복원 - 인라인화로 함수 호출 오버헤드 감소
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
    
    // 스레드 대기 시간 최적화 (ms)
    constexpr auto idle_timeout = std::chrono::milliseconds(30);
    constexpr auto active_timeout = std::chrono::milliseconds(5);
    
    // 활성 상태 트래킹
    bool is_active = false;
    auto last_active_time = std::chrono::steady_clock::now();
    
    // 메모리 할당 최소화를 위한 정적 객체 선언
    static std::vector<cv::Rect> boxes;
    static std::vector<int> classes;
    static AimbotTarget staticTarget(0, 0, 0, 0, 0); // 재사용 가능한 타겟 객체
    
    // 프레임 카운터 추가 (시간 측정 최적화)
    const int FRAMES_BETWEEN_TIME_CHECK = 10;
    int frame_counter = 0;
    
    while (!shouldExit)
    {
        // 적절한 대기 시간 선택 (활성/비활성)
        auto timeout = is_active ? active_timeout : idle_timeout;
        
        // 스레드 제어 상태 한 번에 로드 (원자적 읽기 최소화)
        bool is_aiming = aiming.load();
        bool auto_shooting = config.auto_shoot;
        
        bool newFrameAvailable = false;
        {
            // 탐지 결과에 대한 락 (최소화) - 필요한 데이터만 빠르게 복사
            std::unique_lock<std::mutex> lock(detector.detectionMutex);
            
            // 새 프레임이 준비되거나 종료 신호가 올 때까지 대기
            detector.detectionCV.wait_for(lock, timeout, [&]() {
                return detector.detectionVersion > lastDetectionVersion || shouldExit;
            });
            
            // 종료 검사
            if (shouldExit)
                break;
                
            // 새 프레임이 있으면 데이터 복사
            if (detector.detectionVersion > lastDetectionVersion) {
                newFrameAvailable = true;
                lastDetectionVersion = detector.detectionVersion;
                
                // 메모리 재할당을 방지하기 위해 미리 용량 예약
                if (boxes.capacity() < detector.detectedBoxes.size()) {
                    boxes.reserve(detector.detectedBoxes.size() + 10); // 여유 공간 확보
                }
                if (classes.capacity() < detector.detectedClasses.size()) {
                    classes.reserve(detector.detectedClasses.size() + 10);
                }
                
                boxes = detector.detectedBoxes;
                classes = detector.detectedClasses;
            }
        } // 락 즉시 해제
        
        // 새 프레임이 없으면 반동 제어만 수행하고 계속
        if (!newFrameAvailable) {
            // 비활성 상태 체크 (빈도 제한)
            if (is_active && (++frame_counter >= FRAMES_BETWEEN_TIME_CHECK)) {
                frame_counter = 0;
                if (std::chrono::steady_clock::now() - last_active_time > std::chrono::milliseconds(300)) {
                    is_active = false;
                }
            }
            
            // 반동 제어는 계속 적용
            handleEasyNoRecoil(mouseThread);
            continue;
        }
        
        // 설정 변경 감지 및 처리 (최적화: 조건 체크 후 처리)
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
                    config.sensitivity,
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

        // 타겟 찾기 (정적 객체 대신 기존 인터페이스 유지)
        AimbotTarget *target = sortTargets(boxes, classes, config.detection_resolution, config.detection_resolution, config.disable_headshot);
        
        // 조건 체크 최적화 (조건 통합)
        bool has_target = (target != nullptr);
        
        // 조준 및 발사 로직 단순화
        if (is_aiming && has_target) {
            is_active = true;
            
            // 시간 측정 빈도 제한
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
        
        // 반동 제어 처리
        handleEasyNoRecoil(mouseThread);
        
        // 예측 초기화 검사
        mouseThread.checkAndResetPredictions();
        delete target;
    }
}

// 모델 로딩 중복 코드를 하나의 함수로 통합
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

        // 입력 방식 초기화 코드 간소화
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
            config.sensitivity,
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

        // 모델 로딩 로직을 통합 함수로 대체
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