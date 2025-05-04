#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include "../modules/eigen/include/Eigen/Core"
#include "../modules/eigen/include/Eigen/Dense"
#include <shared_mutex>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>

#include "AimbotTarget.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "InputMethod.h"

// Forward declare PIDController2D
class PIDController2D;

// Forward declarations
class InputMethod;
class SerialConnection;
class GhubMouse;
class IPredictor; // Forward declare the interface
struct Point2D; // Forward declare Point2D

using ErrorTrackingCallback = std::function<void(float error_x, float error_y)>;

class MouseThread
{
private:
    std::unique_ptr<PIDController2D> pid_controller;
    std::unique_ptr<InputMethod> input_method;

    // Performance tracking callback
    ErrorTrackingCallback error_callback;
    std::mutex callback_mutex;
    bool tracking_errors;

    float screen_width;
    float screen_height;
    float center_x;
    float center_y;
    float bScope_multiplier;
    float move_scale_x; // Pre-calculated scaling factor for X movement
    float move_scale_y; // Pre-calculated scaling factor for Y movement
    float norecoil_ms; // Store recoil delay

    std::chrono::steady_clock::time_point last_target_time;
    std::chrono::steady_clock::time_point last_recoil_compensation_time; // Track last recoil time
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    // Simplified target tracking
    AimbotTarget *current_target;

    // Predictor
    std::unique_ptr<IPredictor> predictor_; // Pointer to the current predictor
    std::mutex predictor_mutex_;        // Mutex to protect predictor access/changes

    float calculateTargetDistanceSquared(const AimbotTarget &target) const;
    void initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub);
    void initializeScreen(int resolution, float bScope_multiplier, float norecoil_ms);

public:
    MouseThread(int resolution,
                float kp_x, float ki_x, float kd_x,
                float kp_y, float ki_y, float kd_y,
                float bScope_multiplier,
                float norecoil_ms,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);
    ~MouseThread();

    void updateConfig(int resolution,
                      float kp_x, float ki_x, float kd_x,
                      float kp_y, float ki_y, float kd_y,
                      float bScope_multiplier,
                      float norecoil_ms);

    Eigen::Vector2f calculateMovement(const Eigen::Vector2f &target_pos);
    bool checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor);
    void moveMouse(const AimbotTarget &target);
    void pressMouse(const AimbotTarget &target);
    void releaseMouse();
    void applyRecoilCompensation(float strength);

    void enableErrorTracking(const ErrorTrackingCallback& callback);
    void disableErrorTracking();

    std::mutex input_method_mutex;
    void setInputMethod(std::unique_ptr<InputMethod> new_method);
    void setPredictor(const std::string& algorithm_name); // Method to set the predictor
    
    float& getScreenWidth() { return screen_width; }
    float& getScreenHeight() { return screen_height; }
    float& getScopeMultiplier() { return bScope_multiplier; }
    
    // 타겟 감지 상태를 가져오는 메소드 추가
    bool isTargetDetected() const { return target_detected.load(); }
};

#endif // MOUSE_H