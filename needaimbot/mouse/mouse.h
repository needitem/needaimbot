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

#include "config/config.h"
#include "aimbot_components/AimbotTarget.h"
#include "input_drivers/SerialConnection.h"
#include "input_drivers/ghub.h"
#include "input_drivers/kmboxNet.h"
#include "input_drivers/rzctl.h"
#include "input_drivers/InputMethod.h"

class PIDController2D;


class InputMethod;
class SerialConnection;
class GhubMouse; 
struct Point2D { float x, y; }; 

using ErrorTrackingCallback = std::function<void(float error_x, float error_y)>;

class MouseThread
{
private:
    std::unique_ptr<PIDController2D> pid_controller;
    std::unique_ptr<InputMethod> input_method;
    std::mutex input_method_mutex;
    mutable std::mutex member_data_mutex_;

    
    ErrorTrackingCallback error_callback;
    std::mutex callback_mutex;
    bool tracking_errors;

    float screen_width;
    float screen_height;
    float center_x;
    float center_y;
    float bScope_multiplier;
    float move_scale_x; 
    float move_scale_y; 
    float norecoil_ms; 
    std::chrono::milliseconds silent_aim_click_duration_ms; 

    std::chrono::steady_clock::time_point last_target_time;
    std::chrono::steady_clock::time_point last_recoil_compensation_time; 
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    mutable std::mutex predictor_mutex_;        

    int last_applied_dx_ = 0; 

    float calculateTargetDistanceSquared(const AimbotTarget &target) const;
    void initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub);
    void initializeScreen(int resolution, float bScope_multiplier, float norecoil_ms);

public:
    MouseThread(int resolution,
                float kp_x, float ki_x, float kd_x,
                float kp_y, float ki_y, float kd_y,
                float bScope_multiplier,
                float norecoil_ms,
                float derivative_smoothing_factor,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);
    ~MouseThread();

    void updateConfig(int resolution,
                      float kp_x, float ki_x, float kd_x,
                      float kp_y, float ki_y, float kd_y,
                      float bScope_multiplier,
                      float norecoil_ms,
                      float derivative_smoothing_factor);

    Eigen::Vector2f calculateMovement(const Eigen::Vector2f &target_pos);
    bool checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor);
    void moveMouse(const AimbotTarget &target);
    void pressMouse(const AimbotTarget &target);
    void releaseMouse();
    void applyRecoilCompensation(float strength);

    void enableErrorTracking(const ErrorTrackingCallback& callback);
    void disableErrorTracking();

    void setInputMethod(std::unique_ptr<InputMethod> new_method);
     
    void executeSilentAim(const AimbotTarget& target); 
    
    float getScreenWidth() { std::lock_guard<std::mutex> lock(member_data_mutex_); return screen_width; }
    float getScreenHeight() { std::lock_guard<std::mutex> lock(member_data_mutex_); return screen_height; }
    float getScopeMultiplier() { std::lock_guard<std::mutex> lock(member_data_mutex_); return bScope_multiplier; }
    
    
    bool isTargetDetected() const { return target_detected.load(); }
};

#endif 
