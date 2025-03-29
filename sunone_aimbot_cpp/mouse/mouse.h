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

#include "AimbotTarget.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "InputMethod.h"

// Improved 2D Kalman filter - tracking position, velocity, acceleration
class KalmanFilter2D {
private:
    // State vector: [x, y, vx, vy, ax, ay]
    Eigen::Matrix<float, 6, 6> A;  // State transition matrix
    Eigen::Matrix<float, 2, 6> H;  // Measurement matrix
    Eigen::Matrix<float, 6, 6> Q;  // Process noise
    Eigen::Matrix2f R;              // Measurement noise
    Eigen::Matrix<float, 6, 6> P;  // Error covariance
    Eigen::Matrix<float, 6, 1> x;  // State vector

    void initializeMatrices(float process_noise_q, float measurement_noise_r);

public:
    KalmanFilter2D(float process_noise_q = 0.1f, float measurement_noise_r = 0.1f);
    void predict(float dt);
    void update(const Eigen::Vector2f& measurement);
    Eigen::Matrix<float, 6, 1> getState() const { return x; }
    void reset();
    void updateParameters(float process_noise_q, float measurement_noise_r);
};

// 2D PID controller - for aim correction
class PIDController2D
{
private:
    // PID gains separated for horizontal (X-axis) and vertical (Y-axis)
    float kp_x, kp_y;  // Proportional gain: immediate response to current error (higher value = faster response, lower value = smoother movement)
    float ki_x, ki_y;  // Integral gain: correction of accumulated error (higher value = accurate aiming, lower value = reduced overshoot)
    float kd_x, kd_y;  // Derivative gain: response to error change rate (higher value = faster stopping, lower value = smoother deceleration)
    
    Eigen::Vector2f prev_error;  // Previous error (for derivative term)
    Eigen::Vector2f integral;    // Accumulated error (for integral term)
    Eigen::Vector2f derivative;  // Change rate (derivative term)
    Eigen::Vector2f prev_derivative; // Previous derivative (for derivative filtering)
    std::chrono::steady_clock::time_point last_time_point;  // Previous calculation time (for dt calculation)

public:
    // New constructor with separated X/Y gains
    PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);
    
    Eigen::Vector2f calculate(const Eigen::Vector2f &error);
    void reset();  // Controller reset (used when starting to aim at a new target)
    
    // X/Y separated gain update function
    void updateSeparatedParameters(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);
};

using ErrorTrackingCallback = std::function<void(float error_x, float error_y)>;

class MouseThread
{
private:
    std::unique_ptr<KalmanFilter2D> kalman_filter;
    std::unique_ptr<PIDController2D> pid_controller;
    std::unique_ptr<InputMethod> input_method;

    // Performance tracking callback
    ErrorTrackingCallback error_callback;
    std::mutex callback_mutex;
    bool tracking_errors;

    float screen_width;
    float screen_height;
    float dpi;
    float fov_x;
    float fov_y;
    float center_x;
    float center_y;
    bool auto_shoot;
    float bScope_multiplier;

    std::chrono::steady_clock::time_point last_target_time;
    std::chrono::steady_clock::time_point last_prediction_time;
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    // Simplified target tracking
    AimbotTarget *current_target;

    float calculateTargetDistance(const AimbotTarget &target) const;
    AimbotTarget *findClosestTarget(const std::vector<AimbotTarget> &targets) const;
    void initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub);
    void initializeScreen(int resolution, int dpi, int fovX, int fovY, bool auto_shoot, float bScope_multiplier);

public:
    MouseThread(int resolution, int dpi, int fovX, int fovY,
                float kp_x, float ki_x, float kd_x,
                float kp_y, float ki_y, float kd_y,
                float process_noise_q, float measurement_noise_r,
                bool auto_shoot, float bScope_multiplier,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);

    void updateConfig(int resolution, int dpi, int fovX, int fovY,
                      float kp_x, float ki_x, float kd_x,
                      float kp_y, float ki_y, float kd_y,
                      float process_noise_q, float measurement_noise_r,
                      bool auto_shoot, float bScope_multiplier);

    Eigen::Vector2f predictTargetPosition(float target_x, float target_y);
    Eigen::Vector2f calculateMovement(const Eigen::Vector2f &target_pos);
    bool checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor);
    void moveMouse(const AimbotTarget &target);
    void pressMouse(const AimbotTarget &target);
    void releaseMouse();
    void resetPrediction();
    void applyRecoilCompensation(float strength);

    void enableErrorTracking(const ErrorTrackingCallback& callback);
    void disableErrorTracking();

    std::mutex input_method_mutex;
    void setInputMethod(std::unique_ptr<InputMethod> new_method);
    
    float& getScreenWidth() { return screen_width; }
    float& getScreenHeight() { return screen_height; }
    float& getDPI() { return dpi; }
    float& getFOVX() { return fov_x; }
    float& getFOVY() { return fov_y; }
    bool& getAutoShoot() { return auto_shoot; }
    float& getScopeMultiplier() { return bScope_multiplier; }
    
    // 타겟 감지 상태를 가져오는 메소드 추가
    bool isTargetDetected() const { return target_detected.load(); }
};

#endif // MOUSE_H