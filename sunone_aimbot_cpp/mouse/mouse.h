#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
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
    Eigen::Matrix<double, 6, 6> A;  // State transition matrix
    Eigen::Matrix<double, 2, 6> H;  // Measurement matrix
    Eigen::Matrix<double, 6, 6> Q;  // Process noise
    Eigen::Matrix2d R;              // Measurement noise
    Eigen::Matrix<double, 6, 6> P;  // Error covariance
    Eigen::Matrix<double, 6, 1> x;  // State vector

    void initializeMatrices(double process_noise_q, double measurement_noise_r);

public:
    KalmanFilter2D(double process_noise_q = 0.1, double measurement_noise_r = 0.1);
    void predict(double dt);
    void update(const Eigen::Vector2d& measurement);
    Eigen::Matrix<double, 6, 1> getState() const { return x; }
    void reset();
    void updateParameters(double process_noise_q, double measurement_noise_r);
};

// 2D PID controller - for aim correction
class PIDController2D
{
private:
    // PID gains separated for horizontal (X-axis) and vertical (Y-axis)
    double kp_x, kp_y;  // Proportional gain: immediate response to current error (higher value = faster response, lower value = smoother movement)
    double ki_x, ki_y;  // Integral gain: correction of accumulated error (higher value = accurate aiming, lower value = reduced overshoot)
    double kd_x, kd_y;  // Derivative gain: response to error change rate (higher value = faster stopping, lower value = smoother deceleration)
    
    // Common gains for backward compatibility
    double kp;  
    double ki;  
    double kd;  
    
    Eigen::Vector2d prev_error;  // Previous error (for derivative term)
    Eigen::Vector2d integral;    // Accumulated error (for integral term)
    Eigen::Vector2d derivative;  // Change rate (derivative term)
    Eigen::Vector2d prev_derivative; // Previous derivative (for derivative filtering)
    std::chrono::steady_clock::time_point last_time_point;  // Previous calculation time (for dt calculation)

public:
    // Original constructor (for compatibility)
    PIDController2D(double kp, double ki, double kd);
    
    // New constructor with separated X/Y gains
    PIDController2D(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y);
    
    Eigen::Vector2d calculate(const Eigen::Vector2d &error);
    void reset();  // Controller reset (used when starting to aim at a new target)
    
    // Original parameter update function (for compatibility)
    void updateParameters(double kp, double ki, double kd);
    
    // X/Y separated gain update function
    void updateSeparatedParameters(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y);
};

using ErrorTrackingCallback = std::function<void(double error_x, double error_y)>;

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

    double screen_width;
    double screen_height;
    double dpi;
    double fov_x;
    double fov_y;
    double center_x;
    double center_y;
    bool auto_shoot;
    float bScope_multiplier;

    std::chrono::steady_clock::time_point last_target_time;
    std::chrono::steady_clock::time_point last_prediction_time;
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    // Simplified target tracking
    AimbotTarget *current_target;

    double calculateTargetDistance(const AimbotTarget &target) const;
    AimbotTarget *findClosestTarget(const std::vector<AimbotTarget> &targets) const;
    void initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub);
    void initializeScreen(int resolution, int dpi, int fovX, int fovY, bool auto_shoot, float bScope_multiplier);

public:
    MouseThread(int resolution, int dpi, int fovX, int fovY,
                double kp, double ki, double kd,
                double process_noise_q, double measurement_noise_r,
                bool auto_shoot, float bScope_multiplier,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);

    MouseThread(int resolution, int dpi, int fovX, int fovY,
                double kp_x, double ki_x, double kd_x,
                double kp_y, double ki_y, double kd_y,
                double process_noise_q, double measurement_noise_r,
                bool auto_shoot, float bScope_multiplier,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);

    void updateConfig(int resolution, int dpi, int fovX, int fovY,
                      double kp, double ki, double kd,
                      double process_noise_q, double measurement_noise_r,
                      bool auto_shoot, float bScope_multiplier);
    void updateConfig(int resolution, int dpi, int fovX, int fovY,
                      double kp_x, double ki_x, double kd_x,
                      double kp_y, double ki_y, double kd_y,
                      double process_noise_q, double measurement_noise_r,
                      bool auto_shoot, float bScope_multiplier);

    Eigen::Vector2d predictTargetPosition(double target_x, double target_y);
    Eigen::Vector2d calculateMovement(const Eigen::Vector2d &target_pos);
    bool checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor);
    void moveMouse(const AimbotTarget &target);
    void pressMouse(const AimbotTarget &target);
    void releaseMouse();
    void resetPrediction();
    void applyRecoilCompensation(float strength);

    void enableErrorTracking(const ErrorTrackingCallback& callback);
    void disableErrorTracking();

    std::mutex input_method_mutex;
    void setInputMethod(std::unique_ptr<InputMethod> new_method);
    
    double& getScreenWidth() { return screen_width; }
    double& getScreenHeight() { return screen_height; }
    double& getDPI() { return dpi; }
    double& getFOVX() { return fov_x; }
    double& getFOVY() { return fov_y; }
    bool& getAutoShoot() { return auto_shoot; }
    float& getScopeMultiplier() { return bScope_multiplier; }
    
    // 타겟 감지 상태를 가져오는 메소드 추가
    bool isTargetDetected() const { return target_detected.load(); }
};

#endif // MOUSE_H