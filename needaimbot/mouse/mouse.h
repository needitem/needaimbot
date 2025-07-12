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
#include <random>

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
 

    std::chrono::steady_clock::time_point last_target_time;
    std::chrono::steady_clock::time_point last_recoil_compensation_time; 
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    mutable std::mutex predictor_mutex_;        

    int last_applied_dx_ = 0;
    
    std::vector<std::pair<double, double>> recent_flow_values;
    int optical_flow_recoil_frame_count;
    
    Eigen::Vector2f smoothed_movement; 

    // Target prediction members (moved from static variables)
    Point2D last_target_pos_{0, 0};
    std::chrono::high_resolution_clock::time_point last_target_time_;
    bool prediction_initialized_ = false;
    
    // Velocity history for improved prediction
    static constexpr int VELOCITY_HISTORY_SIZE = 5;
    std::vector<Point2D> velocity_history_;
    std::vector<std::chrono::high_resolution_clock::time_point> velocity_time_history_;
    Point2D current_velocity_{0, 0};
    Point2D current_acceleration_{0, 0};
    Point2D last_velocity_{0, 0};
    
    // Movement accumulation members (moved from static variables)
    float accumulated_x_ = 0.0f;
    float accumulated_y_ = 0.0f;
    
    // Dithering for improved sub-pixel accuracy
    mutable std::mt19937 dither_rng_{std::random_device{}()};
    mutable std::uniform_real_distribution<float> dither_dist_{-0.5f, 0.5f};
    
    // Latency compensation
    static constexpr int LATENCY_HISTORY_SIZE = 10;
    std::vector<float> input_latency_history_;
    std::vector<float> capture_latency_history_;
    float estimated_total_latency_ms_ = 20.0f; // Default assumption
    std::chrono::high_resolution_clock::time_point frame_capture_time_;
    std::chrono::high_resolution_clock::time_point target_detection_time_;
    
    // Constants
    static constexpr float DEAD_ZONE = 0.3f;
    static constexpr float MICRO_MOVEMENT_THRESHOLD = 0.5f;
    static constexpr float MAX_PREDICTION_TIME = 0.03f;
    static constexpr float LARGE_MOVEMENT_THRESHOLD = 100.0f;
    static constexpr float SMALL_ERROR_THRESHOLD = 10.0f;
    static constexpr float MEDIUM_ERROR_THRESHOLD = 50.0f;
    static constexpr float CLOSE_RANGE_SCALE = 0.8f;
    static constexpr float NORMAL_RANGE_SCALE = 1.0f;
    static constexpr float FAR_RANGE_SCALE = 1.1f;
    static constexpr float MIN_DELTA_TIME = 0.001f;
    static constexpr float MAX_DELTA_TIME = 0.1f;
    static constexpr float PREDICTION_TIME_FACTOR = 0.001f;
    static constexpr float SMOOTHING_INCREASE_FACTOR = 0.001f;
    static constexpr float MAX_ADDITIONAL_SMOOTHING = 0.4f;

    float calculateTargetDistanceSquared(const AimbotTarget &target) const;
    void initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub);
    void initializeScreen(int resolution, float bScope_multiplier, float norecoil_ms);
    
    // Latency compensation methods
    void updateLatencyMeasurements(float input_latency_ms, float capture_latency_ms);
    float getEstimatedTotalLatency() const;
    Point2D applyLatencyCompensation(const Point2D& predicted_pos, const Point2D& velocity) const;

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
    void applyWeaponRecoilCompensation(const WeaponRecoilProfile* profile, int scope_magnification);
    void applyOpticalFlowRecoilCompensation();

    void enableErrorTracking(const ErrorTrackingCallback& callback);
    void disableErrorTracking();

    void setInputMethod(std::unique_ptr<InputMethod> new_method);
    
    // Helper methods for moveMouse refactoring
    Point2D calculatePredictedTarget(const AimbotTarget& target, float current_center_x, float current_center_y);
    Eigen::Vector2f applyMovementSmoothing(const Eigen::Vector2f& raw_movement, float error_magnitude, float smoothing_factor);
    std::pair<int, int> processAccumulatedMovement(float move_x, float move_y);
    float calculateAdaptiveScale(float error_magnitude) const;
    void applyRecoilCompensationInternal(float strength, float delay_ms);
     
 
    
    float getScreenWidth() { std::lock_guard<std::mutex> lock(member_data_mutex_); return screen_width; }
    float getScreenHeight() { std::lock_guard<std::mutex> lock(member_data_mutex_); return screen_height; }
    float getScopeMultiplier() { std::lock_guard<std::mutex> lock(member_data_mutex_); return bScope_multiplier; }
    
    
    bool isTargetDetected() const { return target_detected.load(); }
    
    // Add method to access PID controller for resetting
    PIDController2D* getPIDController() { return pid_controller.get(); }
    
    // Add method to reset all accumulated states
    void resetAccumulatedStates();
};

#endif 
