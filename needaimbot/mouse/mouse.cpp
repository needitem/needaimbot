// #include "../constants.h" // File removed
#include "../AppContext.h"
#include "mouse.h"
#include "aimbot_components/AimbotTarget.h"
#include "../capture/capture.h"
// #include "../detector/optical_flow.h" // Optical flow removed
#include "input_drivers/SerialConnection.h"
#include "input_drivers/MakcuConnection.h"
#include "../needaimbot.h"
#include "input_drivers/ghub.h"
#include "../config/config.h"
#include "../keyboard/keyboard_listener.h"
#include "../utils/AutoTuner.h"

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <iostream>
#include <random>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "aimbot_components/PIDController2D.h"
#include "aimbot_components/SnapAimController.h"


extern std::atomic<bool> aiming;
extern std::mutex configMutex;

std::random_device rd;
std::mt19937 gen(rd());

constexpr float SCOPE_MARGIN = 0.15f;




thread_local static LARGE_INTEGER freq;
thread_local static bool freq_initialized = false;

void InitializeHighPrecisionTimer() {
    if (!freq_initialized) {
        QueryPerformanceFrequency(&freq);
        freq_initialized = true;
    }
}

void QueueMove(int dx, int dy, std::function<void(int, int)> move_func) {
    auto& ctx = AppContext::getInstance();
    InitializeHighPrecisionTimer();
    
    if (dx != 0 || dy != 0) {
        // Skip timing measurements if not needed (performance optimization)
        if (ctx.config.verbose || ctx.config.show_metrics) {
            LARGE_INTEGER start;
            QueryPerformanceCounter(&start);
            
            move_func(dx, dy);
            
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            float elapsed_us = ((end.QuadPart - start.QuadPart) * 1000000.0f) / freq.QuadPart;
            
            ctx.add_to_history(ctx.g_input_send_time_history, elapsed_us / 1000.0f, ctx.g_input_send_history_mutex);
            ctx.g_current_input_send_time_ms.store(elapsed_us / 1000.0f);
        } else {
            move_func(dx, dy);
        }
    }
}

MouseThread::MouseThread(
    int resolution,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    float bScope_multiplier,
    float norecoil_ms,
    SerialConnection *serialConnection,
    MakcuConnection *makcuConnection,
    GhubMouse *gHub) : tracking_errors(false), optical_flow_recoil_frame_count(0)
{
    initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    kalman_filter = std::make_unique<TargetKalmanFilter>();
    snap_controller = std::make_unique<SnapAimController>(); // 새로운 스냅 에임 컨트롤러
    initializeInputMethod(serialConnection, makcuConnection, gHub);
    
    // Start async input worker thread
    async_input_thread_ = std::thread(&MouseThread::asyncInputWorker, this);
}

MouseThread::~MouseThread() {
    // Stop the async worker thread
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        should_stop_thread_ = true;
    }
    queue_cv_.notify_all();
    
    if (async_input_thread_.joinable()) {
        async_input_thread_.join();
    }
}

void MouseThread::initializeInputMethod(SerialConnection *serialConnection, MakcuConnection *makcuConnection, GhubMouse *gHub)
{
    if (serialConnection && serialConnection->isOpen())
    {
        input_method = std::make_unique<SerialInputMethod>(serialConnection);
    }
    else if (makcuConnection && makcuConnection->isOpen())
    {
        input_method = std::make_unique<MakcuInputMethod>(makcuConnection);
    }
    else if (gHub)
    {
        input_method = std::make_unique<GHubInputMethod>(gHub);
    }
    else
    {
        input_method = std::make_unique<Win32InputMethod>();
    }
}

void MouseThread::initializeScreen(int resolution, float bScope_multiplier, float norecoil_ms)
{
    this->screen_width = static_cast<float>(resolution);
    this->screen_height = static_cast<float>(resolution); 
    this->bScope_multiplier = bScope_multiplier;
    this->norecoil_ms = norecoil_ms; 
    this->center_x = screen_width / 2.0f;
    this->center_y = screen_height / 2.0f;

    const float SENSITIVITY_FACTOR = 1.0f; // Default sensitivity factor
    float base_scale_x = SENSITIVITY_FACTOR;
    float base_scale_y = SENSITIVITY_FACTOR;

    if (this->bScope_multiplier > 1.0f) {
        this->move_scale_x = base_scale_x / this->bScope_multiplier;
        this->move_scale_y = base_scale_y / this->bScope_multiplier;
    } else {
        this->move_scale_x = base_scale_x;
        this->move_scale_y = base_scale_y;
    }

    this->last_recoil_compensation_time = std::chrono::steady_clock::now();
    this->smoothed_movement = Eigen::Vector2f::Zero();
    this->last_target_time_ = std::chrono::high_resolution_clock::now();
    this->prediction_initialized_ = false;
    this->last_target_class_id_ = -1;
    this->accumulated_x_ = 0.0f;
    this->accumulated_y_ = 0.0f;
}

void MouseThread::updateConfig(
    int resolution,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    float bScope_multiplier,
    float norecoil_ms
    )
{
    auto& ctx = AppContext::getInstance();
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        initializeScreen(resolution, bScope_multiplier, norecoil_ms);
        
    }
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    
}



Eigen::Vector2f MouseThread::calculateMovement(const Eigen::Vector2f &target_pos)
{
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

    float error_x = target_pos[0] - current_center_x;
    float error_y = target_pos[1] - current_center_y;
    
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    float result_x = pid_output[0] * current_move_scale_x;
    float result_y = pid_output[1] * current_move_scale_y;
    
    return Eigen::Vector2f(result_x, result_y);
}

bool MouseThread::checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor)
{
    float current_screen_width, current_screen_height, current_center_x, current_center_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_screen_width = this->screen_width;
        current_screen_height = this->screen_height;
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }

    const float SCOPE_MARGIN = 0.1f; // Default scope margin
    const float screen_margin_x = current_screen_width * SCOPE_MARGIN; 
    const float screen_margin_y = current_screen_height * SCOPE_MARGIN; 
    
    float target_center_x_val = target_x + target_w * 0.5f;
    float target_center_y_val = target_y + target_h * 0.5f;
    
    float diff_x = std::abs(target_center_x_val - current_center_x); 
    float diff_y = std::abs(target_center_y_val - current_center_y); 
    
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    float reduced_half_w = target_w * reduction_factor * 0.5f;
    float reduced_half_h = target_h * reduction_factor * 0.5f;
    
    float min_x = target_center_x_val - reduced_half_w;
    float max_x = target_center_x_val + reduced_half_w;
    float min_y = target_center_y_val - reduced_half_h;
    float max_y = target_center_y_val + reduced_half_h;
    
    return (current_center_x >= min_x && current_center_x <= max_x && 
            current_center_y >= min_y && current_center_y <= max_y); 
}

float MouseThread::calculateTargetDistanceSquared(const AimbotTarget &target) const
{
    auto& ctx = AppContext::getInstance();
    float current_center_x, current_center_y;
    float crosshair_offset_x, crosshair_offset_y;
    
    // Get all values with minimal locking
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        crosshair_offset_x = ctx.config.crosshair_offset_x;
        crosshair_offset_y = ctx.config.crosshair_offset_y;
    }
    
    // Apply crosshair offset correction
    current_center_x += crosshair_offset_x;
    current_center_y += crosshair_offset_y;

    float dx = target.x + target.w * 0.5f - current_center_x;
    float target_center_y_val;

    
    int local_head_class_id_to_use = -1;
    bool local_apply_head_offset = false;
    float local_head_y_offset_val;
    float local_body_y_offset_val;
    std::string local_head_class_name_val;

    {
        auto& ctx = AppContext::getInstance();
        std::lock_guard<std::mutex> lock(configMutex); 
        local_head_class_name_val = ctx.config.head_class_name;
        for (const auto& class_setting : ctx.config.class_settings) {
            if (class_setting.name == local_head_class_name_val) {
                local_head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    local_apply_head_offset = true;
                }
                break;
            }
        }
        local_head_y_offset_val = ctx.config.head_y_offset;
        local_body_y_offset_val = ctx.config.body_y_offset;
    }

    if (local_apply_head_offset && target.classId == local_head_class_id_to_use) {
        target_center_y_val = target.y + target.h * local_head_y_offset_val;
    } else {
        target_center_y_val = target.y + target.h * local_body_y_offset_val;
    }
    float dy = target_center_y_val - current_center_y;
    return dx * dx + dy * dy;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    auto& ctx = AppContext::getInstance();
    
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;
    
    // Cache for config values to reduce mutex locks
    static struct ConfigCache {
        float crosshair_offset_x, crosshair_offset_y;
        bool use_predictive_controller;
        float prediction_time_ms;
        int head_class_id_to_use = -1;
        bool apply_head_offset = false;
        float head_y_offset, body_y_offset;
        std::string head_class_name;
        std::chrono::steady_clock::time_point last_update;
        const std::chrono::milliseconds update_interval{100}; // Update every 100ms
        
        void update(const Config& config) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_update >= update_interval) {
                crosshair_offset_x = config.crosshair_offset_x;
                crosshair_offset_y = config.crosshair_offset_y;
                use_predictive_controller = config.use_predictive_controller;
                prediction_time_ms = config.prediction_time_ms;
                head_class_name = config.head_class_name;
                head_y_offset = config.head_y_offset;
                body_y_offset = config.body_y_offset;
                
                head_class_id_to_use = -1;
                apply_head_offset = false;
                for (const auto& class_setting : config.class_settings) {
                    if (class_setting.name == head_class_name) {
                        head_class_id_to_use = class_setting.id;
                        if (!class_setting.ignore) {
                            apply_head_offset = true;
                        }
                        break;
                    }
                }
                last_update = now;
            }
        }
    } config_cache;

    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

    // Update config cache if needed
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        config_cache.update(ctx.config);
    }

    // Apply crosshair offset correction
    current_center_x += config_cache.crosshair_offset_x;
    current_center_y += config_cache.crosshair_offset_y;

    Point2D raw_target_pos;
    raw_target_pos.x = target.x + target.w * 0.5f;
    
    // Use cached config values
    float local_y_offset_multiplier_val;
    if (config_cache.apply_head_offset && target.classId == config_cache.head_class_id_to_use) {
        local_y_offset_multiplier_val = config_cache.head_y_offset;
    } else {
        local_y_offset_multiplier_val = config_cache.body_y_offset;
    }
    raw_target_pos.y = target.y + target.h * local_y_offset_multiplier_val;

    // Check disable upward aim with copied config
    bool local_disable_upward_aim_active = false;
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        local_disable_upward_aim_active = isAnyKeyPressed(ctx.config.button_disable_upward_aim);
    }

    // Calculate predicted target position
    Point2D predicted_target_pos = calculatePredictedTarget(target, current_center_x, current_center_y);
    
    float error_x = predicted_target_pos.x - current_center_x;
    float error_y = predicted_target_pos.y - current_center_y;

    if (tracking_errors) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback) {
            error_callback(error_x, error_y);
        }
    }

    // Calculate error magnitude first for adaptive control
    float error_magnitude = std::sqrt(error_x * error_x + error_y * error_y);
    
    float move_x, move_y;
    
    // Simple PID without any smoothing or extra algorithms
    Eigen::Vector2f error(error_x, error_y);
    auto pid_start_time = std::chrono::steady_clock::now();
    
    // Basic PID calculation
    Eigen::Vector2f pid_output = pid_controller->calculate(error);
        
    auto pid_end_time = std::chrono::steady_clock::now();
    float pid_duration_ms = std::chrono::duration<float, std::milli>(pid_end_time - pid_start_time).count();
    ctx.g_current_pid_calc_time_ms.store(pid_duration_ms, std::memory_order_relaxed);
    ctx.add_to_history(ctx.g_pid_calc_time_history, pid_duration_ms, ctx.g_pid_calc_history_mutex);

    // Apply error-based scaling to reduce jitter on large errors
    float error_scale_factor = 1.0f;
    
    // Apply the first matching rule
    for (const auto& rule : ctx.config.error_scaling_rules) {
        if (error_magnitude >= rule.error_threshold) {
            error_scale_factor = rule.scale_factor;
            break;
        }
    }
    
    // Apply PID output with error scaling
    move_x = pid_output.x() * current_move_scale_x * error_scale_factor;
    move_y = pid_output.y() * current_move_scale_y * error_scale_factor;

    // Simple dead zone without adaptive threshold
    if (std::abs(move_x) < DEAD_ZONE) move_x = 0.0f;
    if (std::abs(move_y) < DEAD_ZONE) move_y = 0.0f;
    
    // Process accumulated movement
    auto [dx_int, dy_int] = processAccumulatedMovement(move_x, move_y);
    
    if (local_disable_upward_aim_active && dy_int < 0) {
        dy_int = 0; 
    }

    if (dx_int != 0 || dy_int != 0) {
        // Use async queue instead of direct call
        enqueueMouseCommand(MouseCommand::MOVE, dx_int, dy_int);
    }
    
    // Calculate pure detection-to-movement time (excluding FPS waiting)
    auto movement_completion_time = std::chrono::high_resolution_clock::now();
    float detection_to_movement_ms = std::chrono::duration<float, std::milli>(movement_completion_time - target.detection_timestamp).count();
    ctx.g_current_detection_to_movement_time_ms.store(detection_to_movement_ms, std::memory_order_relaxed);
    ctx.add_to_history(ctx.g_detection_to_movement_time_history, detection_to_movement_ms, ctx.g_detection_to_movement_history_mutex);
}

void MouseThread::pressMouse(const AimbotTarget &target)
{
    auto& ctx = AppContext::getInstance();

    // For triggerbot, use a more lenient scope check (1.5x the normal multiplier)
    float triggerbot_scope_multiplier = bScope_multiplier * 1.5f;
    auto bScope = checkTargetInScope(static_cast<float>(target.x), static_cast<float>(target.y), static_cast<float>(target.w), static_cast<float>(target.h), triggerbot_scope_multiplier);

    if (bScope && !mouse_pressed)
    {
        // Add a small delay after release to prevent rapid fire
        auto now = std::chrono::steady_clock::now();
        auto time_since_release = std::chrono::duration<float, std::milli>(now - last_mouse_release_time).count();
        
        // Only press if at least 50ms have passed since last release
        if (time_since_release > 50.0f) {
            // Use async queue instead of direct call
            enqueueMouseCommand(MouseCommand::PRESS);
            mouse_pressed = true;
            last_mouse_press_time = now;
        }
    }
}

void MouseThread::releaseMouse()
{
    auto& ctx = AppContext::getInstance();
    if (!mouse_pressed)
        return;

    // Add a small delay to prevent rapid press/release cycles
    auto now = std::chrono::steady_clock::now();
    auto time_since_press = std::chrono::duration<float, std::milli>(now - last_mouse_press_time).count();
    
    // Only release if at least 100ms have passed since press (prevents rapid clicking)
    if (time_since_press < 100.0f)
        return;

    // Use async queue instead of direct call
    enqueueMouseCommand(MouseCommand::RELEASE);
    mouse_pressed = false;
    last_mouse_release_time = now;
}

void MouseThread::applyRecoilCompensation(float strength)
{
    float current_norecoil_ms;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_norecoil_ms = this->norecoil_ms;
    }
    
    applyRecoilCompensationInternal(strength, current_norecoil_ms);
}

void MouseThread::applyWeaponRecoilCompensation(const WeaponRecoilProfile* profile, int scope_magnification)
{
    if (!profile) {
        return;
    }

    float scope_multiplier = 1.0f;
    switch (scope_magnification) {
        case 1: scope_multiplier = profile->scope_mult_1x; break;
        case 2: scope_multiplier = profile->scope_mult_2x; break;
        case 3: scope_multiplier = profile->scope_mult_3x; break;
        case 4: scope_multiplier = profile->scope_mult_4x; break;
        case 6: scope_multiplier = profile->scope_mult_6x; break;
        case 8: scope_multiplier = profile->scope_mult_8x; break;
        default: scope_multiplier = profile->scope_mult_1x; break;
    }

    float adjusted_strength = profile->base_strength * profile->fire_rate_multiplier * scope_multiplier;
    
    applyRecoilCompensationInternal(adjusted_strength, profile->recoil_ms);
}

void MouseThread::enableErrorTracking(const ErrorTrackingCallback &callback)
{
    std::lock_guard<std::mutex> lock(callback_mutex);
    error_callback = callback;
    tracking_errors = true;
}

void MouseThread::disableErrorTracking()
{
    std::lock_guard<std::mutex> lock(callback_mutex);
    tracking_errors = false;
    error_callback = nullptr; 
}



void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}


Point2D MouseThread::calculatePredictedTarget(const AimbotTarget& target, float current_center_x, float current_center_y)
{
    Point2D raw_target_pos;
    raw_target_pos.x = target.x + target.w * 0.5f;
    
    // Get y offset from target configuration
    float local_y_offset_multiplier_val;
    bool use_kalman_prediction;
    float kalman_measurement_noise;
    float base_prediction_time_ms;
    {
        auto& ctx = AppContext::getInstance();
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        
        bool local_apply_head_offset = false;
        int local_head_class_id_to_use = -1;
        
        for (const auto& class_setting : ctx.config.class_settings) {
            if (class_setting.name == ctx.config.head_class_name) {
                local_head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    local_apply_head_offset = true;
                }
                break;
            }
        }
        
        if (local_apply_head_offset && target.classId == local_head_class_id_to_use) {
            local_y_offset_multiplier_val = ctx.config.head_y_offset;
        } else {
            local_y_offset_multiplier_val = ctx.config.body_y_offset;
        }
        
        use_kalman_prediction = ctx.config.use_predictive_controller;
        kalman_measurement_noise = ctx.config.kalman_measurement_noise;
        base_prediction_time_ms = ctx.config.prediction_time_ms;
        
        // If latency compensation is enabled, include it in Kalman prediction time
        if (use_kalman_prediction && ctx.config.enable_latency_compensation) {
            base_prediction_time_ms += ctx.config.system_latency_ms;
        }
    }
    
    raw_target_pos.y = target.y + target.h * local_y_offset_multiplier_val;
    
    // Calculate initial error for prediction
    float initial_error_x = raw_target_pos.x - current_center_x;
    float initial_error_y = raw_target_pos.y - current_center_y;
    float error_magnitude = std::sqrt(initial_error_x * initial_error_x + initial_error_y * initial_error_y);
    
    Point2D predicted_target_pos = raw_target_pos;
    
    // Use Kalman filter if enabled
    if (use_kalman_prediction && kalman_filter) {
        // Check if target class changed (e.g., head -> body transition)
        bool class_changed = (last_target_class_id_ != -1 && last_target_class_id_ != target.classId);
        
        // Check for large position jumps that might indicate target switch
        float position_jump = 0.0f;
        if (prediction_initialized_) {
            float dx = raw_target_pos.x - last_target_pos_.x;
            float dy = raw_target_pos.y - last_target_pos_.y;
            position_jump = std::sqrt(dx * dx + dy * dy);
        }
        
        // Reset Kalman filter if target changed or large jump detected
        if (class_changed || position_jump > LARGE_MOVEMENT_THRESHOLD) {
            kalman_filter->reset();
            kalman_filter->initialize(raw_target_pos.x, raw_target_pos.y);
            last_target_class_id_ = target.classId;
            predicted_target_pos = raw_target_pos; // No prediction on reset
        } else {
            // Set measurement noise based on config
            kalman_filter->setMeasurementNoise(kalman_measurement_noise);
            
            // Get Kalman prediction
            Eigen::Vector2f kalman_prediction = kalman_filter->predict(raw_target_pos.x, raw_target_pos.y, base_prediction_time_ms);
            
            // Get confidence and apply it
            float confidence = kalman_filter->getConfidence();
            
            // Blend Kalman prediction with raw position based on confidence
            predicted_target_pos.x = raw_target_pos.x + (kalman_prediction.x() - raw_target_pos.x) * confidence;
            predicted_target_pos.y = raw_target_pos.y + (kalman_prediction.y() - raw_target_pos.y) * confidence;
        }
        
        // Update tracking state
        last_target_pos_ = raw_target_pos;
        last_target_class_id_ = target.classId;
        prediction_initialized_ = true;
    } else {
        // Fallback to simple velocity-based prediction
        auto current_time = std::chrono::high_resolution_clock::now();
        
        // Initialize or reset prediction when target changes significantly
        if (!prediction_initialized_ || 
            std::abs(raw_target_pos.x - last_target_pos_.x) > LARGE_MOVEMENT_THRESHOLD || 
            std::abs(raw_target_pos.y - last_target_pos_.y) > LARGE_MOVEMENT_THRESHOLD) {
            last_target_pos_ = raw_target_pos;
            last_target_time_ = current_time;
            prediction_initialized_ = true;
            predicted_target_pos = raw_target_pos; // No prediction on first frame or large jumps
        } else {
            float dt_target = std::chrono::duration<float, std::milli>(current_time - last_target_time_).count() / 1000.0f;
            dt_target = std::clamp(dt_target, MIN_DELTA_TIME, MAX_DELTA_TIME);
            
            if (dt_target > MIN_DELTA_TIME) {
                // Calculate current velocity
                float target_vel_x = (raw_target_pos.x - last_target_pos_.x) / dt_target;
                float target_vel_y = (raw_target_pos.y - last_target_pos_.y) / dt_target;
            
            // Calculate acceleration
            if (last_velocity_.x != 0 || last_velocity_.y != 0) {
                current_acceleration_.x = (target_vel_x - last_velocity_.x) / dt_target;
                current_acceleration_.y = (target_vel_y - last_velocity_.y) / dt_target;
                
                // Limit acceleration to prevent overshooting
                const float MAX_ACCELERATION = 5000.0f; // pixels/s^2
                current_acceleration_.x = std::clamp(current_acceleration_.x, -MAX_ACCELERATION, MAX_ACCELERATION);
                current_acceleration_.y = std::clamp(current_acceleration_.y, -MAX_ACCELERATION, MAX_ACCELERATION);
            }
            
            // Update velocity tracking
            last_velocity_ = current_velocity_;
            current_velocity_.x = target_vel_x;
            current_velocity_.y = target_vel_y;
            
            // Adaptive prediction time based on multiple factors
            float error_magnitude = std::sqrt(initial_error_x * initial_error_x + initial_error_y * initial_error_y);
            float velocity_magnitude = std::sqrt(current_velocity_.x * current_velocity_.x + current_velocity_.y * current_velocity_.y);
            float acceleration_magnitude = std::sqrt(current_acceleration_.x * current_acceleration_.x + 
                                                   current_acceleration_.y * current_acceleration_.y);
            
            float prediction_factor;
            float base_prediction_time_ms;
            {
                auto& ctx = AppContext::getInstance();
                std::lock_guard<std::mutex> lock(ctx.configMutex);
                prediction_factor = ctx.config.prediction_time_factor;
                base_prediction_time_ms = ctx.config.prediction_time_ms;
            }
            
            // Calculate adaptive prediction time
            // Higher velocity = more prediction needed
            // Higher acceleration = less prediction (target changing direction)
            float velocity_factor = std::min(1.0f, velocity_magnitude / 1000.0f); // normalize to 0-1
            float acceleration_penalty = std::min(1.0f, acceleration_magnitude / 5000.0f); // normalize to 0-1
            
            // Combine factors for final prediction time
            float prediction_time = (base_prediction_time_ms / 1000.0f) * velocity_factor * (1.0f - acceleration_penalty * 0.5f);
            
            // Also consider error magnitude
            prediction_time += error_magnitude * prediction_factor;
            prediction_time = std::clamp(prediction_time, 0.0f, MAX_PREDICTION_TIME);
            
            // Apply enhanced prediction with velocity and acceleration (2nd order prediction)
            // Position = current_pos + velocity * t + 0.5 * acceleration * t^2
            predicted_target_pos.x = raw_target_pos.x + current_velocity_.x * prediction_time + 
                                   0.5f * current_acceleration_.x * prediction_time * prediction_time;
            predicted_target_pos.y = raw_target_pos.y + current_velocity_.y * prediction_time + 
                                   0.5f * current_acceleration_.y * prediction_time * prediction_time;
            
            // Apply prediction limits to prevent overshooting
            const float MAX_PREDICTION_DISTANCE = 100.0f; // pixels
            float prediction_distance_x = predicted_target_pos.x - raw_target_pos.x;
            float prediction_distance_y = predicted_target_pos.y - raw_target_pos.y;
            float prediction_distance = std::sqrt(prediction_distance_x * prediction_distance_x + 
                                                prediction_distance_y * prediction_distance_y);
            
            if (prediction_distance > MAX_PREDICTION_DISTANCE) {
                float scale = MAX_PREDICTION_DISTANCE / prediction_distance;
                predicted_target_pos.x = raw_target_pos.x + prediction_distance_x * scale;
                predicted_target_pos.y = raw_target_pos.y + prediction_distance_y * scale;
            }
            }
            
            last_target_pos_ = raw_target_pos;
            last_target_time_ = current_time;
        }
    }
    
    // Apply latency compensation if enabled and not using Kalman (to avoid double compensation)
    bool enable_latency_compensation;
    {
        auto& ctx = AppContext::getInstance();
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        enable_latency_compensation = ctx.config.enable_latency_compensation;
        use_kalman_prediction = ctx.config.use_predictive_controller;
    }
    
    if (enable_latency_compensation && !use_kalman_prediction) {
        predicted_target_pos = applyLatencyCompensation(predicted_target_pos, current_velocity_);
    }
    
    return predicted_target_pos;
}


std::pair<int, int> MouseThread::processAccumulatedMovement(float move_x, float move_y)
{
    // Accumulate sub-pixel movements
    accumulated_x_ += move_x;
    accumulated_y_ += move_y;
    
    int dx_int = 0, dy_int = 0;
    
    // Simple rounding without dithering (dithering was ineffective)
    // Only move when accumulated movement exceeds threshold
    if (std::abs(accumulated_x_) >= MICRO_MOVEMENT_THRESHOLD) {
        dx_int = static_cast<int>(std::round(accumulated_x_));
        accumulated_x_ -= dx_int;
    }
    if (std::abs(accumulated_y_) >= MICRO_MOVEMENT_THRESHOLD) {
        dy_int = static_cast<int>(std::round(accumulated_y_));
        accumulated_y_ -= dy_int;
    }
    
    return {dx_int, dy_int};
}

void MouseThread::updateLatencyMeasurements(float input_latency_ms, float capture_latency_ms)
{
    // Update input latency history
    input_latency_history_.push_back(input_latency_ms);
    if (input_latency_history_.size() > LATENCY_HISTORY_SIZE) {
        input_latency_history_.erase(input_latency_history_.begin());
    }
    
    // Update capture latency history  
    capture_latency_history_.push_back(capture_latency_ms);
    if (capture_latency_history_.size() > LATENCY_HISTORY_SIZE) {
        capture_latency_history_.erase(capture_latency_history_.begin());
    }
    
    // Calculate moving average of total latency
    float avg_input_latency = 0.0f;
    float avg_capture_latency = 0.0f;
    
    if (!input_latency_history_.empty()) {
        for (float latency : input_latency_history_) {
            avg_input_latency += latency;
        }
        avg_input_latency /= input_latency_history_.size();
    }
    
    if (!capture_latency_history_.empty()) {
        for (float latency : capture_latency_history_) {
            avg_capture_latency += latency;
        }
        avg_capture_latency /= capture_latency_history_.size();
    }
    
    // Add estimated processing latency (AI inference + post-processing)
    constexpr float PROCESSING_LATENCY_MS = 3.0f;
    
    estimated_total_latency_ms_ = avg_input_latency + avg_capture_latency + PROCESSING_LATENCY_MS;
    
    // Clamp to reasonable bounds
    estimated_total_latency_ms_ = std::clamp(estimated_total_latency_ms_, 5.0f, 100.0f);
}

float MouseThread::getEstimatedTotalLatency() const
{
    return estimated_total_latency_ms_;
}

Point2D MouseThread::applyLatencyCompensation(const Point2D& predicted_pos, const Point2D& velocity) const
{
    // Get latency compensation setting
    float system_latency_ms;
    {
        auto& ctx = AppContext::getInstance();
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        system_latency_ms = ctx.config.system_latency_ms;
    }
    
    // Convert latency from ms to seconds
    float latency_compensation_time = system_latency_ms / 1000.0f;
    
    // Apply additional prediction based on system latency
    Point2D compensated_pos;
    compensated_pos.x = predicted_pos.x + velocity.x * latency_compensation_time;
    compensated_pos.y = predicted_pos.y + velocity.y * latency_compensation_time;
    
    return compensated_pos;
}

float MouseThread::calculateAdaptiveScale(float error_magnitude) const
{
    // Maintain full speed until very close, then gentle reduction
    // This keeps movement fast while providing precision at the end
    
    if (error_magnitude < 10.0f) {
        // Very close range - slight reduction for precision
        // 10px: 1.0x, 5px: 0.85x, 0px: 0.7x
        return 0.7f + (error_magnitude / 10.0f) * 0.3f;
    } else if (error_magnitude < MEDIUM_ERROR_THRESHOLD) {
        // Normal range - maintain full speed
        return NORMAL_RANGE_SCALE;
    } else {
        // Far range - slightly increased sensitivity for large corrections
        return std::min(FAR_RANGE_SCALE, 1.0f + (error_magnitude - MEDIUM_ERROR_THRESHOLD) * 0.001f);
    }
}

void MouseThread::applyRecoilCompensationInternal(float strength, float delay_ms)
{
    auto& ctx = AppContext::getInstance();

    if (std::abs(strength) < 1e-3f) {
        return;
    }

    auto now_chrono_recoil = std::chrono::steady_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_chrono_recoil - last_recoil_compensation_time);
    auto required_delay = std::chrono::milliseconds(static_cast<long long>(delay_ms));

    if (elapsed >= required_delay) {
        // Apply strength directly without accumulation to prevent over-compensation
        int dy_recoil = static_cast<int>(std::round(strength));
        
        if (dy_recoil != 0) {
            // Use async queue instead of direct call
            enqueueMouseCommand(MouseCommand::MOVE, 0, dy_recoil);
        }
        
        last_recoil_compensation_time = now_chrono_recoil;
    }
}

void MouseThread::applyOpticalFlowRecoilCompensation()
{
    // Optical flow recoil compensation removed
    return;
}

void MouseThread::resetAccumulatedStates()
{
    // Reset PID controller
    if (pid_controller) {
        pid_controller->reset();
    }
    
    // Reset Kalman filter
    if (kalman_filter) {
        kalman_filter->reset();
    }
    
    // Reset smoothed movement
    smoothed_movement = Eigen::Vector2f::Zero();
    
    // Reset movement accumulation
    accumulated_x_ = 0.0f;
    accumulated_y_ = 0.0f;
    
    // Reset prediction state
    prediction_initialized_ = false;
    last_target_class_id_ = -1;
    current_velocity_ = {0, 0};
    last_velocity_ = {0, 0};
    current_acceleration_ = {0, 0};
    last_target_pos_ = {0, 0};
    last_target_time_ = std::chrono::high_resolution_clock::now();
    
    // Reset accumulated movement
    accumulated_x_ = 0.0f;
    accumulated_y_ = 0.0f;
}

void MouseThread::asyncInputWorker()
{
    auto& ctx = AppContext::getInstance();
    
    while (!should_stop_thread_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for commands or stop signal
        queue_cv_.wait(lock, [this] {
            return !mouse_command_queue_.empty() || should_stop_thread_.load();
        });
        
        // Process all pending commands
        while (!mouse_command_queue_.empty() && !should_stop_thread_.load()) {
            MouseCommand cmd = mouse_command_queue_.front();
            mouse_command_queue_.pop();
            
            // Release lock while executing command
            lock.unlock();
            
            // Execute command based on type
            {
                std::lock_guard<std::mutex> input_lock(input_method_mutex);
                if (input_method && input_method->isValid()) {
                    auto exec_start = std::chrono::high_resolution_clock::now();
                    
                    switch (cmd.type) {
                        case MouseCommand::MOVE:
                            if (cmd.dx != 0 || cmd.dy != 0) {
                                input_method->move(cmd.dx, cmd.dy);
                            }
                            break;
                        case MouseCommand::PRESS:
                            input_method->press();
                            break;
                        case MouseCommand::RELEASE:
                            input_method->release();
                            break;
                    }
                    
                    // Track performance if needed
                    if (ctx.config.verbose || ctx.config.show_metrics) {
                        auto exec_end = std::chrono::high_resolution_clock::now();
                        float exec_duration_ms = std::chrono::duration<float, std::milli>(exec_end - exec_start).count();
                        ctx.g_current_input_send_time_ms.store(exec_duration_ms, std::memory_order_relaxed);
                        ctx.add_to_history(ctx.g_input_send_time_history, exec_duration_ms, ctx.g_input_send_history_mutex);
                        
                        // Also track queue latency (time from enqueue to execution)
                        float queue_latency_ms = std::chrono::duration<float, std::milli>(exec_start - cmd.timestamp).count();
                        // You can add a new metric for queue latency if needed
                    }
                }
            }
            
            // Reacquire lock for next iteration
            lock.lock();
        }
    }
}

void MouseThread::enqueueMouseCommand(MouseCommand::Type type, int dx, int dy)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Add command to queue with timestamp
        MouseCommand cmd;
        cmd.type = type;
        cmd.dx = dx;
        cmd.dy = dy;
        cmd.timestamp = std::chrono::high_resolution_clock::now();
        
        mouse_command_queue_.push(cmd);
    }
    
    // Notify worker thread
    queue_cv_.notify_one();
}
