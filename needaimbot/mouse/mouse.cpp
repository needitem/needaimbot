// #include "../constants.h" // File removed
#include "../AppContext.h"
#include "mouse.h"
#include "aimbot_components/AimbotTarget.h"
#include "../detector/capture.h"
// #include "../detector/optical_flow.h" // Optical flow removed
#include "input_drivers/SerialConnection.h"
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


extern std::atomic<bool> aiming;
extern std::mutex configMutex;
// extern Config config;  // Removed - use AppContext::getInstance().config instead


std::random_device rd;
std::mt19937 gen(rd());

constexpr float SCOPE_MARGIN = 0.15f;




thread_local static LARGE_INTEGER freq;
thread_local static bool freq_initialized = false;

void InitializeHighPrecisionTimer() {
    if (!freq_initialized) {
        QueryPerformanceFrequency(&freq);
        freq_initialized = true;
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        SetThreadAffinityMask(GetCurrentThread(), 1 << 1);
    }
}

void QueueMove(int dx, int dy, std::function<void(int, int)> move_func) {
    auto& ctx = AppContext::getInstance();
    InitializeHighPrecisionTimer();
    
    if (dx != 0 || dy != 0) {
        LARGE_INTEGER start;
        QueryPerformanceCounter(&start);
        
        move_func(dx, dy);
        
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        float elapsed_us = ((end.QuadPart - start.QuadPart) * 1000000.0f) / freq.QuadPart;
        
        ctx.add_to_history(ctx.g_input_send_time_history, elapsed_us / 1000.0f, ctx.g_input_send_history_mutex);
        ctx.g_current_input_send_time_ms.store(elapsed_us / 1000.0f);
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
    GhubMouse *gHub) : tracking_errors(false), optical_flow_recoil_frame_count(0)
{
    initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    initializeInputMethod(serialConnection, gHub);
}

MouseThread::~MouseThread() = default;

void MouseThread::initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub)
{
    if (serialConnection && serialConnection->isOpen())
    {
        input_method = std::make_unique<SerialInputMethod>(serialConnection);
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
    
    // Copy all needed config values at once
    float crosshair_offset_x, crosshair_offset_y;
    float movement_smoothing;
    bool use_predictive_controller;
    float prediction_time_ms;
    int head_class_id_to_use = -1;
    bool apply_head_offset = false;
    float head_y_offset, body_y_offset;
    std::string head_class_name;

    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

    // Get all config values in one lock
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        crosshair_offset_x = ctx.config.crosshair_offset_x;
        crosshair_offset_y = ctx.config.crosshair_offset_y;
        movement_smoothing = ctx.config.movement_smoothing;
        use_predictive_controller = ctx.config.use_predictive_controller;
        prediction_time_ms = ctx.config.prediction_time_ms;
        head_class_name = ctx.config.head_class_name;
        head_y_offset = ctx.config.head_y_offset;
        body_y_offset = ctx.config.body_y_offset;
        
        for (const auto& class_setting : ctx.config.class_settings) {
            if (class_setting.name == head_class_name) {
                head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    apply_head_offset = true;
                }
                break;
            }
        }
    }

    // Apply crosshair offset correction
    current_center_x += crosshair_offset_x;
    current_center_y += crosshair_offset_y;

    Point2D raw_target_pos;
    raw_target_pos.x = target.x + target.w * 0.5f;
    
    // Use already copied config values
    float local_y_offset_multiplier_val;
    if (apply_head_offset && target.classId == head_class_id_to_use) {
        local_y_offset_multiplier_val = head_y_offset;
    } else {
        local_y_offset_multiplier_val = body_y_offset;
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
    
    Eigen::Vector2f error(error_x, error_y);
    auto pid_start_time = std::chrono::steady_clock::now();
    
    // Use adaptive or standard PID based on config
    Eigen::Vector2f pid_output;
    if (ctx.config.enable_adaptive_pid) {
        pid_output = pid_controller->calculateAdaptive(error, error_magnitude);
    } else {
        pid_output = pid_controller->calculate(error);
    }
        
    auto pid_end_time = std::chrono::steady_clock::now();
    float pid_duration_ms = std::chrono::duration<float, std::milli>(pid_end_time - pid_start_time).count();
    ctx.g_current_pid_calc_time_ms.store(pid_duration_ms, std::memory_order_relaxed);
    ctx.add_to_history(ctx.g_pid_calc_time_history, pid_duration_ms, ctx.g_pid_calc_history_mutex);

    // Calculate adaptive scale and apply to movement
    float adaptive_scale = calculateAdaptiveScale(error_magnitude);
    float raw_move_x = pid_output.x() * current_move_scale_x * adaptive_scale;
    float raw_move_y = pid_output.y() * current_move_scale_y * adaptive_scale;
    
    Eigen::Vector2f raw_movement(raw_move_x, raw_move_y);
    Eigen::Vector2f smoothed_movement_result = applyMovementSmoothing(raw_movement, error_magnitude, movement_smoothing);
    
    float move_x = smoothed_movement_result.x();
    float move_y = smoothed_movement_result.y();

    // Apply dead zone with adaptive threshold based on error magnitude
    float adaptive_dead_zone = DEAD_ZONE * (1.0f + std::min(1.0f, 10.0f / (error_magnitude + 1.0f)));
    if (std::abs(move_x) < adaptive_dead_zone) move_x = 0.0f;
    if (std::abs(move_y) < adaptive_dead_zone) move_y = 0.0f;
    
    // Process accumulated movement
    auto [dx_int, dy_int] = processAccumulatedMovement(move_x, move_y);
    
    if (local_disable_upward_aim_active && dy_int < 0) {
        dy_int = 0; 
    }

    if (dx_int != 0 || dy_int != 0) {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid()) {
            auto input_send_start_time = std::chrono::steady_clock::now();
            input_method->move(dx_int, dy_int);
            auto input_send_end_time = std::chrono::steady_clock::now();
            float input_send_duration_ms = std::chrono::duration<float, std::milli>(input_send_end_time - input_send_start_time).count();
            ctx.g_current_input_send_time_ms.store(input_send_duration_ms, std::memory_order_relaxed);
            ctx.add_to_history(ctx.g_input_send_time_history, input_send_duration_ms, ctx.g_input_send_history_mutex);
        }
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
    std::lock_guard<std::mutex> lock(input_method_mutex);

    auto bScope = checkTargetInScope(target.x, target.y, target.w, target.h, bScope_multiplier);

    if (bScope && !mouse_pressed)
    {
        if (input_method && input_method->isValid())
        {
            auto input_press_start_time = std::chrono::steady_clock::now();
            input_method->press();
            auto input_press_end_time = std::chrono::steady_clock::now();
            float press_duration_ms = std::chrono::duration<float, std::milli>(input_press_end_time - input_press_start_time).count();
            ctx.g_current_input_send_time_ms.store(press_duration_ms, std::memory_order_relaxed); 
            ctx.add_to_history(ctx.g_input_send_time_history, press_duration_ms, ctx.g_input_send_history_mutex);
        }
        mouse_pressed = true;
    }
}

void MouseThread::releaseMouse()
{
    auto& ctx = AppContext::getInstance();
    if (!mouse_pressed)
        return;

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (input_method && input_method->isValid())
    {
        auto input_release_start_time = std::chrono::steady_clock::now();
        input_method->release();
        auto input_release_end_time = std::chrono::steady_clock::now();
        float release_duration_ms = std::chrono::duration<float, std::milli>(input_release_end_time - input_release_start_time).count();
        ctx.g_current_input_send_time_ms.store(release_duration_ms, std::memory_order_relaxed); 
        ctx.add_to_history(ctx.g_input_send_time_history, release_duration_ms, ctx.g_input_send_history_mutex);
    }
    mouse_pressed = false;
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
    }
    
    raw_target_pos.y = target.y + target.h * local_y_offset_multiplier_val;
    
    // Calculate initial error for prediction
    float initial_error_x = raw_target_pos.x - current_center_x;
    float initial_error_y = raw_target_pos.y - current_center_y;
    
    // Enhanced target prediction using velocity estimation
    Point2D predicted_target_pos = raw_target_pos;
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
            
            // Get velocity history settings from config
            bool use_velocity_history;
            int max_history_size;
            {
                auto& ctx = AppContext::getInstance();
                std::lock_guard<std::mutex> lock(ctx.configMutex);
                use_velocity_history = ctx.config.enable_velocity_history;
                max_history_size = ctx.config.velocity_history_size;
            }
            
            // Update velocity history if enabled
            last_velocity_ = current_velocity_;
            current_velocity_.x = target_vel_x;
            current_velocity_.y = target_vel_y;
            
            if (use_velocity_history) {
                velocity_history_.push_back(current_velocity_);
                velocity_time_history_.push_back(current_time);
                
                // Maintain history size
                while (static_cast<int>(velocity_history_.size()) > max_history_size) {
                    velocity_history_.erase(velocity_history_.begin());
                    velocity_time_history_.erase(velocity_time_history_.begin());
                }
            }
            
            // Calculate weighted average velocity (recent velocities weighted more)
            Point2D avg_velocity{0, 0};
            
            if (use_velocity_history && velocity_history_.size() >= 2) {
                float total_weight = 0.0f;
                for (size_t i = 0; i < velocity_history_.size(); ++i) {
                    float weight = static_cast<float>(i + 1); // Linear weighting, most recent = highest weight
                    avg_velocity.x += velocity_history_[i].x * weight;
                    avg_velocity.y += velocity_history_[i].y * weight;
                    total_weight += weight;
                }
                
                if (total_weight > 0.0f) {
                    avg_velocity.x /= total_weight;
                    avg_velocity.y /= total_weight;
                }
            } else {
                avg_velocity = current_velocity_;
            }
            
            // Adaptive prediction time based on multiple factors
            float error_magnitude = std::sqrt(initial_error_x * initial_error_x + initial_error_y * initial_error_y);
            float velocity_magnitude = std::sqrt(avg_velocity.x * avg_velocity.x + avg_velocity.y * avg_velocity.y);
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
            predicted_target_pos.x = raw_target_pos.x + avg_velocity.x * prediction_time + 
                                   0.5f * current_acceleration_.x * prediction_time * prediction_time;
            predicted_target_pos.y = raw_target_pos.y + avg_velocity.y * prediction_time + 
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
    
    return predicted_target_pos;
}

Eigen::Vector2f MouseThread::applyMovementSmoothing(const Eigen::Vector2f& raw_movement, float error_magnitude, float smoothing_factor)
{
    // Apply movement smoothing for stability without losing speed
    // Less smoothing for small movements, more for large sudden movements
    float smoothing = smoothing_factor;
    if (error_magnitude > LARGE_MOVEMENT_THRESHOLD) {
        smoothing = std::min(MAX_ADDITIONAL_SMOOTHING, smoothing + (error_magnitude - LARGE_MOVEMENT_THRESHOLD) * SMOOTHING_INCREASE_FACTOR);
    }
    
    // Enhanced smoothing with jitter reduction
    // Apply stronger smoothing to small, rapid movements (likely jitter)
    float movement_magnitude = raw_movement.norm();
    if (movement_magnitude < 5.0f && movement_magnitude > 0.1f) {
        // Small movements get extra smoothing to reduce jitter
        smoothing = std::min(0.95f, smoothing + 0.2f);
    }
    
    // Velocity-based smoothing - smooth more when velocity changes rapidly
    Eigen::Vector2f velocity_change = raw_movement - smoothed_movement;
    float velocity_change_magnitude = velocity_change.norm();
    if (velocity_change_magnitude > 10.0f) {
        // Rapid changes get smoothed more
        smoothing = std::min(0.9f, smoothing + 0.1f);
    }
    
    smoothed_movement.x() = smoothing * smoothed_movement.x() + (1.0f - smoothing) * raw_movement.x();
    smoothed_movement.y() = smoothing * smoothed_movement.y() + (1.0f - smoothing) * raw_movement.y();
    
    return smoothed_movement;
}

std::pair<int, int> MouseThread::processAccumulatedMovement(float move_x, float move_y)
{
    // Accumulate sub-pixel movements
    accumulated_x_ += move_x;
    accumulated_y_ += move_y;
    
    int dx_int = 0, dy_int = 0;
    
    // Add dithering to improve sub-pixel accuracy and reduce stepping artifacts
    float dithered_x = accumulated_x_;
    float dithered_y = accumulated_y_;
    
    // Apply dithering if enabled
    {
        auto& ctx = AppContext::getInstance();
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        if (ctx.config.enable_subpixel_dithering) {
            dithered_x += dither_dist_(dither_rng_) * ctx.config.dither_strength;
            dithered_y += dither_dist_(dither_rng_) * ctx.config.dither_strength;
        }
    }
    
    // Only move when accumulated movement exceeds threshold
    if (std::abs(dithered_x) >= MICRO_MOVEMENT_THRESHOLD) {
        dx_int = static_cast<int>(std::round(dithered_x));
        accumulated_x_ -= dx_int;
    }
    if (std::abs(dithered_y) >= MICRO_MOVEMENT_THRESHOLD) {
        dy_int = static_cast<int>(std::round(dithered_y));
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
    // Convert latency from ms to seconds
    float latency_compensation_time = estimated_total_latency_ms_ / 1000.0f;
    
    // Apply additional prediction based on estimated system latency
    Point2D compensated_pos;
    compensated_pos.x = predicted_pos.x + velocity.x * latency_compensation_time;
    compensated_pos.y = predicted_pos.y + velocity.y * latency_compensation_time;
    
    return compensated_pos;
}

float MouseThread::calculateAdaptiveScale(float error_magnitude) const
{
    // Adaptive sensitivity scaling based on error magnitude
    // Close range: higher precision (lower sensitivity)
    // Far range: faster movement (higher sensitivity)
    if (error_magnitude < SMALL_ERROR_THRESHOLD) {
        return CLOSE_RANGE_SCALE; // Higher precision for small errors
    } else if (error_magnitude < MEDIUM_ERROR_THRESHOLD) {
        return NORMAL_RANGE_SCALE; // Normal sensitivity
    } else {
        return FAR_RANGE_SCALE; // Faster movement for large errors
    }
}

void MouseThread::applyRecoilCompensationInternal(float strength, float delay_ms)
{
    auto& ctx = AppContext::getInstance();
    if (!input_method || !input_method->isValid()) {
        return;
    }

    if (std::abs(strength) < 1e-3f) {
        return;
    }

    auto now_chrono_recoil = std::chrono::steady_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_chrono_recoil - last_recoil_compensation_time);
    auto required_delay = std::chrono::milliseconds(static_cast<long long>(delay_ms));

    if (elapsed >= required_delay) {
        std::lock_guard<std::mutex> lock(input_method_mutex); 
        
        // Apply strength directly without accumulation to prevent over-compensation
        int dy_recoil = static_cast<int>(std::round(strength));
        
        if (dy_recoil != 0) {
            auto recoil_send_start_time = std::chrono::steady_clock::now();
            input_method->move(0, dy_recoil);
            auto recoil_send_end_time = std::chrono::steady_clock::now();
            float recoil_send_duration_ms = std::chrono::duration<float, std::milli>(recoil_send_end_time - recoil_send_start_time).count();
            ctx.g_current_input_send_time_ms.store(recoil_send_duration_ms, std::memory_order_relaxed);
            ctx.add_to_history(ctx.g_input_send_time_history, recoil_send_duration_ms, ctx.g_input_send_history_mutex);
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
    
    // Reset smoothed movement
    smoothed_movement = Eigen::Vector2f::Zero();
    
    // Reset movement accumulation
    accumulated_x_ = 0.0f;
    accumulated_y_ = 0.0f;
    
    // Reset prediction state
    prediction_initialized_ = false;
    current_velocity_ = {0, 0};
    last_velocity_ = {0, 0};
    current_acceleration_ = {0, 0};
    velocity_history_.clear();
    velocity_time_history_.clear();
    prediction_initialized_ = false;
    last_target_pos_ = {0, 0};
    last_target_time_ = std::chrono::high_resolution_clock::now();
    
    // Reset accumulated movement
    accumulated_x_ = 0.0f;
    accumulated_y_ = 0.0f;
}
