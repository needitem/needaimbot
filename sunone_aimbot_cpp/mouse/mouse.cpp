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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "PIDController2D.h"
#include "mouse.h"
#include "AimbotTarget.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "config.h"
#include "keyboard/keyboard_listener.h"
#include "IPredictor.h"
#include "VelocityPredictor.h"
#include "LinearRegressionPredictor.h"
#include "ExponentialSmoothingPredictor.h"
#include "KalmanFilterPredictor.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

constexpr float SCOPE_MARGIN = 0.15f;

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
    GhubMouse *gHub) : tracking_errors(false)
{
    initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    initializeInputMethod(serialConnection, gHub);
    
    // Set the initial predictor based on config
    std::string initial_algo;
    {
        std::lock_guard<std::mutex> lock(configMutex);
        initial_algo = config.prediction_algorithm;
    }
    setPredictor(initial_algo); 
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

    const float SENSITIVITY_FACTOR = 0.05f; // Example: Adjust this base sensitivity
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
    initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    // Note: Predictor is NOT updated here. Call setPredictor explicitly if algo changes.
}

void MouseThread::setPredictor(const std::string& algorithm_name) {
    std::lock_guard<std::mutex> lock(predictor_mutex_); // Protect predictor access
    std::lock_guard<std::mutex> config_lock(configMutex); // Protect config access

    std::cout << "[Mouse] Setting predictor algorithm: " << algorithm_name << std::endl;

    if (algorithm_name == "Velocity Based") {
        auto predictor = std::make_unique<VelocityPredictor>();
        predictor->configure(config.velocity_prediction_ms);
        predictor_ = std::move(predictor);
    } else if (algorithm_name == "Linear Regression") {
        auto predictor = std::make_unique<LinearRegressionPredictor>();
        predictor->configure(config.lr_past_points, config.velocity_prediction_ms); // Assuming LR uses same prediction time for now
        predictor_ = std::move(predictor);
    } else if (algorithm_name == "Exponential Smoothing") {
        auto predictor = std::make_unique<ExponentialSmoothingPredictor>();
        predictor->configure(config.es_alpha, config.velocity_prediction_ms); // Assuming ES uses same prediction time for now
        predictor_ = std::move(predictor);
    } else if (algorithm_name == "Kalman Filter") {
        auto predictor = std::make_unique<KalmanFilterPredictor>();
        // Assuming Kalman uses its own prediction time setting from config
        // Note: The config struct in the prompt has kalman_* noise vars but also a separate prediction_time_ms.
        // Let's use the dedicated KF prediction time if it exists, otherwise maybe fallback?
        // For now, assume config has kalman_q, kalman_r, kalman_p and velocity_prediction_ms. We use the latter.
        // TODO: Clarify which prediction time variable Kalman should use. Using velocity_prediction_ms for now.
        predictor->configure(config.kalman_q, config.kalman_r, config.kalman_p, config.velocity_prediction_ms);
        predictor_ = std::move(predictor);
    } else { // "None" or unknown
        std::cout << "[Mouse] No predictor or unknown algorithm specified. Prediction disabled." << std::endl;
        predictor_.reset(); // Set predictor to null
    }

    if (predictor_) {
        predictor_->reset(); // Reset the state of the new predictor
    }
}

Eigen::Vector2f MouseThread::calculateMovement(const Eigen::Vector2f &target_pos)
{
    float error_x = target_pos[0] - center_x;
    float error_y = target_pos[1] - center_y;
    
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    float result_x = pid_output[0] * move_scale_x;
    float result_y = pid_output[1] * move_scale_y;
    
    return Eigen::Vector2f(result_x, result_y);
}

bool MouseThread::checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor)
{
    static const float screen_margin_x = screen_width * SCOPE_MARGIN;
    static const float screen_margin_y = screen_height * SCOPE_MARGIN;
    
    float target_center_x = target_x + target_w * 0.5f;
    float target_center_y = target_y + target_h * 0.5f;
    
    float diff_x = std::abs(target_center_x - center_x);
    float diff_y = std::abs(target_center_y - center_y);
    
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    float reduced_half_w = target_w * reduction_factor * 0.5f;
    float reduced_half_h = target_h * reduction_factor * 0.5f;
    
    float min_x = target_center_x - reduced_half_w;
    float max_x = target_center_x + reduced_half_w;
    float min_y = target_center_y - reduced_half_h;
    float max_y = target_center_y + reduced_half_h;
    
    return (center_x >= min_x && center_x <= max_x && 
            center_y >= min_y && center_y <= max_y);
}

float MouseThread::calculateTargetDistanceSquared(const AimbotTarget &target) const
{
    float dx = target.x + target.w * 0.5f - center_x;
    float target_center_y;

    int head_class_id_to_use = -1;
    bool apply_head_offset = false;
    {
        std::lock_guard<std::mutex> lock(configMutex); // Protects access to config.class_settings and config.head_class_name
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == config.head_class_name) {
                head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) { // Only apply head offset if the designated head class is not ignored
                    apply_head_offset = true;
                }
                break;
            }
        }
    }

    if (apply_head_offset && target.classId == head_class_id_to_use) {
        std::lock_guard<std::mutex> lock(configMutex); // Protects access to config.head_y_offset
        target_center_y = target.y + target.h * config.head_y_offset;
    } else {
        std::lock_guard<std::mutex> lock(configMutex); // Protects access to config.body_y_offset
        target_center_y = target.y + target.h * config.body_y_offset;
    }
    float dy = target_center_y - center_y;
    return dx * dx + dy * dy;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    const float local_center_x = center_x;
    const float local_center_y = center_y;

    // 1. Get Raw Target Position
    Point2D raw_target_pos;
    raw_target_pos.x = target.x + target.w * 0.5f;
    
    float y_offset_multiplier_val; // Renamed from y_offset_multiplier to avoid conflict if it was a member
    int head_class_id_to_use = -1;
    bool apply_head_offset = false;

    {
        std::lock_guard<std::mutex> lock(configMutex);
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == config.head_class_name) {
                head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    apply_head_offset = true;
                }
                break;
            }
        }

        if (apply_head_offset && target.classId == head_class_id_to_use) {
            y_offset_multiplier_val = config.head_y_offset;
        } else {
            y_offset_multiplier_val = config.body_y_offset;
        } 
    }
    raw_target_pos.y = target.y + target.h * y_offset_multiplier_val;

    // 2. Update and Predict using the Predictor
    Point2D predicted_target_pos = raw_target_pos; // Default to raw if no predictor
    auto now = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(predictor_mutex_); // Lock predictor access
        if (predictor_) { 
            predictor_->update(raw_target_pos, now);
            // Predict the full position, but we'll only use the X component later
            predicted_target_pos = predictor_->predict(); 
        } else {
            // No predictor active; predicted_target_pos remains raw_target_pos.
        }
    }

    // 3. Calculate Error based on Predicted X and Raw Y Position
    float error_x = predicted_target_pos.x - local_center_x; // Use predicted X
    float error_y = raw_target_pos.y - local_center_y;      // Use raw Y

    // Error tracking callback (optional)
    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y); // Report error based on prediction
        }
    }

    // 4. Calculate PID Output based on Error
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    // 5. Scale PID Output for Mouse Movement
    float move_x = pid_output.x() * move_scale_x;
    float move_y = pid_output.y() * move_scale_y;

    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    // Check for disable upward aim button
    if (isAnyKeyPressed(config.button_disable_upward_aim) && dy_int < 0)
    {
        dy_int = 0; 
    }

    // 6. Send Mouse Movement Command
    if (dx_int != 0 || dy_int != 0)
    {        
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            input_method->move(dx_int, dy_int);
        }
    }
}

void MouseThread::pressMouse(const AimbotTarget &target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    auto bScope = checkTargetInScope(target.x, target.y, target.w, target.h, bScope_multiplier);

    if (bScope && !mouse_pressed)
    {
        if (input_method)
        {
            input_method->press();
        }
        mouse_pressed = true;
    }
}

void MouseThread::releaseMouse()
{
    if (!mouse_pressed)
        return;

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (input_method)
    {
        input_method->release();
    }
    mouse_pressed = false;
}

void MouseThread::applyRecoilCompensation(float strength)
{
    if (!input_method || !input_method->isValid()) {
        return;
    }

    if (std::abs(strength) < 1e-3f) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_recoil_compensation_time);
    auto required_delay = std::chrono::milliseconds(static_cast<long long>(this->norecoil_ms));

    if (elapsed >= required_delay)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex); 
        
        int dy_recoil = static_cast<int>(std::round(strength));
        if (dy_recoil != 0) {
             input_method->move(0, dy_recoil);
        }
        
        last_recoil_compensation_time = now;
    }
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

void MouseThread::resetPredictor()
{
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    if (predictor_)
    {
        predictor_->reset();
        std::cout << "[Mouse] Predictor state reset." << std::endl;
    }
}

bool MouseThread::hasActivePredictor() const
{
    std::lock_guard<std::mutex> lock(predictor_mutex_); // Ensure thread-safe access
    return predictor_ != nullptr;
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}