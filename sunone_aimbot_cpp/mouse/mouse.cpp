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
#include "mouse.h"
#include "aimbot_components/AimbotTarget.h"
#include "capture/capture.h"
#include "input_drivers/SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "input_drivers/ghub.h"
#include "config/config.h"
#include "keyboard/keyboard_listener.h"
#include "predictors/IPredictor.h"
#include "predictors/VelocityPredictor.h"
#include "predictors/LinearRegressionPredictor.h"
#include "predictors/ExponentialSmoothingPredictor.h"
#include "predictors/KalmanFilterPredictor.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

// For WindMouse
std::random_device rd;
std::mt19937 gen(rd());

constexpr float SCOPE_MARGIN = 0.15f;

// WindMouse function based on the commit and typical implementations
void WindMouse(float target_x, float target_y, float G, float W, float M, float D,
               std::function<void(int, int)> move_func)
{
    float current_x = 0, current_y = 0;
    float v_x = 0, v_y = 0;
    float W_sqrt = sqrt(W);
    float D_sqrt = sqrt(D);

    float dist, random_dist, new_hypot;
    float new_x_vel, new_y_vel;
    float new_x, new_y;

    while (true) {
        dist = hypot(target_x - current_x, target_y - current_y);
        if (dist <= 1.0f) { // Destination reached
            break;
        }

        random_dist = std::uniform_real_distribution<float>(0.0f, dist / D_sqrt)(gen);
        new_hypot = std::uniform_real_distribution<float>(0.0f, dist / M)(gen);

        v_x += W_sqrt * (std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen)) + G * (target_x - current_x - v_x) / dist;
        v_y += W_sqrt * (std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen)) + G * (target_y - current_y - v_y) / dist;

        // Normalize velocity
        float v_magnitude = hypot(v_x, v_y);
        if (v_magnitude > new_hypot) {
            v_x = (new_hypot / v_magnitude) * v_x;
            v_y = (new_hypot / v_magnitude) * v_y;
        }
        
        new_x = current_x + v_x;
        new_y = current_y + v_y;

        // Ensure movement doesn't overshoot.
        if (hypot(target_x - new_x, target_y - new_y) > dist) {
             new_x = current_x + (target_x - current_x) * (dist - random_dist) / dist;
             new_y = current_y + (target_y - current_y) * (dist - random_dist) / dist;
        }
        
        int move_dx = static_cast<int>(std::round(new_x - current_x));
        int move_dy = static_cast<int>(std::round(new_y - current_y));

        if (move_dx != 0 || move_dy != 0) {
            move_func(move_dx, move_dy);
        }
        
        current_x = new_x;
        current_y = new_y;

        Sleep(1); // Small delay to simulate human-like movement pauses
    }
}

// QueueMove function (simple passthrough for now, as its direct usage isn't clear from the commit's MouseThread changes)
void QueueMove(int dx, int dy, std::function<void(int, int)> move_func) {
    if (dx != 0 || dy != 0) {
        move_func(dx, dy);
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
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); // Lock for screen/timing params
        initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    }
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
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); // Protect reads of member data
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
        std::lock_guard<std::mutex> lock(member_data_mutex_); // Protect reads of member data
        current_screen_width = this->screen_width;
        current_screen_height = this->screen_height;
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }

    const float screen_margin_x = current_screen_width * SCOPE_MARGIN; // Use read value
    const float screen_margin_y = current_screen_height * SCOPE_MARGIN; // Use read value
    
    float target_center_x_val = target_x + target_w * 0.5f;
    float target_center_y_val = target_y + target_h * 0.5f;
    
    float diff_x = std::abs(target_center_x_val - current_center_x); // Use read value
    float diff_y = std::abs(target_center_y_val - current_center_y); // Use read value
    
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
            current_center_y >= min_y && current_center_y <= max_y); // Use read values
}

float MouseThread::calculateTargetDistanceSquared(const AimbotTarget &target) const
{
    float current_center_x, current_center_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); // Protect reads of member data
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }

    float dx = target.x + target.w * 0.5f - current_center_x;
    float target_center_y_val;

    int head_class_id_to_use = -1;
    bool apply_head_offset = false;
    float head_y_offset_val; // To store config.head_y_offset
    float body_y_offset_val; // To store config.body_y_offset
    std::string head_class_name_val; // To store config.head_class_name

    {
        std::lock_guard<std::mutex> lock(configMutex); // Protects access to config
        head_class_name_val = config.head_class_name; // Copy needed value
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == head_class_name_val) {
                head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) { // Only apply head offset if the designated head class is not ignored
                    apply_head_offset = true;
                }
                break;
            }
        }
        head_y_offset_val = config.head_y_offset; // Copy needed value
        body_y_offset_val = config.body_y_offset; // Copy needed value
    }

    if (apply_head_offset && target.classId == head_class_id_to_use) {
        target_center_y_val = target.y + target.h * head_y_offset_val;
    } else {
        target_center_y_val = target.y + target.h * body_y_offset_val;
    }
    float dy = target_center_y_val - current_center_y;
    return dx * dx + dy * dy;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;

    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); // Protect reads of member data
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

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
    float error_x = predicted_target_pos.x - current_center_x; // Use read value
    float error_y = raw_target_pos.y - current_center_y;      // Use read value

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
    float move_x = pid_output.x() * current_move_scale_x; // Use read value
    float move_y = pid_output.y() * current_move_scale_y; // Use read value

    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    // Check for disable upward aim button
    bool disable_upward_aim_active = false;
    bool wind_mouse_enabled_val = false;
    float wind_G_val = 18.0f, wind_W_val = 15.0f, wind_M_val = 10.0f, wind_D_val = 8.0f; // Default values

    {
        std::lock_guard<std::mutex> lock(configMutex); // Protect config access
        disable_upward_aim_active = isAnyKeyPressed(config.button_disable_upward_aim);
        wind_mouse_enabled_val = config.wind_mouse_enabled;
        if (wind_mouse_enabled_val) {
            wind_G_val = config.wind_G;
            wind_W_val = config.wind_W;
            wind_M_val = config.wind_M;
            wind_D_val = config.wind_D;
        }
    }

    if (disable_upward_aim_active && dy_int < 0)
    {
        dy_int = 0; 
    }

    // 6. Send Mouse Movement Command
    if (dx_int != 0 || dy_int != 0)
    {        
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            if (wind_mouse_enabled_val) {
                // The target for WindMouse is the calculated dx_int, dy_int
                WindMouse(static_cast<float>(dx_int), static_cast<float>(dy_int),
                          wind_G_val, wind_W_val, wind_M_val, wind_D_val,
                          [this](int mdx, int mdy) {
                              if (this->input_method && this->input_method->isValid()) {
                                 this->input_method->move(mdx, mdy);
                              }
                          });
            } else {
                input_method->move(dx_int, dy_int);
            }
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
    
    long long current_norecoil_ms;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); // Protect read of norecoil_ms
        current_norecoil_ms = static_cast<long long>(this->norecoil_ms);
    }
    auto required_delay = std::chrono::milliseconds(current_norecoil_ms);

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