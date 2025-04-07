#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <iostream> // For debugging recoil timing

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "KalmanFilter2D.h"
#include "PIDController2D.h"
#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "config.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

// Constants for PID controller - Removing hardcoded limits
// constexpr float MAX_OUTPUT_X = 375.0f;  // Removed
// constexpr float MAX_OUTPUT_Y = 300.0f;  // Removed
// constexpr float MAX_INTEGRAL_X = 80.0f; // Removed
// constexpr float MAX_INTEGRAL_Y = 60.0f; // Removed
// constexpr float ERROR_THRESHOLD_X = 80.0f; // Removed
// constexpr float ERROR_THRESHOLD_Y = 60.0f; // Removed
constexpr float SCOPE_MARGIN = 0.15f;

// Constants for Kalman filter - Changed to float
// constexpr float VEL_NOISE_FACTOR = 2.5f; // Remove definition from here, use KalmanFilter2D.h
// ACC_NOISE_FACTOR is removed if using 4D filter
// constexpr float ACC_NOISE_FACTOR = 4.0f; // Commented out assuming 4D filter
constexpr float BASE_PREDICTION_FACTOR = 0.07f;

MouseThread::MouseThread(
    int resolution,
    int dpi,
    int fovX,
    int fovY,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    float process_noise_q,
    float measurement_noise_r,
    bool auto_shoot,
    float bScope_multiplier,
    float norecoil_ms,
    SerialConnection *serialConnection,
    GhubMouse *gHub) : tracking_errors(false)
{
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier, norecoil_ms);
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    initializeInputMethod(serialConnection, gHub);
    last_prediction_time = std::chrono::steady_clock::now();
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

void MouseThread::initializeScreen(int resolution, int dpi, int fovX, int fovY, bool auto_shoot, float bScope_multiplier, float norecoil_ms)
{
    this->screen_width = static_cast<float>(resolution);
    this->screen_height = static_cast<float>(resolution); // Should this be height? Yes, likely. Consider using separate config value if needed.
    this->dpi = static_cast<float>(dpi);
    this->fov_x = static_cast<float>(fovX);
    this->fov_y = static_cast<float>(fovY);
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->norecoil_ms = norecoil_ms; // Initialize norecoil_ms
    this->center_x = screen_width / 2.0f;
    this->center_y = screen_height / 2.0f;

    // Pre-calculate movement scaling factors
    // Avoid division by zero if dpi is zero
    float dpi_safe = (this->dpi > 1e-3f) ? this->dpi : 1.0f;
    float base_scale_x = (this->fov_x / 360.0f) * (1000.0f / dpi_safe);
    float base_scale_y = (this->fov_y / 360.0f) * (1000.0f / dpi_safe);

    // Incorporate scope multiplier into the pre-calculated scale
    if (this->bScope_multiplier > 1.0f) {
        this->move_scale_x = base_scale_x / this->bScope_multiplier;
        this->move_scale_y = base_scale_y / this->bScope_multiplier;
    } else {
        this->move_scale_x = base_scale_x;
        this->move_scale_y = base_scale_y;
    }

    // Initialize last recoil time
    this->last_recoil_compensation_time = std::chrono::steady_clock::now();
}

void MouseThread::updateConfig(
    int resolution,
    int dpi,
    int fovX,
    int fovY,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    float process_noise_q,
    float measurement_noise_r,
    bool auto_shoot,
    float bScope_multiplier,
    float norecoil_ms)
{
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier, norecoil_ms);
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
}

Eigen::Vector2f MouseThread::predictTargetPosition(float target_x, float target_y)
{
    auto current_time = std::chrono::steady_clock::now();
    float dt = std::min(std::chrono::duration<float>(current_time - last_prediction_time).count(), 0.1f);
    last_prediction_time = current_time;

    // Update Kalman filter
    kalman_filter->predict(dt);
    Eigen::Vector2f measurement(target_x, target_y);
    kalman_filter->update(measurement);

    // Get state
    const Eigen::Matrix<float, 4, 1>& state = kalman_filter->getState();
    float pos_x = state(0, 0);
    float pos_y = state(1, 0);
    float vel_x = state(2, 0);
    float vel_y = state(3, 0);

    // Calculate velocity magnitude
    float velocity_magnitude = std::sqrt(vel_x * vel_x + vel_y * vel_y);

    // Simplified prediction time calculation
    float prediction_time = 0.03f; // Fixed base prediction time (30ms)
    
    // Only scale prediction time based on velocity
    if (velocity_magnitude > 50.0f) {
        prediction_time *= std::min(velocity_magnitude / 100.0f, 1.5f);
    }

    // Calculate predicted position
    float future_x = pos_x + vel_x * prediction_time;
    float future_y = pos_y + vel_y * prediction_time;

    return Eigen::Vector2f(future_x, future_y);
}

Eigen::Vector2f MouseThread::calculateMovement(const Eigen::Vector2f &target_pos)
{
    static const float fov_scale_x = fov_x / screen_width;
    static const float fov_scale_y = fov_y / screen_height;
    static const float sens_scale = dpi / 360.0f;
    
    float error_x = target_pos[0] - center_x;
    float error_y = target_pos[1] - center_y;
    
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    float result_x = pid_output[0] * fov_scale_x * sens_scale;
    float result_y = pid_output[1] * fov_scale_y * sens_scale;
    
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

float MouseThread::calculateTargetDistance(const AimbotTarget &target) const
{
    float dx = target.x + target.w * 0.5f - center_x;
    float dy = target.y + target.h * 0.5f - center_y;
    return std::sqrt(dx * dx + dy * dy);
}

AimbotTarget *MouseThread::findClosestTarget(const std::vector<AimbotTarget> &targets) const
{
    if (targets.empty())
    {
        return nullptr;
    }

    AimbotTarget *closest = nullptr;
    float min_distance = std::numeric_limits<float>::max(); // Use float max

    for (const auto &target : targets)
    {
        // Calculate squared distance to avoid sqrt for comparison
        float dx = target.x + target.w * 0.5f - center_x;
        float dy = target.y + target.h * 0.5f - center_y;
        float distance_sq = dx * dx + dy * dy;

        if (distance_sq < min_distance) // Compare squared distances
        {
            min_distance = distance_sq; // Store the minimum squared distance
            closest = const_cast<AimbotTarget *>(&target);
        }
    }

    return closest;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    const float local_center_x = center_x;
    const float local_center_y = center_y;
    const float local_fov_x = fov_x;
    const float local_fov_y = fov_y;
    const float local_dpi = dpi;

    float target_center_x = target.x + target.w * 0.5f;
    float target_center_y = target.y + target.h * 0.5f;

    // resetPrediction(); // Why is this called here? Consider if needed on every move.
    
    Eigen::Vector2f predicted = predictTargetPosition(target_center_x, target_center_y);
    float error_x = predicted.x() - local_center_x;
    float error_y = predicted.y() - local_center_y;

    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y);
        }
    }

    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    // Use pre-calculated scaling factors
    float move_x = pid_output.x() * move_scale_x;
    float move_y = pid_output.y() * move_scale_y;

    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

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

void MouseThread::resetPrediction()
{
    kalman_filter->reset();
    pid_controller->reset();
    last_prediction_time = std::chrono::steady_clock::now();
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}

void MouseThread::applyRecoilCompensation(float strength)
{
    // Ensure input method is valid before proceeding
    if (!input_method || !input_method->isValid()) {
        return;
    }

    // Only apply if strength is non-zero
    if (std::abs(strength) < 1e-3f) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_recoil_compensation_time);
    auto required_delay = std::chrono::milliseconds(static_cast<long long>(this->norecoil_ms));

    // Check if enough time has passed since the last compensation
    if (elapsed >= required_delay)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex); // Lock before accessing input_method
        // Apply recoil compensation (move mouse vertically)
        // Cast strength to int after potential scaling/logic if needed
        int dy_recoil = static_cast<int>(std::round(strength));
        if (dy_recoil != 0) {
             input_method->move(0, dy_recoil);
        }
        // Update the time of the last compensation
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
    error_callback = nullptr; // Clear the callback
}