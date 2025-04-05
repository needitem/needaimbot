#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "config.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

// Constants for PID controller - Changed to float
constexpr float MAX_OUTPUT_X = 1500.0f;
constexpr float MAX_OUTPUT_Y = 1200.0f;
constexpr float MAX_INTEGRAL_X = 80.0f;
constexpr float MAX_INTEGRAL_Y = 60.0f;
constexpr float ERROR_THRESHOLD_X = 100.0f;
constexpr float ERROR_THRESHOLD_Y = 80.0f;
constexpr float SCOPE_MARGIN = 0.15f;

// Constants for Kalman filter - Changed to float
constexpr float VEL_NOISE_FACTOR = 2.5f;
constexpr float ACC_NOISE_FACTOR = 4.0f;
constexpr float BASE_PREDICTION_FACTOR = 0.07f;

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y)
{
    reset();
}

Eigen::Vector2f PIDController2D::calculate(const Eigen::Vector2f &error)
{
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_time_point).count();
    dt = dt > 0.1f ? 0.1f : dt;
    last_time_point = now;

    static auto last_gain_update = now;
    static float cached_kp_x = kp_x;
    static float cached_ki_x = ki_x;
    static float cached_kd_x = kd_x;
    static float cached_kp_y = kp_y;
    static float cached_ki_y = ki_y;
    static float cached_kd_y = kd_y;

    float time_since_update = std::chrono::duration<float>(now - last_gain_update).count();
    if (time_since_update >= 0.05f)
    {
        // Optimization: Skip dynamic gain update if error is already very small
        constexpr float DYNAMIC_GAIN_ERROR_THRESH_SQ = 2.0f * 2.0f; // e.g., skip if error < 2 pixels (squared)
        if (error.squaredNorm() < DYNAMIC_GAIN_ERROR_THRESH_SQ) {
            // Keep last_gain_update = now; to prevent immediate re-entry if error fluctuates slightly
            last_gain_update = now; 
        } else {
            // Original dynamic gain adjustment logic follows
            last_gain_update = now;

            float error_magnitude_x = std::abs(error.x());
            float error_magnitude_y = std::abs(error.y());

            float kp_factor_x = 1.0f + std::min(error_magnitude_x * 0.01f, 0.6f);
            float ki_factor_x = 1.0f - std::min(error_magnitude_x * 0.0025f, 0.8f);
            
            float kp_factor_y = 1.0f + std::min(error_magnitude_y * 0.00833f, 0.5f);
            float ki_factor_y = 1.0f - std::min(error_magnitude_y * 0.00286f, 0.9f);
            
            float target_kp_x = kp_x * kp_factor_x;
            float target_ki_x = ki_x * ki_factor_x;
            float target_kd_x = kd_x * (1.0f + std::min(error_magnitude_x * 0.002f, 0.4f));
            
            float target_kp_y = kp_y * kp_factor_y;
            float target_ki_y = ki_y * ki_factor_y;
            float target_kd_y = kd_y * (1.0f + std::min(error_magnitude_y * 0.00278f, 0.5f));

            float alpha = std::min(time_since_update * 15.0f, 1.0f);
            if (alpha > 0.95f) {
                cached_kp_x = target_kp_x;
                cached_ki_x = target_ki_x;
                cached_kd_x = target_kd_x;
                cached_kp_y = target_kp_y;
                cached_ki_y = target_ki_y;
                cached_kd_y = target_kd_y;
            } else {
                float one_minus_alpha = 1.0f - alpha;
                cached_kp_x = cached_kp_x * one_minus_alpha + target_kp_x * alpha;
                cached_ki_x = cached_ki_x * one_minus_alpha + target_ki_x * alpha;
                cached_kd_x = cached_kd_x * one_minus_alpha + target_kd_x * alpha;
                cached_kp_y = cached_kp_y * one_minus_alpha + target_kp_y * alpha;
                cached_ki_y = cached_ki_y * one_minus_alpha + target_ki_y * alpha;
                cached_kd_y = cached_kd_y * one_minus_alpha + target_kd_y * alpha;
            }
        } // End of the 'else' block for dynamic gain calculation
    }

    if (dt > 0.0001f)
    {
        static const float inv_error_threshold_x = 1.0f / ERROR_THRESHOLD_X;
        static const float inv_error_threshold_y = 1.0f / ERROR_THRESHOLD_Y;
        
        float abs_error_x = std::abs(error.x());
        float abs_error_y = std::abs(error.y());
        
        float integral_factor_x = abs_error_x > ERROR_THRESHOLD_X ? 
                                  ERROR_THRESHOLD_X / abs_error_x : 1.0f;
        float integral_factor_y = abs_error_y > ERROR_THRESHOLD_Y ? 
                                  ERROR_THRESHOLD_Y / abs_error_y : 1.0f;
        
        integral.x() += error.x() * dt * integral_factor_x;
        integral.y() += error.y() * dt * integral_factor_y;
        
        if (integral.x() > MAX_INTEGRAL_X) integral.x() = MAX_INTEGRAL_X;
        else if (integral.x() < -MAX_INTEGRAL_X) integral.x() = -MAX_INTEGRAL_X;
        
        if (integral.y() > MAX_INTEGRAL_Y) integral.y() = MAX_INTEGRAL_Y;
        else if (integral.y() < -MAX_INTEGRAL_Y) integral.y() = -MAX_INTEGRAL_Y;
        
        float derivative_x = (error.x() - prev_error.x()) / dt;
        float derivative_y = (error.y() - prev_error.y()) / dt;
        
        float alpha_x = (std::abs(derivative_x) > 500.0f) ? 0.7f : 0.85f;
        float alpha_y = (std::abs(derivative_y) > 400.0f) ? 0.6f : 0.9f;
        
        derivative.x() = derivative_x * alpha_x + prev_derivative.x() * (1.0f - alpha_x);
        derivative.y() = derivative_y * alpha_y + prev_derivative.y() * (1.0f - alpha_y);
        
        prev_derivative = derivative;
    }
    else
    {
        derivative.setZero();
    }

    Eigen::Vector2f output;
    output.x() = cached_kp_x * error.x() + cached_ki_x * integral.x() + cached_kd_x * derivative.x();
    output.y() = cached_kp_y * error.y() + cached_ki_y * integral.y() + cached_kd_y * derivative.y();
    
    if (output.x() > MAX_OUTPUT_X) output.x() = MAX_OUTPUT_X;
    else if (output.x() < -MAX_OUTPUT_X) output.x() = -MAX_OUTPUT_X;
    
    if (output.y() > MAX_OUTPUT_Y) output.y() = MAX_OUTPUT_Y;
    else if (output.y() < -MAX_OUTPUT_Y) output.y() = -MAX_OUTPUT_Y;

    prev_error = error;
    return output;
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2f::Zero();
    integral = Eigen::Vector2f::Zero();
    derivative = Eigen::Vector2f::Zero();
    prev_derivative = Eigen::Vector2f::Zero();
    last_time_point = std::chrono::steady_clock::now();
}

void PIDController2D::updateSeparatedParameters(float kp_x, float ki_x, float kd_x, 
                                               float kp_y, float ki_y, float kd_y)
{
    this->kp_x = kp_x;
    this->ki_x = ki_x;
    this->kd_x = kd_x;
    this->kp_y = kp_y;
    this->ki_y = ki_y;
    this->kd_y = kd_y;
}

void KalmanFilter2D::initializeMatrices(float process_noise_q, float measurement_noise_r)
{
    Q = Eigen::Matrix<float, 6, 6>::Identity() * process_noise_q;
    
    Q(2, 2) = process_noise_q * VEL_NOISE_FACTOR;
    Q(3, 3) = process_noise_q * VEL_NOISE_FACTOR;
    Q(4, 4) = process_noise_q * ACC_NOISE_FACTOR;
    Q(5, 5) = process_noise_q * ACC_NOISE_FACTOR;
    
    R = Eigen::Matrix2f::Identity() * measurement_noise_r;
}

KalmanFilter2D::KalmanFilter2D(float process_noise_q, float measurement_noise_r)
{
    A = Eigen::Matrix<float, 6, 6>::Identity();

    H = Eigen::Matrix<float, 2, 6>::Zero();
    H(0, 0) = 1.0f;
    H(1, 1) = 1.0f;

    initializeMatrices(process_noise_q, measurement_noise_r);
    P = Eigen::Matrix<float, 6, 6>::Identity();
    x = Eigen::Matrix<float, 6, 1>::Zero();
}

void KalmanFilter2D::predict(float dt)
{
    A(0, 2) = dt;
    A(0, 4) = 0.5f * dt * dt;
    A(1, 3) = dt;
    A(1, 5) = 0.5f * dt * dt;
    A(2, 4) = dt;
    A(3, 5) = dt;

    x = A * x;
    P = A * P * A.transpose() + Q;
}

void KalmanFilter2D::update(const Eigen::Vector2f &measurement)
{
    Eigen::Matrix2f S = H * P * H.transpose() + R;
    Eigen::Matrix<float, 6, 2> K = P * H.transpose() * S.inverse();

    Eigen::Vector2f y = measurement - H * x;
    x = x + K * y;
    P = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * P;
}

void KalmanFilter2D::reset()
{
    x = Eigen::Matrix<float, 6, 1>::Zero();
    P = Eigen::Matrix<float, 6, 6>::Identity();
}

void KalmanFilter2D::updateParameters(float process_noise_q, float measurement_noise_r)
{
    initializeMatrices(process_noise_q, measurement_noise_r);
}

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
    SerialConnection *serialConnection,
    GhubMouse *gHub) : tracking_errors(false)
{
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier);
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    initializeInputMethod(serialConnection, gHub);
    last_prediction_time = std::chrono::steady_clock::now();
}

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

void MouseThread::initializeScreen(int resolution, int dpi, int fovX, int fovY, bool auto_shoot, float bScope_multiplier)
{
    this->screen_width = static_cast<float>(resolution);
    this->screen_height = static_cast<float>(resolution); // Should this be height?
    this->dpi = static_cast<float>(dpi);
    this->fov_x = static_cast<float>(fovX);
    this->fov_y = static_cast<float>(fovY);
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
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
    float bScope_multiplier)
{
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier);
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
}

Eigen::Vector2f MouseThread::predictTargetPosition(float target_x, float target_y)
{
    auto current_time = std::chrono::steady_clock::now();
    float dt = std::min(std::chrono::duration<float>(current_time - last_prediction_time).count(), 0.1f);
    last_prediction_time = current_time;

    kalman_filter->predict(dt);

    Eigen::Vector2f measurement(target_x, target_y);
    kalman_filter->update(measurement);

    const auto &state = kalman_filter->getState();
    
    float pos_x = state(0, 0);
    float pos_y = state(1, 0);
    float vel_x = state(2, 0);
    float vel_y = state(3, 0);
    // float acc_x = state(4, 0); // Acceleration removed in previous optimization
    // float acc_y = state(5, 0);

    // Use squared velocity for magnitude checks to avoid sqrt
    float velocity_squared = vel_x * vel_x + vel_y * vel_y;
    // Threshold for velocity index (squared value of 200.0f)
    constexpr float VEL_THRESH_SQ = 200.0f * 200.0f;

    constexpr float prediction_factors[4] = {1.0f, 1.5f, 2.0f, 2.5f};
    
    // Calculate index based on squared velocity
    int velocity_idx = std::min(static_cast<int>(velocity_squared / VEL_THRESH_SQ), 3);
    float prediction_time = dt * BASE_PREDICTION_FACTOR * prediction_factors[velocity_idx];

    // Simplified prediction: Remove acceleration term (already done)
    float future_x = pos_x + vel_x * prediction_time;
    float future_y = pos_y + vel_y * prediction_time;
    
    static Eigen::Vector2f prev_velocity(0.0f, 0.0f);
    Eigen::Vector2f current_velocity(vel_x, vel_y);
    
    // Check for significant direction change using dot product and squared norms (avoid sqrt and acos)
    float prev_vel_norm_sq = prev_velocity.squaredNorm();
    // Use a threshold based on squared velocity (VEL_THRESH_SQ)
    if (prev_vel_norm_sq > 1e-4f && velocity_squared > VEL_THRESH_SQ) { // Check prev_vel_norm_sq to avoid division by zero
        float dot_product = prev_velocity.dot(current_velocity);
        // Cosine threshold corresponding to angle 0.5 radians (cos(0.5) approx 0.87758)
        constexpr float COS_ANGLE_THRESH = 0.87758f; 
        // Compare cosine directly: dot / (norm1 * norm2) < threshold <=> dot * dot < threshold^2 * norm1^2 * norm2^2
        // To avoid the remaining sqrt in norm2 (velocity), compare dot directly if norms are similar, or use cosine comparison carefully
        // Simpler approach: compare cosine similarity value
        float cos_similarity = dot_product / std::sqrt(prev_vel_norm_sq * velocity_squared); // Need one sqrt here unfortunately, but avoid acos

        if (cos_similarity < COS_ANGLE_THRESH) { // Angle is greater than 0.5 rad if cosine is smaller
            // Calculate reduction factor based on cosine similarity instead of angle
            // Map cosine similarity range [-1, COS_ANGLE_THRESH] to reduction factor [0.3, 1.0]
            // Example linear mapping: factor = 0.3 + 0.7 * (cos_similarity + 1.0) / (COS_ANGLE_THRESH + 1.0)
            float reduction_factor = 0.3f + 0.7f * (cos_similarity + 1.0f) / (COS_ANGLE_THRESH + 1.0f);
            reduction_factor = std::clamp(reduction_factor, 0.3f, 1.0f); // Ensure it stays within bounds

            future_x = pos_x + vel_x * prediction_time * reduction_factor;
            future_y = pos_y + vel_y * prediction_time * reduction_factor;
        }
    }
    
    prev_velocity = current_velocity;
    
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
    if (input_method)
    {
        input_method->move(0, strength);
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
}