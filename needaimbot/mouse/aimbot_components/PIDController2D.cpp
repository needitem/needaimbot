#include "PIDController2D.h"
#include <cmath>
#include <algorithm>
#include "../../config/config.h"

extern Config config;
extern std::mutex configMutex; 

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y)
{
    reset();
}

Eigen::Vector2f PIDController2D::calculate(const Eigen::Vector2f &error)
{
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::nano>(now - last_time_point).count() * 1e-9f;
    last_time_point = now;
    
    static const float MIN_DT = 1e-6f;
    static const float MAX_DT = 0.1f;
    dt = std::clamp(dt, MIN_DT, MAX_DT);

    constexpr float INTEGRAL_CLAMP = 100.0f;
    
    integral.x() = std::clamp(integral.x() + error.x() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    integral.y() = std::clamp(integral.y() + error.y() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);

    const float inv_dt = 1.0f / dt;
    float current_derivative_x = (error.x() - prev_error.x()) * inv_dt;
    float current_derivative_y = (error.y() - prev_error.y()) * inv_dt;

    // No smoothing - direct derivative calculation
    derivative.x() = current_derivative_x;
    derivative.y() = current_derivative_y;

    prev_derivative = derivative;

    
    Eigen::Vector2f output;
    output.x() = kp_x * error.x() + ki_x * integral.x() + kd_x * derivative.x();
    output.y() = kp_y * error.y() + ki_y * integral.y() + kd_y * derivative.y();

    prev_error = error;
    return output;
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2f::Zero();
    integral = Eigen::Vector2f::Zero();
    derivative = Eigen::Vector2f::Zero();
    prev_derivative = Eigen::Vector2f::Zero();
    smoothed_derivative = Eigen::Vector2f::Zero();
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

Eigen::Vector2f PIDController2D::calculateAdaptive(const Eigen::Vector2f &error, float error_magnitude)
{
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::nano>(now - last_time_point).count() * 1e-9f;
    last_time_point = now;
    
    static const float MIN_DT = 1e-6f;
    static const float MAX_DT = 0.1f;
    dt = std::clamp(dt, MIN_DT, MAX_DT);

    // Distance-based PID parameter adaptation
    float distance_factor = std::clamp(error_magnitude / 100.0f, 0.1f, 3.0f);
    
    // Reduce derivative gain for large errors to prevent oscillation
    float adaptive_kd_x = kd_x;
    float adaptive_kd_y = kd_y;
    if (error_magnitude > 50.0f) {
        float damping_factor = 1.0f / (1.0f + (error_magnitude - 50.0f) * 0.01f);
        adaptive_kd_x *= damping_factor;
        adaptive_kd_y *= damping_factor;
    }
    
    // Reduce integral windup for large errors
    float integral_scale = error_magnitude > 30.0f ? 0.5f : 1.0f;
    constexpr float INTEGRAL_CLAMP = 100.0f;
    
    integral.x() = std::clamp(integral.x() + error.x() * dt * integral_scale, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    integral.y() = std::clamp(integral.y() + error.y() * dt * integral_scale, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);

    const float inv_dt = 1.0f / dt;
    float current_derivative_x = (error.x() - prev_error.x()) * inv_dt;
    float current_derivative_y = (error.y() - prev_error.y()) * inv_dt;

    // Adaptive smoothing - less smoothing for close targets, more for far targets
    float smoothing;
    {
        std::lock_guard<std::mutex> lock(configMutex);
        smoothing = config.pid_derivative_smoothing;
    }
    if (error_magnitude > 50.0f) {
        smoothing = std::min(0.6f, smoothing + (error_magnitude - 50.0f) * 0.002f);
    }
    
    // Apply exponential smoothing to derivative
    smoothed_derivative.x() = smoothing * smoothed_derivative.x() + (1.0f - smoothing) * current_derivative_x;
    smoothed_derivative.y() = smoothing * smoothed_derivative.y() + (1.0f - smoothing) * current_derivative_y;
    
    derivative = smoothed_derivative;

    // Calculate output with adaptive gains
    Eigen::Vector2f output;
    output.x() = kp_x * error.x() + ki_x * integral.x() + adaptive_kd_x * derivative.x();
    output.y() = kp_y * error.y() + ki_y * integral.y() + adaptive_kd_y * derivative.y();

    prev_error = error;
    return output;
}

