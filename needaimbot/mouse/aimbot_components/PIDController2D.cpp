#include "PIDController2D.h"
#include <cmath>
#include <algorithm> 

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y, float derivative_smoothing_factor)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y), derivative_smoothing_factor(derivative_smoothing_factor)
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

    // Enhanced derivative filtering with adaptive smoothing
    float error_magnitude = std::sqrt(error.x() * error.x() + error.y() * error.y());
    float adaptive_smoothing = derivative_smoothing_factor;
    
    // Use more smoothing for small errors (higher precision)
    // Use less smoothing for large errors (faster response)
    if (error_magnitude < 5.0f) {
        adaptive_smoothing = std::min(derivative_smoothing_factor * 1.5f, 0.9f);
    } else if (error_magnitude > 20.0f) {
        adaptive_smoothing = std::max(derivative_smoothing_factor * 0.7f, 0.1f);
    }
    
    derivative.x() = current_derivative_x * adaptive_smoothing + prev_derivative.x() * (1.0f - adaptive_smoothing);
    derivative.y() = current_derivative_y * adaptive_smoothing + prev_derivative.y() * (1.0f - adaptive_smoothing);

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
    last_time_point = std::chrono::steady_clock::now();
}

void PIDController2D::updateSeparatedParameters(float kp_x, float ki_x, float kd_x,
                                               float kp_y, float ki_y, float kd_y, float derivative_smoothing_factor)
{
    this->kp_x = kp_x;
    this->ki_x = ki_x;
    this->kd_x = kd_x;
    this->kp_y = kp_y;
    this->ki_y = ki_y;
    this->kd_y = kd_y;
    this->derivative_smoothing_factor = derivative_smoothing_factor;
}

