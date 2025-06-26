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
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_time_point).count();
    last_time_point = now;

    if (dt > 0.0001f)
    {
        integral.x() += error.x() * dt;
        integral.y() += error.y() * dt;

        float current_derivative_x = (error.x() - prev_error.x()) / dt;
        float current_derivative_y = (error.y() - prev_error.y()) / dt;

        // Apply smoothing to the derivative term
        derivative.x() = current_derivative_x * derivative_smoothing_factor + prev_derivative.x() * (1.0f - derivative_smoothing_factor);
        derivative.y() = current_derivative_y * derivative_smoothing_factor + prev_derivative.y() * (1.0f - derivative_smoothing_factor);

        prev_derivative = derivative;
    }
    else
    {
        // If dt is too small, maintain the previous derivative to avoid sudden jumps
        // derivative.setZero(); // Uncomment this if you prefer to zero out derivative for very small dt
    }

    
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

