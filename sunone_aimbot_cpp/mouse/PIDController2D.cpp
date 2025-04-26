#include "PIDController2D.h"
#include <cmath>
#include <algorithm> // For std::min

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y)
{
    reset();
}

Eigen::Vector2f PIDController2D::calculate(const Eigen::Vector2f &error)
{
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_time_point).count();
    dt = dt > 0.1f ? 0.1f : dt; // Clamp dt
    last_time_point = now;

    // Remove static local cached gain variables
    // static auto last_gain_update = now;
    // static float cached_kp_x = kp_x; // Use member variable instead
    // ... remove other static declarations ...

    // Calculate Integral Term
    if (dt > 0.0001f)
    {
        integral.x() += error.x() * dt;
        integral.y() += error.y() * dt;

        // Calculate Derivative Term
        float derivative_x = (error.x() - prev_error.x()) / dt;
        float derivative_y = (error.y() - prev_error.y()) / dt;

        // Smoothing derivative (EMA)
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

    // Calculate Final Output using base member gains
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
                                               float kp_y, float ki_y, float kd_y)
{
    // Update base member gains
    this->kp_x = kp_x;
    this->ki_x = ki_x;
    this->kd_x = kd_x;
    this->kp_y = kp_y;
    this->ki_y = ki_y;
    this->kd_y = kd_y;
}
