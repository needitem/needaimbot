#include "PIDController2D.h"
#include <cmath>
#include <algorithm>

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y)
{
    reset();
}

LA::Vector2f PIDController2D::calculate(const LA::Vector2f &error)
{
    // Calculate dt
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_time_point).count();
    last_time_point = now;
    
    // Clamp dt to reasonable values
    dt = std::clamp(dt, 0.001f, 0.1f);
    
    // Integral
    integral += error * dt;
    
    // Derivative
    LA::Vector2f derivative = (error - prev_error) / dt;
    
    // PID output
    LA::Vector2f output;
    output.x() = kp_x * error.x() + ki_x * integral.x() + kd_x * derivative.x();
    output.y() = kp_y * error.y() + ki_y * integral.y() + kd_y * derivative.y();
    
    // Update previous error
    prev_error = error;
    
    return output;
}

void PIDController2D::reset()
{
    prev_error = LA::Vector2f::Zero();
    integral = LA::Vector2f::Zero();
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


