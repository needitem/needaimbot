#include "PIDController2D.h"
#include <cmath>
#include <algorithm> // For std::min

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y),
      // Initialize cached gains with base gains
      cached_kp_x(kp_x), cached_ki_x(ki_x), cached_kd_x(kd_x),
      cached_kp_y(kp_y), cached_ki_y(ki_y), cached_kd_y(kd_y)
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

    // Optional: Dynamic gain adjustment (consider moving frequency check outside or making interval a member)
    static auto last_gain_update = now; // Keep track of update time locally if adjustment logic stays here
    float time_since_update = std::chrono::duration<float>(now - last_gain_update).count();
    if (time_since_update >= 0.05f)
    {
        constexpr float DYNAMIC_GAIN_ERROR_THRESH_SQ = 2.0f * 2.0f;
        if (error.squaredNorm() < DYNAMIC_GAIN_ERROR_THRESH_SQ) {
            last_gain_update = now;
        } else {
            last_gain_update = now;

            float error_magnitude_x = std::abs(error.x());
            float error_magnitude_y = std::abs(error.y());

            float kp_factor_x = 1.0f + std::min(error_magnitude_x * 0.005f, 0.5f);
            float ki_factor_x = 1.0f - std::min(error_magnitude_x * 0.001f, 0.7f);
            float kp_factor_y = 1.0f + std::min(error_magnitude_y * 0.004f, 0.4f);
            float ki_factor_y = 1.0f - std::min(error_magnitude_y * 0.0015f, 0.8f);

            // Calculate target gains based on base member gains (kp_x, ki_x etc.)
            float target_kp_x = kp_x * kp_factor_x; 
            float target_ki_x = ki_x * ki_factor_x;
            float target_kd_x = kd_x * (1.0f + std::min(error_magnitude_x * 0.001f, 0.3f));
            float target_kp_y = kp_y * kp_factor_y;
            float target_ki_y = ki_y * ki_factor_y;
            float target_kd_y = kd_y * (1.0f + std::min(error_magnitude_y * 0.0015f, 0.4f));

            // Smoothing update to member cached gains
            float alpha = std::min(time_since_update * 10.0f, 1.0f);
            if (alpha > 0.95f) {
                cached_kp_x = target_kp_x;
                cached_ki_x = target_ki_x;
                cached_kd_x = target_kd_x;
                cached_kp_y = target_kp_y;
                cached_ki_y = target_ki_y;
                cached_kd_y = target_kd_y;
            } else {
                float one_minus_alpha = 1.0f - alpha;
                // Update member variables
                cached_kp_x = cached_kp_x * one_minus_alpha + target_kp_x * alpha;
                cached_ki_x = cached_ki_x * one_minus_alpha + target_ki_x * alpha;
                cached_kd_x = cached_kd_x * one_minus_alpha + target_kd_x * alpha;
                cached_kp_y = cached_kp_y * one_minus_alpha + target_kp_y * alpha;
                cached_ki_y = cached_ki_y * one_minus_alpha + target_ki_y * alpha;
                cached_kd_y = cached_kd_y * one_minus_alpha + target_kd_y * alpha;
            }
        }
    }

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

    // Calculate Final Output using member cached gains
    Eigen::Vector2f output;
    output.x() = cached_kp_x * error.x() + cached_ki_x * integral.x() + cached_kd_x * derivative.x();
    output.y() = cached_kp_y * error.y() + cached_ki_y * integral.y() + cached_kd_y * derivative.y();

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
    // Reset cached gains to base gains when controller resets
    cached_kp_x = kp_x;
    cached_ki_x = ki_x;
    cached_kd_x = kd_x;
    cached_kp_y = kp_y;
    cached_ki_y = ki_y;
    cached_kd_y = kd_y;
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

    // Update member cached gains immediately
    this->cached_kp_x = kp_x;
    this->cached_ki_x = ki_x;
    this->cached_kd_x = kd_x;
    this->cached_kp_y = kp_y;
    this->cached_ki_y = ki_y;
    this->cached_kd_y = kd_y;
}
