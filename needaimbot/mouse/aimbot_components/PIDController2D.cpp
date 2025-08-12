#include "PIDController2D.h"
#include <cmath>
#include <algorithm>
#include <mutex>
#include "../../config/config.h"
#include "../../AppContext.h"

extern std::mutex configMutex; 

PIDController2D::PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y)
{
    reset();
}

LA::Vector2f PIDController2D::calculate(const LA::Vector2f &error)
{
    // Use steady_clock consistently for stable dt computation
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float, std::nano>(now - last_time_point).count() * 1e-9f;
    last_time_point = now;
    
    // Avoid extremely small dt causing derivative spikes
    static const float MIN_DT = 1e-3f;  // 1 ms
    static const float MAX_DT = 0.2f;   // 200 ms (relax clamp but scale D down if large)
    dt = std::clamp(dt, MIN_DT, MAX_DT);
    
    auto& ctx = AppContext::getInstance();
    
    // ============ SETPOINT FILTERING ============
    // Smooth sudden target changes to reduce overshoot
    LA::Vector2f current_error = error;
    if (first_error) {
        filtered_error = current_error;
        first_error = false;
    } else {
        float error_alpha = ctx.config.pid_error_smoothing; // 0.3f = more smooth, 1.0f = no smoothing
        filtered_error = error_alpha * current_error + (1.0f - error_alpha) * filtered_error;
    }
    
    // Use filtered error for calculations
    const LA::Vector2f& working_error = ctx.config.pid_use_error_filter ? filtered_error : current_error;

    // ============ IMPROVED ANTI-WINDUP ============
    constexpr float INTEGRAL_CLAMP = 100.0f;
    constexpr float OUTPUT_SATURATION = 100.0f; // Max output before saturation
    
    // Conditional integration - stop integrating if output is saturated and error has same sign
    if (integral_enabled_x) {
        integral.x() = std::clamp(integral.x() + working_error.x() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    }
    if (integral_enabled_y) {
        integral.y() = std::clamp(integral.y() + working_error.y() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    }

    // ============ DERIVATIVE CALCULATION (properly normalized by dt) ============
    float delta_x = working_error.x() - prev_error.x();
    float delta_y = working_error.y() - prev_error.y();

    // Convert to time-normalized derivative (pixels per second)
    float deriv_rate_x = delta_x / dt;
    float deriv_rate_y = delta_y / dt;

    // Store derivative rates in tiny ring buffer and use median-of-three for robustness
    recent_delta_x[delta_index] = deriv_rate_x;
    recent_delta_y[delta_index] = deriv_rate_y;
    delta_index = (delta_index + 1) % 3;

    auto median3 = [](float a, float b, float c) {
        if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
        if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
        return c;
    };

    float med_dx = median3(recent_delta_x[0], recent_delta_x[1], recent_delta_x[2]);
    float med_dy = median3(recent_delta_y[0], recent_delta_y[1], recent_delta_y[2]);

    // Apply exponential moving average (LPF) on derivative to reduce jitter
    // alpha corresponds roughly to cutoff ~ 1/(alpha * dt). Choose alpha based on target ~20-30ms.
    float target_tau_s = 0.03f; // 30ms
    float alpha = dt / (target_tau_s + dt);
    alpha = std::clamp(alpha, 0.0f, 1.0f);
    filtered_deriv_x = (1.0f - alpha) * filtered_deriv_x + alpha * med_dx;
    filtered_deriv_y = (1.0f - alpha) * filtered_deriv_y + alpha * med_dy;

    // Soft scaling for very large dt to avoid stale derivative bursts
    float large_dt_scale = 1.0f;
    if (dt > 0.05f) { // >50ms frame gap
        // Linearly reduce D influence up to 200ms
        large_dt_scale = std::max(0.0f, 1.0f - (dt - 0.05f) / (0.2f - 0.05f));
    }

    derivative.x() = filtered_deriv_x * large_dt_scale;
    derivative.y() = filtered_deriv_y * large_dt_scale;

    
    // ============ PID COMPOSITION ============
    LA::Vector2f output;
    float p_x = kp_x * working_error.x();
    float i_x = ki_x * integral.x();
    float d_x = kd_x * derivative.x();
    float p_y = kp_y * working_error.y();
    float i_y = ki_y * integral.y();
    float d_y = kd_y * derivative.y();
    
    // ============ VELOCITY FEEDFORWARD (Predictive Control) ============
    // Predict overshoot based on current velocity and reduce P gain accordingly
    if (ctx.config.pid_use_velocity_prediction) {
        float prediction_time = ctx.config.pid_prediction_time; // seconds ahead to predict
        float predicted_overshoot_x = derivative.x() * prediction_time;
        float predicted_overshoot_y = derivative.y() * prediction_time;
        
        // If predicted position would overshoot, reduce proportional gain
        if (std::abs(predicted_overshoot_x) > std::abs(working_error.x()) * 0.5f) {
            p_x *= ctx.config.pid_overshoot_suppression; // e.g., 0.5f for aggressive suppression
        }
        if (std::abs(predicted_overshoot_y) > std::abs(working_error.y()) * 0.5f) {
            p_y *= ctx.config.pid_overshoot_suppression;
        }
    }



    // Combine PID components
    output.x() = p_x + i_x + d_x;
    output.y() = p_y + i_y + d_y;
    
    // ============ MOTION PROFILING (Velocity & Acceleration Limiting) ============
    // Limit maximum velocity
    const float MAX_VELOCITY = ctx.config.pid_max_velocity; // pixels per frame
    output.x() = std::clamp(output.x(), -MAX_VELOCITY, MAX_VELOCITY);
    output.y() = std::clamp(output.y(), -MAX_VELOCITY, MAX_VELOCITY);
    
    // Jerk limiting (smooth acceleration changes)
    if (ctx.config.pid_use_jerk_limit) {
        const float MAX_JERK = ctx.config.pid_max_jerk; // max change in acceleration
        float accel_change_x = output.x() - prev_output.x();
        float accel_change_y = output.y() - prev_output.y();
        
        if (std::abs(accel_change_x) > MAX_JERK) {
            output.x() = prev_output.x() + std::copysign(MAX_JERK, accel_change_x);
        }
        if (std::abs(accel_change_y) > MAX_JERK) {
            output.y() = prev_output.y() + std::copysign(MAX_JERK, accel_change_y);
        }
    }
    
    // Update anti-windup state for next iteration
    integral_enabled_x = (std::abs(output.x()) < OUTPUT_SATURATION) || 
                         (std::signbit(working_error.x()) != std::signbit(integral.x()));
    integral_enabled_y = (std::abs(output.y()) < OUTPUT_SATURATION) || 
                         (std::signbit(working_error.y()) != std::signbit(integral.y()));
    
    // Store state for next iteration
    prev_error = working_error;
    prev_output = output;
    
    return output;
}

void PIDController2D::reset()
{
    prev_error = LA::Vector2f::Zero();
    integral = LA::Vector2f::Zero();
    derivative = LA::Vector2f::Zero();
    filtered_error = LA::Vector2f::Zero();
    prev_output = LA::Vector2f::Zero();
    first_error = true;
    integral_enabled_x = true;
    integral_enabled_y = true;
    filtered_deriv_x = 0.0f;
    filtered_deriv_y = 0.0f;
    // Keep clock consistent with calculate()
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


