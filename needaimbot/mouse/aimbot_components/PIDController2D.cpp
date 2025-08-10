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

Eigen::Vector2f PIDController2D::calculate(const Eigen::Vector2f &error)
{
    // Use steady_clock consistently for stable dt computation
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float, std::nano>(now - last_time_point).count() * 1e-9f;
    last_time_point = now;
    
    // Avoid extremely small dt causing derivative spikes
    static const float MIN_DT = 1e-3f;  // 1 ms
    static const float MAX_DT = 0.2f;   // 200 ms (relax clamp but scale D down if large)
    dt = std::clamp(dt, MIN_DT, MAX_DT);

    constexpr float INTEGRAL_CLAMP = 100.0f;
    
    integral.x() = std::clamp(integral.x() + error.x() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    integral.y() = std::clamp(integral.y() + error.y() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);

    // Derivative based on time (unit: pixels per second)
    auto& ctx = AppContext::getInstance();
    const float DERIV_DEADBAND = ctx.config.pid_d_deadband; // pixels (apply on delta before normalization)
    float delta_x = error.x() - prev_error.x();
    float delta_y = error.y() - prev_error.y();
    if (std::fabs(delta_x) < DERIV_DEADBAND) delta_x = 0.0f;
    if (std::fabs(delta_y) < DERIV_DEADBAND) delta_y = 0.0f;

    // Keep D active but will be smoothly scaled later near target

    // During warmup frames, softly scale derivative rather than hard-suppress
    float warmup_scale = 1.0f;
    if (warmup_frames_remaining > 0) {
        warmup_scale = std::max(0.0f, 1.0f - (static_cast<float>(warmup_frames_remaining) / std::max(1, AppContext::getInstance().config.pid_d_warmup_frames)));
        warmup_frames_remaining--;
    }

    // Convert to time-normalized derivative (pixels per second)
    float deriv_rate_x = (delta_x / dt) * warmup_scale;
    float deriv_rate_y = (delta_y / dt) * warmup_scale;

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

    
    // Compose PID with capped D contribution to prevent D-dominant jitter
    Eigen::Vector2f output;
    float p_x = kp_x * error.x();
    float i_x = ki_x * integral.x();
    float d_x = kd_x * derivative.x();
    float p_y = kp_y * error.y();
    float i_y = ki_y * integral.y();
    float d_y = kd_y * derivative.y();

    // Near target, gradually scale D instead of hard-disable to keep damping
    const float ERROR_FOR_D_DISABLE = ctx.config.pid_d_disable_error; // pixels
    auto smooth_step01 = [](float t){ t = std::clamp(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); };
    float d_scale_x = 1.0f;
    float d_scale_y = 1.0f;
    if (ERROR_FOR_D_DISABLE > 0.0f) {
        d_scale_x = smooth_step01(std::fabs(error.x()) / ERROR_FOR_D_DISABLE);
        d_scale_y = smooth_step01(std::fabs(error.y()) / ERROR_FOR_D_DISABLE);
    }

    d_x *= d_scale_x;
    d_y *= d_scale_y;

    // Soft clamp D contribution to avoid domination
    auto soft_clip = [](float v, float limit){
        float a = std::fabs(v);
        if (a <= limit) return v;
        // smoothly compress beyond limit
        float sign = (v >= 0.0f) ? 1.0f : -1.0f;
        return sign * (limit + (a - limit) * 0.3f);
    };
    const float D_LIMIT = 5000.0f; // pixels/sec scaled by kd; conservative large cap
    d_x = soft_clip(d_x, D_LIMIT);
    d_y = soft_clip(d_y, D_LIMIT);

    // Apply output deadzone ONLY to PI, then add D back so damping survives micro deadzone
    float pi_x = p_x + i_x;
    float pi_y = p_y + i_y;
    const float OUTPUT_DEADZONE = ctx.config.pid_output_deadzone; // pixels
    if (std::fabs(pi_x) < OUTPUT_DEADZONE) pi_x = 0.0f;
    if (std::fabs(pi_y) < OUTPUT_DEADZONE) pi_y = 0.0f;

    output.x() = pi_x + d_x;
    output.y() = pi_y + d_y;

    // Remove micro-movements that cause visible jitter but no real correction
    if (std::fabs(output.x()) < OUTPUT_DEADZONE) output.x() = 0.0f;
    if (std::fabs(output.y()) < OUTPUT_DEADZONE) output.y() = 0.0f;

    prev_error = error;
    return output;
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2f::Zero();
    integral = Eigen::Vector2f::Zero();
    derivative = Eigen::Vector2f::Zero();
    // Keep clock consistent with calculate()
    last_time_point = std::chrono::steady_clock::now();
    // Small warmup to avoid initial derivative kick
    warmup_frames_remaining = AppContext::getInstance().config.pid_d_warmup_frames;
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


