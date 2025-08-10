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
    static const float MAX_DT = 0.1f;   // 100 ms
    dt = std::clamp(dt, MIN_DT, MAX_DT);

    constexpr float INTEGRAL_CLAMP = 100.0f;
    
    integral.x() = std::clamp(integral.x() + error.x() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    integral.y() = std::clamp(integral.y() + error.y() * dt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);

    // Derivative based on per-sample delta (unit: pixels per sample),
    // avoids huge spikes when dt fluctuates.
    auto& ctx = AppContext::getInstance();
    const float DERIV_DEADBAND = ctx.config.pid_d_deadband; // pixels
    float delta_x = error.x() - prev_error.x();
    float delta_y = error.y() - prev_error.y();
    if (std::fabs(delta_x) < DERIV_DEADBAND) delta_x = 0.0f;
    if (std::fabs(delta_y) < DERIV_DEADBAND) delta_y = 0.0f;

    // Disable D when very close to target to prevent dithering
    const float ERROR_FOR_D_DISABLE = ctx.config.pid_d_disable_error; // pixels
    if (std::fabs(error.x()) < ERROR_FOR_D_DISABLE) delta_x = 0.0f;
    if (std::fabs(error.y()) < ERROR_FOR_D_DISABLE) delta_y = 0.0f;

    // During warmup frames, suppress derivative entirely
    // Warmup frames configurable via config
    if (warmup_frames_remaining > 0) {
        delta_x = 0.0f;
        delta_y = 0.0f;
        warmup_frames_remaining--;
    }

    // Store deltas in tiny ring buffer and use median-of-three for robustness
    recent_delta_x[delta_index] = delta_x;
    recent_delta_y[delta_index] = delta_y;
    delta_index = (delta_index + 1) % 3;

    auto median3 = [](float a, float b, float c) {
        if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
        if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
        return c;
    };

    float med_dx = median3(recent_delta_x[0], recent_delta_x[1], recent_delta_x[2]);
    float med_dy = median3(recent_delta_y[0], recent_delta_y[1], recent_delta_y[2]);

    // Use median derivative deltas directly without clamping
    derivative.x() = med_dx;
    derivative.y() = med_dy;

    
    // Compose PID with capped D contribution to prevent D-dominant jitter
    Eigen::Vector2f output;
    float p_x = kp_x * error.x();
    float i_x = ki_x * integral.x();
    float d_x = kd_x * derivative.x();
    float p_y = kp_y * error.y();
    float i_y = ki_y * integral.y();
    float d_y = kd_y * derivative.y();

    // No output clamp for D contribution

    output.x() = p_x + i_x + d_x;
    output.y() = p_y + i_y + d_y;

    // Remove micro-movements that cause visible jitter but no real correction
    const float OUTPUT_DEADZONE = ctx.config.pid_output_deadzone; // pixels
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


