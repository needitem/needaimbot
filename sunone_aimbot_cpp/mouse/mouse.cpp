#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <immintrin.h>

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "config.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

// PID Controller implementation
PIDController2D::PIDController2D(double kp, double ki, double kd)
    : kp(kp), ki(ki), kd(kd), kp_x(kp), ki_x(ki), kd_x(kd), kp_y(kp), ki_y(ki), kd_y(kd)
{
    reset();
}

// Implementation of new constructor using separate X/Y gains
PIDController2D::PIDController2D(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y), 
      kp((kp_x + kp_y) / 2), ki((ki_x + ki_y) / 2), kd((kd_x + kd_y) / 2) // Set the average value as common gain
{
    reset();
}

Eigen::Vector2d PIDController2D::calculate(const Eigen::Vector2d &error)
{
    // Calculate time delta with clamping
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time_point).count();
    dt = std::min(dt, 0.1); // Limit time delta
    last_time_point = now;

    // Static cache for gains and time update to reduce mutex contention
    static auto last_gain_update = now;
    static double cached_kp_x = kp_x;
    static double cached_ki_x = ki_x;
    static double cached_kd_x = kd_x;
    static double cached_kp_y = kp_y;
    static double cached_ki_y = ki_y;
    static double cached_kd_y = kd_y;

    // Only update gains every 30ms to reduce calculations
    double time_since_update = std::chrono::duration<double>(now - last_gain_update).count();
    if (time_since_update >= 0.03)
    {
        last_gain_update = now;
        
        // Extract error components with absolute value
        double error_magnitude_x = std::abs(error.x());
        double error_magnitude_y = std::abs(error.y());

        // Remove unnecessary SIMD and replace with direct calculation
        // X-axis gain calculation elements
        double kp_factor_x = 1.0 + std::min(error_magnitude_x / 100.0, 0.6);
        double ki_factor_x = 1.0 - std::min(error_magnitude_x / 400.0, 0.8);
        
        // Y-axis gain calculation elements
        double kp_factor_y = 1.0 + std::min(error_magnitude_y / 120.0, 0.5);
        double ki_factor_y = 1.0 - std::min(error_magnitude_y / 350.0, 0.9);
        
        // Target gain calculation
        double target_kp_x = kp_x * kp_factor_x;
        double target_ki_x = ki_x * ki_factor_x;
        double target_kd_x = kd_x * (1.0 + std::min(error_magnitude_x / 200.0, 0.4));
        
        double target_kp_y = kp_y * kp_factor_y;
        double target_ki_y = ki_y * ki_factor_y;
        double target_kd_y = kd_y * (1.0 + std::min(error_magnitude_y / 180.0, 0.5));

        // Fast interpolation to avoid sudden changes
        double alpha = std::min(time_since_update * 15.0, 1.0);
        double one_minus_alpha = 1.0 - alpha;
        
        // Remove unnecessary SIMD and perform direct interpolation calculation
        cached_kp_x = cached_kp_x * one_minus_alpha + target_kp_x * alpha;
        cached_ki_x = cached_ki_x * one_minus_alpha + target_ki_x * alpha;
        cached_kp_y = cached_kp_y * one_minus_alpha + target_kp_y * alpha;
        cached_ki_y = cached_ki_y * one_minus_alpha + target_ki_y * alpha;
        
        // Update Kd with simple interpolation
        cached_kd_x = cached_kd_x * one_minus_alpha + target_kd_x * alpha;
        cached_kd_y = cached_kd_y * one_minus_alpha + target_kd_y * alpha;
    }

    // Only update integral and derivative if time delta is significant
    if (dt > 0.0001)
    {
        // Calculate integral limiting factors
        double integral_factor_x = (std::abs(error.x()) > 100.0) ? 
                                  100.0 / std::abs(error.x()) : 1.0;
        double integral_factor_y = (std::abs(error.y()) > 80.0) ? 
                                  80.0 / std::abs(error.y()) : 1.0;
        
        // Update integral terms
        integral.x() += error.x() * dt * integral_factor_x;
        integral.y() += error.y() * dt * integral_factor_y;
        
        // Hard clamp integral to prevent windup
        integral.x() = std::clamp(integral.x(), -80.0, 80.0);
        integral.y() = std::clamp(integral.y(), -60.0, 60.0);
        
        // Remove unnecessary SIMD and perform direct derivative calculation
        double derivative_x = (error.x() - prev_error.x()) / dt;
        double derivative_y = (error.y() - prev_error.y()) / dt;
        
        // Apply different filtering based on derivative magnitude
        double alpha_x = (std::abs(derivative_x) > 500.0) ? 0.7 : 0.85;
        double alpha_y = (std::abs(derivative_y) > 400.0) ? 0.6 : 0.9;
        
        // Update derivative with filtering
        derivative.x() = derivative_x * alpha_x + prev_derivative.x() * (1.0 - alpha_x);
        derivative.y() = derivative_y * alpha_y + prev_derivative.y() * (1.0 - alpha_y);
        
        prev_derivative = derivative;
    }
    else
    {
        derivative.setZero();
    }

    // Remove unnecessary SIMD and perform direct PID output calculation
    double p_term_x = cached_kp_x * error.x();
    double p_term_y = cached_kp_y * error.y();
    
    double i_term_x = cached_ki_x * integral.x();
    double i_term_y = cached_ki_y * integral.y();
    
    double d_term_x = cached_kd_x * derivative.x();
    double d_term_y = cached_kd_y * derivative.y();
    
    // Sum the terms
    double output_x = p_term_x + i_term_x + d_term_x;
    double output_y = p_term_y + i_term_y + d_term_y;
    
    // Output limits (different limits for X and Y)
    const double max_output_x = 1500.0;
    const double max_output_y = 1200.0;
    
    output_x = std::clamp(output_x, -max_output_x, max_output_x);
    output_y = std::clamp(output_y, -max_output_y, max_output_y);

    // Update previous error
    prev_error = error;
    
    // Return the final output
    return Eigen::Vector2d(output_x, output_y);
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2d::Zero();               // Initialize previous error
    integral = Eigen::Vector2d::Zero();                 // Initialize integral term
    derivative = Eigen::Vector2d::Zero();               // Initialize derivative term
    prev_derivative = Eigen::Vector2d::Zero();          // Initialize previous derivative
    last_time_point = std::chrono::steady_clock::now(); // Initialize time
}

void PIDController2D::updateParameters(double kp, double ki, double kd)
{
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
    
    // Common gain is also set to X/Y gain (backwards compatibility)
    this->kp_x = kp;
    this->ki_x = ki;
    this->kd_x = kd;
    this->kp_y = kp;
    this->ki_y = ki;
    this->kd_y = kd;
}

// X/Y separate gain update function implementation
void PIDController2D::updateSeparatedParameters(double kp_x, double ki_x, double kd_x, 
                                               double kp_y, double ki_y, double kd_y)
{
    this->kp_x = kp_x;
    this->ki_x = ki_x;
    this->kd_x = kd_x;
    this->kp_y = kp_y;
    this->ki_y = ki_y;
    this->kd_y = kd_y;
    
    // Common gain is set to X/Y average (for compatibility with existing code)
    this->kp = (kp_x + kp_y) / 2;
    this->ki = (ki_x + ki_y) / 2;
    this->kd = (kd_x + kd_y) / 2;
}

// Kalman filter implementation
KalmanFilter2D::KalmanFilter2D(double process_noise_q, double measurement_noise_r)
{
    // Initialize state transition matrix
    A = Eigen::Matrix<double, 6, 6>::Identity();

    // Initialize measurement matrix (position only)
    H = Eigen::Matrix<double, 2, 6>::Zero();
    H(0, 0) = 1.0; // x position
    H(1, 1) = 1.0; // y position

    // Initialize noise matrices - apply different noise values to position, velocity, acceleration
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
    
    // Increase noise for velocity and acceleration to be more sensitive to rapid movements
    Q(2, 2) = process_noise_q * 2.5; // Increase process noise for vx
    Q(3, 3) = process_noise_q * 2.5; // Increase process noise for vy
    Q(4, 4) = process_noise_q * 4.0; // Increase process noise for ax
    Q(5, 5) = process_noise_q * 4.0; // Increase process noise for ay
    
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
    P = Eigen::Matrix<double, 6, 6>::Identity();

    x = Eigen::Matrix<double, 6, 1>::Zero();
}

void KalmanFilter2D::predict(double dt)
{
    // Update state transition matrix according to dt
    A(0, 2) = dt; // x = x + vx*dt + 0.5*ax*dt^2
    A(0, 4) = 0.5 * dt * dt;
    A(1, 3) = dt; // y = y + vy*dt + 0.5*ay*dt^2
    A(1, 5) = 0.5 * dt * dt;
    A(2, 4) = dt; // vx = vx + ax*dt
    A(3, 5) = dt; // vy = vy + ay*dt

    x = A * x;
    P = A * P * A.transpose() + Q;
}

void KalmanFilter2D::update(const Eigen::Vector2d &measurement)
{
    Eigen::Matrix2d S = H * P * H.transpose() + R;
    Eigen::Matrix<double, 6, 2> K = P * H.transpose() * S.inverse();

    Eigen::Vector2d y = measurement - H * x;
    x = x + K * y;
    P = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * P;
}

void KalmanFilter2D::reset()
{
    x = Eigen::Matrix<double, 6, 1>::Zero();
    P = Eigen::Matrix<double, 6, 6>::Identity();
}

void KalmanFilter2D::updateParameters(double process_noise_q, double measurement_noise_r)
{
    // Update basic noise
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
    
    // Increase velocity and acceleration noise to be more sensitive to rapid movements
    Q(2, 2) = process_noise_q * 2.5; // Increase process noise for vx
    Q(3, 3) = process_noise_q * 2.5; // Increase process noise for vy
    Q(4, 4) = process_noise_q * 4.0; // Increase process noise for ax
    Q(5, 5) = process_noise_q * 4.0; // Increase process noise for ay
    
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
}

// MouseThread Implementation
MouseThread::MouseThread(
    int resolution,
    int dpi,
    int fovX,
    int fovY,
    double kp,
    double ki,
    double kd,
    double process_noise_q,
    double measurement_noise_r,
    bool auto_shoot,
    float bScope_multiplier,
    SerialConnection *serialConnection,
    GhubMouse *gHub) : screen_width(static_cast<double>(resolution * 16) / 9.0),
                       screen_height(static_cast<double>(resolution)),
                       dpi(static_cast<double>(dpi)),
                       fov_x(static_cast<double>(fovX)),
                       fov_y(static_cast<double>(fovY)),
                       center_x(screen_width / 2),
                       center_y(screen_height / 2),
                       auto_shoot(auto_shoot),
                       bScope_multiplier(bScope_multiplier),
                       current_target(nullptr),
                       tracking_errors(false)
{
    // Initialize Kalman filter and PID controller
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp, ki, kd);

    // Initialize InputMethod
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

    last_target_time = std::chrono::steady_clock::now();
    last_prediction_time = last_target_time;
}

// Implementation of new constructor with separated X/Y PID controllers
MouseThread::MouseThread(
    int resolution,
    int dpi,
    int fovX,
    int fovY,
    double kp_x,
    double ki_x,
    double kd_x,
    double kp_y,
    double ki_y,
    double kd_y,
    double process_noise_q,
    double measurement_noise_r,
    bool auto_shoot,
    float bScope_multiplier,
    SerialConnection *serialConnection,
    GhubMouse *gHub) : screen_width(static_cast<double>(resolution * 16) / 9.0),
                       screen_height(static_cast<double>(resolution)),
                       dpi(static_cast<double>(dpi)),
                       fov_x(static_cast<double>(fovX)),
                       fov_y(static_cast<double>(fovY)),
                       center_x(screen_width / 2),
                       center_y(screen_height / 2),
                       auto_shoot(auto_shoot),
                       bScope_multiplier(bScope_multiplier),
                       current_target(nullptr),
                       tracking_errors(false)
{
    // Initialize Kalman filter and separated PID controller
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);

    // Initialize InputMethod
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

    last_target_time = std::chrono::steady_clock::now();
    last_prediction_time = last_target_time;
}

void MouseThread::updateConfig(
    int resolution,
    int dpi,
    int fovX,
    int fovY,
    double kp,
    double ki,
    double kd,
    double process_noise_q,
    double measurement_noise_r,
    bool auto_shoot,
    float bScope_multiplier)
{
    this->screen_width = static_cast<double>(resolution);
    this->screen_height = static_cast<double>(resolution);
    this->dpi = static_cast<double>(dpi);
    this->fov_x = static_cast<double>(fovX);
    this->fov_y = static_cast<double>(fovY);
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = screen_width / 2.0;
    this->center_y = screen_height / 2.0;

    // Update Kalman filter
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);

    // Update legacy PID controller (same gains for X/Y axes)
    pid_controller->updateParameters(kp, ki, kd);
}

// Implementation of updateConfig method using separated X/Y PID gains
void MouseThread::updateConfig(
    int resolution,
    int dpi,
    int fovX,
    int fovY,
    double kp_x,
    double ki_x,
    double kd_x,
    double kp_y,
    double ki_y,
    double kd_y,
    double process_noise_q,
    double measurement_noise_r,
    bool auto_shoot,
    float bScope_multiplier)
{
    this->screen_width = static_cast<double>(resolution);
    this->screen_height = static_cast<double>(resolution);
    this->dpi = static_cast<double>(dpi);
    this->fov_x = static_cast<double>(fovX);
    this->fov_y = static_cast<double>(fovY);
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = screen_width / 2.0;
    this->center_y = screen_height / 2.0;

    // Update Kalman filter
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);

    // Update separated PID controller (different gains for X/Y axes)
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
}

Eigen::Vector2d MouseThread::predictTargetPosition(double target_x, double target_y)
{
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_prediction_time).count();
    dt = std::min(dt, 0.1);

    last_prediction_time = current_time;

    // Always predict first, irrespective of target detection
    kalman_filter->predict(dt);

    // Early return if no target
    if (!target_detected.load())
    {
        const auto &state = kalman_filter->getState();
        return Eigen::Vector2d(state(0, 0), state(1, 0));
    }

    // Removed unnecessary SIMD: direct creation is more efficient when simply setting two values
    Eigen::Vector2d measurement(target_x, target_y);
    kalman_filter->update(measurement);

    const auto &state = kalman_filter->getState();
    
    // Simple variable extraction is more efficient without SIMD with direct assignment
    double pos_x = state(0, 0);
    double pos_y = state(1, 0);
    double vel_x = state(2, 0);
    double vel_y = state(3, 0);
    double acc_x = state(4, 0);
    double acc_y = state(5, 0);

    // Vector size calculation without SIMD
    double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y);
    double acceleration = std::sqrt(acc_x * acc_x + acc_y * acc_y);

    // Use lookup table approach to eliminate branches for prediction_time calculation
    constexpr double base_prediction_factor = 0.07;
    constexpr double prediction_factors[4] = {1.0, 1.5, 2.0, 2.5};
    
    int velocity_idx = std::min(static_cast<int>(velocity / 200.0), 3);
    double prediction_time = dt * base_prediction_factor * prediction_factors[velocity_idx];

    // SIMD 대신 직접 계산
    double half_pred_time_squared = 0.5 * prediction_time * prediction_time;
    double future_x = pos_x + vel_x * prediction_time + acc_x * half_pred_time_squared;
    double future_y = pos_y + vel_y * prediction_time + acc_y * half_pred_time_squared;
    
    // Direction change detection and correction
    static Eigen::Vector2d prev_velocity(0, 0);
    Eigen::Vector2d current_velocity(vel_x, vel_y);
    
    if (prev_velocity.norm() > 0 && velocity > 200.0) {
        double angle_change = std::acos(
            std::clamp(prev_velocity.dot(current_velocity) / (prev_velocity.norm() * velocity), -1.0, 1.0)
        );
        
        // Apply correction only for significant direction changes
        if (angle_change > 0.5) {
            double reduction_factor = std::max(0.3, 1.0 - angle_change / 3.14);
            future_x = pos_x + vel_x * prediction_time * reduction_factor;
            future_y = pos_y + vel_y * prediction_time * reduction_factor;
        }
    }
    
    prev_velocity = current_velocity;
    
    target_detected.store(true);
    return Eigen::Vector2d(future_x, future_y);
}

Eigen::Vector2d MouseThread::calculateMovement(const Eigen::Vector2d &target_pos)
{
    // Pre-compute scaling factors for better cache locality
    static const double fov_scale_x = fov_x / screen_width;
    static const double fov_scale_y = fov_y / screen_height;
    static const double sens_scale = dpi / 360.0; // Removed mouse_sensitivity reference
    
    // Remove unnecessary SIMD and calculate error directly
    double error_x = target_pos[0] - center_x;
    double error_y = target_pos[1] - center_y;
    
    // Convert to Eigen vector
    Eigen::Vector2d error(error_x, error_y);

    // Calculate PID output
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    // Remove unnecessary SIMD and directly scale output
    double result_x = pid_output[0] * fov_scale_x * sens_scale;
    double result_y = pid_output[1] * fov_scale_y * sens_scale;
    
    return Eigen::Vector2d(result_x, result_y);
}

bool MouseThread::checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    // Fast boundary check - first do a quick approximate check
    constexpr double SCOPE_MARGIN = 0.15; // 15% of screen width/height
    
    // Cache the screen boundaries
    static const double screen_margin_x = screen_width * SCOPE_MARGIN;
    static const double screen_margin_y = screen_height * SCOPE_MARGIN;
    
    // Remove unnecessary SIMD and perform simple center point calculation
    double target_center_x = target_x + target_w * 0.5;
    double target_center_y = target_y + target_h * 0.5;
    
    // Absolute difference calculation
    double diff_x = std::abs(target_center_x - center_x);
    double diff_y = std::abs(target_center_y - center_y);
    
    // Fast early rejection (avoid unnecessary calculations)
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    // Calculate reduced target size
    double reduced_half_w = target_w * reduction_factor * 0.5;
    double reduced_half_h = target_h * reduction_factor * 0.5;
    
    // Target range calculation
    double min_x = target_center_x - reduced_half_w;
    double max_x = target_center_x + reduced_half_w;
    double min_y = target_center_y - reduced_half_h;
    double max_y = target_center_y + reduced_half_h;
    
    // Check if screen center is within reduced target range
    return (center_x >= min_x && center_x <= max_x && 
            center_y >= min_y && center_y <= max_y);
}

double MouseThread::calculateTargetDistance(const AimbotTarget &target) const
{
    // SIMD removed: simple 2D distance calculation
    double dx = target.x + target.w * 0.5 - center_x;
    double dy = target.y + target.h * 0.5 - center_y;
    return std::sqrt(dx * dx + dy * dy);
}

AimbotTarget *MouseThread::findClosestTarget(const std::vector<AimbotTarget> &targets) const
{
    if (targets.empty())
    {
        return nullptr;
    }

    AimbotTarget *closest = nullptr;
    double min_distance = std::numeric_limits<double>::max();

    for (const auto &target : targets)
    {
        double distance = calculateTargetDistance(target);
        if (distance < min_distance)
        {
            min_distance = distance;
            closest = const_cast<AimbotTarget *>(&target);
        }
    }

    return closest;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    // Cache frequently used values locally
    const double local_center_x = center_x;
    const double local_center_y = center_y;
    const double local_fov_x = fov_x;
    const double local_fov_y = fov_y;
    const double local_dpi = dpi;

    // Calculate target center point
    double target_center_x = target.x + target.w * 0.5;
    double target_center_y = target.y + target.h * 0.5;

    // Calculate error without SIMD
    double error_x = target_center_x - local_center_x;
    double error_y = target_center_y - local_center_y;

    // Reset prediction for first detection
    if (!target_detected.load())
    {
        resetPrediction();
        Eigen::Vector2d measurement(target_center_x, target_center_y);
        kalman_filter->update(measurement);
    }

    // Predict target position
    Eigen::Vector2d predicted = predictTargetPosition(target_center_x, target_center_y);

    // Calculate adjusted error
    error_x = predicted.x() - local_center_x;
    error_y = predicted.y() - local_center_y;

    // Performance tracking callback
    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y);
        }
    }

    // Input error to PID controller
    Eigen::Vector2d error(error_x, error_y);
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    // Calculate mouse movement - sensitivity removed
    double move_x = pid_output.x() * (local_fov_x / 360.0) * (1000.0 / local_dpi);
    double move_y = pid_output.y() * (local_fov_y / 360.0) * (1000.0 / local_dpi);

    // Apply scope multiplier
    if (bScope_multiplier > 1.0f)
    {
        move_x /= bScope_multiplier;
        move_y /= bScope_multiplier;
    }

    // Round to integers
    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    // Actual mouse movement (only if non-zero)
    if (dx_int != 0 || dy_int != 0)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            input_method->move(dx_int, dy_int);
        }
    }

    last_target_time = std::chrono::steady_clock::now();
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
    target_detected = false;
    last_prediction_time = std::chrono::steady_clock::now();
}

void MouseThread::checkAndResetPredictions()
{
    if (target_detected)
    {
        const auto current_time = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();

        // Reduced target loss detection time from 250ms to 150ms - quicker acquisition of new targets
        if (elapsed > 0.1) // 150ms timeout
        {
            resetPrediction();
            target_detected = false;
        }
    }
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}

void MouseThread::applyRecoilCompensation(float strength)
{
    // Move mouse atomically
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