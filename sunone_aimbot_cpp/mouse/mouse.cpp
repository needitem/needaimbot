#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "config.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

// Constants for PID controller
constexpr double MAX_OUTPUT_X = 1500.0;
constexpr double MAX_OUTPUT_Y = 1200.0;
constexpr double MAX_INTEGRAL_X = 80.0;
constexpr double MAX_INTEGRAL_Y = 60.0;
constexpr double ERROR_THRESHOLD_X = 100.0;
constexpr double ERROR_THRESHOLD_Y = 80.0;
constexpr double SCOPE_MARGIN = 0.15;

// Constants for Kalman filter
constexpr double VEL_NOISE_FACTOR = 2.5;
constexpr double ACC_NOISE_FACTOR = 4.0;
constexpr double BASE_PREDICTION_FACTOR = 0.07;

PIDController2D::PIDController2D(double kp, double ki, double kd)
    : kp(kp), ki(ki), kd(kd), kp_x(kp), ki_x(ki), kd_x(kd), kp_y(kp), ki_y(ki), kd_y(kd)
{
    reset();
}

PIDController2D::PIDController2D(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y), 
      kp((kp_x + kp_y) / 2), ki((ki_x + ki_y) / 2), kd((kd_x + kd_y) / 2)
{
    reset();
}

Eigen::Vector2d PIDController2D::calculate(const Eigen::Vector2d &error)
{
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time_point).count();
    dt = dt > 0.1 ? 0.1 : dt;
    last_time_point = now;

    static auto last_gain_update = now;
    static double cached_kp_x = kp_x;
    static double cached_ki_x = ki_x;
    static double cached_kd_x = kd_x;
    static double cached_kp_y = kp_y;
    static double cached_ki_y = ki_y;
    static double cached_kd_y = kd_y;

    double time_since_update = std::chrono::duration<double>(now - last_gain_update).count();
    if (time_since_update >= 0.05)
    {
        last_gain_update = now;
        
        double error_magnitude_x = std::abs(error.x());
        double error_magnitude_y = std::abs(error.y());

        double kp_factor_x = 1.0 + std::min(error_magnitude_x * 0.01, 0.6);
        double ki_factor_x = 1.0 - std::min(error_magnitude_x * 0.0025, 0.8);
        
        double kp_factor_y = 1.0 + std::min(error_magnitude_y * 0.00833, 0.5);
        double ki_factor_y = 1.0 - std::min(error_magnitude_y * 0.00286, 0.9);
        
        double target_kp_x = kp_x * kp_factor_x;
        double target_ki_x = ki_x * ki_factor_x;
        double target_kd_x = kd_x * (1.0 + std::min(error_magnitude_x * 0.002, 0.4));
        
        double target_kp_y = kp_y * kp_factor_y;
        double target_ki_y = ki_y * ki_factor_y;
        double target_kd_y = kd_y * (1.0 + std::min(error_magnitude_y * 0.00278, 0.5));

        double alpha = std::min(time_since_update * 15.0, 1.0);
        if (alpha > 0.95) {
            cached_kp_x = target_kp_x;
            cached_ki_x = target_ki_x;
            cached_kd_x = target_kd_x;
            cached_kp_y = target_kp_y;
            cached_ki_y = target_ki_y;
            cached_kd_y = target_kd_y;
        } else {
            double one_minus_alpha = 1.0 - alpha;
            cached_kp_x = cached_kp_x * one_minus_alpha + target_kp_x * alpha;
            cached_ki_x = cached_ki_x * one_minus_alpha + target_ki_x * alpha;
            cached_kd_x = cached_kd_x * one_minus_alpha + target_kd_x * alpha;
            cached_kp_y = cached_kp_y * one_minus_alpha + target_kp_y * alpha;
            cached_ki_y = cached_ki_y * one_minus_alpha + target_ki_y * alpha;
            cached_kd_y = cached_kd_y * one_minus_alpha + target_kd_y * alpha;
        }
    }

    if (dt > 0.0001)
    {
        static const double inv_error_threshold_x = 1.0 / ERROR_THRESHOLD_X;
        static const double inv_error_threshold_y = 1.0 / ERROR_THRESHOLD_Y;
        
        double abs_error_x = std::abs(error.x());
        double abs_error_y = std::abs(error.y());
        
        double integral_factor_x = abs_error_x > ERROR_THRESHOLD_X ? 
                                  ERROR_THRESHOLD_X / abs_error_x : 1.0;
        double integral_factor_y = abs_error_y > ERROR_THRESHOLD_Y ? 
                                  ERROR_THRESHOLD_Y / abs_error_y : 1.0;
        
        integral.x() += error.x() * dt * integral_factor_x;
        integral.y() += error.y() * dt * integral_factor_y;
        
        if (integral.x() > MAX_INTEGRAL_X) integral.x() = MAX_INTEGRAL_X;
        else if (integral.x() < -MAX_INTEGRAL_X) integral.x() = -MAX_INTEGRAL_X;
        
        if (integral.y() > MAX_INTEGRAL_Y) integral.y() = MAX_INTEGRAL_Y;
        else if (integral.y() < -MAX_INTEGRAL_Y) integral.y() = -MAX_INTEGRAL_Y;
        
        double derivative_x = (error.x() - prev_error.x()) / dt;
        double derivative_y = (error.y() - prev_error.y()) / dt;
        
        double alpha_x = (std::abs(derivative_x) > 500.0) ? 0.7 : 0.85;
        double alpha_y = (std::abs(derivative_y) > 400.0) ? 0.6 : 0.9;
        
        derivative.x() = derivative_x * alpha_x + prev_derivative.x() * (1.0 - alpha_x);
        derivative.y() = derivative_y * alpha_y + prev_derivative.y() * (1.0 - alpha_y);
        
        prev_derivative = derivative;
    }
    else
    {
        derivative.setZero();
    }

    Eigen::Vector2d output;
    output.x() = cached_kp_x * error.x() + cached_ki_x * integral.x() + cached_kd_x * derivative.x();
    output.y() = cached_kp_y * error.y() + cached_ki_y * integral.y() + cached_kd_y * derivative.y();
    
    if (output.x() > MAX_OUTPUT_X) output.x() = MAX_OUTPUT_X;
    else if (output.x() < -MAX_OUTPUT_X) output.x() = -MAX_OUTPUT_X;
    
    if (output.y() > MAX_OUTPUT_Y) output.y() = MAX_OUTPUT_Y;
    else if (output.y() < -MAX_OUTPUT_Y) output.y() = -MAX_OUTPUT_Y;

    prev_error = error;
    return output;
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2d::Zero();
    integral = Eigen::Vector2d::Zero();
    derivative = Eigen::Vector2d::Zero();
    prev_derivative = Eigen::Vector2d::Zero();
    last_time_point = std::chrono::steady_clock::now();
}

void PIDController2D::updateParameters(double kp, double ki, double kd)
{
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
    
    this->kp_x = kp;
    this->ki_x = ki;
    this->kd_x = kd;
    this->kp_y = kp;
    this->ki_y = ki;
    this->kd_y = kd;
}

void PIDController2D::updateSeparatedParameters(double kp_x, double ki_x, double kd_x, 
                                               double kp_y, double ki_y, double kd_y)
{
    this->kp_x = kp_x;
    this->ki_x = ki_x;
    this->kd_x = kd_x;
    this->kp_y = kp_y;
    this->ki_y = ki_y;
    this->kd_y = kd_y;
    
    this->kp = (kp_x + kp_y) / 2;
    this->ki = (ki_x + ki_y) / 2;
    this->kd = (kd_x + kd_y) / 2;
}

void KalmanFilter2D::initializeMatrices(double process_noise_q, double measurement_noise_r)
{
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
    
    Q(2, 2) = process_noise_q * VEL_NOISE_FACTOR;
    Q(3, 3) = process_noise_q * VEL_NOISE_FACTOR;
    Q(4, 4) = process_noise_q * ACC_NOISE_FACTOR;
    Q(5, 5) = process_noise_q * ACC_NOISE_FACTOR;
    
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
}

KalmanFilter2D::KalmanFilter2D(double process_noise_q, double measurement_noise_r)
{
    A = Eigen::Matrix<double, 6, 6>::Identity();

    H = Eigen::Matrix<double, 2, 6>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;

    initializeMatrices(process_noise_q, measurement_noise_r);
    P = Eigen::Matrix<double, 6, 6>::Identity();
    x = Eigen::Matrix<double, 6, 1>::Zero();
}

void KalmanFilter2D::predict(double dt)
{
    A(0, 2) = dt;
    A(0, 4) = 0.5 * dt * dt;
    A(1, 3) = dt;
    A(1, 5) = 0.5 * dt * dt;
    A(2, 4) = dt;
    A(3, 5) = dt;

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
    initializeMatrices(process_noise_q, measurement_noise_r);
}

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
    GhubMouse *gHub) : tracking_errors(false)
{
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier);
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp, ki, kd);
    initializeInputMethod(serialConnection, gHub);
    last_prediction_time = std::chrono::steady_clock::now();
}

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
    GhubMouse *gHub) : tracking_errors(false)
{
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier);
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    initializeInputMethod(serialConnection, gHub);
    last_prediction_time = std::chrono::steady_clock::now();
}

void MouseThread::initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub)
{
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
}

void MouseThread::initializeScreen(int resolution, int dpi, int fovX, int fovY, bool auto_shoot, float bScope_multiplier)
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
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier);
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);
    pid_controller->updateParameters(kp, ki, kd);
}

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
    initializeScreen(resolution, dpi, fovX, fovY, auto_shoot, bScope_multiplier);
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
}

Eigen::Vector2d MouseThread::predictTargetPosition(double target_x, double target_y)
{
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::min(std::chrono::duration<double>(current_time - last_prediction_time).count(), 0.1);
    last_prediction_time = current_time;

    kalman_filter->predict(dt);

    Eigen::Vector2d measurement(target_x, target_y);
    kalman_filter->update(measurement);

    const auto &state = kalman_filter->getState();
    
    double pos_x = state(0, 0);
    double pos_y = state(1, 0);
    double vel_x = state(2, 0);
    double vel_y = state(3, 0);
    double acc_x = state(4, 0);
    double acc_y = state(5, 0);

    double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y);

    constexpr double prediction_factors[4] = {1.0, 1.5, 2.0, 2.5};
    
    int velocity_idx = std::min(static_cast<int>(velocity / 200.0), 3);
    double prediction_time = dt * BASE_PREDICTION_FACTOR * prediction_factors[velocity_idx];

    double half_pred_time_squared = 0.5 * prediction_time * prediction_time;
    double future_x = pos_x + vel_x * prediction_time + acc_x * half_pred_time_squared;
    double future_y = pos_y + vel_y * prediction_time + acc_y * half_pred_time_squared;
    
    static Eigen::Vector2d prev_velocity(0, 0);
    Eigen::Vector2d current_velocity(vel_x, vel_y);
    
    if (prev_velocity.norm() > 0 && velocity > 200.0) {
        double angle_change = std::acos(
            std::clamp(prev_velocity.dot(current_velocity) / (prev_velocity.norm() * velocity), -1.0, 1.0)
        );
        
        if (angle_change > 0.5) {
            double reduction_factor = std::max(0.3, 1.0 - angle_change / 3.14);
            future_x = pos_x + vel_x * prediction_time * reduction_factor;
            future_y = pos_y + vel_y * prediction_time * reduction_factor;
        }
    }
    
    prev_velocity = current_velocity;
    
    return Eigen::Vector2d(future_x, future_y);
}

Eigen::Vector2d MouseThread::calculateMovement(const Eigen::Vector2d &target_pos)
{
    static const double fov_scale_x = fov_x / screen_width;
    static const double fov_scale_y = fov_y / screen_height;
    static const double sens_scale = dpi / 360.0;
    
    double error_x = target_pos[0] - center_x;
    double error_y = target_pos[1] - center_y;
    
    Eigen::Vector2d error(error_x, error_y);
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    double result_x = pid_output[0] * fov_scale_x * sens_scale;
    double result_y = pid_output[1] * fov_scale_y * sens_scale;
    
    return Eigen::Vector2d(result_x, result_y);
}

bool MouseThread::checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    static const double screen_margin_x = screen_width * SCOPE_MARGIN;
    static const double screen_margin_y = screen_height * SCOPE_MARGIN;
    
    double target_center_x = target_x + target_w * 0.5;
    double target_center_y = target_y + target_h * 0.5;
    
    double diff_x = std::abs(target_center_x - center_x);
    double diff_y = std::abs(target_center_y - center_y);
    
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    double reduced_half_w = target_w * reduction_factor * 0.5;
    double reduced_half_h = target_h * reduction_factor * 0.5;
    
    double min_x = target_center_x - reduced_half_w;
    double max_x = target_center_x + reduced_half_w;
    double min_y = target_center_y - reduced_half_h;
    double max_y = target_center_y + reduced_half_h;
    
    return (center_x >= min_x && center_x <= max_x && 
            center_y >= min_y && center_y <= max_y);
}

double MouseThread::calculateTargetDistance(const AimbotTarget &target) const
{
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
    const double local_center_x = center_x;
    const double local_center_y = center_y;
    const double local_fov_x = fov_x;
    const double local_fov_y = fov_y;
    const double local_dpi = dpi;

    double target_center_x = target.x + target.w * 0.5;
    double target_center_y = target.y + target.h * 0.5;

    resetPrediction();
    
    Eigen::Vector2d predicted = predictTargetPosition(target_center_x, target_center_y);
    double error_x = predicted.x() - local_center_x;
    double error_y = predicted.y() - local_center_y;

    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y);
        }
    }

    Eigen::Vector2d error(error_x, error_y);
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    double move_x = pid_output.x() * (local_fov_x / 360.0) * (1000.0 / local_dpi);
    double move_y = pid_output.y() * (local_fov_y / 360.0) * (1000.0 / local_dpi);

    if (bScope_multiplier > 1.0f)
    {
        move_x /= bScope_multiplier;
        move_y /= bScope_multiplier;
    }

    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    if (dx_int != 0 || dy_int != 0)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            input_method->move(dx_int, dy_int);
        }
    }
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
    last_prediction_time = std::chrono::steady_clock::now();
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}

void MouseThread::applyRecoilCompensation(float strength)
{
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