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

// PID 컨트롤러 구현
PIDController2D::PIDController2D(double kp, double ki, double kd)
    : kp(kp), ki(ki), kd(kd), kp_x(kp), ki_x(ki), kd_x(kd), kp_y(kp), ki_y(ki), kd_y(kd)
{
    reset();
}

// X/Y 분리 게인을 사용하는 새 생성자 구현
PIDController2D::PIDController2D(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y)
    : kp_x(kp_x), ki_x(ki_x), kd_x(kd_x), kp_y(kp_y), ki_y(ki_y), kd_y(kd_y), 
      kp((kp_x + kp_y) / 2), ki((ki_x + ki_y) / 2), kd((kd_x + kd_y) / 2) // 평균값을 공통 게인으로 설정
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

        // 불필요한 SIMD 제거하고 직접 계산으로 대체
        // X-축 게인 계산 요소
        double kp_factor_x = 1.0 + std::min(error_magnitude_x / 100.0, 0.6);
        double ki_factor_x = 1.0 - std::min(error_magnitude_x / 400.0, 0.8);
        
        // Y-축 게인 계산 요소
        double kp_factor_y = 1.0 + std::min(error_magnitude_y / 120.0, 0.5);
        double ki_factor_y = 1.0 - std::min(error_magnitude_y / 350.0, 0.9);
        
        // 목표 게인 계산
        double target_kp_x = kp_x * kp_factor_x;
        double target_ki_x = ki_x * ki_factor_x;
        double target_kd_x = kd_x * (1.0 + std::min(error_magnitude_x / 200.0, 0.4));
        
        double target_kp_y = kp_y * kp_factor_y;
        double target_ki_y = ki_y * ki_factor_y;
        double target_kd_y = kd_y * (1.0 + std::min(error_magnitude_y / 180.0, 0.5));

        // Fast interpolation to avoid sudden changes
        double alpha = std::min(time_since_update * 15.0, 1.0);
        double one_minus_alpha = 1.0 - alpha;
        
        // 불필요한 SIMD 제거하고 직접 보간 계산
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
        
        // 불필요한 SIMD 제거하고 직접 미분 계산
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

    // 불필요한 SIMD 제거하고 직접 PID 출력 계산
    double p_term_x = cached_kp_x * error.x();
    double p_term_y = cached_kp_y * error.y();
    
    double i_term_x = cached_ki_x * integral.x();
    double i_term_y = cached_ki_y * integral.y();
    
    double d_term_x = cached_kd_x * derivative.x();
    double d_term_y = cached_kd_y * derivative.y();
    
    // 항 합산
    double output_x = p_term_x + i_term_x + d_term_x;
    double output_y = p_term_y + i_term_y + d_term_y;
    
    // 출력 제한 (X와 Y에 다른 한계 적용)
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
    prev_error = Eigen::Vector2d::Zero();               // 이전 오차 초기화
    integral = Eigen::Vector2d::Zero();                 // 적분항 초기화
    derivative = Eigen::Vector2d::Zero();               // 미분항 초기화
    prev_derivative = Eigen::Vector2d::Zero();          // 이전 미분항 초기화
    last_time_point = std::chrono::steady_clock::now(); // 시간 초기화
}

void PIDController2D::updateParameters(double kp, double ki, double kd)
{
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
    
    // 공통 게인도 X/Y 게인으로 설정 (하위 호환성)
    this->kp_x = kp;
    this->ki_x = ki;
    this->kd_x = kd;
    this->kp_y = kp;
    this->ki_y = ki;
    this->kd_y = kd;
}

// X/Y 분리 게인 업데이트 함수 구현
void PIDController2D::updateSeparatedParameters(double kp_x, double ki_x, double kd_x, 
                                               double kp_y, double ki_y, double kd_y)
{
    this->kp_x = kp_x;
    this->ki_x = ki_x;
    this->kd_x = kd_x;
    this->kp_y = kp_y;
    this->ki_y = ki_y;
    this->kd_y = kd_y;
    
    // 공통 게인은 X/Y의 평균으로 설정 (기존 코드와의 호환성 위해)
    this->kp = (kp_x + kp_y) / 2;
    this->ki = (ki_x + ki_y) / 2;
    this->kd = (kd_x + kd_y) / 2;
}

// 칼만 필터 구현
KalmanFilter2D::KalmanFilter2D(double process_noise_q, double measurement_noise_r)
{
    // 상태 전이 행렬 초기화
    A = Eigen::Matrix<double, 6, 6>::Identity();

    // 측정 행렬 초기화 (위치만 측정)
    H = Eigen::Matrix<double, 2, 6>::Zero();
    H(0, 0) = 1.0; // x 위치
    H(1, 1) = 1.0; // y 위치

    // 노이즈 매트릭스 초기화 - 위치, 속도, 가속도에 다른 노이즈 값 적용
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
    
    // 급격한 움직임에 더 민감하게 반응하도록 속도와 가속도 노이즈 증가
    Q(2, 2) = process_noise_q * 2.5; // vx에 대한 프로세스 노이즈 증가
    Q(3, 3) = process_noise_q * 2.5; // vy에 대한 프로세스 노이즈 증가
    Q(4, 4) = process_noise_q * 4.0; // ax에 대한 프로세스 노이즈 증가
    Q(5, 5) = process_noise_q * 4.0; // ay에 대한 프로세스 노이즈 증가
    
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
    P = Eigen::Matrix<double, 6, 6>::Identity();

    x = Eigen::Matrix<double, 6, 1>::Zero();
}

void KalmanFilter2D::predict(double dt)
{
    // dt에 따른 상태 전이 행렬 업데이트
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
    // 기본 노이즈 업데이트
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
    
    // 급격한 움직임에 더 민감하게 반응하도록 속도와 가속도 노이즈 증가
    Q(2, 2) = process_noise_q * 2.5; // vx에 대한 프로세스 노이즈 증가
    Q(3, 3) = process_noise_q * 2.5; // vy에 대한 프로세스 노이즈 증가
    Q(4, 4) = process_noise_q * 4.0; // ax에 대한 프로세스 노이즈 증가
    Q(5, 5) = process_noise_q * 4.0; // ay에 대한 프로세스 노이즈 증가
    
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
}

// MouseThread Implementation
MouseThread::MouseThread(
    int resolution,
    int dpi,
    double sensitivity,
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
                       mouse_sensitivity(sensitivity),
                       fov_x(static_cast<double>(fovX)),
                       fov_y(static_cast<double>(fovY)),
                       center_x(screen_width / 2),
                       center_y(screen_height / 2),
                       auto_shoot(auto_shoot),
                       bScope_multiplier(bScope_multiplier),
                       current_target(nullptr),
                       tracking_errors(false)
{
    // 칼만 필터와 PID 컨트롤러 초기화
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp, ki, kd);

    // InputMethod 초기화
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

// X/Y 분리 PID 컨트롤러를 지원하는 새 생성자 구현
MouseThread::MouseThread(
    int resolution,
    int dpi,
    double sensitivity,
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
                       mouse_sensitivity(sensitivity),
                       fov_x(static_cast<double>(fovX)),
                       fov_y(static_cast<double>(fovY)),
                       center_x(screen_width / 2),
                       center_y(screen_height / 2),
                       auto_shoot(auto_shoot),
                       bScope_multiplier(bScope_multiplier),
                       current_target(nullptr),
                       tracking_errors(false)
{
    // 칼만 필터와 분리된 PID 컨트롤러 초기화
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);

    // InputMethod 초기화
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
    double sensitivity,
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
    this->mouse_sensitivity = sensitivity;
    this->fov_x = static_cast<double>(fovX);
    this->fov_y = static_cast<double>(fovY);
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = screen_width / 2.0;
    this->center_y = screen_height / 2.0;

    // 칼만 필터 업데이트
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);

    // 레거시 PID 컨트롤러 업데이트 (X/Y 축 동일 게인)
    pid_controller->updateParameters(kp, ki, kd);
}

// 분리된 X/Y PID 게인을 사용하는 새 updateConfig 메서드 구현
void MouseThread::updateConfig(
    int resolution,
    int dpi,
    double sensitivity,
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
    this->mouse_sensitivity = sensitivity;
    this->fov_x = static_cast<double>(fovX);
    this->fov_y = static_cast<double>(fovY);
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = screen_width / 2.0;
    this->center_y = screen_height / 2.0;

    // 칼만 필터 업데이트
    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);

    // 분리된 PID 컨트롤러 업데이트 (X/Y 축 별도 게인)
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

    // 불필요한 SIMD 제거: 단순히 두 값만 설정하는 경우 직접 생성이 더 효율적
    Eigen::Vector2d measurement(target_x, target_y);
    kalman_filter->update(measurement);

    const auto &state = kalman_filter->getState();
    
    // 간단한 변수 추출은 SIMD 없이 직접 할당하는 것이 더 효율적
    double pos_x = state(0, 0);
    double pos_y = state(1, 0);
    double vel_x = state(2, 0);
    double vel_y = state(3, 0);
    double acc_x = state(4, 0);
    double acc_y = state(5, 0);

    // SIMD 없이 벡터 크기 계산
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
    static const double sens_scale = dpi * (1.0 / mouse_sensitivity) / 360.0;
    
    // 불필요한 SIMD 제거하고 직접 오차 계산
    double error_x = target_pos[0] - center_x;
    double error_y = target_pos[1] - center_y;
    
    // Eigen 벡터로 변환
    Eigen::Vector2d error(error_x, error_y);

    // Calculate PID output
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    // 불필요한 SIMD 제거하고 직접 출력 스케일링
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
    
    // 불필요한 SIMD 제거: 간단한 중심점 계산
    double target_center_x = target_x + target_w * 0.5;
    double target_center_y = target_y + target_h * 0.5;
    
    // 절대 차이 계산
    double diff_x = std::abs(target_center_x - center_x);
    double diff_y = std::abs(target_center_y - center_y);
    
    // Fast early rejection (avoid unnecessary calculations)
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    // 축소된 타겟 사이즈 계산
    double reduced_half_w = target_w * reduction_factor * 0.5;
    double reduced_half_h = target_h * reduction_factor * 0.5;
    
    // 타겟 범위 계산
    double min_x = target_center_x - reduced_half_w;
    double max_x = target_center_x + reduced_half_w;
    double min_y = target_center_y - reduced_half_h;
    double max_y = target_center_y + reduced_half_h;
    
    // 스크린 중심점이 축소된 타겟 범위 내에 있는지 체크
    return (center_x >= min_x && center_x <= max_x && 
            center_y >= min_y && center_y <= max_y);
}

double MouseThread::calculateTargetDistance(const AimbotTarget &target) const
{
    // SIMD 제거: 간단한 2D 거리 계산
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
    // 자주 사용되는 값들을 로컬에 캐시
    const double local_center_x = center_x;
    const double local_center_y = center_y;
    const double local_fov_x = fov_x;
    const double local_fov_y = fov_y;
    const double local_sensitivity = mouse_sensitivity;
    const double local_dpi = dpi;

    // 타겟 중심점 계산
    double target_center_x = target.x + target.w * 0.5;
    double target_center_y = target.y + target.h * 0.5;

    // SIMD 없이 오차 계산
    double error_x = target_center_x - local_center_x;
    double error_y = target_center_y - local_center_y;

    // 첫 번째 탐지인 경우 예측 초기화
    if (!target_detected.load())
    {
        resetPrediction();
        Eigen::Vector2d measurement(target_center_x, target_center_y);
        kalman_filter->update(measurement);
    }

    // 대상 위치 예측
    Eigen::Vector2d predicted = predictTargetPosition(target_center_x, target_center_y);

    // 수정된 오차 계산
    error_x = predicted.x() - local_center_x;
    error_y = predicted.y() - local_center_y;

    // 성능 측정 콜백
    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y);
        }
    }

    // PID 컨트롤러에 오차 입력
    Eigen::Vector2d error(error_x, error_y);
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    // 마우스 이동 계산
    double move_x = pid_output.x() * (local_fov_x / 360.0) * (1000.0 / (local_sensitivity * local_dpi));
    double move_y = pid_output.y() * (local_fov_y / 360.0) * (1000.0 / (local_sensitivity * local_dpi));

    // 스코프 배율 적용
    if (bScope_multiplier > 1.0f)
    {
        move_x /= bScope_multiplier;
        move_y /= bScope_multiplier;
    }

    // 정수로 반올림
    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    // 실제 마우스 이동 (0이 아닌 경우에만)
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

        // 타겟 손실 감지 시간을 250ms에서 150ms로 줄임 - 더 빠른 새 타겟 획득
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