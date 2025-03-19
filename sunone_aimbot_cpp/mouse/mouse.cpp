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

        // SIMD for gain adjustments
        __m128d error_mag = _mm_set_pd(error_magnitude_y, error_magnitude_x);
        
        // X-axis gain calculation factors
        __m128d x_factors = _mm_set_pd(
            1.0 - std::min(error_magnitude_x / 400.0, 0.8), // ki factor
            1.0 + std::min(error_magnitude_x / 100.0, 0.6)  // kp factor
        );
        
        // Y-axis gain calculation factors
        __m128d y_factors = _mm_set_pd(
            1.0 - std::min(error_magnitude_y / 350.0, 0.9), // ki factor
            1.0 + std::min(error_magnitude_y / 120.0, 0.5)  // kp factor
        );
        
        // Calculate target gains
        double target_kp_x = kp_x * _mm_cvtsd_f64(x_factors);
        double target_ki_x = ki_x * _mm_cvtsd_f64(_mm_unpackhi_pd(x_factors, x_factors));
        double target_kd_x = kd_x * (1.0 + std::min(error_magnitude_x / 200.0, 0.4));
        
        double target_kp_y = kp_y * _mm_cvtsd_f64(y_factors);
        double target_ki_y = ki_y * _mm_cvtsd_f64(_mm_unpackhi_pd(y_factors, y_factors));
        double target_kd_y = kd_y * (1.0 + std::min(error_magnitude_y / 180.0, 0.5));

        // Fast interpolation to avoid sudden changes
        double alpha = std::min(time_since_update * 15.0, 1.0);
        double one_minus_alpha = 1.0 - alpha;
        
        // SIMD for interpolation
        __m128d current_x_gains = _mm_set_pd(cached_ki_x, cached_kp_x);
        __m128d target_x_gains = _mm_set_pd(target_ki_x, target_kp_x);
        __m128d alpha_vec = _mm_set1_pd(alpha);
        __m128d one_minus_alpha_vec = _mm_set1_pd(one_minus_alpha);
        
        __m128d new_x_gains = _mm_add_pd(
            _mm_mul_pd(current_x_gains, one_minus_alpha_vec),
            _mm_mul_pd(target_x_gains, alpha_vec)
        );
        
        cached_kp_x = _mm_cvtsd_f64(new_x_gains);
        cached_ki_x = _mm_cvtsd_f64(_mm_unpackhi_pd(new_x_gains, new_x_gains));
        
        // Same for Y gains
        __m128d current_y_gains = _mm_set_pd(cached_ki_y, cached_kp_y);
        __m128d target_y_gains = _mm_set_pd(target_ki_y, target_kp_y);
        
        __m128d new_y_gains = _mm_add_pd(
            _mm_mul_pd(current_y_gains, one_minus_alpha_vec),
            _mm_mul_pd(target_y_gains, alpha_vec)
        );
        
        cached_kp_y = _mm_cvtsd_f64(new_y_gains);
        cached_ki_y = _mm_cvtsd_f64(_mm_unpackhi_pd(new_y_gains, new_y_gains));
        
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
        
        // Calculate derivative
        __m128d error_vec = _mm_set_pd(error.y(), error.x());
        __m128d prev_error_vec = _mm_set_pd(prev_error.y(), prev_error.x());
        __m128d dt_vec = _mm_set1_pd(1.0 / dt);
        
        __m128d derivative_vec = _mm_mul_pd(_mm_sub_pd(error_vec, prev_error_vec), dt_vec);
        
        // Extract derivative components
        double derivative_x = _mm_cvtsd_f64(derivative_vec);
        double derivative_y = _mm_cvtsd_f64(_mm_unpackhi_pd(derivative_vec, derivative_vec));
        
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

    // Calculate PID output using SIMD
    __m128d error_vec = _mm_set_pd(error.y(), error.x());
    __m128d integral_vec = _mm_set_pd(integral.y(), integral.x());
    __m128d derivative_vec = _mm_set_pd(derivative.y(), derivative.x());
    
    __m128d kp_vec = _mm_set_pd(cached_kp_y, cached_kp_x);
    __m128d ki_vec = _mm_set_pd(cached_ki_y, cached_ki_x);
    __m128d kd_vec = _mm_set_pd(cached_kd_y, cached_kd_x);
    
    __m128d p_term = _mm_mul_pd(kp_vec, error_vec);
    __m128d i_term = _mm_mul_pd(ki_vec, integral_vec);
    __m128d d_term = _mm_mul_pd(kd_vec, derivative_vec);
    
    // Sum the terms
    __m128d output_vec = _mm_add_pd(_mm_add_pd(p_term, i_term), d_term);
    
    // Clamp the output using different limits for X and Y
    double output_x = _mm_cvtsd_f64(output_vec);
    double output_y = _mm_cvtsd_f64(_mm_unpackhi_pd(output_vec, output_vec));
    
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

    // Use SIMD to create measurement vector
    __m128d measurement_vec = _mm_set_pd(target_y, target_x);
    Eigen::Vector2d measurement(_mm_cvtsd_f64(measurement_vec), 
                               _mm_cvtsd_f64(_mm_unpackhi_pd(measurement_vec, measurement_vec)));
    
    kalman_filter->update(measurement);

    const auto &state = kalman_filter->getState();
    
    // Use SIMD to gather state variables
    __m128d pos = _mm_set_pd(state(1, 0), state(0, 0));
    __m128d vel = _mm_set_pd(state(3, 0), state(2, 0));
    __m128d acc = _mm_set_pd(state(5, 0), state(4, 0));
    
    // Extract position components
    double pos_x = _mm_cvtsd_f64(pos);
    double pos_y = _mm_cvtsd_f64(_mm_unpackhi_pd(pos, pos));
    
    // Extract velocity components
    double vel_x = _mm_cvtsd_f64(vel);
    double vel_y = _mm_cvtsd_f64(_mm_unpackhi_pd(vel, vel));
    
    // Extract acceleration components
    double acc_x = _mm_cvtsd_f64(acc);
    double acc_y = _mm_cvtsd_f64(_mm_unpackhi_pd(acc, acc));

    // Compute velocity magnitude using SIMD
    __m128d vel_squared = _mm_mul_pd(vel, vel);
    __m128d sum_vel = _mm_hadd_pd(vel_squared, vel_squared);
    double velocity = _mm_cvtsd_f64(_mm_sqrt_pd(sum_vel));

    // Compute acceleration magnitude using SIMD
    __m128d acc_squared = _mm_mul_pd(acc, acc);
    __m128d sum_acc = _mm_hadd_pd(acc_squared, acc_squared);
    double acceleration = _mm_cvtsd_f64(_mm_sqrt_pd(sum_acc));

    // Use lookup table approach to eliminate branches for prediction_time calculation
    constexpr double base_prediction_factor = 0.07;
    constexpr double prediction_factors[4] = {1.0, 1.5, 2.0, 2.5};
    
    int velocity_idx = std::min(static_cast<int>(velocity / 200.0), 3);
    double prediction_time = dt * base_prediction_factor * prediction_factors[velocity_idx];

    // SIMD for position prediction calculation
    __m128d pred_time = _mm_set1_pd(prediction_time);
    __m128d pred_time_squared = _mm_mul_pd(pred_time, pred_time);
    __m128d half = _mm_set1_pd(0.5);
    
    // Calculate: pos + vel*t + 0.5*acc*t^2
    __m128d term1 = _mm_mul_pd(vel, pred_time);
    __m128d term2 = _mm_mul_pd(_mm_mul_pd(acc, pred_time_squared), half);
    __m128d future_pos = _mm_add_pd(_mm_add_pd(pos, term1), term2);
    
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
            __m128d reduction = _mm_set1_pd(reduction_factor);
            __m128d corrected_term1 = _mm_mul_pd(vel, _mm_mul_pd(pred_time, reduction));
            future_pos = _mm_add_pd(pos, corrected_term1);
        }
    }
    
    prev_velocity = current_velocity;
    
    // Extract final position
    double future_x = _mm_cvtsd_f64(future_pos);
    double future_y = _mm_cvtsd_f64(_mm_unpackhi_pd(future_pos, future_pos));

    target_detected.store(true);
    return Eigen::Vector2d(future_x, future_y);
}

Eigen::Vector2d MouseThread::calculateMovement(const Eigen::Vector2d &target_pos)
{
    // Pre-compute scaling factors for better cache locality
    static const double fov_scale_x = fov_x / screen_width;
    static const double fov_scale_y = fov_y / screen_height;
    static const double sens_scale = dpi * (1.0 / mouse_sensitivity) / 360.0;
    
    // Use SIMD to calculate error
    __m128d target = _mm_set_pd(target_pos[1], target_pos[0]);
    __m128d center = _mm_set_pd(center_y, center_x);
    __m128d error_simd = _mm_sub_pd(target, center);
    
    // Convert to Eigen vector for PID controller
    Eigen::Vector2d error(_mm_cvtsd_f64(error_simd), 
                          _mm_cvtsd_f64(_mm_unpackhi_pd(error_simd, error_simd)));

    // Calculate PID output
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    // Use SIMD for output scaling
    __m128d pid_vec = _mm_set_pd(pid_output[1], pid_output[0]);
    __m128d scale = _mm_set_pd(fov_scale_y * sens_scale, fov_scale_x * sens_scale);
    __m128d result = _mm_mul_pd(pid_vec, scale);
    
    // Convert back to Eigen vector
    return Eigen::Vector2d(_mm_cvtsd_f64(result), 
                          _mm_cvtsd_f64(_mm_unpackhi_pd(result, result)));
}

bool MouseThread::checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    // Fast boundary check using SIMD - first do a quick approximate check
    constexpr double SCOPE_MARGIN = 0.15; // 25% of screen width/height
    
    // Cache the screen boundaries
    static const double screen_margin_x = screen_width * SCOPE_MARGIN;
    static const double screen_margin_y = screen_height * SCOPE_MARGIN;
    
    // Calculate target center using SIMD
    __m128d target_pos = _mm_set_pd(target_y, target_x);
    __m128d target_size = _mm_set_pd(target_h, target_w);
    __m128d half = _mm_set1_pd(0.5);
    __m128d target_center = _mm_add_pd(target_pos, _mm_mul_pd(target_size, half));
    
    // Calculate absolute difference from screen center
    __m128d screen_center = _mm_set_pd(center_y, center_x);
    __m128d diff = _mm_sub_pd(target_center, screen_center);
    __m128d abs_diff = _mm_andnot_pd(_mm_set1_pd(-0.0), diff); // Fast absolute value
    
    // Extract x and y differences
    double diff_x = _mm_cvtsd_f64(abs_diff);
    double diff_y = _mm_cvtsd_f64(_mm_unpackhi_pd(abs_diff, abs_diff));
    
    // Fast early rejection (avoid unnecessary calculations)
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    // Calculate reduced target size
    __m128d reduction = _mm_set1_pd(reduction_factor * 0.5);
    __m128d reduced_size = _mm_mul_pd(target_size, reduction);
    
    // Calculate target bounds
    __m128d min_bound = _mm_sub_pd(target_center, reduced_size);
    __m128d max_bound = _mm_add_pd(target_center, reduced_size);
    
    // Check if screen center is within reduced target bounds
    __m128d compare_min = _mm_cmpge_pd(screen_center, min_bound);
    __m128d compare_max = _mm_cmple_pd(screen_center, max_bound);
    __m128d result = _mm_and_pd(compare_min, compare_max);
    
    // Both conditions must be true for target to be in scope
    return _mm_movemask_pd(result) == 0x3;
}

double MouseThread::calculateTargetDistance(const AimbotTarget &target) const
{
    // SIMD optimized distance calculation
    __m128d pos = _mm_set_pd(target.y - center_y, target.x - center_x);
    __m128d squared = _mm_mul_pd(pos, pos);
    __m128d sum = _mm_hadd_pd(squared, squared);
    return _mm_cvtsd_f64(_mm_sqrt_pd(sum));
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

    // SIMD 최적화를 위한 데이터 정렬
    alignas(16) double target_data[2] = {
        target.x + target.w * 0.5,
        target.y + target.h * 0.5};

    // SIMD를 사용한 오차 계산
    __m128d target_pos = _mm_load_pd(target_data);
    __m128d center_pos = _mm_set_pd(local_center_y, local_center_x);
    __m128d error_vec = _mm_sub_pd(target_pos, center_pos);

    double error_x = _mm_cvtsd_f64(error_vec);
    double error_y = _mm_cvtsd_f64(_mm_unpackhi_pd(error_vec, error_vec));

    // 첫 번째 탐지인 경우 예측 초기화
    if (!target_detected.load())
    {
        resetPrediction();
        Eigen::Vector2d measurement(target_data[0], target_data[1]);
        kalman_filter->update(measurement);
    }

    // 대상 위치 예측
    Eigen::Vector2d predicted = predictTargetPosition(target_data[0], target_data[1]);

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

    // 마우스 이동 계산 최적화
    __m128d pid_vec = _mm_set_pd(pid_output.y(), pid_output.x());
    __m128d scale_vec = _mm_set_pd(
        (local_fov_y / 360.0) * (1000.0 / (local_sensitivity * local_dpi)),
        (local_fov_x / 360.0) * (1000.0 / (local_sensitivity * local_dpi)));
    __m128d move_vec = _mm_mul_pd(pid_vec, scale_vec);

    // 스코프 배율 적용
    if (bScope_multiplier > 1.0f)
    {
        __m128d scope_vec = _mm_set1_pd(1.0 / bScope_multiplier);
        move_vec = _mm_mul_pd(move_vec, scope_vec);
    }

    // 정수로 반올림
    int dx_int = static_cast<int>(std::round(_mm_cvtsd_f64(move_vec)));
    int dy_int = static_cast<int>(std::round(_mm_cvtsd_f64(_mm_unpackhi_pd(move_vec, move_vec))));

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