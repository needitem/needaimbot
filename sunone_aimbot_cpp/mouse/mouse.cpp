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
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time_point).count();
    dt = std::min(dt, 0.1); // 시간 간격 제한

    last_time_point = now;

    // 시간 기반 게인 조정 (30ms 간격으로 변경 - 더 빠른 업데이트)
    static auto last_gain_update = now;
    // X축/Y축 게인 캐싱
    static double cached_kp_x = kp_x;
    static double cached_ki_x = ki_x;
    static double cached_kd_x = kd_x;
    static double cached_kp_y = kp_y;
    static double cached_ki_y = ki_y;
    static double cached_kd_y = kd_y;

    double time_since_update = std::chrono::duration<double>(now - last_gain_update).count();
    if (time_since_update >= 0.03) // 30ms로 속도 향상
    {
        last_gain_update = now;
        
        // X축과 Y축 오차 분리
        double error_magnitude_x = std::abs(error.x());
        double error_magnitude_y = std::abs(error.y());

        // X축 게인 조정 (수평 움직임)
        double target_kp_x = kp_x * (1.0 + std::min(error_magnitude_x / 100.0, 0.6));
        double target_ki_x = ki_x * (1.0 - std::min(error_magnitude_x / 400.0, 0.8));
        double target_kd_x = kd_x * (1.0 + std::min(error_magnitude_x / 200.0, 0.4));

        // Y축 게인 조정 (수직 움직임) - 조금 더 보수적으로 조정
        double target_kp_y = kp_y * (1.0 + std::min(error_magnitude_y / 120.0, 0.5)); // 수직은 좀 더 보수적
        double target_ki_y = ki_y * (1.0 - std::min(error_magnitude_y / 350.0, 0.9)); // 수직 적분 효과 더 감소
        double target_kd_y = kd_y * (1.0 + std::min(error_magnitude_y / 180.0, 0.5)); // 더 강한 수직 미분 반응

        // 빠른 보간으로 신속하게 변경
        double alpha = std::min(time_since_update * 15.0, 1.0); // 최대 67ms에 걸쳐 완전히 변경
        cached_kp_x = cached_kp_x * (1.0 - alpha) + target_kp_x * alpha;
        cached_ki_x = cached_ki_x * (1.0 - alpha) + target_ki_x * alpha;
        cached_kd_x = cached_kd_x * (1.0 - alpha) + target_kd_x * alpha;
        cached_kp_y = cached_kp_y * (1.0 - alpha) + target_kp_y * alpha;
        cached_ki_y = cached_ki_y * (1.0 - alpha) + target_ki_y * alpha;
        cached_kd_y = cached_kd_y * (1.0 - alpha) + target_kd_y * alpha;
    }

    if (dt > 0.0001)
    {
        // 적분항 업데이트 - 오차 크기에 따른 적분 제한
        double error_norm = error.norm();
        
        // X축/Y축 적분 인자 별도 계산
        double integral_factor_x = 1.0;
        double integral_factor_y = 1.0;
        
        // 오차가 크면 적분 효과 감소 (과도한 누적 방지)
        if (std::abs(error.x()) > 100.0) {
            integral_factor_x = 100.0 / std::abs(error.x());
        }
        if (std::abs(error.y()) > 80.0) { // Y축은 더 빨리 제한 (하향 조준 문제 방지)
            integral_factor_y = 80.0 / std::abs(error.y());
        }
        
        // X축/Y축 분리 적분
        integral.x() += error.x() * dt * integral_factor_x;
        integral.y() += error.y() * dt * integral_factor_y;

        // 적분 항 제한 - 동적 제한
        integral.x() = std::clamp(integral.x(), -80.0, 80.0);
        integral.y() = std::clamp(integral.y(), -60.0, 60.0); // Y축 적분을 좀 더 제한
        
        // 미분항 계산 및 필터링 - X축/Y축 별도 계산
        derivative = (error - prev_error) / dt;
        
        // X축/Y축 별도의 미분 필터링
        double derivative_norm_x = std::abs(derivative.x());
        double derivative_norm_y = std::abs(derivative.y());
        
        double alpha_x = derivative_norm_x > 500.0 ? 0.7 : 0.85;
        double alpha_y = derivative_norm_y > 400.0 ? 0.6 : 0.9; // Y축은 더 빠르게 반응, 더 강한 필터링
        
        derivative.x() = derivative.x() * alpha_x + prev_derivative.x() * (1.0 - alpha_x);
        derivative.y() = derivative.y() * alpha_y + prev_derivative.y() * (1.0 - alpha_y);
        
        prev_derivative = derivative;
    }
    else
    {
        derivative.setZero();
    }

    // X축/Y축 분리 게인으로 PID 출력 계산
    Eigen::Vector2d output;
    output.x() = cached_kp_x * error.x() + cached_ki_x * integral.x() + cached_kd_x * derivative.x();
    output.y() = cached_kp_y * error.y() + cached_ki_y * integral.y() + cached_kd_y * derivative.y();

    // 출력 제한 - X축/Y축 별도 제한
    const double max_output_x = 1500.0;
    const double max_output_y = 1200.0; // Y축은 조금 더 제한 (과도한 하향 움직임 방지)
    
    output.x() = std::clamp(output.x(), -max_output_x, max_output_x);
    output.y() = std::clamp(output.y(), -max_output_y, max_output_y);

    prev_error = error;
    return output;
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

    if (!target_detected.load())
    {
        kalman_filter->predict(dt);
        return Eigen::Vector2d(kalman_filter->getState()(0, 0), kalman_filter->getState()(1, 0));
    }

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

    // 빠른 움직임을 더 잘 감지하기 위한 속도 및 가속도 계산
    double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y);
    double acceleration = std::sqrt(acc_x * acc_x + acc_y * acc_y);

    // 속도에 따른 예측 시간 조정 (이전보다 더 적극적으로)
    constexpr double base_prediction_factor = 0.07; // 기본값 증가
    double prediction_time;

    // 속도와 가속도에 따른 예측 시간 동적 조정 - 급격히 빠른 움직임에 더 빠르게 반응
    if (velocity < 200.0)
    {
        prediction_time = dt * base_prediction_factor;
    }
    else if (velocity < 500.0)
    {
        prediction_time = dt * base_prediction_factor * 1.5;
    }
    else if (velocity < 800.0)
    {
        prediction_time = dt * base_prediction_factor * 2.0;
    }
    else
    {
        // 매우 빠른 움직임에 대한 더 적극적인 예측
        prediction_time = dt * base_prediction_factor * 2.5;
    }

    // 가속도를 고려한 이차 예측 (고속 움직임에 더 정확함)
    double future_x = pos_x + vel_x * prediction_time + 0.5 * acc_x * prediction_time * prediction_time;
    double future_y = pos_y + vel_y * prediction_time + 0.5 * acc_y * prediction_time * prediction_time;

    // 급격한 방향 전환 감지 및 보정
    static Eigen::Vector2d prev_velocity(0, 0);
    Eigen::Vector2d current_velocity(vel_x, vel_y);
    
    if (prev_velocity.norm() > 0 && current_velocity.norm() > 200.0) {
        double angle_change = std::acos(
            std::clamp(prev_velocity.dot(current_velocity) / (prev_velocity.norm() * current_velocity.norm()), -1.0, 1.0)
        );
        
        // 급격한 방향 전환 시 예측을 덜 적극적으로 조정
        if (angle_change > 0.5) { // ~30도 이상 변화 감지
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
    // Calculate error directly
    Eigen::Vector2d error(target_pos[0] - center_x, target_pos[1] - center_y);

    // Calculate PID output
    Eigen::Vector2d pid_output = pid_controller->calculate(error);

    // Scale output with FOV and sensitivity
    double fov_scale_x = fov_x / screen_width;
    double fov_scale_y = fov_y / screen_height;
    double sens_scale = dpi * (1.0 / mouse_sensitivity) / 360.0;

    return Eigen::Vector2d(
        pid_output[0] * fov_scale_x * sens_scale,
        pid_output[1] * fov_scale_y * sens_scale);
}

bool MouseThread::checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    // SIMD 최적화된 중심점 계산
    __m128d target_dims = _mm_set_pd(target_h, target_w);
    __m128d half = _mm_set1_pd(0.5);
    __m128d target_pos = _mm_set_pd(target_y, target_x);
    __m128d target_center = _mm_add_pd(target_pos, _mm_mul_pd(target_dims, half));

    // 화면 중심과의 차이 계산
    __m128d screen_center = _mm_set_pd(center_y, center_x);
    __m128d diff = _mm_sub_pd(target_center, screen_center);
    __m128d abs_diff = _mm_andnot_pd(_mm_set1_pd(-0.0), diff);

    // 빠른 경계 검사
    double max_diff_x = _mm_cvtsd_f64(abs_diff);
    double max_diff_y = _mm_cvtsd_f64(_mm_unpackhi_pd(abs_diff, abs_diff));

    if (max_diff_x > screen_width * 0.25 || max_diff_y > screen_height * 0.25)
    {
        return false;
    }

    // 축소된 타겟 크기 계산
    __m128d reduced_dims = _mm_mul_pd(target_dims, _mm_set1_pd(reduction_factor * 0.5));

    // 경계 검사
    __m128d min_bound = _mm_sub_pd(target_center, reduced_dims);
    __m128d max_bound = _mm_add_pd(target_center, reduced_dims);

    // 화면 중심이 축소된 타겟 영역 내에 있는지 확인
    __m128d compare_min = _mm_cmpge_pd(screen_center, min_bound);
    __m128d compare_max = _mm_cmple_pd(screen_center, max_bound);
    __m128d result = _mm_and_pd(compare_min, compare_max);

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
    std::lock_guard<std::mutex> lock(input_method_mutex);

    // Pre-compute the scaling factor
    static const double vertical_scale = (fov_y / screen_height) * (dpi * (1.0 / mouse_sensitivity)) / 360.0;

    // Apply strength with pre-computed scale
    int compensation = static_cast<int>(strength * vertical_scale);

    if (input_method)
    {
        input_method->move(0, compensation);
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