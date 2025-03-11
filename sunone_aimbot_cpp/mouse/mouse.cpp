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
    : kp(kp), ki(ki), kd(kd)
{
    reset();
}

Eigen::Vector2d PIDController2D::calculate(const Eigen::Vector2d &error)
{
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time_point).count();
    dt = std::min(dt, 0.1); // 시간 간격 제한

    last_time_point = now;

    // 시간 기반 게인 조정 (50ms 간격)
    static auto last_gain_update = now;
    static double cached_kp = kp;
    static double cached_kd = kd;

    double time_since_update = std::chrono::duration<double>(now - last_gain_update).count();
    if (time_since_update >= 0.05) // 50ms
    {
        last_gain_update = now;
        double error_magnitude = error.norm();

        // 부드러운 게인 보간
        double target_kp = kp * (1.0 + std::min(error_magnitude / 150.0, 0.3));
        double target_kd = kd * (1.0 + std::min(error_magnitude / 300.0, 0.2));

        // 선형 보간으로 부드럽게 변경
        double alpha = std::min(time_since_update * 10.0, 1.0); // 최대 100ms에 걸쳐 완전히 변경
        cached_kp = cached_kp * (1.0 - alpha) + target_kp * alpha;
        cached_kd = cached_kd * (1.0 - alpha) + target_kd * alpha;
    }

    if (dt > 0.0001)
    {
        // 적분항 업데이트 - 간단한 제한
        integral += error * dt;

        // 적분 항 제한 - 단순화된 버전
        const double max_integral = 50.0;
        integral.x() = std::clamp(integral.x(), -max_integral, max_integral);
        integral.y() = std::clamp(integral.y(), -max_integral, max_integral);

        // 미분항 계산 및 필터링 - 더 안정적인 필터링
        derivative = (error - prev_error) / dt;
        static const double alpha = 0.8; // 더 강한 필터링
        derivative = derivative * alpha + prev_derivative * (1.0 - alpha);
        prev_derivative = derivative;
    }
    else
    {
        derivative.setZero();
    }

    // 캐시된 게인으로 PID 출력 계산
    Eigen::Vector2d output = cached_kp * error + ki * integral + cached_kd * derivative;

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

    // 노이즈 매트릭스 초기화
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
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
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
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

    last_prediction_time = std::chrono::steady_clock::now();
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
    this->screen_width = resolution;
    this->screen_height = resolution;
    this->dpi = dpi;
    this->mouse_sensitivity = sensitivity;
    this->fov_x = fovX;
    this->fov_y = fovY;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = resolution / 2;
    this->center_y = resolution / 2;

    kalman_filter->updateParameters(process_noise_q, measurement_noise_r);
    pid_controller->updateParameters(kp, ki, kd);
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

    // 단순화된 속도 계산 (hypot 대신 sqrt 사용)
    double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y);

    // 단순화된 예측 시간 조정
    constexpr double base_prediction_factor = 0.05;
    double prediction_time;

    if (velocity < 300.0)
    {
        prediction_time = dt * base_prediction_factor;
    }
    else if (velocity < 600.0)
    {
        prediction_time = dt * base_prediction_factor * 1.3;
    }
    else
    {
        prediction_time = dt * base_prediction_factor * 1.5;
    }

    // 선형 예측만 사용 (가속도 제외)
    double future_x = pos_x + vel_x * prediction_time;
    double future_y = pos_y + vel_y * prediction_time;

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

        if (elapsed > 0.25) // 250ms timeout
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