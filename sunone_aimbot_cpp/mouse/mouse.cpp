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
    // 시간 간격 계산 - std::chrono 사용으로 최적화
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time_point).count();
    
    // 시간 간격 제한 (너무 큰 dt는 불안정성 유발)
    dt = std::min(dt, 0.1);
    
    last_time_point = now;

    // SIMD 최적화를 위한 벡터 연산 개선
    if (dt > 0.0001) {  // 0으로 나누기 방지
        // 적분항 업데이트 (오차의 누적) - 적분 항 제한 추가
        integral += error * dt;
        
        // 적분 항 제한 (anti-windup)
        const double max_integral = 50.0;
        if (integral.x() > max_integral) integral.x() = max_integral;
        else if (integral.x() < -max_integral) integral.x() = -max_integral;
        
        if (integral.y() > max_integral) integral.y() = max_integral;
        else if (integral.y() < -max_integral) integral.y() = -max_integral;
        
        // 미분항 계산 (변화율)
        derivative = (error - prev_error) / dt;
    }
    else {
        derivative.setZero();
    }

    // PID 출력 계산 - 한 번의 연산으로 처리
    Eigen::Vector2d output = kp * error + ki * integral + kd * derivative;
    
    prev_error = error;  // 다음 계산을 위해 현재 오차 저장
    return output;
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2d::Zero();  // 이전 오차 초기화
    integral = Eigen::Vector2d::Zero();    // 적분항 초기화
    derivative = Eigen::Vector2d::Zero();  // 미분항 초기화
    last_time_point = std::chrono::steady_clock::now(); // 시간 초기화
}

void PIDController2D::updateParameters(double kp, double ki, double kd)
{
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
}

// 칼만 필터 구현
KalmanFilter2D::KalmanFilter2D(double process_noise_q, double measurement_noise_r) {
    // 상태 전이 행렬 초기화
    A = Eigen::Matrix<double, 6, 6>::Identity();
    
    // 측정 행렬 초기화 (위치만 측정)
    H = Eigen::Matrix<double, 2, 6>::Zero();
    H(0,0) = 1.0;  // x 위치
    H(1,1) = 1.0;  // y 위치
    
    // 노이즈 매트릭스 초기화
    Q = Eigen::Matrix<double, 6, 6>::Identity() * process_noise_q;
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
    P = Eigen::Matrix<double, 6, 6>::Identity();
    
    x = Eigen::Matrix<double, 6, 1>::Zero();
}

void KalmanFilter2D::predict(double dt) {
    // dt에 따른 상태 전이 행렬 업데이트
    A(0,2) = dt;      // x = x + vx*dt + 0.5*ax*dt^2
    A(0,4) = 0.5*dt*dt;
    A(1,3) = dt;      // y = y + vy*dt + 0.5*ay*dt^2
    A(1,5) = 0.5*dt*dt;
    A(2,4) = dt;      // vx = vx + ax*dt
    A(3,5) = dt;      // vy = vy + ay*dt

    x = A * x;
    P = A * P * A.transpose() + Q;
}

void KalmanFilter2D::update(const Eigen::Vector2d& measurement) {
    Eigen::Matrix2d S = H * P * H.transpose() + R;
    Eigen::Matrix<double, 6, 2> K = P * H.transpose() * S.inverse();
    
    Eigen::Vector2d y = measurement - H * x;
    x = x + K * y;
    P = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * P;
}

void KalmanFilter2D::reset() {
    x = Eigen::Matrix<double, 6, 1>::Zero();
    P = Eigen::Matrix<double, 6, 6>::Identity();
}

void KalmanFilter2D::updateParameters(double process_noise_q, double measurement_noise_r) {
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

Eigen::Vector2d MouseThread::predictTargetPosition(double target_x, double target_y) {
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_prediction_time).count();
    dt = std::min(dt, 0.1); // 너무 긴 시간 간격은 제한
    
    last_prediction_time = current_time;
    
    // 대상이 탐지되지 않았으면 칼만 필터 예측만 수행
    if (!target_detected.load()) {
        kalman_filter->predict(dt);
        return Eigen::Vector2d(kalman_filter->getState()(0, 0), kalman_filter->getState()(1, 0));
    }
    
    // 칼만 필터 시간 스텝 예측
    kalman_filter->predict(dt);
    
    // 측정값 업데이트
    Eigen::Vector2d measurement(target_x, target_y);
    kalman_filter->update(measurement);
    
    // 현재 상태 벡터에서 위치, 속도, 가속도 추출
    const auto& state = kalman_filter->getState();
    double pos_x = state(0, 0);
    double pos_y = state(1, 0);
    double vel_x = state(2, 0);
    double vel_y = state(3, 0);
    double acc_x = state(4, 0);
    double acc_y = state(5, 0);
    
    // 미래 위치 예측 (dt * 예측 시간 조정 계수)
    // 반응성과 안정성의 밸런스를 위한 계수
    constexpr double prediction_factor = 0.05;
    double prediction_time = dt * prediction_factor;
    
    // x = x0 + v*t + 0.5*a*t^2 공식 사용
    double future_x = pos_x + vel_x * prediction_time + 0.5 * acc_x * prediction_time * prediction_time;
    double future_y = pos_y + vel_y * prediction_time + 0.5 * acc_y * prediction_time * prediction_time;
    
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
        pid_output[1] * fov_scale_y * sens_scale
    );
}

bool MouseThread::checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    // Fast path: first check if the target center is within reasonable bounds
    double center_target_x = target_x + target_w / 2;
    double center_target_y = target_y + target_h / 2;

    // Quick check against screen center with a margin
    double dx = std::abs(center_target_x - center_x);
    double dy = std::abs(center_target_y - center_y);

    // If center is far away, avoid more complex calculations
    if (dx > screen_width / 4 || dy > screen_height / 4)
    {
        return false;
    }

    // More precise check using the reduced target dimensions
    double reduced_w = target_w * reduction_factor;
    double reduced_h = target_h * reduction_factor;

    // Fast AABB check using pre-calculated boundaries
    double min_x = center_target_x - reduced_w / 2;
    double max_x = center_target_x + reduced_w / 2;
    double min_y = center_target_y - reduced_h / 2;
    double max_y = center_target_y + reduced_h / 2;

    // Check if screen center is within the reduced target bounds
    return (center_x >= min_x && center_x <= max_x &&
            center_y >= min_y && center_y <= max_y);
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

void MouseThread::moveMouse(const AimbotTarget &target) {
    // 잠금 없이 로컬 변수에 값 복사 (스레드 안전성 개선)
    double center_x_local = center_x;
    double center_y_local = center_y;
    
    // 대상 위치 가져오기 - 바운딩 박스의 중앙 좌표로 계산
    double target_x = target.x + target.w / 2; // 중앙 X 좌표 계산
    double target_y = target.y + target.h / 2; // 중앙 Y 좌표 계산
        
    // 화면 중앙에서 타겟까지의 오차 계산
    double error_x = target_x - center_x_local;
    double error_y = target_y - center_y_local;
    
    // 첫 번째 탐지인 경우 예측 초기화
    if (!target_detected.load()) {
        resetPrediction();
        // 기본 윈도우 중앙 좌표로 상태 초기화
        Eigen::Vector2d measurement(target_x, target_y);
        kalman_filter->update(measurement);
    }
    
    // 대상 위치 예측
    Eigen::Vector2d predicted = predictTargetPosition(target_x, target_y);
    
    // 수정된 오차 계산 (중앙과 예측 위치 사이)
    error_x = predicted.x() - center_x_local;
    error_y = predicted.y() - center_y_local;
    
    // 성능 측정 콜백 사용
    if (tracking_errors) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback) {
            error_callback(error_x, error_y);
        }
    }
    
    // PID 컨트롤러에 오차 입력
    Eigen::Vector2d error(error_x, error_y);
    Eigen::Vector2d pid_output = pid_controller->calculate(error);
    
    // 마우스 이동 크기를 게임 설정에 맞게 변환
    // 화면상 거리(픽셀) -> 마우스 움직임(미크론) -> 윈도우 좌표
    double dx = (pid_output.x() / 360.0) * (fov_x / mouse_sensitivity) * (1000.0 / dpi);
    double dy = (pid_output.y() / 360.0) * (fov_y / mouse_sensitivity) * (1000.0 / dpi);
    
    // 스코프 배율 적용
    if (bScope_multiplier > 1.0f) {
        dx /= bScope_multiplier;
        dy /= bScope_multiplier;
    }
    
    // 정수로 반올림하여 마우스 오차 최소화
    int dx_int = static_cast<int>(std::round(dx));
    int dy_int = static_cast<int>(std::round(dy));
    
    // 실제 마우스 이동 (0이 아닌 경우에만)
    if (dx_int != 0 || dy_int != 0) {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid()) {
            input_method->move(dx_int, dy_int);
        }
    }
    
    // 타겟 시간 업데이트
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

void MouseThread::enableErrorTracking(const ErrorTrackingCallback& callback)
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