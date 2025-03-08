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

extern std::atomic<bool> aiming;
extern std::mutex configMutex;

// PID 컨트롤러 구현
PIDController2D::PIDController2D(double kp, double ki, double kd)
    : kp(kp), ki(ki), kd(kd)
{
    reset();
}

Eigen::Vector2d PIDController2D::calculate(const Eigen::Vector2d &error)
{
    // 시간 간격 계산
    double now = time(NULL);
    double dt = now - last_time;
    if (dt > 0.1) dt = 0.1;  // 시간 간격 제한 (너무 큰 dt는 불안정성 유발)
    
    last_time = now;

    // 미분항 계산 (오차의 변화율)
    Eigen::Vector2d derivative = error - prev_error;
    
    if (dt > 0.0001) {  // 0으로 나누기 방지
        // 적분항 업데이트 (오차의 누적)
        integral += error * dt;
        // 미분항 계산 (변화율)
        derivative /= dt;
    }

    // PID 출력 계산
    // P항: 현재 오차에 비례
    // I항: 누적 오차에 비례
    // D항: 오차 변화율에 비례
    Eigen::Vector2d output = kp * error + ki * integral + kd * derivative;
    
    prev_error = error;  // 다음 계산을 위해 현재 오차 저장
    return output;
}

void PIDController2D::reset()
{
    prev_error = Eigen::Vector2d::Zero();  // 이전 오차 초기화
    integral = Eigen::Vector2d::Zero();    // 적분항 초기화
    last_time = time(NULL);                // 시간 초기화
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
    const auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_prediction_time).count();
    last_prediction_time = current_time;

    // 시간 간격이 너무 크면 리셋
    if (dt > 0.1) {
        kalman_filter->reset();
        dt = 0.001;  // 안전한 작은 값 사용
    }

    // 측정값 업데이트
    Eigen::Vector2d measurement(target_x, target_y);
    
    // 예측 및 업데이트
    kalman_filter->predict(dt);
    kalman_filter->update(measurement);
    
    // 예측된 다음 위치 계산 (현재 속도와 가속도 고려)
    Eigen::Matrix<double, 6, 1> state = kalman_filter->getState();
    double predicted_x = state(0) + state(2) * dt + 0.5 * state(4) * dt * dt;
    double predicted_y = state(1) + state(3) * dt + 0.5 * state(5) * dt * dt;
    
    return Eigen::Vector2d(predicted_x, predicted_y);
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

void MouseThread::moveMouse(const AimbotTarget &target)
{
    // Calculate target center
    double center_target_x = target.x + target.w / 2;
    double center_target_y = target.y + target.h / 2;

    // 칼만 필터로 다음 위치 예측
    Eigen::Vector2d predicted_pos = predictTargetPosition(center_target_x, center_target_y);

    // Calculate movement based on predicted position
    Eigen::Vector2d movement = calculateMovement(predicted_pos);

    // Move mouse using InputMethod
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method)
        {
            // easynorecoil이 켜져 있으면 y축 방향(아래쪽)으로 추가 이동
            int moveX = static_cast<int>(movement[0]);
            int moveY = static_cast<int>(movement[1]);
            
            if (config.easynorecoil && shooting.load() && zooming.load())
            {
                // Add recoil compensation to vertical movement
                int recoil_compensation = static_cast<int>(config.easynorecoilstrength);
                moveY += recoil_compensation;
            }
            
            // 기본 마우스 이동 적용
            input_method->move(moveX, moveY);
        }
    }

    // Update tracking state
    target_detected = true;
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