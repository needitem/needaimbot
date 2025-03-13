#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include "../modules/eigen/include/Eigen/Dense"
#include <shared_mutex>
#include <memory>
#include <functional>
#include <chrono>  // 시간 측정을 위한 헤더 추가

#include "AimbotTarget.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "InputMethod.h"

// 개선된 2D 칼만 필터 - 위치, 속도, 가속도 추적
class KalmanFilter2D {
private:
    // 상태 벡터: [x, y, vx, vy, ax, ay]
    Eigen::Matrix<double, 6, 6> A;  // 상태 전이 행렬
    Eigen::Matrix<double, 2, 6> H;  // 측정 행렬
    Eigen::Matrix<double, 6, 6> Q;  // 프로세스 노이즈
    Eigen::Matrix2d R;              // 측정 노이즈
    Eigen::Matrix<double, 6, 6> P;  // 오차 공분산
    Eigen::Matrix<double, 6, 1> x;  // 상태 벡터

public:
    KalmanFilter2D(double process_noise_q = 0.1, double measurement_noise_r = 0.1);
    void predict(double dt);
    void update(const Eigen::Vector2d& measurement);
    Eigen::Matrix<double, 6, 1> getState() const { return x; }
    void reset();
    void updateParameters(double process_noise_q, double measurement_noise_r);
};

// 2D PID 컨트롤러 - 에임 보정용
class PIDController2D
{
private:
    // 수평(X축)과 수직(Y축)으로 분리된 PID 게인
    double kp_x, kp_y;  // 비례 게인: 현재 오차에 대한 즉각적인 반응 (큰 값 = 빠른 반응, 작은 값 = 부드러운 움직임)
    double ki_x, ki_y;  // 적분 게인: 누적 오차 보정 (큰 값 = 정확한 조준, 작은 값 = 오버슈트 감소)
    double kd_x, kd_y;  // 미분 게인: 오차 변화율에 대한 반응 (큰 값 = 빠른 정지, 작은 값 = 부드러운 감속)
    
    // 기존 공통 게인 (호환성 유지용)
    double kp;  
    double ki;  
    double kd;  
    
    Eigen::Vector2d prev_error;  // 이전 오차 (미분항 계산용)
    Eigen::Vector2d integral;    // 누적 오차 (적분항 계산용)
    Eigen::Vector2d derivative;  // 변화율 저장 (미분항)
    Eigen::Vector2d prev_derivative; // 이전 미분값 (미분항 필터링용)
    std::chrono::steady_clock::time_point last_time_point;  // 이전 계산 시간 (dt 계산용)

public:
    // 기존 생성자 (호환성 유지)
    PIDController2D(double kp, double ki, double kd);
    
    // X/Y 분리 게인을 사용하는 새 생성자
    PIDController2D(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y);
    
    Eigen::Vector2d calculate(const Eigen::Vector2d &error);
    void reset();  // 컨트롤러 초기화 (새로운 타겟 조준 시작시 사용)
    
    // 기존 파라미터 업데이트 함수 (호환성 유지)
    void updateParameters(double kp, double ki, double kd);
    
    // X/Y 분리 게인 업데이트 함수
    void updateSeparatedParameters(double kp_x, double ki_x, double kd_x, double kp_y, double ki_y, double kd_y);
};

using ErrorTrackingCallback = std::function<void(double error_x, double error_y)>;

class MouseThread
{
private:
    std::unique_ptr<KalmanFilter2D> kalman_filter;
    std::unique_ptr<PIDController2D> pid_controller;
    std::unique_ptr<InputMethod> input_method;

    // 성능 추적을 위한 콜백 함수
    ErrorTrackingCallback error_callback;
    std::mutex callback_mutex;
    bool tracking_errors;

    double screen_width;
    double screen_height;
    double dpi;
    double mouse_sensitivity;
    double fov_x;
    double fov_y;
    double center_x;
    double center_y;
    bool auto_shoot;
    float bScope_multiplier;

    std::chrono::steady_clock::time_point last_target_time;
    std::chrono::steady_clock::time_point last_prediction_time;
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    // Simplified target tracking
    AimbotTarget *current_target;

    double calculateTargetDistance(const AimbotTarget &target) const;
    AimbotTarget *findClosestTarget(const std::vector<AimbotTarget> &targets) const;

public:
    MouseThread(int resolution, int dpi, double sensitivity, int fovX, int fovY,
                double kp, double ki, double kd,
                double process_noise_q, double measurement_noise_r,
                bool auto_shoot, float bScope_multiplier,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);

    // X/Y 분리 PID 컨트롤러를 지원하는 새 생성자
    MouseThread(int resolution, int dpi, double sensitivity, int fovX, int fovY,
                double kp_x, double ki_x, double kd_x,
                double kp_y, double ki_y, double kd_y,
                double process_noise_q, double measurement_noise_r,
                bool auto_shoot, float bScope_multiplier,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);

    void updateConfig(int resolution, int dpi, double sensitivity, int fovX, int fovY,
                      double kp, double ki, double kd,
                      double process_noise_q, double measurement_noise_r,
                      bool auto_shoot, float bScope_multiplier);

    // 분리된 X/Y PID 게인을 사용하는 새 updateConfig 메서드
    void updateConfig(int resolution, int dpi, double sensitivity, int fovX, int fovY,
                      double kp_x, double ki_x, double kd_x,
                      double kp_y, double ki_y, double kd_y,
                      double process_noise_q, double measurement_noise_r,
                      bool auto_shoot, float bScope_multiplier);

    Eigen::Vector2d predictTargetPosition(double target_x, double target_y);
    Eigen::Vector2d calculateMovement(const Eigen::Vector2d &target_pos);
    bool checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor);
    void moveMouse(const AimbotTarget &target);
    void pressMouse(const AimbotTarget &target);
    void releaseMouse();
    void resetPrediction();
    void checkAndResetPredictions();
    void applyRecoilCompensation(float strength);

    void enableErrorTracking(const ErrorTrackingCallback& callback);
    void disableErrorTracking();

    std::mutex input_method_mutex;
    void setInputMethod(std::unique_ptr<InputMethod> new_method);
    
    double& getScreenWidth() { return screen_width; }
    double& getScreenHeight() { return screen_height; }
    double& getDPI() { return dpi; }
    double& getSensitivity() { return mouse_sensitivity; }
    double& getFOVX() { return fov_x; }
    double& getFOVY() { return fov_y; }
    bool& getAutoShoot() { return auto_shoot; }
    float& getScopeMultiplier() { return bScope_multiplier; }
};

#endif // MOUSE_H