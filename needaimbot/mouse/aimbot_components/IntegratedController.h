#ifndef INTEGRATED_CONTROLLER_H
#define INTEGRATED_CONTROLLER_H

#include "../../modules/eigen/include/Eigen/Dense"
#include <chrono>
#include <memory>
#include <mutex>
#include <atomic>
#include "PIDController2D.h"
#include "../../config/config.h"

/**
 * @brief 통합 마우스 컨트롤러 - PID 조준과 반동 제어를 충돌 없이 결합
 * 
 * 두 시스템이 독립적으로 작동하면서도 지능적으로 협력하도록 설계.
 * 반동 제어는 타겟 없이도 작동하고, PID는 타겟이 있을 때만 활성화.
 */
class IntegratedController {
public:
    // 움직임 전략 타입
    enum class MovementStrategy {
        NONE,                   // 움직임 없음
        RECOIL_ONLY,           // 반동 제어만
        PID_ONLY,              // PID 조준만
        ALIGNED_COMBINE,       // 같은 방향 - 단순 합
        CONFLICT_WEIGHTED,     // 반대 방향 - 가중 평균
        PERPENDICULAR_WEIGHTED // 수직 방향 - 축별 가중치
    };

    // 디버그 정보 구조체
    struct DebugInfo {
        Eigen::Vector2f pid_movement;
        Eigen::Vector2f recoil_movement;
        Eigen::Vector2f final_movement;
        MovementStrategy strategy;
        float conflict_score;
        bool has_target;
        bool is_firing;
    };

    IntegratedController(float kp_x, float ki_x, float kd_x, 
                        float kp_y, float ki_y, float kd_y);
    ~IntegratedController() = default;

    /**
     * @brief 통합 움직임 계산 - 메인 함수
     * @return 최종 마우스 움직임 벡터와 디버그 정보
     */
    std::pair<Eigen::Vector2f, DebugInfo> calculateMovement();

    /**
     * @brief 타겟 정보 업데이트
     */
    void updateTarget(float target_x, float target_y, float center_x, float center_y);
    
    /**
     * @brief 타겟 클리어
     */
    void clearTarget();

    /**
     * @brief 사격 시작
     */
    void startFiring();
    
    /**
     * @brief 사격 종료
     */
    void stopFiring();

    /**
     * @brief 무기 프로파일 설정
     */
    void setWeaponProfile(const WeaponRecoilProfile* profile, int scope_magnification);

    /**
     * @brief PID 파라미터 업데이트
     */
    void updatePIDParameters(float kp_x, float ki_x, float kd_x,
                           float kp_y, float ki_y, float kd_y);

    /**
     * @brief 컨트롤러 리셋
     */
    void reset();

    /**
     * @brief 스케일 팩터 설정
     */
    void setScaleFactors(float scale_x, float scale_y);

private:
    // PID 컨트롤러
    std::unique_ptr<PIDController2D> pid_controller;
    
    // 상태 변수
    std::atomic<bool> has_target{false};
    std::atomic<bool> is_firing{false};
    
    // 타겟 정보
    struct {
        float x, y;
        float center_x, center_y;
    } target_info;
    
    // 반동 제어 변수
    const WeaponRecoilProfile* current_weapon_profile{nullptr};
    int current_scope_magnification{1};
    std::chrono::steady_clock::time_point last_recoil_time;
    std::chrono::steady_clock::time_point fire_start_time;
    
    // 스케일 팩터
    float move_scale_x{1.0f};
    float move_scale_y{1.0f};
    
    // 스무딩 변수
    Eigen::Vector2f smoothed_movement{0.0f, 0.0f};
    static constexpr float SMOOTHING_FACTOR = 0.7f;
    
    // 뮤텍스
    mutable std::mutex data_mutex;

    /**
     * @brief PID 움직임 계산
     */
    Eigen::Vector2f calculatePIDMovement();
    
    /**
     * @brief 반동 제어 움직임 계산
     */
    Eigen::Vector2f calculateRecoilMovement();
    
    /**
     * @brief 충돌 해결 알고리즘
     */
    std::pair<Eigen::Vector2f, MovementStrategy> resolveConflict(
        const Eigen::Vector2f& pid_move, 
        const Eigen::Vector2f& recoil_move);
    
    /**
     * @brief 움직임 스무딩 적용
     */
    Eigen::Vector2f applySmoothingSimple(const Eigen::Vector2f& raw_movement);
};

#endif // INTEGRATED_CONTROLLER_H