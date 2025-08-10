#include "IntegratedController.h"
#include <cmath>
#include <algorithm>
#include <iostream>

IntegratedController::IntegratedController(float kp_x, float ki_x, float kd_x, 
                                         float kp_y, float ki_y, float kd_y)
    : pid_controller(std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y))
    , last_recoil_time(std::chrono::steady_clock::now())
    , fire_start_time(std::chrono::steady_clock::now())
{
}

std::pair<Eigen::Vector2f, IntegratedController::DebugInfo> 
IntegratedController::calculateMovement() {
    std::lock_guard<std::mutex> lock(data_mutex);
    
    DebugInfo debug;
    debug.has_target = has_target.load();
    debug.is_firing = is_firing.load();
    
    // 1. 각 시스템에서 독립적으로 움직임 계산
    Eigen::Vector2f pid_movement(0.0f, 0.0f);
    Eigen::Vector2f recoil_movement(0.0f, 0.0f);
    
    if (debug.has_target) {
        pid_movement = calculatePIDMovement();
    }
    
    if (debug.is_firing) {
        recoil_movement = calculateRecoilMovement();
    }
    
    debug.pid_movement = pid_movement;
    debug.recoil_movement = recoil_movement;
    
    // 2. 충돌 해결 및 전략 선택
    auto [final_movement, strategy] = resolveConflict(pid_movement, recoil_movement);
    debug.strategy = strategy;
    
    // 3. 스무딩 제거 (원 요청: EMA 삭제)
    debug.final_movement = final_movement;
    
    // 충돌 점수 계산 (디버그용)
    if (pid_movement.norm() > 0 && recoil_movement.norm() > 0) {
        debug.conflict_score = pid_movement.normalized().dot(recoil_movement.normalized());
    } else {
        debug.conflict_score = 0.0f;
    }
    
    return {debug.final_movement, debug};
}

Eigen::Vector2f IntegratedController::calculatePIDMovement() {
    if (!has_target.load()) {
        return Eigen::Vector2f(0.0f, 0.0f);
    }
    
    // 에러 계산
    float error_x = target_info.x - target_info.center_x;
    float error_y = target_info.y - target_info.center_y;
    
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);
    
    // 스케일 적용
    pid_output.x() *= move_scale_x;
    pid_output.y() *= move_scale_y;
    
    return pid_output;
}

Eigen::Vector2f IntegratedController::calculateRecoilMovement() {
    if (!is_firing.load() || !current_weapon_profile) {
        return Eigen::Vector2f(0.0f, 0.0f);
    }
    
    auto now = std::chrono::steady_clock::now();
    
    // 발사 시작 후 경과 시간
    auto fire_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - fire_start_time).count();
    
    // 시작 딜레이 체크
    if (fire_duration < current_weapon_profile->start_delay_ms) {
        return Eigen::Vector2f(0.0f, 0.0f);
    }
    
    // 종료 딜레이 체크
    if (current_weapon_profile->end_delay_ms > 0 && 
        fire_duration > current_weapon_profile->end_delay_ms) {
        return Eigen::Vector2f(0.0f, 0.0f);
    }
    
    // 반동 적용 타이밍 체크
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_recoil_time).count();
    
    if (time_since_last < current_weapon_profile->recoil_ms) {
        return Eigen::Vector2f(0.0f, 0.0f);
    }
    
    // 스코프 배율 계산
    float scope_multiplier = 1.0f;
    switch (current_scope_magnification) {
        case 1: scope_multiplier = current_weapon_profile->scope_mult_1x; break;
        case 2: scope_multiplier = current_weapon_profile->scope_mult_2x; break;
        case 3: scope_multiplier = current_weapon_profile->scope_mult_3x; break;
        case 4: scope_multiplier = current_weapon_profile->scope_mult_4x; break;
        case 6: scope_multiplier = current_weapon_profile->scope_mult_6x; break;
        case 8: scope_multiplier = current_weapon_profile->scope_mult_8x; break;
    }
    
    // 최종 반동 강도 계산
    float recoil_strength = current_weapon_profile->base_strength * 
                           current_weapon_profile->fire_rate_multiplier * 
                           scope_multiplier;
    
    last_recoil_time = now;
    
    // 반동은 Y축 아래 방향으로만 적용
    return Eigen::Vector2f(0.0f, recoil_strength);
}

std::pair<Eigen::Vector2f, IntegratedController::MovementStrategy> 
IntegratedController::resolveConflict(const Eigen::Vector2f& pid_move, 
                                    const Eigen::Vector2f& recoil_move) {
    // 케이스 1: 움직임 없음
    if (pid_move.norm() < 0.001f && recoil_move.norm() < 0.001f) {
        return {Eigen::Vector2f(0.0f, 0.0f), MovementStrategy::NONE};
    }
    
    // 케이스 2: 반동만
    if (pid_move.norm() < 0.001f) {
        return {recoil_move, MovementStrategy::RECOIL_ONLY};
    }
    
    // 케이스 3: PID만
    if (recoil_move.norm() < 0.001f) {
        return {pid_move, MovementStrategy::PID_ONLY};
    }
    
    // 케이스 4: 둘 다 있을 때 - 충돌 분석
    float dot_product = pid_move.normalized().dot(recoil_move.normalized());
    
    // 4-1: 같은 방향 (dot product > 0.8)
    if (dot_product > 0.8f) {
        return {pid_move + recoil_move, MovementStrategy::ALIGNED_COMBINE};
    }
    
    // 4-2: 반대 방향 (dot product < -0.8)
    if (dot_product < -0.8f) {
        // Y축은 반동 우선, X축은 PID 우선
        Eigen::Vector2f combined;
        combined.x() = pid_move.x() * 0.7f + recoil_move.x() * 0.3f;
        combined.y() = pid_move.y() * 0.3f + recoil_move.y() * 0.7f;
        return {combined, MovementStrategy::CONFLICT_WEIGHTED};
    }
    
    // 4-3: 수직 또는 각도가 있는 경우
    // 축별로 다른 가중치 적용
    Eigen::Vector2f combined;
    
    // X축: PID가 주로 담당 (좌우 조준)
    combined.x() = pid_move.x() * 0.8f + recoil_move.x() * 0.2f;
    
    // Y축: 반동과 PID 균형있게 적용
    if (pid_move.y() > 0 && recoil_move.y() > 0) {
        // 둘 다 아래로 향하면 합산
        combined.y() = pid_move.y() + recoil_move.y();
    } else {
        // 그 외의 경우 가중 평균
        combined.y() = pid_move.y() * 0.5f + recoil_move.y() * 0.5f;
    }
    
    return {combined, MovementStrategy::PERPENDICULAR_WEIGHTED};
}

Eigen::Vector2f IntegratedController::applySmoothingSimple(const Eigen::Vector2f& raw_movement) {
    // 간단한 지수 이동 평균 스무딩
    smoothed_movement = smoothed_movement * (1.0f - SMOOTHING_FACTOR) + 
                       raw_movement * SMOOTHING_FACTOR;
    
    // 작은 움직임 제거 (노이즈 필터)
    if (smoothed_movement.norm() < 0.1f) {
        return Eigen::Vector2f(0.0f, 0.0f);
    }
    
    return smoothed_movement;
}

void IntegratedController::updateTarget(float target_x, float target_y, 
                                      float center_x, float center_y) {
    std::lock_guard<std::mutex> lock(data_mutex);
    target_info.x = target_x;
    target_info.y = target_y;
    target_info.center_x = center_x;
    target_info.center_y = center_y;
    has_target.store(true);
}

void IntegratedController::clearTarget() {
    std::lock_guard<std::mutex> lock(data_mutex);
    has_target.store(false);
    pid_controller->reset();
}

void IntegratedController::startFiring() {
    std::lock_guard<std::mutex> lock(data_mutex);
    if (!is_firing.load()) {
        fire_start_time = std::chrono::steady_clock::now();
        last_recoil_time = fire_start_time;
    }
    is_firing.store(true);
}

void IntegratedController::stopFiring() {
    std::lock_guard<std::mutex> lock(data_mutex);
    is_firing.store(false);
}

void IntegratedController::setWeaponProfile(const WeaponRecoilProfile* profile, 
                                           int scope_magnification) {
    std::lock_guard<std::mutex> lock(data_mutex);
    current_weapon_profile = profile;
    current_scope_magnification = scope_magnification;
}

void IntegratedController::updatePIDParameters(float kp_x, float ki_x, float kd_x,
                                             float kp_y, float ki_y, float kd_y) {
    std::lock_guard<std::mutex> lock(data_mutex);
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
}

void IntegratedController::reset() {
    std::lock_guard<std::mutex> lock(data_mutex);
    pid_controller->reset();
    smoothed_movement = Eigen::Vector2f(0.0f, 0.0f);
    has_target.store(false);
    is_firing.store(false);
}

void IntegratedController::setScaleFactors(float scale_x, float scale_y) {
    std::lock_guard<std::mutex> lock(data_mutex);
    move_scale_x = scale_x;
    move_scale_y = scale_y;
}