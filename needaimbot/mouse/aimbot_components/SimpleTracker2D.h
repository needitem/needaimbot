#ifndef SIMPLE_TRACKER_2D_H
#define SIMPLE_TRACKER_2D_H

#include <chrono>
#include "../../include/simple_math.h"

// Eigen Kalman 필터를 대체하는 간단한 지수 평활화 추적기
class SimpleTracker2D
{
private:
    Vec2f position;
    Vec2f velocity;
    float alpha_pos;      // 위치 평활화 계수
    float alpha_vel;      // 속도 평활화 계수
    bool initialized;
    std::chrono::steady_clock::time_point last_time_point;
    
public:
    SimpleTracker2D(float pos_alpha = 0.3f, float vel_alpha = 0.5f)
        : alpha_pos(pos_alpha), alpha_vel(vel_alpha), initialized(false) {}
    
    // 새로운 측정값으로 업데이트 (최적화된 버전)
    Vec2f update(const Vec2f& measurement) {
        auto now = std::chrono::steady_clock::now();
        
        if (!initialized) {
            position = measurement;
            velocity = Vec2f(0, 0);
            initialized = true;
            last_time_point = now;
            return position;
        }
        
        // 최적화: 나노초 단위로 직접 계산하여 정밀도 향상
        float dt = std::chrono::duration<float, std::nano>(now - last_time_point).count() * 1e-9f;
        dt = std::max(0.001f, std::min(dt, 0.1f)); // 1ms~100ms 제한
        
        // 최적화: 역수 미리 계산하여 나눗셈을 곱셈으로 변환
        const float inv_dt = 1.0f / dt;
        const float one_minus_alpha_vel = 1.0f - alpha_vel;
        const float one_minus_alpha_pos = 1.0f - alpha_pos;
        
        // 벡터화된 위치 차이 계산
        Vec2f pos_diff = measurement - position;
        
        // 즉시 속도 계산 (최적화된 방식)
        Vec2f inst_velocity = pos_diff * inv_dt;
        
        // 지수 평활화로 속도 스무딩 (벡터화)
        velocity = inst_velocity * alpha_vel + velocity * one_minus_alpha_vel;
        
        // 지수 평활화로 위치 스무딩 (벡터화)
        position = measurement * alpha_pos + position * one_minus_alpha_pos;
        
        last_time_point = now;
        return position;
    }
    
    // 미래 위치 예측
    Vec2f predict(float future_time_ms = 0.0f) {
        if (!initialized) return Vec2f(0, 0);
        
        float dt = future_time_ms * 0.001f; // ms -> 초
        return Vec2f(
            position.x + velocity.x * dt,
            position.y + velocity.y * dt
        );
    }
    
    // 현재 속도 반환
    Vec2f getVelocity() const {
        return velocity;
    }
    
    // 추적기 리셋
    void reset() {
        position.reset();
        velocity.reset();
        initialized = false;
    }
    
    // 평활화 매개변수 업데이트
    void updateParameters(float pos_alpha, float vel_alpha) {
        alpha_pos = pos_alpha;
        alpha_vel = vel_alpha;
    }
    
    // 초기화 여부 확인
    bool isInitialized() const { 
        return initialized; 
    }
    
    // 현재 위치 반환
    Vec2f getPosition() const {
        return position;
    }
};

#endif // SIMPLE_TRACKER_2D_H