#ifndef SNAP_AIM_CONTROLLER_H
#define SNAP_AIM_CONTROLLER_H

#include <algorithm>
#include <cmath>

class SnapAimController {
public:
    struct Settings {
        float snap_speed = 1.0f;           // 스냅 속도 배율 (1.0 = 기본)
        float min_movement = 1.0f;         // 최소 이동 단위 (지터 방지)
        float close_range_threshold = 5.0f; // 근거리 임계값
        float acceleration_factor = 2.0f;   // 가속 계수
        bool enable_instant_snap = true;    // 즉시 스냅 모드
    };

    SnapAimController(const Settings& settings = Settings()) 
        : settings_(settings) {}

    // 에러를 입력받아 즉시 이동할 거리 계산
    std::pair<float, float> calculate(float error_x, float error_y) {
        float distance = std::sqrt(error_x * error_x + error_y * error_y);
        
        // 거리가 매우 가까우면 이동 안함 (지터 방지)
        if (distance < settings_.close_range_threshold) {
            // 근거리에서는 정확히 타겟에 맞추기
            if (std::abs(error_x) < settings_.min_movement && 
                std::abs(error_y) < settings_.min_movement) {
                return {0.0f, 0.0f};
            }
            
            // 근거리에서는 천천히 이동
            return {
                error_x * 0.8f * settings_.snap_speed,
                error_y * 0.8f * settings_.snap_speed
            };
        }
        
        // 즉시 스냅 모드
        if (settings_.enable_instant_snap) {
            // 한 번에 타겟까지 이동 (가장 빠름)
            return {
                error_x * settings_.snap_speed * settings_.acceleration_factor,
                error_y * settings_.snap_speed * settings_.acceleration_factor
            };
        }
        
        // 거리 기반 가속 이동
        float speed_multiplier = 1.0f + (distance / 100.0f) * settings_.acceleration_factor;
        speed_multiplier = std::min(speed_multiplier, 5.0f); // 최대 5배속
        
        return {
            error_x * settings_.snap_speed * speed_multiplier,
            error_y * settings_.snap_speed * speed_multiplier
        };
    }
    
    // 설정 업데이트
    void updateSettings(const Settings& new_settings) {
        settings_ = new_settings;
    }
    
    // 타겟 거리에 따른 우선순위 계산 (가장 가까운 적 선택용)
    static float calculatePriority(float error_x, float error_y) {
        return error_x * error_x + error_y * error_y; // 거리의 제곱 (계산 효율성)
    }

private:
    Settings settings_;
};

#endif // SNAP_AIM_CONTROLLER_H