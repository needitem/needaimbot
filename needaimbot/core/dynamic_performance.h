#pragma once

#include "../AppContext.h"
#include <chrono>
#include <atomic>

class DynamicPerformanceManager {
public:
    static DynamicPerformanceManager& getInstance() {
        static DynamicPerformanceManager instance;
        return instance;
    }

    void update() {
        auto& ctx = AppContext::getInstance();
        auto now = std::chrono::steady_clock::now();
        
        // 타겟이 없을 때 성능 설정 낮추기
        if (!ctx.overlay_has_target.load()) {
            if (now - last_target_time > std::chrono::seconds(2)) {
                // 2초 이상 타겟 없음 - 저성능 모드
                setLowPerformanceMode();
            }
        } else {
            // 타겟 있음 - 고성능 모드
            last_target_time = now;
            setHighPerformanceMode();
        }
        
        // CPU 사용률에 따른 동적 조절
        float cpu_usage = ctx.g_current_total_cycle_time_ms.load();
        if (cpu_usage > 10.0f) {  // 10ms 이상
            reducePerformance();
        } else if (cpu_usage < 5.0f) {  // 5ms 이하
            increasePerformance();
        }
    }

private:
    void setLowPerformanceMode() {
        auto& ctx = AppContext::getInstance();
        
        // 캡처 FPS 낮추기
        if (ctx.config.capture_fps > 30) {
            ctx.config.capture_fps = 30;
            ctx.capture_fps_changed = true;
        }
        
        // 탐지 해상도 낮추기
        if (ctx.config.detection_resolution > 320) {
            ctx.config.detection_resolution = 320;
            ctx.detection_resolution_changed = true;
        }
        
        // 캡처 영역 줄이기
        if (capture_width > 400) {
            capture_width = 400;
            capture_height = 400;
            ctx.capture_borders_changed = true;
        }
    }
    
    void setHighPerformanceMode() {
        auto& ctx = AppContext::getInstance();
        
        // 원래 설정으로 복원
        if (ctx.config.capture_fps < original_fps) {
            ctx.config.capture_fps = original_fps;
            ctx.capture_fps_changed = true;
        }
        
        if (ctx.config.detection_resolution < 640) {
            ctx.config.detection_resolution = 640;
            ctx.detection_resolution_changed = true;
        }
        
        if (capture_width < 640) {
            capture_width = 640;
            capture_height = 640;
            ctx.capture_borders_changed = true;
        }
    }
    
    void reducePerformance() {
        auto& ctx = AppContext::getInstance();
        
        // FPS를 10 단위로 감소
        if (ctx.config.capture_fps > 30) {
            ctx.config.capture_fps -= 10;
            ctx.capture_fps_changed = true;
        }
    }
    
    void increasePerformance() {
        auto& ctx = AppContext::getInstance();
        
        // FPS를 10 단위로 증가 (최대값까지)
        if (ctx.config.capture_fps < original_fps) {
            ctx.config.capture_fps = std::min(ctx.config.capture_fps + 10.0f, original_fps);
            ctx.capture_fps_changed = true;
        }
    }

private:
    std::chrono::steady_clock::time_point last_target_time;
    float original_fps = 60.0f;
    int capture_width = 640;
    int capture_height = 640;
    
    DynamicPerformanceManager() {
        auto& ctx = AppContext::getInstance();
        original_fps = ctx.config.capture_fps;
        last_target_time = std::chrono::steady_clock::now();
    }
};