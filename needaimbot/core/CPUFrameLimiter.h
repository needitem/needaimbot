#pragma once

#include <chrono>
#include <thread>
#include <atomic>

namespace needaimbot {

class CPUFrameLimiter {
private:
    std::chrono::steady_clock::time_point last_frame;
    std::chrono::microseconds target_frame_time;
    std::atomic<int> target_fps;
    std::atomic<bool> enabled;
    
    // Performance metrics
    std::atomic<float> actual_fps;
    std::atomic<float> gpu_utilization;
    std::chrono::steady_clock::time_point fps_calc_start;
    std::atomic<int> frame_count;
    
public:
    CPUFrameLimiter(int fps = 120) 
        : target_fps(fps)
        , target_frame_time(1000000 / fps)
        , enabled(true)
        , actual_fps(0.0f)
        , gpu_utilization(0.0f)
        , frame_count(0) {
        last_frame = std::chrono::steady_clock::now();
        fps_calc_start = last_frame;
    }
    
    // Limit frame rate to reduce GPU usage
    void limitFrame() {
        if (!enabled) return;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame);
        
        // If we're ahead of schedule, sleep to limit FPS
        if (elapsed < target_frame_time) {
            // Use a combination of sleep and busy wait for precision
            auto sleep_time = target_frame_time - elapsed - std::chrono::microseconds(100);
            
            if (sleep_time > std::chrono::microseconds(0)) {
                std::this_thread::sleep_for(sleep_time);
            }
            
            // Busy wait for the remaining time for precision
            while (std::chrono::steady_clock::now() - last_frame < target_frame_time) {
                std::this_thread::yield();
            }
        }
        
        // Update frame timing
        last_frame = std::chrono::steady_clock::now();
        
        // Calculate actual FPS every second
        frame_count++;
        auto fps_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            last_frame - fps_calc_start).count();
        
        if (fps_elapsed >= 1000) {
            actual_fps = static_cast<float>(frame_count) * 1000.0f / fps_elapsed;
            frame_count = 0;
            fps_calc_start = last_frame;
        }
    }
    
    // Dynamic FPS adjustment based on GPU utilization
    void adjustFPSBasedOnGPU(float gpu_util) {
        gpu_utilization = gpu_util;
        
        // If GPU is heavily loaded, reduce FPS
        if (gpu_util > 90.0f && target_fps > 60) {
            setTargetFPS(target_fps - 10);
        }
        // If GPU has headroom, increase FPS
        else if (gpu_util < 70.0f && target_fps < 120) {
            setTargetFPS(target_fps + 10);
        }
    }
    
    void setTargetFPS(int fps) {
        if (fps > 0 && fps <= 240) {
            target_fps = fps;
            target_frame_time = std::chrono::microseconds(1000000 / fps);
        }
    }
    
    int getTargetFPS() const { return target_fps; }
    float getActualFPS() const { return actual_fps; }
    float getGPUUtilization() const { return gpu_utilization; }
    
    void enable(bool state) { enabled = state; }
    bool isEnabled() const { return enabled; }
    
    // Get frame time statistics
    std::chrono::microseconds getTargetFrameTime() const { return target_frame_time; }
    
    // Reset timing (useful after long operations)
    void reset() {
        last_frame = std::chrono::steady_clock::now();
        fps_calc_start = last_frame;
        frame_count = 0;
    }
};

} // namespace needaimbot