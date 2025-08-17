#pragma once
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <string>

namespace Core {
    class PerformanceMetrics {
    private:
        struct Metric {
            std::atomic<float> current{0.0f};
            std::vector<float> history;
            mutable std::mutex historyMutex;
            size_t maxHistorySize = 100;
            
            void update(float value);
            float getAverage() const;
            float getMin() const;
            float getMax() const;
        };
        
        std::unordered_map<std::string, std::unique_ptr<Metric>> metrics_;
        mutable std::mutex metricsMapMutex_;
        
    public:
        PerformanceMetrics();
        ~PerformanceMetrics() = default;
        
        // 메트릭 등록 및 업데이트
        void registerMetric(const std::string& name);
        void updateMetric(const std::string& name, float value);
        
        // 조회
        float getCurrentValue(const std::string& name) const;
        std::vector<float> getHistory(const std::string& name) const;
        
        // 통계
        struct Stats {
            float current = 0.0f;
            float average = 0.0f;
            float min = 0.0f;
            float max = 0.0f;
        };
        Stats getStats(const std::string& name) const;
        
        // 타이머 헬퍼
        class ScopedTimer {
            PerformanceMetrics& metrics_;
            std::string metricName_;
            std::chrono::high_resolution_clock::time_point start_;
            
        public:
            ScopedTimer(PerformanceMetrics& metrics, const std::string& name);
            ~ScopedTimer();
        };
        
        // 기본 메트릭들을 위한 편의 함수들
        void updateFrameAcquisitionTime(float ms) { updateMetric("frame_acquisition_time", ms); }
        void updateCaptureFps(float fps) { updateMetric("capture_fps", fps); }
        void updateCaptureCycleTime(float ms) { updateMetric("capture_cycle_time", ms); }
        void updateProcessFrameTime(float ms) { updateMetric("process_frame_time", ms); }
        void updateDetectorCycleTime(float ms) { updateMetric("detector_cycle_time", ms); }
        void updateInferenceTime(float ms) { updateMetric("inference_time", ms); }
        void updateInputSendTime(float ms) { updateMetric("input_send_time", ms); }
        void updateDetectionToMovementTime(float ms) { updateMetric("detection_to_movement_time", ms); }
        void updateFpsDelayTime(float ms) { updateMetric("fps_delay_time", ms); }
        void updateTotalCycleTime(float ms) { updateMetric("total_cycle_time", ms); }
        
        // 현재 값 조회 편의 함수들
        float getCurrentFrameAcquisitionTime() const { return getCurrentValue("frame_acquisition_time"); }
        float getCurrentCaptureFps() const { return getCurrentValue("capture_fps"); }
        float getCurrentCaptureCycleTime() const { return getCurrentValue("capture_cycle_time"); }
        float getCurrentProcessFrameTime() const { return getCurrentValue("process_frame_time"); }
        float getCurrentDetectorCycleTime() const { return getCurrentValue("detector_cycle_time"); }
        float getCurrentInferenceTime() const { return getCurrentValue("inference_time"); }
        float getCurrentInputSendTime() const { return getCurrentValue("input_send_time"); }
        float getCurrentDetectionToMovementTime() const { return getCurrentValue("detection_to_movement_time"); }
        float getCurrentFpsDelayTime() const { return getCurrentValue("fps_delay_time"); }
        float getCurrentTotalCycleTime() const { return getCurrentValue("total_cycle_time"); }
    };
}