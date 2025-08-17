#pragma once
#include <atomic>
#include <mutex>
#include <vector>
#include "../../core/Target.h"

namespace Core {
    class DetectionState {
    private:
        // 타겟 정보
        mutable std::mutex targetMutex_;
        Target currentTarget_;
        std::vector<Target> allTargets_;
        std::atomic<bool> hasTarget_{false};
        
        // 오버레이용 타겟 정보 (UI 동기화용)
        std::atomic<bool> overlayHasTarget_{false};
        Target overlayTargetInfo_{};
        mutable std::mutex overlayTargetMutex_;
        
        // 상태 플래그
        std::atomic<bool> detectionPaused_{false};
        std::atomic<bool> modelChanged_{false};
        
        // 성능 메트릭
        std::atomic<float> inferenceTime_{0.0f};
        std::atomic<float> postProcessTime_{0.0f};
        std::atomic<float> processFrameTime_{0.0f};
        std::atomic<float> detectorCycleTime_{0.0f};
        std::atomic<float> detectionToMovementTime_{0.0f};
        std::atomic<float> totalCycleTime_{0.0f};
        
    public:
        DetectionState() = default;
        ~DetectionState() = default;
        
        // 타겟 관리
        void updateTargets(const std::vector<Target>& targets);
        void updateCurrentTarget(const Target& target);
        Target getBestTarget() const;
        std::vector<Target> getAllTargets() const;
        bool hasValidTarget() const { return hasTarget_.load(); }
        void clearTargets();
        
        // 오버레이 타겟 정보
        void updateOverlayTarget(const Target& target);
        Target getOverlayTarget() const;
        bool hasOverlayTarget() const { return overlayHasTarget_.load(); }
        void clearOverlayTarget();
        
        // 상태 관리
        void pauseDetection() { detectionPaused_ = true; }
        void resumeDetection() { detectionPaused_ = false; }
        bool isPaused() const { return detectionPaused_.load(); }
        
        void markModelChanged() { modelChanged_ = true; }
        bool checkAndResetModelChange();
        
        // 성능 메트릭
        void setInferenceTime(float ms) { inferenceTime_ = ms; }
        float getInferenceTime() const { return inferenceTime_.load(); }
        
        void setPostProcessTime(float ms) { postProcessTime_ = ms; }
        float getPostProcessTime() const { return postProcessTime_.load(); }
        
        void setProcessFrameTime(float ms) { processFrameTime_ = ms; }
        float getProcessFrameTime() const { return processFrameTime_.load(); }
        
        void setDetectorCycleTime(float ms) { detectorCycleTime_ = ms; }
        float getDetectorCycleTime() const { return detectorCycleTime_.load(); }
        
        void setDetectionToMovementTime(float ms) { detectionToMovementTime_ = ms; }
        float getDetectionToMovementTime() const { return detectionToMovementTime_.load(); }
        
        void setTotalCycleTime(float ms) { totalCycleTime_ = ms; }
        float getTotalCycleTime() const { return totalCycleTime_.load(); }
    };
}