#include "DetectionState.h"
#include <algorithm>

namespace Core {
    void DetectionState::updateTargets(const std::vector<Target>& targets) {
        std::lock_guard<std::mutex> lock(targetMutex_);
        allTargets_ = targets;
        hasTarget_ = !targets.empty();
        
        // 가장 좋은 타겟을 현재 타겟으로 설정
        if (!targets.empty()) {
            // confidence가 가장 높은 타겟을 선택
            auto bestTarget = std::max_element(targets.begin(), targets.end(),
                [](const Target& a, const Target& b) {
                    return a.confidence < b.confidence;
                });
            currentTarget_ = *bestTarget;
        }
    }
    
    void DetectionState::updateCurrentTarget(const Target& target) {
        std::lock_guard<std::mutex> lock(targetMutex_);
        currentTarget_ = target;
        hasTarget_ = true;
    }
    
    Target DetectionState::getBestTarget() const {
        std::lock_guard<std::mutex> lock(targetMutex_);
        return currentTarget_;
    }
    
    std::vector<Target> DetectionState::getAllTargets() const {
        std::lock_guard<std::mutex> lock(targetMutex_);
        return allTargets_;
    }
    
    void DetectionState::clearTargets() {
        std::lock_guard<std::mutex> lock(targetMutex_);
        allTargets_.clear();
        hasTarget_ = false;
        currentTarget_ = Target{}; // 기본값으로 리셋
    }
    
    void DetectionState::updateOverlayTarget(const Target& target) {
        std::lock_guard<std::mutex> lock(overlayTargetMutex_);
        overlayTargetInfo_ = target;
        overlayHasTarget_ = true;
    }
    
    Target DetectionState::getOverlayTarget() const {
        std::lock_guard<std::mutex> lock(overlayTargetMutex_);
        return overlayTargetInfo_;
    }
    
    void DetectionState::clearOverlayTarget() {
        std::lock_guard<std::mutex> lock(overlayTargetMutex_);
        overlayTargetInfo_ = Target{};
        overlayHasTarget_ = false;
    }
    
    bool DetectionState::checkAndResetModelChange() {
        return modelChanged_.exchange(false);
    }
}