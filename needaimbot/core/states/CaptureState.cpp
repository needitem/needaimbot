#include "CaptureState.h"

namespace Core {
    CaptureState::CaptureState(size_t bufferCount) {
        gpuBuffers_.resize(bufferCount);
        cpuBuffers_.resize(bufferCount);
    }
    
    SimpleCudaMat& CaptureState::getGpuWriteBuffer() {
        return gpuBuffers_[gpuWriteIdx_.load()];
    }
    
    const SimpleCudaMat& CaptureState::getGpuReadBuffer() const {
        // 읽기 버퍼는 쓰기 버퍼의 이전 인덱스
        int readIdx = (gpuWriteIdx_.load() - 1 + gpuBuffers_.size()) % gpuBuffers_.size();
        return gpuBuffers_[readIdx];
    }
    
    SimpleMat& CaptureState::getCpuWriteBuffer() {
        return cpuBuffers_[cpuWriteIdx_.load()];
    }
    
    const SimpleMat& CaptureState::getCpuReadBuffer() const {
        // 읽기 버퍼는 쓰기 버퍼의 이전 인덱스
        int readIdx = (cpuWriteIdx_.load() - 1 + cpuBuffers_.size()) % cpuBuffers_.size();
        return cpuBuffers_[readIdx];
    }
    
    void CaptureState::swapGpuBuffers() {
        gpuWriteIdx_ = (gpuWriteIdx_ + 1) % gpuBuffers_.size();
        frameReady_ = true;
        frameCV_.notify_one();
    }
    
    void CaptureState::swapCpuBuffers() {
        cpuWriteIdx_ = (cpuWriteIdx_ + 1) % cpuBuffers_.size();
    }
    
    void CaptureState::notifyFrameReady() {
        {
            std::lock_guard<std::mutex> lock(frameMutex_);
            frameReady_ = true;
        }
        frameCV_.notify_one();
    }
    
    bool CaptureState::waitForFrame(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(frameMutex_);
        return frameCV_.wait_for(lock, timeout, 
            [this] { return frameReady_.load(); });
    }
    
    bool CaptureState::checkAndResetResolutionChange() {
        return resolutionChanged_.exchange(false);
    }
    
    bool CaptureState::checkAndResetMethodChange() {
        return methodChanged_.exchange(false);
    }
    
    bool CaptureState::checkAndResetFpsChange() {
        return fpsChanged_.exchange(false);
    }
    
    bool CaptureState::checkAndResetCursorChange() {
        return cursorChanged_.exchange(false);
    }
    
    bool CaptureState::checkAndResetBordersChange() {
        return bordersChanged_.exchange(false);
    }
    
    bool CaptureState::checkAndResetTimeoutChange() {
        return timeoutChanged_.exchange(false);
    }
    
    bool CaptureState::checkAndResetCrosshairOffsetChange() {
        return crosshairOffsetChanged_.exchange(false);
    }
    
    void CaptureState::resizeBuffers(size_t newSize) {
        std::lock_guard<std::mutex> lock(frameMutex_);
        gpuBuffers_.resize(newSize);
        cpuBuffers_.resize(newSize);
        
        // 인덱스가 유효한 범위를 벗어나면 리셋
        if (gpuWriteIdx_.load() >= static_cast<int>(newSize)) {
            gpuWriteIdx_ = 0;
        }
        if (cpuWriteIdx_.load() >= static_cast<int>(newSize)) {
            cpuWriteIdx_ = 0;
        }
    }
}