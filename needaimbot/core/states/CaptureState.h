#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include "../../cuda/simple_cuda_mat.h"

namespace Core {
    class CaptureState {
    private:
        // 캡처 버퍼
        std::vector<SimpleCudaMat> gpuBuffers_;
        std::vector<SimpleMat> cpuBuffers_;
        std::atomic<int> gpuWriteIdx_{0};
        std::atomic<int> cpuWriteIdx_{0};
        
        // 동기화
        mutable std::mutex frameMutex_;
        std::condition_variable frameCV_;
        std::atomic<bool> frameReady_{false};
        
        // 설정 변경 플래그
        std::atomic<bool> resolutionChanged_{false};
        std::atomic<bool> methodChanged_{false};
        std::atomic<bool> fpsChanged_{false};
        std::atomic<bool> cursorChanged_{false};
        std::atomic<bool> bordersChanged_{false};
        std::atomic<bool> timeoutChanged_{false};
        std::atomic<bool> crosshairOffsetChanged_{false};
        
        // 캡처 방법 (0 = Desktop Duplication, 1 = Region Capture)
        std::atomic<int> captureMethod_{0};
        
    public:
        CaptureState(size_t bufferCount = 4);
        ~CaptureState() = default;
        
        // 버퍼 접근
        SimpleCudaMat& getGpuWriteBuffer();
        const SimpleCudaMat& getGpuReadBuffer() const;
        SimpleMat& getCpuWriteBuffer();
        const SimpleMat& getCpuReadBuffer() const;
        
        void swapGpuBuffers();
        void swapCpuBuffers();
        
        // 동기화
        void notifyFrameReady();
        bool waitForFrame(std::chrono::milliseconds timeout);
        void resetFrameReady() { frameReady_ = false; }
        bool isFrameReady() const { return frameReady_.load(); }
        
        // 설정 변경 관리
        void markResolutionChanged() { resolutionChanged_ = true; }
        bool checkAndResetResolutionChange();
        
        void markMethodChanged() { methodChanged_ = true; }
        bool checkAndResetMethodChange();
        
        void markFpsChanged() { fpsChanged_ = true; }
        bool checkAndResetFpsChange();
        
        void markCursorChanged() { cursorChanged_ = true; }
        bool checkAndResetCursorChange();
        
        void markBordersChanged() { bordersChanged_ = true; }
        bool checkAndResetBordersChange();
        
        void markTimeoutChanged() { timeoutChanged_ = true; }
        bool checkAndResetTimeoutChange();
        
        void markCrosshairOffsetChanged() { crosshairOffsetChanged_ = true; }
        bool checkAndResetCrosshairOffsetChange();
        
        // 캡처 방법 관리
        void setCaptureMethod(int method) { captureMethod_ = method; }
        int getCaptureMethod() const { return captureMethod_.load(); }
        
        // 버퍼 크기 관리
        size_t getBufferCount() const { return gpuBuffers_.size(); }
        void resizeBuffers(size_t newSize);
    };
}