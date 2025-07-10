#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <mutex>
#include <memory>

class IScreenCapture;

// Encapsulates all capture-related global state
class CaptureContext {
public:
    static CaptureContext& getInstance() {
        static CaptureContext instance;
        return instance;
    }

    // Delete copy constructor and assignment operator
    CaptureContext(const CaptureContext&) = delete;
    CaptureContext& operator=(const CaptureContext&) = delete;

    // Frame buffers
    cv::cuda::GpuMat& getLatestFrameGpu() { 
        std::lock_guard<std::mutex> lock(frameMutex_);
        return latestFrameGpu_; 
    }
    
    cv::Mat& getLatestFrameCpu() { 
        std::lock_guard<std::mutex> lock(frameMutex_);
        return latestFrameCpu_; 
    }

    void setLatestFrameGpu(const cv::cuda::GpuMat& frame) {
        std::lock_guard<std::mutex> lock(frameMutex_);
        latestFrameGpu_ = frame;
    }

    void setLatestFrameCpu(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(frameMutex_);
        latestFrameCpu_ = frame;
    }

    // Capture instance management
    std::shared_ptr<IScreenCapture> getCapturer() {
        std::lock_guard<std::mutex> lock(capturerMutex_);
        return capturer_;
    }

    void setCapturer(std::shared_ptr<IScreenCapture> newCapturer) {
        std::lock_guard<std::mutex> lock(capturerMutex_);
        capturer_ = newCapturer;
    }

    // Frame synchronization
    std::mutex& getFrameMutex() { return frameMutex_; }
    std::mutex& getCapturerMutex() { return capturerMutex_; }

private:
    CaptureContext() = default;
    ~CaptureContext() = default;

    // Frame buffers
    cv::cuda::GpuMat latestFrameGpu_;
    cv::Mat latestFrameCpu_;
    std::mutex frameMutex_;

    // Capture instance
    std::shared_ptr<IScreenCapture> capturer_;
    std::mutex capturerMutex_;
};