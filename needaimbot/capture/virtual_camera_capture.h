#ifndef VIRTUAL_CAMERA_CAPTURE_H
#define VIRTUAL_CAMERA_CAPTURE_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>
#include "capture.h"

class VirtualCameraCapture : public IScreenCapture
{
public:
    VirtualCameraCapture(int width, int height, int device_id = 0);
    ~VirtualCameraCapture();
    
    bool IsInitialized() const { return initialized_; }
    
    cv::cuda::GpuMat GetNextFrameGpu() override;
    cv::Mat GetNextFrameCpu() override;
    cudaEvent_t GetCaptureDoneEvent() const override;
    
    // Virtual camera specific methods
    bool SetResolution(int width, int height);
    void SetFrameRate(double fps);
    std::vector<std::string> ListAvailableDevices() const;

private:
    bool initializeCamera();
    void cleanupResources();
    bool processFrame(cv::Mat& frame);
    
    cv::VideoCapture capture_;
    cv::cuda::GpuMat gpu_frame_;
    cv::Mat cpu_frame_;
    
    int width_;
    int height_;
    int device_id_;
    double target_fps_;
    
    std::atomic<bool> initialized_{false};
    std::atomic<bool> should_stop_{false};
    
    mutable std::mutex frame_mutex_;
    cudaEvent_t capture_event_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point last_frame_time_;
    std::atomic<float> current_fps_{0.0f};
};

#endif // VIRTUAL_CAMERA_CAPTURE_H