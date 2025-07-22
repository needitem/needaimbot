#ifndef NDI_CAPTURE_H
#define NDI_CAPTURE_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include "capture.h"

// NDI SDK includes (if available)
#ifdef NDI_SDK_AVAILABLE
#include <Processing.NDI.Lib.h>
#endif

class NDICapture : public IScreenCapture
{
public:
    NDICapture(int width, int height, const std::string& source_name = "");
    ~NDICapture();
    
    bool IsInitialized() const { return initialized_; }
    
    cv::cuda::GpuMat GetNextFrameGpu() override;
    cv::Mat GetNextFrameCpu() override;
    cudaEvent_t GetCaptureDoneEvent() const override;
    
    // NDI specific methods
    std::vector<std::string> FindNDISources() const;
    bool ConnectToSource(const std::string& source_name);
    void SetLowLatencyMode(bool enable);
    float GetCurrentBandwidth() const;
    
    // Fallback to network stream when NDI SDK is not available
    bool ConnectToNetworkStream(const std::string& url);

private:
    bool initializeNDI();
    bool initializeNetworkStream();
    void cleanupResources();
    bool processFrame(cv::Mat& frame);
    void captureLoop();
    
#ifdef NDI_SDK_AVAILABLE
    NDIlib_recv_instance_t ndi_receiver_;
    NDIlib_find_instance_t ndi_finder_;
    std::vector<NDIlib_source_t> ndi_sources_;
#endif
    
    // Network stream fallback
    cv::VideoCapture network_capture_;
    std::string network_url_;
    
    cv::cuda::GpuMat gpu_frame_;
    cv::Mat cpu_frame_;
    
    int width_;
    int height_;
    std::string source_name_;
    bool low_latency_mode_;
    
    std::atomic<bool> initialized_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> use_network_fallback_{false};
    
    mutable std::mutex frame_mutex_;
    std::mutex source_mutex_;
    cudaEvent_t capture_event_;
    
    std::thread capture_thread_;
    
    // Performance tracking
    std::atomic<float> current_fps_{0.0f};
    std::atomic<float> bandwidth_mbps_{0.0f};
    std::chrono::steady_clock::time_point last_frame_time_;
};

#endif // NDI_CAPTURE_H