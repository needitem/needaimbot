#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <array>

extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);


extern int g_captureRegionWidth;
extern int g_captureRegionHeight;

extern std::atomic<int> captureFrameCount;
extern std::atomic<int> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

extern cv::cuda::GpuMat latestFrameGpu;
extern cv::Mat latestFrameCpu;


constexpr int FRAME_BUFFER_COUNT = 4;
extern std::array<cv::cuda::GpuMat, FRAME_BUFFER_COUNT> captureGpuBuffer;
extern std::array<cv::Mat, FRAME_BUFFER_COUNT> captureCpuBuffer;
extern std::atomic<int> captureGpuWriteIdx;
extern std::atomic<int> captureCpuWriteIdx;

extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;
extern std::atomic<bool> show_window_changed;
extern std::atomic<bool> newFrameAvailable;


typedef struct CUevent_st* cudaEvent_t;

class IScreenCapture
{
public:
    virtual ~IScreenCapture() = default; 
    virtual cv::cuda::GpuMat GetNextFrameGpu() = 0;
    virtual cv::Mat GetNextFrameCpu() = 0;
    virtual cudaEvent_t GetCaptureDoneEvent() const = 0; 
};

#endif 
