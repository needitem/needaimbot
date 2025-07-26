#ifndef CAPTURE_H
#define CAPTURE_H

#include "../cuda/simple_cuda_mat.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <array>
#include <cuda_runtime.h>

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

extern SimpleCudaMat latestFrameGpu;
extern SimpleMat latestFrameCpu;


constexpr int FRAME_BUFFER_COUNT = 4;
extern std::array<SimpleCudaMat, FRAME_BUFFER_COUNT> captureGpuBuffer;
extern std::array<SimpleMat, FRAME_BUFFER_COUNT> captureCpuBuffer;
extern std::atomic<int> captureGpuWriteIdx;
extern std::atomic<int> captureCpuWriteIdx;

extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> should_exit;
extern std::atomic<bool> show_window_changed;
extern std::atomic<bool> newFrameAvailable;


typedef struct CUevent_st* cudaEvent_t;

class IScreenCapture
{
public:
    virtual ~IScreenCapture() = default; 
    virtual SimpleCudaMat GetNextFrameGpu() = 0;
    virtual SimpleMat GetNextFrameCpu() = 0;
    virtual cudaEvent_t GetCaptureDoneEvent() const = 0; 
};

#endif 