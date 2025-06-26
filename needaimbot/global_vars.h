#ifndef GLOBAL_VARS_H
#define GLOBAL_VARS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>

// Capture buffers
extern std::vector<cv::cuda::GpuMat> captureGpuBuffer;
extern std::atomic<int> captureGpuWriteIdx;
extern std::vector<cv::Mat> captureCpuBuffer;
extern std::atomic<int> captureCpuWriteIdx;

// Frame synchronization
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> newFrameAvailable;

// Detection synchronization
extern std::mutex detectionMutex;
extern std::condition_variable detectionCV;

#endif // GLOBAL_VARS_H
