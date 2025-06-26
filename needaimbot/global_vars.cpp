#include "global_vars.h"

// Capture buffers
std::vector<cv::cuda::GpuMat> captureGpuBuffer;
std::atomic<int> captureGpuWriteIdx(0);
std::vector<cv::Mat> captureCpuBuffer;
std::atomic<int> captureCpuWriteIdx(0);

// Frame synchronization
std::mutex frameMutex;
std::condition_variable frameCV;
std::atomic<bool> newFrameAvailable(false);

// Detection synchronization
std::mutex detectionMutex;
std::condition_variable detectionCV;
