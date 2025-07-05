#ifndef APP_CONTEXT_H
#define APP_CONTEXT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "config/config.h"

class MouseThread; // Forward declaration
class Detector;
class OpticalFlow;

struct AppContext {
public:
    AppContext(const AppContext&) = delete;
    AppContext& operator=(const AppContext&) = delete;

    static AppContext& getInstance() {
        static AppContext instance;
        return instance;
    }

    // Config
    Config config;

    // Capture buffers
    std::vector<cv::cuda::GpuMat> captureGpuBuffer;
    std::atomic<int> captureGpuWriteIdx{0};
    std::vector<cv::Mat> captureCpuBuffer;
    std::atomic<int> captureCpuWriteIdx{0};

    // Frame synchronization
    std::mutex frameMutex;
    std::condition_variable frameCV;
    std::atomic<bool> newFrameAvailable{false};

    // Detection synchronization
    std::mutex detectionMutex;
    std::condition_variable detectionCV;

    // Application state
    std::atomic<bool> shouldExit{false};
    std::atomic<bool> shooting{false};
    std::atomic<bool> zooming{false};
    std::atomic<bool> config_optical_flow_changed{false};
    std::atomic<bool> input_method_changed{false};

    // Modules
    MouseThread* globalMouseThread = nullptr;
    Detector detector;
    OpticalFlow opticalFlow;

private:
    AppContext() : captureGpuWriteIdx(0), captureCpuWriteIdx(0), newFrameAvailable(false) {}
};

#endif // APP_CONTEXT_H
