#pragma once
#ifdef USE_CUDA

#include <chrono>
#include <mutex>
#include <string>
#include <utility>

namespace nvinfer1 {
    class ILogger;
}

namespace depth_anything {

class DepthAnythingTrt;

struct DepthMaskOptions {
    bool enabled = false;
    int fps = 5;
    int near_percent = 20;
    bool invert = false;
};

struct DepthMaskDebugState {
    bool initialized = false;
    bool has_model = false;
    bool model_ready = false;
    std::string last_model_path;
};

class DepthMaskGenerator {
public:
    void update(const float* frameRgb, int width, int height,
                const DepthMaskOptions& options,
                const std::string& modelPath, nvinfer1::ILogger& logger);
    
    // Get the binary mask (0 or 255, same size as input)
    bool getMask(unsigned char* outMask, int& outWidth, int& outHeight) const;
    bool ready() const;
    std::string lastError() const;
    std::chrono::steady_clock::time_point lastAttemptTime() const;
    std::pair<int, int> lastFrameSize() const;
    DepthMaskDebugState debugState() const;
    void reset();

private:
    mutable std::mutex state_mutex;
    std::vector<unsigned char> mask_binary;
    int mask_width = 0;
    int mask_height = 0;
    std::chrono::steady_clock::time_point last_update = std::chrono::steady_clock::time_point::min();
    std::chrono::steady_clock::time_point last_attempt = std::chrono::steady_clock::time_point::min();
    int last_frame_w = 0;
    int last_frame_h = 0;
    std::string last_model_path;
    std::string last_error;
    bool initialized = false;

    DepthAnythingTrt* model = nullptr;
};

DepthMaskGenerator& GetDepthMaskGenerator();

} // namespace depth_anything

#endif // USE_CUDA
