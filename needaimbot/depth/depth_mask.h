#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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
    DepthMaskGenerator();
    ~DepthMaskGenerator();

    // Update mask from CPU RGB image
    void update(const uint8_t* frameRgb, int width, int height,
                const DepthMaskOptions& options,
                const std::string& modelPath, nvinfer1::ILogger& logger);
    
    // Update mask from GPU RGBA image
    void updateGpu(const uint8_t* d_frameRgba, int width, int height, int pitch,
                   const DepthMaskOptions& options,
                   const std::string& modelPath, nvinfer1::ILogger& logger,
                   cudaStream_t stream = nullptr);
    
    // Get the binary mask (0 or 255, same size as input frame)
    // Returns true if mask is available
    bool getMask(std::vector<uint8_t>& outMask, int& outWidth, int& outHeight) const;
    
    // Get normalized depth map (0-255)
    bool getDepthMap(std::vector<uint8_t>& outDepth, int& outWidth, int& outHeight) const;
    
    // Check if a point (in frame coordinates) is in the "near" region
    bool isPointNear(int x, int y) const;
    
    // Get depth value at a point (0-255, lower = nearer by default)
    uint8_t getDepthAt(int x, int y) const;
    
    bool ready() const;
    std::string lastError() const;
    std::chrono::steady_clock::time_point lastAttemptTime() const;
    std::pair<int, int> lastFrameSize() const;
    DepthMaskDebugState debugState() const;
    void reset();

private:
    mutable std::mutex state_mutex;
    std::vector<uint8_t> depth_map;      // Normalized depth (0-255)
    std::vector<uint8_t> mask_binary;    // Binary mask (0 or 255)
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
    
    // Internal: generate binary mask from depth map
    void generateMask(int near_percent, bool invert);
};

DepthMaskGenerator& GetDepthMaskGenerator();

} // namespace depth_anything

#endif // USE_CUDA
