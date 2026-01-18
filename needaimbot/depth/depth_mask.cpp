#include "depth_mask.h"
#include "depth_anything_trt.h"

#ifdef USE_CUDA

#include <algorithm>
#include <cmath>

namespace depth_anything {

// Global singleton instance
static DepthMaskGenerator g_depthMaskGenerator;

DepthMaskGenerator& GetDepthMaskGenerator() {
    return g_depthMaskGenerator;
}

void DepthMaskGenerator::update(const float* frameRgb, int width, int height,
                                 const DepthMaskOptions& options,
                                 const std::string& modelPath, nvinfer1::ILogger& logger) {
    std::lock_guard<std::mutex> lock(state_mutex);
    
    last_attempt = std::chrono::steady_clock::now();
    last_frame_w = width;
    last_frame_h = height;

    if (!options.enabled) {
        return;
    }

    // Rate limiting based on FPS setting
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
    int targetInterval = 1000 / std::max(1, options.fps);
    
    if (elapsed < targetInterval) {
        return;
    }

    // Initialize model if needed
    if (!model && !modelPath.empty()) {
        model = new DepthAnythingTrt();
        if (!model->initialize(modelPath, logger)) {
            last_error = model->lastError();
            delete model;
            model = nullptr;
            return;
        }
        last_model_path = modelPath;
        initialized = true;
    }

    if (!model || !model->ready()) {
        last_error = "Depth model not ready";
        return;
    }

    // Run depth estimation
    std::vector<float> depthMap(width * height);
    int outW, outH;
    
    if (!model->predictDepth(frameRgb, width, height, depthMap.data(), outW, outH)) {
        last_error = model->lastError();
        return;
    }

    // Generate binary mask based on near_percent threshold
    // Find min/max depth values
    float minDepth = *std::min_element(depthMap.begin(), depthMap.end());
    float maxDepth = *std::max_element(depthMap.begin(), depthMap.end());
    float range = maxDepth - minDepth;
    
    if (range < 1e-6f) {
        // All same depth, return full mask
        mask_binary.resize(outW * outH, 255);
        mask_width = outW;
        mask_height = outH;
        last_update = now;
        return;
    }

    // Calculate threshold based on near_percent
    float threshold;
    if (options.invert) {
        // Far objects (higher depth values)
        threshold = minDepth + range * (1.0f - options.near_percent / 100.0f);
    } else {
        // Near objects (lower depth values)
        threshold = minDepth + range * (options.near_percent / 100.0f);
    }

    // Generate binary mask
    mask_binary.resize(outW * outH);
    for (int i = 0; i < outW * outH; ++i) {
        bool isNear;
        if (options.invert) {
            isNear = depthMap[i] >= threshold;  // Far objects
        } else {
            isNear = depthMap[i] <= threshold;  // Near objects
        }
        mask_binary[i] = isNear ? 255 : 0;
    }

    mask_width = outW;
    mask_height = outH;
    last_update = now;
    last_error.clear();
}

bool DepthMaskGenerator::getMask(unsigned char* outMask, int& outWidth, int& outHeight) const {
    std::lock_guard<std::mutex> lock(state_mutex);
    
    if (mask_binary.empty()) {
        return false;
    }

    outWidth = mask_width;
    outHeight = mask_height;
    std::copy(mask_binary.begin(), mask_binary.end(), outMask);
    return true;
}

bool DepthMaskGenerator::ready() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    return initialized && model && model->ready() && !mask_binary.empty();
}

std::string DepthMaskGenerator::lastError() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    return last_error;
}

std::chrono::steady_clock::time_point DepthMaskGenerator::lastAttemptTime() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    return last_attempt;
}

std::pair<int, int> DepthMaskGenerator::lastFrameSize() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    return {last_frame_w, last_frame_h};
}

DepthMaskDebugState DepthMaskGenerator::debugState() const {
    std::lock_guard<std::mutex> lock(state_mutex);
    DepthMaskDebugState state;
    state.initialized = initialized;
    state.has_model = (model != nullptr);
    state.model_ready = (model && model->ready());
    state.last_model_path = last_model_path;
    return state;
}

void DepthMaskGenerator::reset() {
    std::lock_guard<std::mutex> lock(state_mutex);
    
    if (model) {
        model->reset();
        delete model;
        model = nullptr;
    }
    
    mask_binary.clear();
    mask_width = 0;
    mask_height = 0;
    last_update = std::chrono::steady_clock::time_point::min();
    last_attempt = std::chrono::steady_clock::time_point::min();
    last_model_path.clear();
    last_error.clear();
    initialized = false;
}

} // namespace depth_anything

#endif // USE_CUDA
