#ifdef USE_CUDA

#include "depth_mask.h"
#include "depth_anything_trt.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace depth_anything {

// Global singleton instance
static DepthMaskGenerator g_depthMaskGenerator;

DepthMaskGenerator& GetDepthMaskGenerator() {
    return g_depthMaskGenerator;
}

DepthMaskGenerator::DepthMaskGenerator() : model(nullptr) {
}

DepthMaskGenerator::~DepthMaskGenerator() {
    reset();
}

void DepthMaskGenerator::reset() {
    std::lock_guard<std::mutex> lk(state_mutex);
    
    depth_map.clear();
    mask_binary.clear();
    mask_width = 0;
    mask_height = 0;
    last_error.clear();
    last_model_path.clear();
    initialized = false;
    last_update = std::chrono::steady_clock::time_point::min();
    last_attempt = std::chrono::steady_clock::time_point::min();
    last_frame_w = 0;
    last_frame_h = 0;
    
    if (model) {
        delete model;
        model = nullptr;
    }
}

bool DepthMaskGenerator::ready() const {
    std::lock_guard<std::mutex> lk(state_mutex);
    return initialized && model && model->ready() && !depth_map.empty();
}

std::string DepthMaskGenerator::lastError() const {
    std::lock_guard<std::mutex> lk(state_mutex);
    return last_error;
}

std::chrono::steady_clock::time_point DepthMaskGenerator::lastAttemptTime() const {
    std::lock_guard<std::mutex> lk(state_mutex);
    return last_attempt;
}

std::pair<int, int> DepthMaskGenerator::lastFrameSize() const {
    std::lock_guard<std::mutex> lk(state_mutex);
    return {last_frame_w, last_frame_h};
}

DepthMaskDebugState DepthMaskGenerator::debugState() const {
    std::lock_guard<std::mutex> lk(state_mutex);
    DepthMaskDebugState state;
    state.initialized = initialized;
    state.has_model = (model != nullptr);
    state.model_ready = (model != nullptr) ? model->ready() : false;
    state.last_model_path = last_model_path;
    return state;
}

void DepthMaskGenerator::generateMask(int near_percent, bool invert) {
    if (depth_map.empty() || mask_width <= 0 || mask_height <= 0) {
        return;
    }

    const int total = mask_width * mask_height;
    const int target = std::max(1, (total * near_percent) / 100);

    // Build histogram of depth values
    int hist[256] = {0};
    for (const auto& val : depth_map) {
        hist[val]++;
    }

    // Find threshold based on near_percent
    int threshold = 0;
    if (!invert) {
        // Near objects have LOWER depth values (closer to camera)
        int count = 0;
        for (int i = 0; i < 256; ++i) {
            count += hist[i];
            if (count >= target) {
                threshold = i;
                break;
            }
        }
    } else {
        // Invert: select FAR objects (higher depth values)
        int count = 0;
        for (int i = 255; i >= 0; --i) {
            count += hist[i];
            if (count >= target) {
                threshold = i;
                break;
            }
        }
    }

    // Generate binary mask
    mask_binary.resize(total);
    for (int i = 0; i < total; ++i) {
        if (!invert) {
            mask_binary[i] = (depth_map[i] <= threshold) ? 255 : 0;
        } else {
            mask_binary[i] = (depth_map[i] >= threshold) ? 255 : 0;
        }
    }
}

void DepthMaskGenerator::update(const uint8_t* frameRgb, int width, int height,
                                 const DepthMaskOptions& options,
                                 const std::string& modelPath, nvinfer1::ILogger& logger) {
    if (!options.enabled) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    
    if (!frameRgb || width <= 0 || height <= 0) {
        std::lock_guard<std::mutex> lk(state_mutex);
        last_error = "Depth mask frame is empty.";
        last_attempt = now;
        last_frame_w = 0;
        last_frame_h = 0;
        return;
    }

    std::lock_guard<std::mutex> lk(state_mutex);
    last_attempt = now;
    last_frame_w = width;
    last_frame_h = height;

    // Initialize model if needed
    if (!model) {
        model = new DepthAnythingTrt();
    }

    if (modelPath.empty()) {
        last_error = "Depth mask model path is empty.";
        return;
    }

    // Initialize or reinitialize if model path changed
    if (!initialized || modelPath != last_model_path || !model->ready()) {
        if (!model->initialize(modelPath, logger)) {
            last_error = model->lastError();
            initialized = false;
            return;
        }
        last_model_path = modelPath;
        initialized = true;
        last_error.clear();
        std::cout << "[DepthMask] Model initialized: " << modelPath << std::endl;
    }

    // Rate limiting based on FPS setting
    const int fps = options.fps > 0 ? options.fps : 5;
    const auto interval = std::chrono::milliseconds(1000 / fps);
    if (now - last_update < interval) {
        return;
    }

    last_update = now;

    // Allocate output buffer
    depth_map.resize(width * height);
    int outW, outH;

    // Run depth prediction
    if (!model->predictDepth(frameRgb, width, height, depth_map.data(), outW, outH)) {
        last_error = model->lastError();
        if (last_error.empty()) {
            last_error = "Depth mask inference returned empty output.";
        }
        return;
    }

    mask_width = outW;
    mask_height = outH;

    // Generate binary mask
    generateMask(options.near_percent, options.invert);
    
    last_error.clear();
}

void DepthMaskGenerator::updateGpu(const uint8_t* d_frameRgba, int width, int height, int pitch,
                                    const DepthMaskOptions& options,
                                    const std::string& modelPath, nvinfer1::ILogger& logger,
                                    cudaStream_t stream) {
    if (!options.enabled) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    
    if (!d_frameRgba || width <= 0 || height <= 0) {
        std::lock_guard<std::mutex> lk(state_mutex);
        last_error = "Depth mask GPU frame is empty.";
        last_attempt = now;
        last_frame_w = 0;
        last_frame_h = 0;
        return;
    }

    std::lock_guard<std::mutex> lk(state_mutex);
    last_attempt = now;
    last_frame_w = width;
    last_frame_h = height;

    // Initialize model if needed
    if (!model) {
        model = new DepthAnythingTrt();
    }

    if (modelPath.empty()) {
        last_error = "Depth mask model path is empty.";
        return;
    }

    // Initialize or reinitialize if model path changed
    if (!initialized || modelPath != last_model_path || !model->ready()) {
        if (!model->initialize(modelPath, logger)) {
            last_error = model->lastError();
            initialized = false;
            return;
        }
        last_model_path = modelPath;
        initialized = true;
        last_error.clear();
        std::cout << "[DepthMask] Model initialized (GPU): " << modelPath << std::endl;
    }

    // Rate limiting based on FPS setting
    const int fps = options.fps > 0 ? options.fps : 5;
    const auto interval = std::chrono::milliseconds(1000 / fps);
    if (now - last_update < interval) {
        return;
    }

    last_update = now;

    // Allocate GPU output buffer
    uint8_t* d_depth_output = nullptr;
    cudaMalloc(&d_depth_output, width * height);

    int outW, outH;

    // Run depth prediction on GPU
    if (!model->predictDepthGpu(d_frameRgba, width, height, pitch, 
                                 d_depth_output, outW, outH, stream)) {
        last_error = model->lastError();
        if (last_error.empty()) {
            last_error = "Depth mask GPU inference failed.";
        }
        cudaFree(d_depth_output);
        return;
    }

    // Copy result to host
    depth_map.resize(outW * outH);
    cudaMemcpy(depth_map.data(), d_depth_output, outW * outH, cudaMemcpyDeviceToHost);
    cudaFree(d_depth_output);

    mask_width = outW;
    mask_height = outH;

    // Generate binary mask
    generateMask(options.near_percent, options.invert);
    
    last_error.clear();
}

bool DepthMaskGenerator::getMask(std::vector<uint8_t>& outMask, int& outWidth, int& outHeight) const {
    std::lock_guard<std::mutex> lk(state_mutex);
    
    if (mask_binary.empty() || mask_width <= 0 || mask_height <= 0) {
        return false;
    }

    outMask = mask_binary;
    outWidth = mask_width;
    outHeight = mask_height;
    return true;
}

bool DepthMaskGenerator::getDepthMap(std::vector<uint8_t>& outDepth, int& outWidth, int& outHeight) const {
    std::lock_guard<std::mutex> lk(state_mutex);
    
    if (depth_map.empty() || mask_width <= 0 || mask_height <= 0) {
        return false;
    }

    outDepth = depth_map;
    outWidth = mask_width;
    outHeight = mask_height;
    return true;
}

bool DepthMaskGenerator::isPointNear(int x, int y) const {
    std::lock_guard<std::mutex> lk(state_mutex);
    
    if (mask_binary.empty() || x < 0 || y < 0 || x >= mask_width || y >= mask_height) {
        return false;
    }

    return mask_binary[y * mask_width + x] > 0;
}

uint8_t DepthMaskGenerator::getDepthAt(int x, int y) const {
    std::lock_guard<std::mutex> lk(state_mutex);
    
    if (depth_map.empty() || x < 0 || y < 0 || x >= mask_width || y >= mask_height) {
        return 255;  // Return max depth (far) if invalid
    }

    return depth_map[y * mask_width + x];
}

} // namespace depth_anything

#endif // USE_CUDA
