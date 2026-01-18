#include "depth_anything_trt.h"

#ifdef USE_CUDA

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace depth_anything {

DepthAnythingTrt::DepthAnythingTrt()
    : input_w(0)
    , input_h(0)
    , min_input_size(252)
    , max_input_size(756)
    , dynamic_input(false)
    , colormap_type(COLORMAP_INFERNO)
    , stream(nullptr)
    , initialized(false)
{
    mean[0] = 0.485f; mean[1] = 0.456f; mean[2] = 0.406f;
    std_val[0] = 0.229f; std_val[1] = 0.224f; std_val[2] = 0.225f;
    buffer[0] = nullptr;
    buffer[1] = nullptr;
}

DepthAnythingTrt::~DepthAnythingTrt() {
    reset();
}

bool DepthAnythingTrt::initialize(const std::string& modelPath, nvinfer1::ILogger& logger) {
    if (initialized) {
        return true;
    }

    if (!loadEngine(modelPath, logger)) {
        return false;
    }

    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        last_error = "Failed to create CUDA stream: " + std::string(cudaGetErrorString(err));
        return false;
    }

    initialized = true;
    return true;
}

bool DepthAnythingTrt::loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        last_error = "Failed to open engine file: " + modelPath;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    if (!file.read(engineData.data(), size)) {
        last_error = "Failed to read engine file";
        return false;
    }
    file.close();

    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        last_error = "Failed to create TensorRT runtime";
        return false;
    }

    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    if (!engine) {
        last_error = "Failed to deserialize CUDA engine";
        return false;
    }

    context.reset(engine->createExecutionContext());
    if (!context) {
        last_error = "Failed to create execution context";
        return false;
    }

    // Get input dimensions
    int inputIndex = 0;
    auto inputDims = engine->getTensorShape(engine->getIOTensorName(inputIndex));
    
    if (inputDims.d[0] == -1) {
        dynamic_input = true;
        input_w = 518;  // Default size
        input_h = 518;
    } else {
        dynamic_input = false;
        input_h = inputDims.d[2];
        input_w = inputDims.d[3];
    }

    // Allocate buffers
    size_t inputSize = input_h * input_w * 3 * sizeof(float);
    size_t outputSize = input_h * input_w * sizeof(float);

    cudaError_t err = cudaMalloc(&buffer[0], inputSize);
    if (err != cudaSuccess) {
        last_error = "Failed to allocate input buffer: " + std::string(cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(&buffer[1], outputSize);
    if (err != cudaSuccess) {
        last_error = "Failed to allocate output buffer: " + std::string(cudaGetErrorString(err));
        cudaFree(buffer[0]);
        buffer[0] = nullptr;
        return false;
    }

    depth_data.resize(input_h * input_w);

    return true;
}

bool DepthAnythingTrt::setInputShape(int w, int h) {
    if (!dynamic_input) {
        return (w == input_w && h == input_h);
    }

    // Resize buffers if needed
    if (w != input_w || h != input_h) {
        input_w = w;
        input_h = h;

        if (buffer[0]) cudaFree(buffer[0]);
        if (buffer[1]) cudaFree(buffer[1]);

        size_t inputSize = input_h * input_w * 3 * sizeof(float);
        size_t outputSize = input_h * input_w * sizeof(float);

        cudaError_t err = cudaMalloc(&buffer[0], inputSize);
        if (err != cudaSuccess) {
            last_error = "Failed to reallocate input buffer";
            return false;
        }

        err = cudaMalloc(&buffer[1], outputSize);
        if (err != cudaSuccess) {
            last_error = "Failed to reallocate output buffer";
            cudaFree(buffer[0]);
            buffer[0] = nullptr;
            return false;
        }

        depth_data.resize(input_h * input_w);

        // Update context input shape
        nvinfer1::Dims4 dims{1, 3, input_h, input_w};
        const char* inputName = engine->getIOTensorName(0);
        context->setInputShape(inputName, dims);
    }

    return true;
}

int DepthAnythingTrt::selectInputSize(int width, int height) const {
    int maxDim = std::max(width, height);
    
    // Select appropriate size based on input
    if (maxDim <= 336) return 252;
    if (maxDim <= 420) return 336;
    if (maxDim <= 504) return 420;
    if (maxDim <= 588) return 504;
    if (maxDim <= 672) return 588;
    return 756;
}

bool DepthAnythingTrt::predictDepth(const float* inputRgb, int width, int height,
                                     float* outputDepth, int& outWidth, int& outHeight) {
    if (!initialized || !inputRgb || !outputDepth) {
        last_error = "Not initialized or invalid input";
        return false;
    }

    // For now, just return a stub (all zeros)
    // Full implementation would:
    // 1. Preprocess input (resize, normalize)
    // 2. Copy to GPU
    // 3. Run inference
    // 4. Copy output back
    // 5. Postprocess (normalize to 0-1 range)

    outWidth = input_w;
    outHeight = input_h;
    
    // Zero-fill output for stub
    std::fill(outputDepth, outputDepth + input_w * input_h, 0.5f);

    return true;
}

void DepthAnythingTrt::setColormap(int type) {
    colormap_type = type;
}

int DepthAnythingTrt::colormapType() const {
    return colormap_type;
}

bool DepthAnythingTrt::ready() const {
    return initialized;
}

const std::string& DepthAnythingTrt::lastError() const {
    return last_error;
}

void DepthAnythingTrt::reset() {
    if (buffer[0]) {
        cudaFree(buffer[0]);
        buffer[0] = nullptr;
    }
    if (buffer[1]) {
        cudaFree(buffer[1]);
        buffer[1] = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    context.reset();
    engine.reset();
    runtime.reset();

    depth_data.clear();
    initialized = false;
    last_error.clear();
}

} // namespace depth_anything

#endif // USE_CUDA
