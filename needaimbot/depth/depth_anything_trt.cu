#ifdef USE_CUDA

#include "depth_anything_trt.h"

#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>

namespace depth_anything {

// CUDA kernel: preprocess RGB uint8 to float CHW with normalization
__global__ void preprocessKernel(
    const uint8_t* __restrict__ input,  // HWC uint8 RGB
    float* __restrict__ output,          // CHW float normalized
    int srcWidth, int srcHeight, int srcStep,
    int dstWidth, int dstHeight,
    float meanR, float meanG, float meanB,
    float stdR, float stdG, float stdB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // Compute source coordinates with simple resize
    int srcX = x * srcWidth / dstWidth;
    int srcY = y * srcHeight / dstHeight;
    
    srcX = min(srcX, srcWidth - 1);
    srcY = min(srcY, srcHeight - 1);
    
    // Read RGB values
    const uint8_t* srcPixel = input + srcY * srcStep + srcX * 3;
    float r = srcPixel[0];
    float g = srcPixel[1];
    float b = srcPixel[2];
    
    // Normalize and write to CHW format
    int pixelIdx = y * dstWidth + x;
    int planeSize = dstWidth * dstHeight;
    
    output[0 * planeSize + pixelIdx] = (r - meanR) / stdR;  // R channel
    output[1 * planeSize + pixelIdx] = (g - meanG) / stdG;  // G channel
    output[2 * planeSize + pixelIdx] = (b - meanB) / stdB;  // B channel
}

// CUDA kernel: preprocess RGBA uint8 (GPU) to float CHW with normalization
__global__ void preprocessRgbaKernel(
    const uint8_t* __restrict__ input,  // HWC uint8 RGBA
    float* __restrict__ output,          // CHW float normalized
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight,
    float meanR, float meanG, float meanB,
    float stdR, float stdG, float stdB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // Compute source coordinates with simple resize
    int srcX = x * srcWidth / dstWidth;
    int srcY = y * srcHeight / dstHeight;
    
    srcX = min(srcX, srcWidth - 1);
    srcY = min(srcY, srcHeight - 1);
    
    // Read RGBA values (skip alpha)
    const uint8_t* srcPixel = input + srcY * srcPitch + srcX * 4;
    float r = srcPixel[0];
    float g = srcPixel[1];
    float b = srcPixel[2];
    
    // Normalize and write to CHW format
    int pixelIdx = y * dstWidth + x;
    int planeSize = dstWidth * dstHeight;
    
    output[0 * planeSize + pixelIdx] = (r - meanR) / stdR;
    output[1 * planeSize + pixelIdx] = (g - meanG) / stdG;
    output[2 * planeSize + pixelIdx] = (b - meanB) / stdB;
}

// CUDA kernel: normalize depth output to 0-255 uint8
__global__ void normalizeDepthKernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int width, int height,
    float minVal, float maxVal)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float val = input[idx];
    
    // Normalize to 0-255
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;
    
    float normalized = (val - minVal) / range * 255.0f;
    output[idx] = static_cast<uint8_t>(fminf(fmaxf(normalized, 0.0f), 255.0f));
}

// CUDA kernel: find min/max in parallel (reduction)
__global__ void minMaxReductionKernel(
    const float* __restrict__ input,
    float* __restrict__ minOut,
    float* __restrict__ maxOut,
    int size)
{
    extern __shared__ float shared[];
    float* smin = shared;
    float* smax = shared + blockDim.x;
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with extreme values
    float localMin = 1e30f;
    float localMax = -1e30f;
    
    if (i < size) {
        localMin = input[i];
        localMax = input[i];
    }
    
    smin[tid] = localMin;
    smax[tid] = localMax;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < size) {
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin(reinterpret_cast<int*>(minOut), __float_as_int(smin[0]));
        atomicMax(reinterpret_cast<int*>(maxOut), __float_as_int(smax[0]));
    }
}

DepthAnythingTrt::DepthAnythingTrt()
    : input_w(kOptInputSize)
    , input_h(kOptInputSize)
    , min_input_size(kMinInputSize)
    , max_input_size(kMaxInputSize)
    , dynamic_input(false)
    , mean{123.675f, 116.28f, 103.53f}
    , std_val{58.395f, 57.12f, 57.375f}
    , colormap_type(COLORMAP_TWILIGHT)
    , stream(nullptr)
    , d_input_float(nullptr)
    , d_output_float(nullptr)
    , initialized(false)
{
    buffer[0] = nullptr;
    buffer[1] = nullptr;
}

DepthAnythingTrt::~DepthAnythingTrt() {
    reset();
}

bool DepthAnythingTrt::ready() const {
    return initialized;
}

const std::string& DepthAnythingTrt::lastError() const {
    return last_error;
}

void DepthAnythingTrt::setColormap(int type) {
    if (type < COLORMAP_AUTUMN || type > COLORMAP_DEEPGREEN) {
        colormap_type = COLORMAP_TWILIGHT;
        return;
    }
    colormap_type = type;
}

int DepthAnythingTrt::colormapType() const {
    return colormap_type;
}

void DepthAnythingTrt::reset() {
    initialized = false;
    last_error.clear();

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    if (buffer[0]) {
        cudaFree(buffer[0]);
        buffer[0] = nullptr;
    }
    if (buffer[1]) {
        cudaFree(buffer[1]);
        buffer[1] = nullptr;
    }
    if (d_input_float) {
        cudaFree(d_input_float);
        d_input_float = nullptr;
    }
    if (d_output_float) {
        cudaFree(d_output_float);
        d_output_float = nullptr;
    }

    h_depth_data.clear();
    context.reset();
    engine.reset();
    runtime.reset();
}

bool DepthAnythingTrt::initialize(const std::string& modelPath, nvinfer1::ILogger& logger) {
    reset();
    dynamic_input = false;
    min_input_size = kMinInputSize;
    max_input_size = kMaxInputSize;

    if (!std::filesystem::exists(modelPath)) {
        last_error = "Depth model file not found: " + modelPath;
        return false;
    }

    if (!loadEngine(modelPath, logger)) {
        if (last_error.empty()) {
            last_error = "Failed to load depth model: " + modelPath;
        }
        return false;
    }

    // Get input dimensions
    auto input_name = engine->getIOTensorName(0);
    auto input_dims = engine->getTensorShape(input_name);
    bool has_dynamic = false;
    for (int i = 0; i < input_dims.nbDims; i++) {
        if (input_dims.d[i] == -1) {
            has_dynamic = true;
            break;
        }
    }

    if (has_dynamic) {
        dynamic_input = true;
        min_input_size = kMinInputSize;
        max_input_size = kMaxInputSize;
        input_h = max_input_size;
        input_w = max_input_size;
    } else {
        input_h = input_dims.d[2];
        input_w = input_dims.d[3];
        min_input_size = input_w;
        max_input_size = input_w;
    }

    cudaStreamCreate(&stream);

    // Allocate GPU buffers
    const size_t max_input = static_cast<size_t>(input_h) * static_cast<size_t>(input_w);
    cudaMalloc(&buffer[0], 3 * max_input * sizeof(float));
    cudaMalloc(&buffer[1], max_input * sizeof(float));
    
    d_input_float = static_cast<float*>(buffer[0]);
    d_output_float = static_cast<float*>(buffer[1]);

    h_depth_data.resize(max_input);

    initialized = true;
    std::cout << "[DepthAnything] Initialized: " << input_w << "x" << input_h 
              << " (dynamic=" << (dynamic_input ? "true" : "false") << ")" << std::endl;
    return true;
}

int DepthAnythingTrt::selectInputSize(int width, int height) const {
    if (min_input_size <= 0 || max_input_size <= 0) {
        return input_w;
    }
    int long_side = std::max(width, height);
    return std::clamp(long_side, min_input_size, max_input_size);
}

bool DepthAnythingTrt::setInputShape(int w, int h) {
    const char* input_name = engine->getIOTensorName(0);
    if (!context->setInputShape(input_name, nvinfer1::Dims4{1, 3, h, w})) {
        last_error = "Failed to set depth input shape.";
        return false;
    }
    input_w = w;
    input_h = h;
    return true;
}

bool DepthAnythingTrt::preprocess(const uint8_t* inputRgb, int srcWidth, int srcHeight) {
    // Upload to GPU and preprocess
    size_t srcSize = srcWidth * srcHeight * 3;
    uint8_t* d_input_rgb = nullptr;
    cudaMalloc(&d_input_rgb, srcSize);
    cudaMemcpyAsync(d_input_rgb, inputRgb, srcSize, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((input_w + 15) / 16, (input_h + 15) / 16);

    preprocessKernel<<<grid, block, 0, stream>>>(
        d_input_rgb, d_input_float,
        srcWidth, srcHeight, srcWidth * 3,
        input_w, input_h,
        mean[0], mean[1], mean[2],
        std_val[0], std_val[1], std_val[2]
    );

    cudaFree(d_input_rgb);
    return true;
}

bool DepthAnythingTrt::preprocessGpu(const uint8_t* d_inputRgba, int srcWidth, int srcHeight, 
                                      int srcPitch, cudaStream_t extStream) {
    cudaStream_t useStream = extStream ? extStream : stream;
    
    dim3 block(16, 16);
    dim3 grid((input_w + 15) / 16, (input_h + 15) / 16);

    preprocessRgbaKernel<<<grid, block, 0, useStream>>>(
        d_inputRgba, d_input_float,
        srcWidth, srcHeight, srcPitch,
        input_w, input_h,
        mean[0], mean[1], mean[2],
        std_val[0], std_val[1], std_val[2]
    );

    return true;
}

bool DepthAnythingTrt::postprocess(uint8_t* outputDepth, int dstWidth, int dstHeight) {
    // Copy output to host
    const size_t output_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
    cudaMemcpyAsync(h_depth_data.data(), d_output_float, output_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Find min/max on CPU
    float minVal = h_depth_data[0];
    float maxVal = h_depth_data[0];
    for (size_t i = 1; i < h_depth_data.size(); i++) {
        if (h_depth_data[i] < minVal) minVal = h_depth_data[i];
        if (h_depth_data[i] > maxVal) maxVal = h_depth_data[i];
    }

    // Normalize to 0-255
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;

    // Resize to output dimensions
    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x++) {
            int srcX = x * input_w / dstWidth;
            int srcY = y * input_h / dstHeight;
            srcX = std::min(srcX, input_w - 1);
            srcY = std::min(srcY, input_h - 1);
            
            float val = h_depth_data[srcY * input_w + srcX];
            float normalized = (val - minVal) / range * 255.0f;
            outputDepth[y * dstWidth + x] = static_cast<uint8_t>(std::clamp(normalized, 0.0f, 255.0f));
        }
    }

    return true;
}

bool DepthAnythingTrt::predictDepth(const uint8_t* inputRgb, int width, int height,
                                     uint8_t* outputDepth, int& outWidth, int& outHeight) {
    if (!initialized || !inputRgb || !outputDepth) {
        last_error = "Not initialized or invalid input";
        return false;
    }

    // Select input size
    int target_size = selectInputSize(width, height);
    if (dynamic_input) {
        if (!setInputShape(target_size, target_size)) {
            return false;
        }
    }

    // Preprocess
    if (!preprocess(inputRgb, width, height)) {
        return false;
    }

    // Run inference
    context->setTensorAddress(engine->getIOTensorName(0), buffer[0]);
    context->setTensorAddress(engine->getIOTensorName(1), buffer[1]);
    
    if (!context->enqueueV3(stream)) {
        last_error = "Depth inference failed";
        return false;
    }

    // Postprocess - resize to original dimensions
    outWidth = width;
    outHeight = height;
    if (!postprocess(outputDepth, width, height)) {
        return false;
    }

    return true;
}

bool DepthAnythingTrt::predictDepthGpu(const uint8_t* d_inputRgba, int width, int height, int inputPitch,
                                        uint8_t* d_outputDepth, int& outWidth, int& outHeight, 
                                        cudaStream_t extStream) {
    if (!initialized || !d_inputRgba || !d_outputDepth) {
        last_error = "Not initialized or invalid input";
        return false;
    }

    cudaStream_t useStream = extStream ? extStream : stream;

    // Select input size
    int target_size = selectInputSize(width, height);
    if (dynamic_input) {
        if (!setInputShape(target_size, target_size)) {
            return false;
        }
    }

    // Preprocess on GPU
    if (!preprocessGpu(d_inputRgba, width, height, inputPitch, useStream)) {
        return false;
    }

    // Run inference
    context->setTensorAddress(engine->getIOTensorName(0), buffer[0]);
    context->setTensorAddress(engine->getIOTensorName(1), buffer[1]);
    
    if (!context->enqueueV3(useStream)) {
        last_error = "Depth inference failed";
        return false;
    }

    // For GPU output, we need to normalize on GPU
    // First find min/max
    const size_t output_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
    cudaMemcpyAsync(h_depth_data.data(), d_output_float, output_bytes, cudaMemcpyDeviceToHost, useStream);
    cudaStreamSynchronize(useStream);

    float minVal = h_depth_data[0];
    float maxVal = h_depth_data[0];
    for (size_t i = 1; i < h_depth_data.size(); i++) {
        if (h_depth_data[i] < minVal) minVal = h_depth_data[i];
        if (h_depth_data[i] > maxVal) maxVal = h_depth_data[i];
    }

    // Normalize on GPU
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    // Note: For simplicity, we resize on CPU and upload. 
    // A full GPU implementation would use a resize+normalize kernel
    std::vector<uint8_t> h_output(width * height);
    postprocess(h_output.data(), width, height);
    cudaMemcpyAsync(d_outputDepth, h_output.data(), width * height, cudaMemcpyHostToDevice, useStream);

    outWidth = width;
    outHeight = height;
    return true;
}

bool DepthAnythingTrt::loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger) {
    // Check if ONNX file
    if (modelPath.find(".onnx") != std::string::npos) {
        if (!buildEngine(modelPath, logger)) {
            return false;
        }
        saveEngine(modelPath);
        return true;
    }

    std::ifstream engineStream(modelPath, std::ios::binary);
    if (!engineStream.is_open()) {
        last_error = "Unable to open depth engine: " + modelPath;
        return false;
    }

    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::vector<char> engineData(modelSize);
    engineStream.read(engineData.data(), modelSize);
    engineStream.close();

    runtime.reset(nvinfer1::createInferRuntime(logger));
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), modelSize));
    if (!engine) {
        last_error = "Failed to deserialize depth engine: " + modelPath;
        return false;
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
        last_error = "Failed to create depth execution context.";
        return false;
    }
    return true;
}

bool DepthAnythingTrt::buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger) {
    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        last_error = "Failed to create TensorRT builder.";
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // Enable FP16 for better performance
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        last_error = "Failed to parse depth ONNX model.";
        delete parser;
        delete config;
        delete network;
        delete builder;
        return false;
    }

    auto input = network->getInput(0);
    if (input) {
        auto input_dims = input->getDimensions();
        bool has_dynamic = false;
        for (int i = 0; i < input_dims.nbDims; i++) {
            if (input_dims.d[i] == -1) {
                has_dynamic = true;
                break;
            }
        }

        if (has_dynamic) {
            auto profile = builder->createOptimizationProfile();
            int opt_size = std::clamp(kOptInputSize, kMinInputSize, kMaxInputSize);
            const char* input_name = input->getName();
            bool ok = profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, 
                                             nvinfer1::Dims4{1, 3, kMinInputSize, kMinInputSize});
            ok = ok && profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, 
                                              nvinfer1::Dims4{1, 3, opt_size, opt_size});
            ok = ok && profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, 
                                              nvinfer1::Dims4{1, 3, kMaxInputSize, kMaxInputSize});
            if (!ok || !profile->isValid()) {
                last_error = "Failed to set depth input optimization profile.";
                delete parser;
                delete config;
                delete network;
                delete builder;
                return false;
            }
            config->addOptimizationProfile(profile);
        }
    }

    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    runtime.reset(nvinfer1::createInferRuntime(logger));
    engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    context.reset(engine->createExecutionContext());

    delete plan;
    delete parser;
    delete config;
    delete network;
    delete builder;

    if (!engine || !context) {
        last_error = "Failed to build depth engine from ONNX.";
        return false;
    }

    return true;
}

bool DepthAnythingTrt::saveEngine(const std::string& onnxPath) {
    if (!engine) {
        return false;
    }

    size_t dotIndex = onnxPath.find_last_of(".");
    if (dotIndex == std::string::npos) {
        return false;
    }

    std::string engine_path = onnxPath.substr(0, dotIndex) + ".engine";
    nvinfer1::IHostMemory* data = engine->serialize();
    std::ofstream file(engine_path, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        delete data;
        return false;
    }

    file.write(reinterpret_cast<const char*>(data->data()), data->size());
    file.close();
    delete data;
    
    std::cout << "[DepthAnything] Saved engine to: " << engine_path << std::endl;
    return true;
}

} // namespace depth_anything

#endif // USE_CUDA
