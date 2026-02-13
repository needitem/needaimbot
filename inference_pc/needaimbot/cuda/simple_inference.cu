// Simple TensorRT inference with CUDA Graph optimization
// Supports FP16 and FP32 models natively
// GPU postprocessing for minimal latency
// BGRA/RGB input with fused channel conversion + normalization
#include "simple_inference.h"
#include "simple_postprocess.h"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <exception>
#include <vector>
#include <NvInferVersion.h>

// TensorRT API version compatibility
// TensorRT 10.x removed legacy binding APIs
#if NV_TENSORRT_MAJOR >= 10
    #define TRT_USE_NEW_API 1
#else
    #define TRT_USE_NEW_API 0
#endif

// =============================================================================
// Unified Preprocessing Kernels (RGB/BGRA -> CHW normalized)
// =============================================================================
// src_bpp: bytes per pixel (3=RGB, 4=BGRA)
// r_ch, g_ch, b_ch: source channel byte offsets for R, G, B output
//   RGB:  r_ch=0, g_ch=1, b_ch=2
//   BGRA: r_ch=2, g_ch=1, b_ch=0

// Same resolution, no resize -> FP16
__global__ void preprocessKernelFP16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int width, int height,
    int src_bpp, int r_ch, int g_ch, int b_ch,
    float scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * src_bpp;
    int hw_size = width * height;
    int dst_idx = y * width + x;

    dst[dst_idx] = __float2half(src[src_idx + r_ch] * scale_factor);
    dst[dst_idx + hw_size] = __float2half(src[src_idx + g_ch] * scale_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(src[src_idx + b_ch] * scale_factor);
}

// Same resolution, no resize -> FP32
__global__ void preprocessKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int width, int height,
    int src_bpp, int r_ch, int g_ch, int b_ch,
    float scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * src_bpp;
    int hw_size = width * height;
    int dst_idx = y * width + x;

    dst[dst_idx] = src[src_idx + r_ch] * scale_factor;
    dst[dst_idx + hw_size] = src[src_idx + g_ch] * scale_factor;
    dst[dst_idx + 2 * hw_size] = src[src_idx + b_ch] * scale_factor;
}

// BGRA specialized (no channel offset indirection) -> FP16
__global__ void preprocessKernelBGRAFP16(
    const uchar4* __restrict__ src,
    __half* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int hw_size = width * height;
    int dst_idx = y * width + x;
    uchar4 px = src[dst_idx];  // BGRA

    dst[dst_idx] = __float2half(px.z * scale_factor);             // R
    dst[dst_idx + hw_size] = __float2half(px.y * scale_factor);   // G
    dst[dst_idx + 2 * hw_size] = __float2half(px.x * scale_factor); // B
}

// BGRA specialized (no channel offset indirection) -> FP32
__global__ void preprocessKernelBGRA(
    const uchar4* __restrict__ src,
    float* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int hw_size = width * height;
    int dst_idx = y * width + x;
    uchar4 px = src[dst_idx];  // BGRA

    dst[dst_idx] = px.z * scale_factor;              // R
    dst[dst_idx + hw_size] = px.y * scale_factor;    // G
    dst[dst_idx + 2 * hw_size] = px.x * scale_factor; // B
}

// =============================================================================
// Bilinear Resize + Preprocessing Kernels (fused for efficiency)
// =============================================================================

// Optimized bilinear interpolation - reads 4 pixels once for all 3 output channels
__device__ __forceinline__ void bilinearSample(
    const uint8_t* __restrict__ src,
    int src_w, int src_h, int src_bpp,
    int r_ch, int g_ch, int b_ch,
    float src_x, float src_y,
    float& r, float& g, float& b
) {
    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = src_x - x0;
    float fy = src_y - y0;

    // Read 4 pixels (HWC layout with src_bpp stride)
    const uint8_t* p00 = src + (y0 * src_w + x0) * src_bpp;
    const uint8_t* p10 = src + (y0 * src_w + x1) * src_bpp;
    const uint8_t* p01 = src + (y1 * src_w + x0) * src_bpp;
    const uint8_t* p11 = src + (y1 * src_w + x1) * src_bpp;

    // R channel
    float r00 = p00[r_ch], r10 = p10[r_ch], r01 = p01[r_ch], r11 = p11[r_ch];
    float r0 = r00 + fx * (r10 - r00);
    float r1 = r01 + fx * (r11 - r01);
    r = r0 + fy * (r1 - r0);

    // G channel
    float g00 = p00[g_ch], g10 = p10[g_ch], g01 = p01[g_ch], g11 = p11[g_ch];
    float g0 = g00 + fx * (g10 - g00);
    float g1 = g01 + fx * (g11 - g01);
    g = g0 + fy * (g1 - g0);

    // B channel
    float b00 = p00[b_ch], b10 = p10[b_ch], b01 = p01[b_ch], b11 = p11[b_ch];
    float b0 = b00 + fx * (b10 - b00);
    float b1 = b01 + fx * (b11 - b01);
    b = b0 + fy * (b1 - b0);
}

// Optimized bilinear interpolation for BGRA uchar4 source.
__device__ __forceinline__ void bilinearSampleBGRA(
    const uchar4* __restrict__ src,
    int src_w, int src_h,
    float src_x, float src_y,
    float& r, float& g, float& b
) {
    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = src_x - x0;
    float fy = src_y - y0;

    const uchar4 p00 = src[y0 * src_w + x0];
    const uchar4 p10 = src[y0 * src_w + x1];
    const uchar4 p01 = src[y1 * src_w + x0];
    const uchar4 p11 = src[y1 * src_w + x1];

    // R from .z
    float r00 = p00.z, r10 = p10.z, r01 = p01.z, r11 = p11.z;
    float r0 = r00 + fx * (r10 - r00);
    float r1 = r01 + fx * (r11 - r01);
    r = r0 + fy * (r1 - r0);

    // G from .y
    float g00 = p00.y, g10 = p10.y, g01 = p01.y, g11 = p11.y;
    float g0 = g00 + fx * (g10 - g00);
    float g1 = g01 + fx * (g11 - g01);
    g = g0 + fy * (g1 - g0);

    // B from .x
    float b00 = p00.x, b10 = p10.x, b01 = p01.x, b11 = p11.x;
    float b0 = b00 + fx * (b10 - b00);
    float b1 = b01 + fx * (b11 - b01);
    b = b0 + fy * (b1 - b0);
}

// Bilinear resize + HWC->CHW + normalize -> FP16
__global__ void resizePreprocessKernelFP16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int src_w, int src_h, int src_bpp,
    int r_ch, int g_ch, int b_ch,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    float norm_factor
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = dx * scale_x;
    float sy = dy * scale_y;

    int hw_size = dst_w * dst_h;
    int dst_idx = dy * dst_w + dx;

    float r, g, b;
    bilinearSample(src, src_w, src_h, src_bpp, r_ch, g_ch, b_ch, sx, sy, r, g, b);

    dst[dst_idx] = __float2half(r * norm_factor);
    dst[dst_idx + hw_size] = __float2half(g * norm_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(b * norm_factor);
}

// Bilinear resize + HWC->CHW + normalize -> FP32
__global__ void resizePreprocessKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h, int src_bpp,
    int r_ch, int g_ch, int b_ch,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    float norm_factor
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = dx * scale_x;
    float sy = dy * scale_y;

    int hw_size = dst_w * dst_h;
    int dst_idx = dy * dst_w + dx;

    float r, g, b;
    bilinearSample(src, src_w, src_h, src_bpp, r_ch, g_ch, b_ch, sx, sy, r, g, b);

    dst[dst_idx] = r * norm_factor;
    dst[dst_idx + hw_size] = g * norm_factor;
    dst[dst_idx + 2 * hw_size] = b * norm_factor;
}

// Bilinear resize + BGRA->CHW + normalize -> FP16
__global__ void resizePreprocessKernelBGRAFP16(
    const uchar4* __restrict__ src,
    __half* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    float norm_factor
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = dx * scale_x;
    float sy = dy * scale_y;

    int hw_size = dst_w * dst_h;
    int dst_idx = dy * dst_w + dx;

    float r, g, b;
    bilinearSampleBGRA(src, src_w, src_h, sx, sy, r, g, b);

    dst[dst_idx] = __float2half(r * norm_factor);
    dst[dst_idx + hw_size] = __float2half(g * norm_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(b * norm_factor);
}

// Bilinear resize + BGRA->CHW + normalize -> FP32
__global__ void resizePreprocessKernelBGRA(
    const uchar4* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    float norm_factor
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = dx * scale_x;
    float sy = dy * scale_y;

    int hw_size = dst_w * dst_h;
    int dst_idx = dy * dst_w + dx;

    float r, g, b;
    bilinearSampleBGRA(src, src_w, src_h, sx, sy, r, g, b);

    dst[dst_idx] = r * norm_factor;
    dst[dst_idx + hw_size] = g * norm_factor;
    dst[dst_idx + 2 * hw_size] = b * norm_factor;
}

// Unified preprocessing wrapper (handles RGB/BGRA, resize/no-resize, FP16/FP32)
extern "C" cudaError_t cuda_preprocessing(
    const void* src_data,
    void* dst_chw,
    int src_width, int src_height,
    int src_bpp,              // 3=RGB, 4=BGRA
    bool bgra,                // true=BGRA channel order, false=RGB
    int target_width, int target_height,
    bool use_fp16,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((target_width + block.x - 1) / block.x,
              (target_height + block.y - 1) / block.y);

    const float norm_factor = 1.0f / 255.0f;

    // Channel offsets: RGB -> 0,1,2; BGRA -> 2,1,0
    int r_ch = bgra ? 2 : 0;
    int g_ch = 1;
    int b_ch = bgra ? 0 : 2;

    bool need_resize = (src_width != target_width) || (src_height != target_height);
    const bool bgraFastPath = (bgra && src_bpp == 4);

    if (need_resize) {
        float scale_x = (float)(src_width - 1) / (float)(target_width - 1);
        float scale_y = (float)(src_height - 1) / (float)(target_height - 1);

        if (bgraFastPath && use_fp16) {
            resizePreprocessKernelBGRAFP16<<<grid, block, 0, stream>>>(
                static_cast<const uchar4*>(src_data),
                static_cast<__half*>(dst_chw),
                src_width, src_height,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        } else if (bgraFastPath) {
            resizePreprocessKernelBGRA<<<grid, block, 0, stream>>>(
                static_cast<const uchar4*>(src_data),
                static_cast<float*>(dst_chw),
                src_width, src_height,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        } else if (use_fp16) {
            resizePreprocessKernelFP16<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
                static_cast<__half*>(dst_chw),
                src_width, src_height, src_bpp, r_ch, g_ch, b_ch,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        } else {
            resizePreprocessKernel<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
                static_cast<float*>(dst_chw),
                src_width, src_height, src_bpp, r_ch, g_ch, b_ch,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        }
    } else {
        if (bgraFastPath && use_fp16) {
            preprocessKernelBGRAFP16<<<grid, block, 0, stream>>>(
                static_cast<const uchar4*>(src_data),
                static_cast<__half*>(dst_chw),
                target_width, target_height,
                norm_factor
            );
        } else if (bgraFastPath) {
            preprocessKernelBGRA<<<grid, block, 0, stream>>>(
                static_cast<const uchar4*>(src_data),
                static_cast<float*>(dst_chw),
                target_width, target_height,
                norm_factor
            );
        } else if (use_fp16) {
            preprocessKernelFP16<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
                static_cast<__half*>(dst_chw),
                target_width, target_height,
                src_bpp, r_ch, g_ch, b_ch, norm_factor
            );
        } else {
            preprocessKernel<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
                static_cast<float*>(dst_chw),
                target_width, target_height,
                src_bpp, r_ch, g_ch, b_ch, norm_factor
            );
        }
    }

    return cudaGetLastError();
}

namespace gpa {

void SimpleInference::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

SimpleInference::SimpleInference() {
}

SimpleInference::~SimpleInference() {
    // Flush pending stream work so callback state is no longer in-flight.
    if (m_stream) cudaStreamSynchronize(m_stream);

    // Destroy CUDA graph
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);

    // Free GPU memory
    if (m_d_rawInput) cudaFree(m_d_rawInput);
    if (m_d_chwInput) cudaFree(m_d_chwInput);
    if (m_d_output) cudaFree(m_d_output);

    // Free GPU postprocessing buffers
    if (m_d_decoded) cudaFree(m_d_decoded);
    if (m_d_decodedCount) cudaFree(m_d_decodedCount);
    if (m_d_bestTarget) cudaFree(m_d_bestTarget);
    if (m_d_hasTarget) cudaFree(m_d_hasTarget);

    // Free GPU fused pipeline buffers
    if (m_d_selectedTarget) cudaFree(m_d_selectedTarget);
    if (m_d_pidState) cudaFree(m_d_pidState);
    if (m_d_mouseMovement) cudaFree(m_d_mouseMovement);

    // Free combined result buffer
    if (m_d_inferenceResult) cudaFree(m_d_inferenceResult);
    if (m_h_inferenceResultPinned) cudaFreeHost(m_h_inferenceResultPinned);

    // Free pinned host memory
    if (m_h_rawPinned) cudaFreeHost(m_h_rawPinned);

    if (m_stream) cudaStreamDestroy(m_stream);
    if (m_context) delete m_context;
    if (m_engine) delete m_engine;
    if (m_runtime) delete m_runtime;
}

bool SimpleInference::loadEngine(const std::string& enginePath) {
    // Load engine file
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        std::cerr << "[SimpleInference] Failed to open engine: " << enginePath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // Create runtime & engine
    m_runtime = nvinfer1::createInferRuntime(m_logger);
    if (!m_runtime) {
        std::cerr << "[SimpleInference] Failed to create runtime" << std::endl;
        return false;
    }

    m_engine = m_runtime->deserializeCudaEngine(engineData.data(), size);
    if (!m_engine) {
        std::cerr << "[SimpleInference] Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create context
    m_context = m_engine->createExecutionContext();
    if (!m_context) {
        std::cerr << "[SimpleInference] Failed to create context" << std::endl;
        return false;
    }

    // Get dimensions - API differs between TensorRT versions
#if TRT_USE_NEW_API
    const char* inputName = "images";
    const char* outputName = "output0";

    auto inputDims = m_engine->getTensorShape(inputName);
    auto outputDims = m_engine->getTensorShape(outputName);

    if (inputDims.nbDims <= 0 || outputDims.nbDims <= 0) {
        std::cerr << "[SimpleInference] Invalid tensor names" << std::endl;
        return false;
    }

    auto inputType = m_engine->getTensorDataType(inputName);
    auto outputType = m_engine->getTensorDataType(outputName);
#else
    int inputIdx = m_engine->getBindingIndex("images");
    int outputIdx = m_engine->getBindingIndex("output0");

    if (inputIdx < 0 || outputIdx < 0) {
        std::cerr << "[SimpleInference] Invalid binding names" << std::endl;
        return false;
    }

    auto inputDims = m_engine->getBindingDimensions(inputIdx);
    auto outputDims = m_engine->getBindingDimensions(outputIdx);

    auto inputType = m_engine->getBindingDataType(inputIdx);
    auto outputType = m_engine->getBindingDataType(outputIdx);
#endif

    m_inputFP16 = (inputType == nvinfer1::DataType::kHALF);
    m_outputFP16 = (outputType == nvinfer1::DataType::kHALF);

    m_inputH = inputDims.d[2];
    m_inputW = inputDims.d[3];
    m_numBoxes = outputDims.d[2];
    m_numClasses = outputDims.d[1] - 4;

    std::cout << "[SimpleInference] Input: " << m_inputW << "x" << m_inputH
              << " (" << (m_inputFP16 ? "FP16" : "FP32") << ")" << std::endl;
    std::cout << "[SimpleInference] Output: " << outputDims.d[1] << "x" << m_numBoxes
              << " (" << m_numClasses << " classes, " << (m_outputFP16 ? "FP16" : "FP32") << ")" << std::endl;

    // Create CUDA stream with high priority
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, greatestPriority);

    // Allocate GPU memory (use max of RGB/BGRA for raw input)
    size_t rawInputSize = m_inputH * m_inputW * 4;  // Allocate for BGRA (max)
    size_t chwInputSize = 1 * 3 * m_inputH * m_inputW * (m_inputFP16 ? sizeof(__half) : sizeof(float));
    size_t outputSizeGPU = 1 * outputDims.d[1] * m_numBoxes * (m_outputFP16 ? sizeof(__half) : sizeof(float));

    cudaMalloc(&m_d_rawInput, rawInputSize);
    cudaMalloc(&m_d_chwInput, chwInputSize);
    cudaMalloc(&m_d_output, outputSizeGPU);

    // Allocate GPU postprocessing buffers
    cudaMalloc(&m_d_decoded, kMaxDetections * sizeof(Detection));
    cudaMalloc(&m_d_decodedCount, sizeof(int));
    cudaMalloc(&m_d_bestTarget, sizeof(Detection));
    cudaMalloc(&m_d_hasTarget, sizeof(int));

    // Allocate GPU fused pipeline buffers
    cudaMalloc(&m_d_selectedTarget, sizeof(Detection));
    cudaMalloc(&m_d_pidState, sizeof(PIDState));
    cudaMalloc(&m_d_mouseMovement, sizeof(MouseMovement));

    // Initialize GPU state buffers to zero
    cudaMemset(m_d_selectedTarget, 0, sizeof(Detection));
    cudaMemset(m_d_pidState, 0, sizeof(PIDState));
    cudaMemset(m_d_mouseMovement, 0, sizeof(MouseMovement));

    // Allocate pinned host memory (max size for BGRA)
    cudaMallocHost(&m_h_rawPinned, rawInputSize);

    m_loaded = true;
    std::cout << "[SimpleInference] Engine loaded successfully" << std::endl;

    // Warm up TensorRT
    std::cout << "[SimpleInference] Warming up..." << std::endl;
    size_t warmupSize = m_inputH * m_inputW * inputBytesPerPixel();
    memset(m_h_rawPinned, 128, warmupSize);
    for (int i = 0; i < 3; i++) {
        cudaMemcpyAsync(m_d_rawInput, m_h_rawPinned, warmupSize, cudaMemcpyHostToDevice, m_stream);
        cuda_preprocessing(m_d_rawInput, m_d_chwInput,
                           m_inputW, m_inputH, inputBytesPerPixel(), m_bgraInput,
                           m_inputW, m_inputH, m_inputFP16, m_stream);
#if TRT_USE_NEW_API
        m_context->setTensorAddress("images", m_d_chwInput);
        m_context->setTensorAddress("output0", m_d_output);
        m_context->enqueueV3(m_stream);
#else
        void* bindings[2] = { m_d_chwInput, m_d_output };
        m_context->enqueueV2(bindings, m_stream, nullptr);
#endif
        cudaStreamSynchronize(m_stream);
    }
    std::cout << "[SimpleInference] Warmup complete" << std::endl;

    return true;
}

// =============================================================================
// OPTIMIZED API: Zero-copy + Single D2H Transfer + Full CUDA Graph
// =============================================================================

// Pipeline without H2D transfer - for CUDA Graph capture
void SimpleInference::executeFusedPipelinePostH2D(int width, int height,
                                                   float confThreshold, int headClassId, float headBonus,
                                                   uint32_t allowedClassMask, const PIDConfig& pidConfig,
                                                   float iouThreshold, float headYOffset, float bodyYOffset) {
    // GPU preprocessing (handles RGB/BGRA + resize)
    cuda_preprocessing(
        m_d_rawInput, m_d_chwInput,
        width, height, inputBytesPerPixel(), m_bgraInput,
        m_inputW, m_inputH, m_inputFP16, m_stream
    );

    // TensorRT inference
#if TRT_USE_NEW_API
    m_context->setTensorAddress("images", m_d_chwInput);
    m_context->setTensorAddress("output0", m_d_output);
    m_context->enqueueV3(m_stream);
#else
    void* bindings[2] = { m_d_chwInput, m_d_output };
    m_context->enqueueV2(bindings, m_stream, nullptr);
#endif

    // GPU decode
    decodeYoloGpu(
        m_d_output, m_outputFP16, m_numBoxes, m_numClasses,
        confThreshold, allowedClassMask,
        m_d_decoded, m_d_decodedCount, kMaxDetections, m_stream
    );

    // Fused target selection + PID + result packing
    float crosshairX = m_inputW * 0.5f;
    float crosshairY = m_inputH * 0.5f;

    fusedTargetSelectionAndMovementGpu(
        m_d_decoded, m_d_decodedCount, kMaxDetections,
        crosshairX, crosshairY,
        headClassId, headBonus, pidConfig,
        iouThreshold, headYOffset, bodyYOffset,
        m_d_selectedTarget, m_d_bestTarget, m_d_hasTarget,
        m_d_mouseMovement, m_d_pidState,
        m_d_inferenceResult, m_stream
    );

    // Single D2H transfer (40 bytes)
    cudaMemcpyAsync(m_h_inferenceResultPinned, m_d_inferenceResult,
                    sizeof(InferenceResult), cudaMemcpyDeviceToHost, m_stream);
}

void SimpleInference::executeFusedPipeline(void* rawInput, int width, int height,
                                            float confThreshold, int headClassId, float headBonus,
                                            uint32_t allowedClassMask, const PIDConfig& pidConfig,
                                            float iouThreshold, float headYOffset, float bodyYOffset) {
    size_t rawSize = width * height * inputBytesPerPixel();

    // H2D: Upload directly from pinned memory
    cudaMemcpyAsync(m_d_rawInput, rawInput, rawSize, cudaMemcpyHostToDevice, m_stream);

    executeFusedPipelinePostH2D(width, height, confThreshold, headClassId, headBonus,
                                 allowedClassMask, pidConfig, iouThreshold, headYOffset, bodyYOffset);
}

bool SimpleInference::captureFullGraph(float confThreshold, int headClassId, float headBonus,
                                        uint32_t allowedClassMask, const PIDConfig& pidConfig,
                                        float iouStickinessThreshold, float headYOffset, float bodyYOffset) {
    if (m_graphCaptured) {
        if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
        if (m_graph) cudaGraphDestroy(m_graph);
        m_graphExec = nullptr;
        m_graph = nullptr;
        m_graphCaptured = false;
    }

    // Allocate combined result buffer if not already
    if (!m_d_inferenceResult) {
        cudaMalloc(&m_d_inferenceResult, sizeof(InferenceResult));
    }
    if (!m_h_inferenceResultPinned) {
        cudaMallocHost(&m_h_inferenceResultPinned, sizeof(InferenceResult));
    }

    // Cache parameters
    m_cachedConfThreshold = confThreshold;
    m_cachedHeadClassId = headClassId;
    m_cachedHeadBonus = headBonus;
    m_cachedAllowedClassMask = allowedClassMask;
    m_cachedPidConfig = pidConfig;
    m_cachedIouThreshold = iouStickinessThreshold;
    m_cachedHeadYOffset = headYOffset;
    m_cachedBodyYOffset = bodyYOffset;

    // Fill pinned buffer with dummy data and upload to GPU (H2D outside graph)
    size_t rawSize = m_inputH * m_inputW * inputBytesPerPixel();
    memset(m_h_rawPinned, 128, rawSize);
    cudaMemcpyAsync(m_d_rawInput, m_h_rawPinned, rawSize, cudaMemcpyHostToDevice, m_stream);
    cudaStreamSynchronize(m_stream);

    // Begin graph capture
    cudaError_t err = cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to begin full graph capture: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Execute pipeline WITHOUT H2D for capture
    executeFusedPipelinePostH2D(m_inputW, m_inputH,
                                 confThreshold, headClassId, headBonus,
                                 allowedClassMask, pidConfig,
                                 iouStickinessThreshold, headYOffset, bodyYOffset);

    err = cudaStreamEndCapture(m_stream, &m_graph);
    if (err != cudaSuccess || !m_graph) {
        std::cerr << "[SimpleInference] Failed to end full graph capture: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to instantiate full graph: "
                  << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }

    m_graphCaptured = true;
    std::cout << "[SimpleInference] Full CUDA graph captured (preprocess+inference+postprocess)" << std::endl;
    return true;
}

// =============================================================================
// GPU CALLBACK API - Lowest latency via cudaLaunchHostFunc
// =============================================================================

// CUDA host function called when GPU work completes
void CUDART_CB SimpleInference::inferenceCompleteCallback(void* data) {
    auto* cbData = static_cast<SimpleInference::CallbackData*>(data);
    if (!cbData) return;

    try {
        if (cbData->callback) {
            cbData->callback(*cbData->resultPtr, cbData->userData);
        }
    } catch (const std::exception& e) {
        std::cerr << "[SimpleInference] Callback exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[SimpleInference] Callback exception: unknown" << std::endl;
    }

    if (cbData->owner) {
        cbData->owner->m_callbackInFlight.store(false, std::memory_order_release);
    }
}

bool SimpleInference::runInferenceWithCallback(void* pinnedData, int width, int height,
                                                float confThreshold, int headClassId, float headBonus,
                                                uint32_t allowedClassMask,
                                                const PIDConfig& pidConfig,
                                                float iouStickinessThreshold,
                                                float headYOffset, float bodyYOffset,
                                                InferenceCallback callback, void* userData) {
    if (!m_loaded) return false;
    if (m_callbackInFlight.exchange(true, std::memory_order_acq_rel)) return false;

    // Allocate result buffers if needed
    if (!m_d_inferenceResult) {
        cudaError_t err = cudaMalloc(&m_d_inferenceResult, sizeof(InferenceResult));
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaMalloc(m_d_inferenceResult) failed: "
                      << cudaGetErrorString(err) << std::endl;
            m_callbackInFlight.store(false, std::memory_order_release);
            return false;
        }
    }
    if (!m_h_inferenceResultPinned) {
        cudaError_t err = cudaMallocHost(&m_h_inferenceResultPinned, sizeof(InferenceResult));
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaMallocHost(m_h_inferenceResultPinned) failed: "
                      << cudaGetErrorString(err) << std::endl;
            m_callbackInFlight.store(false, std::memory_order_release);
            return false;
        }
    }

    // Use CUDA Graph only when shape and parameters match captured constants.
    const bool samePidConfig = (std::memcmp(&pidConfig, &m_cachedPidConfig, sizeof(PIDConfig)) == 0);
    const bool canUseGraph =
        m_graphCaptured &&
        (width == m_inputW) &&
        (height == m_inputH) &&
        (confThreshold == m_cachedConfThreshold) &&
        (headClassId == m_cachedHeadClassId) &&
        (headBonus == m_cachedHeadBonus) &&
        (allowedClassMask == m_cachedAllowedClassMask) &&
        samePidConfig &&
        (iouStickinessThreshold == m_cachedIouThreshold) &&
        (headYOffset == m_cachedHeadYOffset) &&
        (bodyYOffset == m_cachedBodyYOffset);

    if (canUseGraph) {
        // H2D outside graph - copy directly from user's pinned buffer
        size_t rawSize = width * height * inputBytesPerPixel();
        cudaError_t err = cudaMemcpyAsync(m_d_rawInput, pinnedData, rawSize, cudaMemcpyHostToDevice, m_stream);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaMemcpyAsync(m_d_rawInput) failed: "
                      << cudaGetErrorString(err) << std::endl;
            m_callbackInFlight.store(false, std::memory_order_release);
            return false;
        }

        // Launch graph (preprocess + inference + postprocess + D2H)
        err = cudaGraphLaunch(m_graphExec, m_stream);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaGraphLaunch failed: " << cudaGetErrorString(err) << std::endl;
            m_callbackInFlight.store(false, std::memory_order_release);
            return false;
        }
    } else {
        // Standard pipeline execution
        executeFusedPipeline(pinnedData, width, height,
                             confThreshold, headClassId, headBonus,
                             allowedClassMask, pidConfig,
                             iouStickinessThreshold, headYOffset, bodyYOffset);
    }

    // Setup pre-allocated callback data (no heap allocation)
    m_callbackData = {callback, userData, m_h_inferenceResultPinned, this};

    // Launch host function - fires immediately when GPU finishes
    cudaError_t err = cudaLaunchHostFunc(m_stream, SimpleInference::inferenceCompleteCallback, &m_callbackData);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] cudaLaunchHostFunc failed: " << cudaGetErrorString(err) << std::endl;
        m_callbackInFlight.store(false, std::memory_order_release);
        return false;
    }

    return true;
}

} // namespace gpa
