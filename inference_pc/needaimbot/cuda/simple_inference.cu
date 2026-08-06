// Simple TensorRT inference with CUDA Graph optimization
// Supports FP16 and FP32 models natively
// GPU postprocessing for minimal latency
// RGB input with fused normalization
#include "simple_inference.h"
#include "simple_postprocess.h"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <exception>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <NvInferVersion.h>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

// TensorRT API version compatibility
// TensorRT 10.x removed legacy binding APIs
#if NV_TENSORRT_MAJOR >= 10
    #define TRT_USE_NEW_API 1
#else
    #define TRT_USE_NEW_API 0
#endif

// RGB Preprocessing Kernels (RGB -> CHW normalized)

// Same resolution, no resize -> FP16
// src_slot is a device cell holding the actual source pointer; reading it here
// (instead of taking the pointer by value) lets a captured CUDA graph keep a
// stable kernel argument while the source buffer varies per frame (Tegra
// zero-copy: the mapped receive buffer; dGPU: the constant H2D staging buffer).
__global__ void preprocessKernelFP16(
    const uint8_t* const* __restrict__ src_slot,
    __half* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
    const uint8_t* __restrict__ src = *src_slot;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;
    int hw_size = width * height;
    int dst_idx = y * width + x;

    dst[dst_idx] = __float2half(src[src_idx] * scale_factor);
    dst[dst_idx + hw_size] = __float2half(src[src_idx + 1] * scale_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(src[src_idx + 2] * scale_factor);
}

// Same resolution, no resize -> FP32
__global__ void preprocessKernel(
    const uint8_t* const* __restrict__ src_slot,
    float* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
    const uint8_t* __restrict__ src = *src_slot;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;
    int hw_size = width * height;
    int dst_idx = y * width + x;

    dst[dst_idx] = src[src_idx] * scale_factor;
    dst[dst_idx + hw_size] = src[src_idx + 1] * scale_factor;
    dst[dst_idx + 2 * hw_size] = src[src_idx + 2] * scale_factor;
}

// Bilinear Resize + Preprocessing Kernels (fused for efficiency)

// Optimized bilinear interpolation - reads 4 pixels once for all 3 output channels
__device__ __forceinline__ void bilinearSample(
    const uint8_t* __restrict__ src,
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

    const uint8_t* p00 = src + (y0 * src_w + x0) * 3;
    const uint8_t* p10 = src + (y0 * src_w + x1) * 3;
    const uint8_t* p01 = src + (y1 * src_w + x0) * 3;
    const uint8_t* p11 = src + (y1 * src_w + x1) * 3;

    // R channel
    float r00 = p00[0], r10 = p10[0], r01 = p01[0], r11 = p11[0];
    float r0 = r00 + fx * (r10 - r00);
    float r1 = r01 + fx * (r11 - r01);
    r = r0 + fy * (r1 - r0);

    // G channel
    float g00 = p00[1], g10 = p10[1], g01 = p01[1], g11 = p11[1];
    float g0 = g00 + fx * (g10 - g00);
    float g1 = g01 + fx * (g11 - g01);
    g = g0 + fy * (g1 - g0);

    // B channel
    float b00 = p00[2], b10 = p10[2], b01 = p01[2], b11 = p11[2];
    float b0 = b00 + fx * (b10 - b00);
    float b1 = b01 + fx * (b11 - b01);
    b = b0 + fy * (b1 - b0);
}

// Bilinear resize + HWC->CHW + normalize -> FP16
__global__ void resizePreprocessKernelFP16(
    const uint8_t* const* __restrict__ src_slot,
    __half* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    float norm_factor
) {
    const uint8_t* __restrict__ src = *src_slot;
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = dx * scale_x;
    float sy = dy * scale_y;

    int hw_size = dst_w * dst_h;
    int dst_idx = dy * dst_w + dx;

    float r, g, b;
    bilinearSample(src, src_w, src_h, sx, sy, r, g, b);

    dst[dst_idx] = __float2half(r * norm_factor);
    dst[dst_idx + hw_size] = __float2half(g * norm_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(b * norm_factor);
}

// Bilinear resize + HWC->CHW + normalize -> FP32
__global__ void resizePreprocessKernel(
    const uint8_t* const* __restrict__ src_slot,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    float norm_factor
) {
    const uint8_t* __restrict__ src = *src_slot;
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = dx * scale_x;
    float sy = dy * scale_y;

    int hw_size = dst_w * dst_h;
    int dst_idx = dy * dst_w + dx;

    float r, g, b;
    bilinearSample(src, src_w, src_h, sx, sy, r, g, b);

    dst[dst_idx] = r * norm_factor;
    dst[dst_idx + hw_size] = g * norm_factor;
    dst[dst_idx + 2 * hw_size] = b * norm_factor;
}

// Preprocessing wrapper (handles RGB resize/no-resize, FP16/FP32)
// d_src_slot is a device pointer to a device cell that holds the actual source
// buffer pointer. Passing the cell (rather than the source pointer directly) lets
// a captured graph bake a stable kernel argument while the source varies per frame.
extern "C" cudaError_t cuda_preprocessing(
    const uint8_t* const* d_src_slot,
    void* dst_chw,
    int src_width, int src_height,
    int target_width, int target_height,
    bool use_fp16,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((target_width + block.x - 1) / block.x,
              (target_height + block.y - 1) / block.y);

    const float norm_factor = 1.0f / 255.0f;

    bool need_resize = (src_width != target_width) || (src_height != target_height);

    if (need_resize) {
        float scale_x = (float)(src_width - 1) / (float)(target_width - 1);
        float scale_y = (float)(src_height - 1) / (float)(target_height - 1);

        if (use_fp16) {
            resizePreprocessKernelFP16<<<grid, block, 0, stream>>>(
                d_src_slot,
                static_cast<__half*>(dst_chw),
                src_width, src_height,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        } else {
            resizePreprocessKernel<<<grid, block, 0, stream>>>(
                d_src_slot,
                static_cast<float*>(dst_chw),
                src_width, src_height,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        }
    } else {
        if (use_fp16) {
            preprocessKernelFP16<<<grid, block, 0, stream>>>(
                d_src_slot,
                static_cast<__half*>(dst_chw),
                target_width, target_height,
                norm_factor
            );
        } else {
            preprocessKernel<<<grid, block, 0, stream>>>(
                d_src_slot,
                static_cast<float*>(dst_chw),
                target_width, target_height,
                norm_factor
            );
        }
    }

    return cudaGetLastError();
}

namespace gpa {

namespace {
constexpr float kGraphParamEpsilon = 1e-6f;

inline bool nearlyEqual(float a, float b) {
    return std::fabs(a - b) <= kGraphParamEpsilon;
}

inline bool aimConfigNearlyEqual(const AimConfig& a, const AimConfig& b) {
    return nearlyEqual(a.kp_x, b.kp_x) &&
           nearlyEqual(a.kp_y, b.kp_y) &&
           nearlyEqual(a.p_softness_x, b.p_softness_x) &&
           nearlyEqual(a.p_softness_y, b.p_softness_y) &&
           nearlyEqual(a.kd_x, b.kd_x) &&
           nearlyEqual(a.kd_y, b.kd_y) &&
           nearlyEqual(a.max_step, b.max_step) &&
           nearlyEqual(a.distance_stickiness_factor, b.distance_stickiness_factor) &&
           (a.track_persistence_frames == b.track_persistence_frames) &&
           nearlyEqual(a.inflight_comp, b.inflight_comp) &&
           nearlyEqual(a.deadtime_frames, b.deadtime_frames) &&
           nearlyEqual(a.ff_gain, b.ff_gain) &&
           nearlyEqual(a.ff_ego_lag, b.ff_ego_lag) &&
           nearlyEqual(a.ff_v_ema, b.ff_v_ema) &&
           nearlyEqual(a.lead_vgate, b.lead_vgate) &&
           nearlyEqual(a.lead_err_gate, b.lead_err_gate) &&
           nearlyEqual(a.predict_frames, b.predict_frames) &&
           nearlyEqual(a.class_switch_reject, b.class_switch_reject) &&
           nearlyEqual(a.head_deprioritized, b.head_deprioritized) &&
           nearlyEqual(a.aim_h_ema, b.aim_h_ema) &&
           nearlyEqual(a.aim_y_scale, b.aim_y_scale) &&
           nearlyEqual(a.oneeuro_enabled, b.oneeuro_enabled) &&
           nearlyEqual(a.oneeuro_min_cutoff, b.oneeuro_min_cutoff) &&
           nearlyEqual(a.oneeuro_beta, b.oneeuro_beta) &&
           nearlyEqual(a.shoot_offset_x, b.shoot_offset_x) &&
           nearlyEqual(a.shoot_offset_y, b.shoot_offset_y);
}

cudaGraphNode_t findGraphH2DMemcpyNode(cudaGraph_t graph, const void* dst, size_t bytes) {
    if (!graph || !dst || bytes == 0) return nullptr;

    size_t nodeCount = 0;
    cudaError_t err = cudaGraphGetNodes(graph, nullptr, &nodeCount);
    if (err != cudaSuccess || nodeCount == 0) {
        return nullptr;
    }

    std::vector<cudaGraphNode_t> nodes(nodeCount);
    err = cudaGraphGetNodes(graph, nodes.data(), &nodeCount);
    if (err != cudaSuccess) {
        return nullptr;
    }

    for (size_t i = 0; i < nodeCount; ++i) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess ||
            type != cudaGraphNodeTypeMemcpy) {
            continue;
        }

        cudaMemcpy3DParms params{};
        if (cudaGraphMemcpyNodeGetParams(nodes[i], &params) != cudaSuccess) {
            continue;
        }
        if (params.kind == cudaMemcpyHostToDevice &&
            params.dstPtr.ptr == dst &&
            params.extent.width == bytes &&
            params.extent.height == 1 &&
            params.extent.depth == 1) {
            return nodes[i];
        }
    }
    return nullptr;
}
} // namespace

void SimpleInference::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

SimpleInference::SimpleInference() {
    for (int i = 0; i < kMaxCallbacksInFlight; ++i) {
        m_callbackSlotBusy[static_cast<size_t>(i)].store(false, std::memory_order_relaxed);
        m_callbackSlotPending[static_cast<size_t>(i)].store(false, std::memory_order_relaxed);
        m_callbackEvents[static_cast<size_t>(i)] = nullptr;
        m_stageEvtStart[static_cast<size_t>(i)] = nullptr;
        m_stageEvtPostH2D[static_cast<size_t>(i)] = nullptr;
        m_stageEvtPostPreprocess[static_cast<size_t>(i)] = nullptr;
        m_stageEvtPostInference[static_cast<size_t>(i)] = nullptr;
        m_stageEvtPostPostprocess[static_cast<size_t>(i)] = nullptr;
        m_stageEvtEnd[static_cast<size_t>(i)] = nullptr;
        m_stageEvtValid[static_cast<size_t>(i)].store(false, std::memory_order_relaxed);
    }
}

SimpleInference::~SimpleInference() {
    // Flush pending stream work so callback state is no longer in-flight.
    if (m_stream) cudaStreamSynchronize(m_stream);

    {
        std::lock_guard<std::mutex> lock(m_callbackWorkerMutex);
        m_callbackWorkerRunning.store(false, std::memory_order_release);
    }
    m_callbackWorkerCv.notify_all();
    if (m_callbackWorkerThread.joinable()) {
        m_callbackWorkerThread.join();
    }

    for (int i = 0; i < kMaxCallbacksInFlight; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (m_callbackEvents[idx]) {
            cudaEventDestroy(m_callbackEvents[idx]);
            m_callbackEvents[idx] = nullptr;
        }
        if (m_stageEvtStart[idx]) { cudaEventDestroy(m_stageEvtStart[idx]); m_stageEvtStart[idx] = nullptr; }
        if (m_stageEvtPostH2D[idx]) { cudaEventDestroy(m_stageEvtPostH2D[idx]); m_stageEvtPostH2D[idx] = nullptr; }
        if (m_stageEvtPostPreprocess[idx]) { cudaEventDestroy(m_stageEvtPostPreprocess[idx]); m_stageEvtPostPreprocess[idx] = nullptr; }
        if (m_stageEvtPostInference[idx]) { cudaEventDestroy(m_stageEvtPostInference[idx]); m_stageEvtPostInference[idx] = nullptr; }
        if (m_stageEvtPostPostprocess[idx]) { cudaEventDestroy(m_stageEvtPostPostprocess[idx]); m_stageEvtPostPostprocess[idx] = nullptr; }
        if (m_stageEvtEnd[idx]) { cudaEventDestroy(m_stageEvtEnd[idx]); m_stageEvtEnd[idx] = nullptr; }
    }

    // Destroy CUDA graphs
    destroyFullGraphs();

    // Free GPU memory
    if (m_d_rawInput) cudaFree(m_d_rawInput);
    if (m_d_chwInput) cudaFree(m_d_chwInput);
    if (m_d_output) cudaFree(m_d_output);
    if (m_d_srcPtr) cudaFree(m_d_srcPtr);
    // m_d_resultMapped[] are device aliases of the mapped pinned result buffers,
    // not separate allocations - they are released when m_h_inferenceResultPinned
    // is freed below with cudaFreeHost().

    // Free GPU fused pipeline buffers
    if (m_d_selectedTarget) cudaFree(m_d_selectedTarget);
    if (m_d_aimState) cudaFree(m_d_aimState);
    if (m_d_runtimeAimConfig) cudaFree(m_d_runtimeAimConfig);
    if (m_d_stage1BestDist) cudaFree(m_d_stage1BestDist);
    if (m_d_stage1DistScore) cudaFree(m_d_stage1DistScore);
    if (m_d_stage1BestIou) cudaFree(m_d_stage1BestIou);
    if (m_d_stage1IouScore) cudaFree(m_d_stage1IouScore);

    // Free result buffers
    for (int i = 0; i < kMaxCallbacksInFlight; ++i) {
        if (m_d_inferenceResult[i]) cudaFree(m_d_inferenceResult[i]);
        if (m_h_inferenceResultPinned[i]) cudaFreeHost(m_h_inferenceResultPinned[i]);
    }

    // Free pinned host memory
    if (m_h_rawPinned) cudaFreeHost(m_h_rawPinned);

    if (m_stream) cudaStreamDestroy(m_stream);
#if TRT_USE_NEW_API
    if (m_context) delete m_context;
    if (m_engine) delete m_engine;
    if (m_runtime) delete m_runtime;
#else
    if (m_context) m_context->destroy();
    if (m_engine) m_engine->destroy();
    if (m_runtime) m_runtime->destroy();
#endif
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
    m_crosshairX = m_inputW * 0.5f;
    m_crosshairY = m_inputH * 0.5f;
    if (m_maxDetections > m_numBoxes) {
        m_maxDetections = m_numBoxes;
    }
    int stage1ThreadsRequired = 256;
    if (m_numBoxes <= 128) stage1ThreadsRequired = 128;
    if (m_numBoxes <= 64) stage1ThreadsRequired = 64;
    if (m_numBoxes <= 32) stage1ThreadsRequired = 32;
    const int stage1BlocksRequired =
        (m_numBoxes + stage1ThreadsRequired - 1) / stage1ThreadsRequired;
    if (m_maxDetections < stage1BlocksRequired) {
        std::cout << "[SimpleInference] Raising max detections from " << m_maxDetections
                  << " to " << stage1BlocksRequired
                  << " (minimum required for stage-1 candidate blocks)" << std::endl;
        m_maxDetections = stage1BlocksRequired;
    }

    std::cout << "[SimpleInference] Input: " << m_inputW << "x" << m_inputH
              << " (" << (m_inputFP16 ? "FP16" : "FP32") << ")" << std::endl;
    std::cout << "[SimpleInference] Output: " << outputDims.d[1] << "x" << m_numBoxes
              << " (" << m_numClasses << " classes, " << (m_outputFP16 ? "FP16" : "FP32") << ")" << std::endl;
    std::cout << "[SimpleInference] Max detections: " << m_maxDetections << std::endl;

    // Create CUDA stream with high priority
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, greatestPriority);

    // Detect Tegra/Orin unified memory: an integrated GPU that can map host
    // memory lets the preprocess kernel read the pinned receive buffer directly
    // and the postprocess write the result straight into mapped pinned memory -
    // so we can drop the per-frame H2D and D2H copies entirely.
    {
        int dev = 0;
        cudaGetDevice(&dev);
        int integrated = 0, canMapHost = 0;
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        cudaDeviceGetAttribute(&canMapHost, cudaDevAttrCanMapHostMemory, dev);
        m_zeroCopy = (integrated != 0) && (canMapHost != 0);
        std::cout << "[SimpleInference] Unified-memory zero-copy: "
                  << (m_zeroCopy ? "ENABLED (Tegra: no H2D/D2H copy)" : "DISABLED (discrete GPU)")
                  << std::endl;
    }

    // Allocate GPU memory. Keep enough raw-input space for the common 640x640
    // capture case even when the model input is smaller and preprocessing resizes.
    constexpr size_t kDefaultRawInputPixels = 640ull * 640ull;
    size_t rawInputSize =
        std::max(static_cast<size_t>(m_inputH) * static_cast<size_t>(m_inputW),
                 kDefaultRawInputPixels) *
        static_cast<size_t>(inputBytesPerPixel());
    m_rawInputCapacityBytes = rawInputSize;
    size_t chwInputSize = 1 * 3 * m_inputH * m_inputW * (m_inputFP16 ? sizeof(__half) : sizeof(float));
    size_t outputSizeGPU = 1 * outputDims.d[1] * m_numBoxes * (m_outputFP16 ? sizeof(__half) : sizeof(float));

    cudaMalloc(&m_d_rawInput, rawInputSize);
    cudaMalloc(&m_d_chwInput, chwInputSize);
    cudaMalloc(&m_d_output, outputSizeGPU);

    // Source-pointer indirection cell (see m_d_srcPtr). Seed it with the H2D
    // staging buffer so warmup and the discrete-GPU path read m_d_rawInput; on
    // Tegra it is overwritten per frame with the current receive buffer's device
    // pointer (no H2D copy).
    cudaMalloc(&m_d_srcPtr, sizeof(void*));
    cudaMemcpy(m_d_srcPtr, &m_d_rawInput, sizeof(void*), cudaMemcpyHostToDevice);

    // Allocate GPU fused pipeline buffers
    cudaMalloc(&m_d_selectedTarget, sizeof(Detection));
    cudaMalloc(&m_d_aimState, sizeof(AimState));
    cudaMalloc(&m_d_runtimeAimConfig, sizeof(AimConfig));
    cudaMalloc(&m_d_stage1BestDist, static_cast<size_t>(m_maxDetections) * sizeof(Detection));
    cudaMalloc(&m_d_stage1DistScore, static_cast<size_t>(m_maxDetections) * sizeof(float));
    cudaMalloc(&m_d_stage1BestIou, static_cast<size_t>(m_maxDetections) * sizeof(Detection));
    cudaMalloc(&m_d_stage1IouScore, static_cast<size_t>(m_maxDetections) * sizeof(float));

    // Initialize GPU state buffers to zero
    cudaMemset(m_d_selectedTarget, 0, sizeof(Detection));
    cudaMemset(m_d_aimState, 0, sizeof(AimState));
    cudaMemset(m_d_runtimeAimConfig, 0, sizeof(AimConfig));

    // Allocate pinned host memory for RGB input.
    cudaMallocHost(&m_h_rawPinned, rawInputSize);
    // Allocate result buffers per callback slot.
    for (int i = 0; i < kMaxCallbacksInFlight; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (m_zeroCopy) {
            // Mapped pinned result buffer: the postprocess kernel writes directly
            // into host-visible memory (via the device alias), so there is no D2H
            // copy. The callback reads m_h_inferenceResultPinned[i] as before.
            cudaHostAlloc(&m_h_inferenceResultPinned[i], sizeof(InferenceResult),
                          cudaHostAllocMapped);
            cudaHostGetDevicePointer(&m_d_resultMapped[i], m_h_inferenceResultPinned[i], 0);
            m_d_inferenceResult[i] = nullptr;  // unused in zero-copy mode
        } else {
            cudaMalloc(&m_d_inferenceResult[i], sizeof(InferenceResult));
            cudaMallocHost(&m_h_inferenceResultPinned[i], sizeof(InferenceResult));
            m_d_resultMapped[i] = nullptr;
        }
        cudaEventCreateWithFlags(&m_callbackEvents[idx], cudaEventDisableTiming);
        m_callbackSlotPending[idx].store(false, std::memory_order_relaxed);
        m_callbackSlotBusy[idx].store(false, std::memory_order_relaxed);
        if (m_stageTimingEnabled) {
            cudaEventCreate(&m_stageEvtStart[idx]);
            cudaEventCreate(&m_stageEvtPostH2D[idx]);
            cudaEventCreate(&m_stageEvtPostPreprocess[idx]);
            cudaEventCreate(&m_stageEvtPostInference[idx]);
            cudaEventCreate(&m_stageEvtPostPostprocess[idx]);
            cudaEventCreate(&m_stageEvtEnd[idx]);
        }
        m_stageEvtValid[idx].store(false, std::memory_order_relaxed);
    }

#if TRT_USE_NEW_API
    // Tensor addresses are static in this pipeline, bind once.
    m_context->setTensorAddress("images", m_d_chwInput);
    m_context->setTensorAddress("output0", m_d_output);
    m_tensorAddressesBound = true;
#endif

    m_loaded = true;
    m_callbackWorkerRunning.store(true, std::memory_order_release);
    m_callbackWorkerThread = std::thread(&SimpleInference::callbackWorkerLoop, this);
    std::cout << "[SimpleInference] Engine loaded successfully" << std::endl;

    // Warm up TensorRT
    std::cout << "[SimpleInference] Warming up..." << std::endl;
    size_t warmupSize = m_inputH * m_inputW * inputBytesPerPixel();
    memset(m_h_rawPinned, 128, warmupSize);
    for (int i = 0; i < 3; i++) {
        cudaMemcpyAsync(m_d_rawInput, m_h_rawPinned, warmupSize, cudaMemcpyHostToDevice, m_stream);
        // m_d_srcPtr currently points at m_d_rawInput (seeded above), so the
        // preprocess reads the just-uploaded warmup frame.
        cuda_preprocessing(reinterpret_cast<const uint8_t* const*>(m_d_srcPtr), m_d_chwInput,
                           m_inputW, m_inputH, m_inputW, m_inputH, m_inputFP16, m_stream);
#if TRT_USE_NEW_API
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

// OPTIMIZED API: Pinned H2D + Single D2H Transfer + Full CUDA Graph

// Resolve the device pointer for a mapped pinned host buffer (Tegra zero-copy).
// Do not cache this mapping: UDPCapture can free and replace its mapped buffer
// pool on a resolution increase, and a later allocation may reuse the same host
// address with a different device alias - an address-keyed cache would then hand
// back a freed alias.
const uint8_t* SimpleInference::pinnedDevicePtr(void* hostPtr) {
    if (!hostPtr) return nullptr;
    void* devPtr = nullptr;
    if (cudaHostGetDevicePointer(&devPtr, hostPtr, 0) != cudaSuccess || !devPtr) {
        cudaGetLastError();  // swallow so the caller can fall back to a normal H2D
        return nullptr;
    }
    return static_cast<const uint8_t*>(devPtr);
}

// Pipeline without H2D transfer - for CUDA Graph capture
bool SimpleInference::executeFusedPipelinePostH2D(int width, int height,
                                                  float confThreshold, int headClassId,
                                                  uint32_t allowedClassMask, const AimConfig& aimConfig,
                                                  float iouThreshold, float headYOffset, float bodyYOffset,
                                                  int resultSlot) {
    (void)aimConfig;
    if (resultSlot < 0 || resultSlot >= kMaxCallbacksInFlight) {
        std::cerr << "[SimpleInference] Invalid result slot: " << resultSlot << std::endl;
        return false;
    }
    InferenceResult* hResultSlot = m_h_inferenceResultPinned[static_cast<size_t>(resultSlot)];
    // On Tegra the postprocess writes straight into the mapped pinned result
    // (device alias); on a discrete GPU it writes a device buffer that is then
    // copied D2H into the pinned buffer below.
    InferenceResult* dResultSlot = m_zeroCopy
        ? m_d_resultMapped[static_cast<size_t>(resultSlot)]
        : m_d_inferenceResult[static_cast<size_t>(resultSlot)];
    if (!dResultSlot || !hResultSlot) {
        std::cerr << "[SimpleInference] Result slot not allocated: " << resultSlot << std::endl;
        return false;
    }

    const size_t slotIdx = static_cast<size_t>(resultSlot);
    if (m_stageTimingEnabled && m_stageEvtPostH2D[slotIdx]) {
        cudaEventRecord(m_stageEvtPostH2D[slotIdx], m_stream);
    }

    // GPU preprocessing (RGB + optional resize). Reads the source pointer from
    // m_d_srcPtr (the current receive buffer on Tegra, or m_d_rawInput on dGPU).
    cudaError_t preprocessErr = cuda_preprocessing(
        reinterpret_cast<const uint8_t* const*>(m_d_srcPtr), m_d_chwInput,
        width, height, m_inputW, m_inputH, m_inputFP16, m_stream
    );
    if (preprocessErr != cudaSuccess) {
        std::cerr << "[SimpleInference] cuda_preprocessing failed: "
                  << cudaGetErrorString(preprocessErr) << std::endl;
        return false;
    }
    if (m_stageTimingEnabled && m_stageEvtPostPreprocess[slotIdx]) {
        cudaEventRecord(m_stageEvtPostPreprocess[slotIdx], m_stream);
    }

    // TensorRT inference
    bool enqueueOk = false;
#if TRT_USE_NEW_API
    if (!m_tensorAddressesBound) {
        m_context->setTensorAddress("images", m_d_chwInput);
        m_context->setTensorAddress("output0", m_d_output);
        m_tensorAddressesBound = true;
    }
    enqueueOk = m_context->enqueueV3(m_stream);
#else
    void* bindings[2] = { m_d_chwInput, m_d_output };
    enqueueOk = m_context->enqueueV2(bindings, m_stream, nullptr);
#endif
    if (!enqueueOk) {
        std::cerr << "[SimpleInference] TensorRT enqueue failed" << std::endl;
        return false;
    }
    if (m_stageTimingEnabled && m_stageEvtPostInference[slotIdx]) {
        cudaEventRecord(m_stageEvtPostInference[slotIdx], m_stream);
    }

    // One-pass GPU postprocess: decode + target select + movement + result packing
    cudaError_t postErr = postprocessYoloFusedGpu(
        m_d_output, m_outputFP16, m_numBoxes, m_numClasses,
        confThreshold, allowedClassMask, m_maxDetections,
        static_cast<float>(std::max(m_inputW, m_inputH)),
        m_crosshairX, m_crosshairY,
        static_cast<float>(width) / static_cast<float>(m_inputW),
        static_cast<float>(height) / static_cast<float>(m_inputH),
        headClassId, m_d_runtimeAimConfig,
        iouThreshold, headYOffset, bodyYOffset,
        m_d_selectedTarget, m_d_aimState,
        dResultSlot,
        m_d_stage1BestDist, m_d_stage1DistScore,
        m_d_stage1BestIou, m_d_stage1IouScore,
        m_stream
    );
    if (postErr != cudaSuccess) {
        std::cerr << "[SimpleInference] postprocessYoloFusedGpu failed: "
                  << cudaGetErrorString(postErr) << std::endl;
        return false;
    }
    if (m_stageTimingEnabled && m_stageEvtPostPostprocess[slotIdx]) {
        cudaEventRecord(m_stageEvtPostPostprocess[slotIdx], m_stream);
    }

    // Single D2H transfer (dGPU only). On Tegra the postprocess already wrote the
    // result into mapped pinned memory, so no copy is needed - the host reads it
    // once the completion event fires.
    if (!m_zeroCopy) {
        cudaError_t d2hErr = cudaMemcpyAsync(hResultSlot, dResultSlot,
                                             sizeof(InferenceResult), cudaMemcpyDeviceToHost, m_stream);
        if (d2hErr != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaMemcpyAsync(result D2H) failed: "
                      << cudaGetErrorString(d2hErr) << std::endl;
            return false;
        }
    }
    if (m_stageTimingEnabled && m_stageEvtEnd[slotIdx]) {
        cudaEventRecord(m_stageEvtEnd[slotIdx], m_stream);
    }
    return true;
}

bool SimpleInference::executeFusedPipeline(void* rawInput, int width, int height,
                                           float confThreshold, int headClassId,
                                           uint32_t allowedClassMask, const AimConfig& aimConfig,
                                           float iouThreshold, float headYOffset, float bodyYOffset,
                                           int resultSlot) {
    size_t rawSize = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(inputBytesPerPixel());

    const size_t slotIdx = static_cast<size_t>(resultSlot);
    if (m_stageTimingEnabled && resultSlot >= 0 && resultSlot < kMaxCallbacksInFlight &&
        m_stageEvtStart[slotIdx]) {
        cudaEventRecord(m_stageEvtStart[slotIdx], m_stream);
    }

    if (m_zeroCopy) {
        // Zero-copy: repoint the preprocess source cell at the frame's mapped
        // device pointer instead of copying the frame. An 8-byte pointer update
        // replaces the ~1.2MB H2D. Falls back to a real H2D if this particular
        // buffer turns out not to be device-mappable.
        const uint8_t* dev = pinnedDevicePtr(rawInput);
        if (dev) {
            m_h_srcPtrStage[slotIdx] = const_cast<uint8_t*>(dev);
        } else {
            cudaError_t h2dErr = cudaMemcpyAsync(m_d_rawInput, rawInput, rawSize,
                                                 cudaMemcpyHostToDevice, m_stream);
            if (h2dErr != cudaSuccess) {
                std::cerr << "[SimpleInference] cudaMemcpyAsync(m_d_rawInput) failed: "
                          << cudaGetErrorString(h2dErr) << std::endl;
                return false;
            }
            m_h_srcPtrStage[slotIdx] = m_d_rawInput;
        }
        cudaError_t ptrErr = cudaMemcpyAsync(m_d_srcPtr, &m_h_srcPtrStage[slotIdx],
                                             sizeof(void*), cudaMemcpyHostToDevice, m_stream);
        if (ptrErr != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaMemcpyAsync(src ptr) failed: "
                      << cudaGetErrorString(ptrErr) << std::endl;
            return false;
        }
    } else {
        // Discrete GPU: H2D into the staging buffer. m_d_srcPtr already points at
        // m_d_rawInput (seeded at load), so the preprocess reads the uploaded frame.
        cudaError_t h2dErr = cudaMemcpyAsync(m_d_rawInput, rawInput, rawSize,
                                             cudaMemcpyHostToDevice, m_stream);
        if (h2dErr != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaMemcpyAsync(m_d_rawInput) failed: "
                      << cudaGetErrorString(h2dErr) << std::endl;
            return false;
        }
    }

    return executeFusedPipelinePostH2D(width, height, confThreshold, headClassId,
                                       allowedClassMask, aimConfig, iouThreshold,
                                       headYOffset, bodyYOffset, resultSlot);
}

void SimpleInference::destroyBucket(GraphShapeBucket& bucket) {
    for (int i = 0; i < kMaxCallbacksInFlight; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (bucket.graphExecs[idx]) {
            cudaGraphExecDestroy(bucket.graphExecs[idx]);
            bucket.graphExecs[idx] = nullptr;
        }
        if (bucket.graphs[idx]) {
            cudaGraphDestroy(bucket.graphs[idx]);
            bucket.graphs[idx] = nullptr;
        }
        bucket.h2dNodes[idx] = nullptr;
        bucket.rawSizes[idx] = 0;
    }
    bucket.used = false;
    bucket.sourceW = 0;
    bucket.sourceH = 0;
    bucket.slotCount = 0;
    bucket.lastUseTick = 0;
}

void SimpleInference::destroyFullGraphs() {
    for (auto& bucket : m_shapeBuckets) {
        destroyBucket(bucket);
    }
    m_graphUseTickCounter = 0;
}

int SimpleInference::findBucketIndex(int sourceWidth, int sourceHeight) const {
    if (sourceWidth <= 0 || sourceHeight <= 0) return -1;
    for (int i = 0; i < kMaxGraphShapes; ++i) {
        const auto& b = m_shapeBuckets[static_cast<size_t>(i)];
        if (b.used && b.sourceW == sourceWidth && b.sourceH == sourceHeight) {
            return i;
        }
    }
    return -1;
}

int SimpleInference::pickBucketForCapture(int sourceWidth, int sourceHeight) {
    // Reuse if shape already present.
    int existing = findBucketIndex(sourceWidth, sourceHeight);
    if (existing >= 0) return existing;
    // Prefer an empty slot.
    for (int i = 0; i < kMaxGraphShapes; ++i) {
        if (!m_shapeBuckets[static_cast<size_t>(i)].used) return i;
    }
    // Evict the LRU bucket.
    int victim = 0;
    uint64_t oldest = m_shapeBuckets[0].lastUseTick;
    for (int i = 1; i < kMaxGraphShapes; ++i) {
        const auto& b = m_shapeBuckets[static_cast<size_t>(i)];
        if (b.lastUseTick < oldest) {
            oldest = b.lastUseTick;
            victim = i;
        }
    }
    destroyBucket(m_shapeBuckets[static_cast<size_t>(victim)]);
    return victim;
}

void SimpleInference::touchBucket(int bucketIndex) {
    if (bucketIndex < 0 || bucketIndex >= kMaxGraphShapes) return;
    m_shapeBuckets[static_cast<size_t>(bucketIndex)].lastUseTick =
        ++m_graphUseTickCounter;
}

bool SimpleInference::uploadRuntimeAimConfig(const AimConfig& aimConfig, bool force) {
    if (!m_d_runtimeAimConfig) return false;
    if (!force && m_hasEnqueuedRuntimeAimConfig &&
        aimConfigNearlyEqual(aimConfig, m_enqueuedRuntimeAimConfig)) {
        return true;
    }

    const cudaError_t err = cudaMemcpyAsync(
        m_d_runtimeAimConfig, &aimConfig, sizeof(AimConfig),
        cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] cudaMemcpyAsync(runtime AimConfig) failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    m_enqueuedRuntimeAimConfig = aimConfig;
    m_hasEnqueuedRuntimeAimConfig = true;
    return true;
}

bool SimpleInference::nonShapeParamsMatch(float confThreshold, int headClassId,
                                          uint32_t allowedClassMask, const AimConfig& aimConfig,
                                          float iouStickinessThreshold, float headYOffset,
                                          float bodyYOffset) const {
    (void)aimConfig;
    return (headClassId == m_cachedHeadClassId) &&
           (allowedClassMask == m_cachedAllowedClassMask) &&
           nearlyEqual(confThreshold, m_cachedConfThreshold) &&
           nearlyEqual(iouStickinessThreshold, m_cachedIouThreshold) &&
           nearlyEqual(headYOffset, m_cachedHeadYOffset) &&
           nearlyEqual(bodyYOffset, m_cachedBodyYOffset);
}

bool SimpleInference::isFullGraphReadyForShape(int sourceWidth, int sourceHeight,
                                               int graphSlotCount,
                                               float confThreshold, int headClassId,
                                               uint32_t allowedClassMask, const AimConfig& aimConfig,
                                               float iouStickinessThreshold, float headYOffset,
                                               float bodyYOffset) const {
    if (graphSlotCount <= 0 || graphSlotCount > kMaxCallbacksInFlight) return false;
    if (!nonShapeParamsMatch(confThreshold, headClassId,
                             allowedClassMask, aimConfig, iouStickinessThreshold,
                             headYOffset, bodyYOffset)) {
        return false;
    }
    const int bucketIndex = findBucketIndex(sourceWidth, sourceHeight);
    if (bucketIndex < 0) return false;
    const auto& b = m_shapeBuckets[static_cast<size_t>(bucketIndex)];
    if (graphSlotCount > b.slotCount) return false;
    for (int i = 0; i < graphSlotCount; ++i) {
        if (!b.graphExecs[static_cast<size_t>(i)]) return false;
    }
    return true;
}

SimpleInference::LaunchStats SimpleInference::takeLaunchStats() {
    LaunchStats stats;
    stats.graph = m_graphLaunchCount.exchange(0, std::memory_order_relaxed);
    stats.standard = m_standardLaunchCount.exchange(0, std::memory_order_relaxed);
    stats.graphFallback = m_graphFallbackCount.exchange(0, std::memory_order_relaxed);
    return stats;
}

namespace {
inline uint64_t millisToUs(float millis) {
    if (!(millis >= 0.0f)) return 0;
    const double us = static_cast<double>(millis) * 1000.0;
    if (us >= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(us);
}

inline void atomicUpdateMax(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}
} // namespace

void SimpleInference::recordStageTimings(int slotIndex) {
    if (slotIndex < 0 || slotIndex >= kMaxCallbacksInFlight) return;
    const size_t idx = static_cast<size_t>(slotIndex);
    cudaEvent_t e0 = m_stageEvtStart[idx];
    cudaEvent_t e1 = m_stageEvtPostH2D[idx];
    cudaEvent_t e2 = m_stageEvtPostPreprocess[idx];
    cudaEvent_t e3 = m_stageEvtPostInference[idx];
    cudaEvent_t e4 = m_stageEvtPostPostprocess[idx];
    cudaEvent_t e5 = m_stageEvtEnd[idx];
    if (!e0 || !e1 || !e2 || !e3 || !e4 || !e5) return;

    // Report the first failure instead of swallowing it: every stage read coming
    // back 0 is indistinguishable from "the GPU really took 0us", which is how a
    // broken probe silently reads as a fast pipeline.
    auto delta = [](cudaEvent_t a, cudaEvent_t b) -> uint64_t {
        float ms = 0.0f;
        const cudaError_t err = cudaEventElapsedTime(&ms, a, b);
        if (err != cudaSuccess) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                std::cerr << "[SimpleInference] stage timing unavailable: "
                          << cudaGetErrorString(err)
                          << " (events recorded inside a captured CUDA graph cannot"
                             " always be timed; run with idle_graph_precapture off"
                             " or compare Cb/E2E instead)" << std::endl;
            }
            cudaGetLastError();
            return 0;
        }
        return millisToUs(ms);
    };
    const uint64_t h2dUs = delta(e0, e1);
    const uint64_t preprocessUs = delta(e1, e2);
    const uint64_t inferenceUs = delta(e2, e3);
    const uint64_t postprocessUs = delta(e3, e4);
    const uint64_t d2hUs = delta(e4, e5);

    m_stageSamples.fetch_add(1, std::memory_order_relaxed);
    m_stageH2DUsTotal.fetch_add(h2dUs, std::memory_order_relaxed);
    m_stagePreprocessUsTotal.fetch_add(preprocessUs, std::memory_order_relaxed);
    m_stageInferenceUsTotal.fetch_add(inferenceUs, std::memory_order_relaxed);
    m_stagePostprocessUsTotal.fetch_add(postprocessUs, std::memory_order_relaxed);
    m_stageD2HUsTotal.fetch_add(d2hUs, std::memory_order_relaxed);
    atomicUpdateMax(m_stageH2DUsMax, h2dUs);
    atomicUpdateMax(m_stagePreprocessUsMax, preprocessUs);
    atomicUpdateMax(m_stageInferenceUsMax, inferenceUs);
    atomicUpdateMax(m_stagePostprocessUsMax, postprocessUs);
    atomicUpdateMax(m_stageD2HUsMax, d2hUs);
}

SimpleInference::StageTimingStats SimpleInference::takeStageTimingStats() {
    StageTimingStats out;
    out.samples = m_stageSamples.exchange(0, std::memory_order_relaxed);
    out.h2dUsTotal = m_stageH2DUsTotal.exchange(0, std::memory_order_relaxed);
    out.preprocessUsTotal = m_stagePreprocessUsTotal.exchange(0, std::memory_order_relaxed);
    out.inferenceUsTotal = m_stageInferenceUsTotal.exchange(0, std::memory_order_relaxed);
    out.postprocessUsTotal = m_stagePostprocessUsTotal.exchange(0, std::memory_order_relaxed);
    out.d2hUsTotal = m_stageD2HUsTotal.exchange(0, std::memory_order_relaxed);
    out.h2dUsMax = m_stageH2DUsMax.exchange(0, std::memory_order_relaxed);
    out.preprocessUsMax = m_stagePreprocessUsMax.exchange(0, std::memory_order_relaxed);
    out.inferenceUsMax = m_stageInferenceUsMax.exchange(0, std::memory_order_relaxed);
    out.postprocessUsMax = m_stagePostprocessUsMax.exchange(0, std::memory_order_relaxed);
    out.d2hUsMax = m_stageD2HUsMax.exchange(0, std::memory_order_relaxed);
    return out;
}

bool SimpleInference::ensureRawInputCapacity(size_t requiredBytes) {
    if (requiredBytes == 0) return false;
    if (requiredBytes <= m_rawInputCapacityBytes && m_d_rawInput && m_h_rawPinned) {
        return true;
    }
    if (m_callbacksInFlight.load(std::memory_order_acquire) > 0) {
        return false;
    }

    if (m_stream) {
        cudaStreamSynchronize(m_stream);
    }

    void* newDeviceRaw = nullptr;
    uint8_t* newHostRaw = nullptr;
    cudaError_t err = cudaMalloc(&newDeviceRaw, requiredBytes);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to grow raw device buffer to "
                  << requiredBytes << " bytes: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    err = cudaMallocHost(&newHostRaw, requiredBytes);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to grow raw pinned buffer to "
                  << requiredBytes << " bytes: " << cudaGetErrorString(err) << std::endl;
        cudaFree(newDeviceRaw);
        return false;
    }

    destroyFullGraphs();
    if (m_d_rawInput) cudaFree(m_d_rawInput);
    if (m_h_rawPinned) cudaFreeHost(m_h_rawPinned);
    m_d_rawInput = newDeviceRaw;
    m_h_rawPinned = newHostRaw;
    m_rawInputCapacityBytes = requiredBytes;
    // Re-seed the preprocess source cell: it still holds the OLD (now freed)
    // m_d_rawInput. On the discrete-GPU path the cell is a constant alias of
    // m_d_rawInput, so leaving it stale would make the preprocess read freed
    // memory. (On Tegra the cell is overwritten per frame, but re-seeding is
    // harmless.) Safe to do synchronously: the stream was drained above and no
    // callbacks are in flight.
    if (m_d_srcPtr) {
        cudaMemcpy(m_d_srcPtr, &m_d_rawInput, sizeof(void*), cudaMemcpyHostToDevice);
    }
    return true;
}

bool SimpleInference::captureFullGraphForShape(int sourceWidth, int sourceHeight,
                                                int graphSlotCount,
                                                float confThreshold, int headClassId,
                                                uint32_t allowedClassMask, const AimConfig& aimConfig,
                                                float iouStickinessThreshold, float headYOffset,
                                                float bodyYOffset) {
    if (!m_loaded || sourceWidth <= 0 || sourceHeight <= 0) {
        return false;
    }
    if (graphSlotCount < 1) graphSlotCount = 1;
    if (graphSlotCount > kMaxCallbacksInFlight) graphSlotCount = kMaxCallbacksInFlight;
    if (m_callbacksInFlight.load(std::memory_order_acquire) > 0) {
        std::cerr << "[SimpleInference] Cannot recapture CUDA graph while callbacks are in flight"
                  << std::endl;
        return false;
    }

    const size_t rawSize = static_cast<size_t>(sourceWidth) * static_cast<size_t>(sourceHeight) *
                           static_cast<size_t>(inputBytesPerPixel());
    if (!ensureRawInputCapacity(rawSize)) {
        std::cerr << "[SimpleInference] Raw input buffer is too small for graph source shape "
                  << sourceWidth << "x" << sourceHeight << std::endl;
        return false;
    }

    // Non-shape parameter change invalidates every cached bucket.
    const bool paramsChanged = !nonShapeParamsMatch(
        confThreshold, headClassId,
        allowedClassMask, aimConfig, iouStickinessThreshold,
        headYOffset, bodyYOffset);
    if (paramsChanged) {
        if (m_stream) cudaStreamSynchronize(m_stream);
        destroyFullGraphs();
        m_cachedConfThreshold = confThreshold;
        m_cachedHeadClassId = headClassId;
        m_cachedAllowedClassMask = allowedClassMask;
        m_cachedIouThreshold = iouStickinessThreshold;
        m_cachedHeadYOffset = headYOffset;
        m_cachedBodyYOffset = bodyYOffset;
    }

    if (!uploadRuntimeAimConfig(aimConfig, paramsChanged)) {
        return false;
    }

    const int bucketIndex = pickBucketForCapture(sourceWidth, sourceHeight);
    if (bucketIndex < 0) {
        std::cerr << "[SimpleInference] No bucket available for shape "
                  << sourceWidth << "x" << sourceHeight << std::endl;
        return false;
    }
    GraphShapeBucket& bucket = m_shapeBuckets[static_cast<size_t>(bucketIndex)];

    // If bucket already holds the requested shape with enough slots and execs,
    // reuse it; only top up missing slots.
    const bool shapeMatches =
        bucket.used && bucket.sourceW == sourceWidth && bucket.sourceH == sourceHeight;
    if (!shapeMatches) {
        destroyBucket(bucket);
    }

    // Fill pinned buffer with dummy data. The H2D copy itself is captured into
    // the graph, then its source pointer is patched per frame before launch.
    memset(m_h_rawPinned, 128, rawSize);
    cudaError_t err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to sync before graph capture: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    bucket.sourceW = sourceWidth;
    bucket.sourceH = sourceHeight;
    const int slotsToCapture = std::max(bucket.slotCount, graphSlotCount);

    for (int slot = 0; slot < slotsToCapture; ++slot) {
        const size_t slotIdx = static_cast<size_t>(slot);
        if (bucket.graphExecs[slotIdx]) {
            continue;  // Slot already captured for this shape.
        }
        err = cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] Failed to begin full graph capture for slot "
                      << slot << ": " << cudaGetErrorString(err) << std::endl;
            destroyBucket(bucket);
            return false;
        }

        // Capture stage timing start event into the graph (if enabled).
        if (m_stageTimingEnabled && m_stageEvtStart[slotIdx]) {
            cudaEventRecord(m_stageEvtStart[slotIdx], m_stream);
        }

        // dGPU: capture the frame H2D into the graph (its source pointer is
        // patched per frame at launch). Tegra zero-copy: no H2D in the graph -
        // the preprocess reads the receive buffer directly and the source cell is
        // updated on the stream just before each launch.
        if (!m_zeroCopy) {
            err = cudaMemcpyAsync(m_d_rawInput, m_h_rawPinned, rawSize,
                                  cudaMemcpyHostToDevice, m_stream);
            if (err != cudaSuccess) {
                cudaGraph_t capturedGraph = nullptr;
                cudaError_t abortErr = cudaStreamEndCapture(m_stream, &capturedGraph);
                if (abortErr == cudaSuccess && capturedGraph) {
                    cudaGraphDestroy(capturedGraph);
                }
                std::cerr << "[SimpleInference] Failed to capture H2D memcpy for slot "
                          << slot << ": " << cudaGetErrorString(err) << std::endl;
                destroyBucket(bucket);
                return false;
            }
        }

        // Execute pipeline after captured H2D. Each graph writes to its own
        // result slot so callbacks cannot observe overwritten slot-0 results.
        if (!executeFusedPipelinePostH2D(sourceWidth, sourceHeight,
                                         confThreshold, headClassId,
                                         allowedClassMask, aimConfig,
                                         iouStickinessThreshold, headYOffset, bodyYOffset,
                                         slot)) {
            cudaGraph_t capturedGraph = nullptr;
            cudaError_t abortErr = cudaStreamEndCapture(m_stream, &capturedGraph);
            if (abortErr == cudaSuccess && capturedGraph) {
                cudaGraphDestroy(capturedGraph);
            }
            std::cerr << "[SimpleInference] Failed to launch full pipeline during graph capture"
                      << " for slot " << slot << std::endl;
            destroyBucket(bucket);
            return false;
        }

        err = cudaStreamEndCapture(m_stream, &bucket.graphs[slotIdx]);
        if (err != cudaSuccess || !bucket.graphs[slotIdx]) {
            std::cerr << "[SimpleInference] Failed to end full graph capture for slot "
                      << slot << ": " << cudaGetErrorString(err) << std::endl;
            destroyBucket(bucket);
            return false;
        }

        bucket.rawSizes[slotIdx] = rawSize;
        if (m_zeroCopy) {
            // No H2D node exists in the zero-copy graph; the per-frame source is
            // supplied via the m_d_srcPtr cell updated before each launch.
            bucket.h2dNodes[slotIdx] = nullptr;
        } else {
            bucket.h2dNodes[slotIdx] = findGraphH2DMemcpyNode(bucket.graphs[slotIdx], m_d_rawInput, rawSize);
            if (!bucket.h2dNodes[slotIdx]) {
                std::cerr << "[SimpleInference] Failed to locate captured H2D memcpy node for slot "
                          << slot << std::endl;
                destroyBucket(bucket);
                return false;
            }
        }

        err = cudaGraphInstantiate(&bucket.graphExecs[slotIdx], bucket.graphs[slotIdx], nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] Failed to instantiate full graph for slot "
                      << slot << ": " << cudaGetErrorString(err) << std::endl;
            destroyBucket(bucket);
            return false;
        }
        cudaGraphUpload(bucket.graphExecs[slotIdx], m_stream);
    }
    cudaStreamSynchronize(m_stream);

    bucket.used = true;
    bucket.slotCount = std::max(bucket.slotCount, slotsToCapture);
    touchBucket(bucketIndex);

    std::cout << "[SimpleInference] Full CUDA graph captured for " << sourceWidth << "x"
              << sourceHeight << " source (" << bucket.slotCount
              << " slots, bucket " << bucketIndex << "/" << kMaxGraphShapes
              << ", H2D+preprocess+inference+postprocess"
              << (m_stageTimingEnabled ? "+timing" : "") << ")" << std::endl;
    return true;
}

// Callback Completion Worker
void SimpleInference::callbackWorkerLoop() {
#ifndef _WIN32
    pthread_setname_np(pthread_self(), "infer-cb");
    const int fifoMax = sched_get_priority_max(SCHED_FIFO);
    if (fifoMax > 0) {
        sched_param param{};
        param.sched_priority = std::max(1, fifoMax - 1);
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
            const int rrMax = sched_get_priority_max(SCHED_RR);
            if (rrMax > 0) {
                param.sched_priority = std::max(1, rrMax - 1);
                pthread_setschedparam(pthread_self(), SCHED_RR, &param);
            }
        }
    }
    const long cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
    // Configured override wins; otherwise default to the second-to-last core.
    // A configured value < 0 means "unpinned".
    int targetCore = (m_callbackAffinityCore != kCallbackAffinityUnset)
                         ? m_callbackAffinityCore
                         : (cpuCount > 2 ? static_cast<int>(cpuCount - 2) : -1);
    if (targetCore >= 0 && targetCore < cpuCount) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(targetCore, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
#endif
    auto hasPendingCallback = [this]() {
        for (int slot = 0; slot < kMaxCallbacksInFlight; ++slot) {
            if (m_callbackSlotPending[static_cast<size_t>(slot)].load(std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    };

    while (m_callbackWorkerRunning.load(std::memory_order_acquire) ||
           m_callbacksInFlight.load(std::memory_order_acquire) > 0) {
        int pendingSlot = -1;
        for (int slot = 0; slot < kMaxCallbacksInFlight; ++slot) {
            if (m_callbackSlotPending[static_cast<size_t>(slot)].load(std::memory_order_acquire)) {
                pendingSlot = slot;
                break;
            }
        }

        if (pendingSlot < 0) {
            std::unique_lock<std::mutex> lock(m_callbackWorkerMutex);
            m_callbackWorkerCv.wait(lock, [&]() {
                if (!m_callbackWorkerRunning.load(std::memory_order_acquire)) {
                    return true;
                }
                return hasPendingCallback();
            });
            continue;
        }

        cudaError_t eventStatus = cudaEventSynchronize(
            m_callbackEvents[static_cast<size_t>(pendingSlot)]);
        m_callbackSlotPending[static_cast<size_t>(pendingSlot)].store(false, std::memory_order_release);

        if (eventStatus == cudaSuccess) {
            if (m_stageTimingEnabled &&
                m_stageEvtValid[static_cast<size_t>(pendingSlot)].exchange(false, std::memory_order_acquire)) {
                recordStageTimings(pendingSlot);
            }
            CallbackData& cbData = m_callbackDataSlots[static_cast<size_t>(pendingSlot)];
            try {
                if (cbData.callback && cbData.resultPtr) {
                    cbData.callback(*cbData.resultPtr, cbData.userData);
                }
            } catch (const std::exception& e) {
                std::cerr << "[SimpleInference] Callback exception: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[SimpleInference] Callback exception: unknown" << std::endl;
            }
        } else {
            std::cerr << "[SimpleInference] cudaEventSynchronize failed for slot " << pendingSlot
                      << ": " << cudaGetErrorString(eventStatus) << std::endl;
            cudaGetLastError();
        }

        m_callbackSlotBusy[static_cast<size_t>(pendingSlot)].store(false, std::memory_order_release);
        m_callbacksInFlight.fetch_sub(1, std::memory_order_acq_rel);

        // Slot and in-flight counter are now released, so a resubmit from here
        // sees an accurate free-slot count (a kick from inside the user callback
        // above would still count this frame and be rejected at max in-flight,
        // dropping the newest queued frame). Kick the next queued frame straight
        // from this completion thread; the hook is non-blocking (try_lock).
        if (m_postCompletionHook) m_postCompletionHook();
    }
}

bool SimpleInference::runInferenceWithCallback(void* pinnedData, int width, int height,
                                                float confThreshold, int headClassId,
                                                uint32_t allowedClassMask,
                                                const AimConfig& aimConfig,
                                                float iouStickinessThreshold,
                                                float headYOffset, float bodyYOffset,
                                                InferenceCallback callback, void* userData) {
    if (!m_loaded || !pinnedData || width <= 0 || height <= 0) return false;

    const size_t rawSize = static_cast<size_t>(width) * static_cast<size_t>(height) *
                           static_cast<size_t>(inputBytesPerPixel());
    if (rawSize == 0) return false;
    if (rawSize > m_rawInputCapacityBytes && !ensureRawInputCapacity(rawSize)) {
        std::cerr << "[SimpleInference] Input frame too large for raw buffer: "
                  << rawSize << " > " << m_rawInputCapacityBytes << std::endl;
        return false;
    }

    if (m_callbacksInFlight.load(std::memory_order_acquire) >= kMaxCallbacksInFlight) return false;

    const bool nonShapeOk = nonShapeParamsMatch(
        confThreshold, headClassId,
        allowedClassMask, aimConfig, iouStickinessThreshold,
        headYOffset, bodyYOffset);
    const int bucketIndex = nonShapeOk ? findBucketIndex(width, height) : -1;
    const int bucketSlotCount = (bucketIndex >= 0)
        ? m_shapeBuckets[static_cast<size_t>(bucketIndex)].slotCount
        : 0;
    const bool graphAvailable = (bucketIndex >= 0) && (bucketSlotCount > 0);

    int callbackSlot = -1;
    if (graphAvailable) {
        for (int attempt = 0; attempt < bucketSlotCount; ++attempt) {
            const int idx = static_cast<int>((m_callbackSlotCursor + static_cast<uint32_t>(attempt)) %
                                             static_cast<uint32_t>(bucketSlotCount));
            bool expected = false;
            if (m_callbackSlotBusy[idx].compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                callbackSlot = idx;
                m_callbackSlotCursor =
                    (static_cast<uint32_t>(idx) + 1u) % static_cast<uint32_t>(bucketSlotCount);
                break;
            }
        }
    }

    for (int attempt = 0; attempt < kMaxCallbacksInFlight; ++attempt) {
        if (callbackSlot >= 0) break;
        const int idx = static_cast<int>((m_callbackSlotCursor + static_cast<uint32_t>(attempt)) %
                                         static_cast<uint32_t>(kMaxCallbacksInFlight));
        bool expected = false;
        if (m_callbackSlotBusy[idx].compare_exchange_strong(
                expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            callbackSlot = idx;
            m_callbackSlotCursor =
                (static_cast<uint32_t>(idx) + 1u) % static_cast<uint32_t>(kMaxCallbacksInFlight);
            break;
        }
    }
    if (callbackSlot < 0) {
        return false;
    }

    m_callbacksInFlight.fetch_add(1, std::memory_order_acq_rel);

    auto clearInFlightAndFail = [this, callbackSlot](bool drainStream) {
        if (drainStream && m_stream) {
            cudaStreamSynchronize(m_stream);
        }
        if (callbackSlot >= 0 && callbackSlot < kMaxCallbacksInFlight) {
            m_callbackSlotBusy[callbackSlot].store(false, std::memory_order_release);
        }
        m_callbacksInFlight.fetch_sub(1, std::memory_order_acq_rel);
        return false;
    };

    if (!uploadRuntimeAimConfig(aimConfig)) {
        return clearInFlightAndFail(false);
    }

    // Use CUDA Graph only when shape and parameters match captured constants.
    GraphShapeBucket* bucket = (bucketIndex >= 0)
        ? &m_shapeBuckets[static_cast<size_t>(bucketIndex)]
        : nullptr;
    const bool canUseGraph =
        graphAvailable && bucket &&
        callbackSlot < bucket->slotCount &&
        bucket->graphExecs[static_cast<size_t>(callbackSlot)] != nullptr &&
        // dGPU needs the captured H2D node to patch; Tegra has none (zero-copy).
        (m_zeroCopy || bucket->h2dNodes[static_cast<size_t>(callbackSlot)] != nullptr) &&
        bucket->rawSizes[static_cast<size_t>(callbackSlot)] == rawSize;

    if (canUseGraph) {
        // Prepare the per-frame source before launching the graph.
        // - Tegra zero-copy: update the m_d_srcPtr cell with the frame's mapped
        //   device pointer (an 8-byte stream copy; no H2D node exists).
        // - dGPU: patch the captured H2D memcpy node's source host pointer.
        cudaError_t err = cudaSuccess;
        if (m_zeroCopy) {
            const uint8_t* dev = pinnedDevicePtr(pinnedData);
            if (!dev) {
                err = cudaErrorInvalidValue;  // force the standard-path fallback below
            } else {
                m_h_srcPtrStage[static_cast<size_t>(callbackSlot)] = const_cast<uint8_t*>(dev);
                err = cudaMemcpyAsync(m_d_srcPtr,
                                      &m_h_srcPtrStage[static_cast<size_t>(callbackSlot)],
                                      sizeof(void*), cudaMemcpyHostToDevice, m_stream);
            }
        } else {
            err = cudaGraphExecMemcpyNodeSetParams1D(
                bucket->graphExecs[static_cast<size_t>(callbackSlot)],
                bucket->h2dNodes[static_cast<size_t>(callbackSlot)],
                m_d_rawInput,
                pinnedData,
                rawSize,
                cudaMemcpyHostToDevice);
        }
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] Graph source update failed: "
                      << cudaGetErrorString(err) << std::endl;
            if (!executeFusedPipeline(pinnedData, width, height,
                                      confThreshold, headClassId,
                                      allowedClassMask, aimConfig,
                                      iouStickinessThreshold, headYOffset, bodyYOffset, callbackSlot)) {
                return clearInFlightAndFail(true);
            }
            m_graphFallbackCount.fetch_add(1, std::memory_order_relaxed);
            m_standardLaunchCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Launch graph (preprocess + inference + postprocess [+ H2D/D2H on dGPU])
            err = cudaGraphLaunch(bucket->graphExecs[static_cast<size_t>(callbackSlot)], m_stream);
            if (err != cudaSuccess) {
                std::cerr << "[SimpleInference] cudaGraphLaunch failed: "
                          << cudaGetErrorString(err) << std::endl;
                return clearInFlightAndFail(true);
            }
            m_graphLaunchCount.fetch_add(1, std::memory_order_relaxed);
            touchBucket(bucketIndex);
        }
    } else {
        // Standard pipeline execution
        if (!executeFusedPipeline(pinnedData, width, height,
                                  confThreshold, headClassId,
                                  allowedClassMask, aimConfig,
                                  iouStickinessThreshold, headYOffset, bodyYOffset, callbackSlot)) {
            return clearInFlightAndFail(true);
        }
        m_standardLaunchCount.fetch_add(1, std::memory_order_relaxed);
    }

    if (m_stageTimingEnabled) {
        m_stageEvtValid[static_cast<size_t>(callbackSlot)].store(true, std::memory_order_release);
    }

    // Setup pre-allocated callback data (no heap allocation)
    CallbackData& cbData = m_callbackDataSlots[callbackSlot];
    cbData = {callback, userData,
              m_h_inferenceResultPinned[static_cast<size_t>(callbackSlot)]};

    // Record completion event and let callback worker invoke host callback.
    cudaError_t err = cudaEventRecord(m_callbackEvents[static_cast<size_t>(callbackSlot)], m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] cudaEventRecord failed: " << cudaGetErrorString(err) << std::endl;
        return clearInFlightAndFail(true);
    }
    {
        std::lock_guard<std::mutex> lock(m_callbackWorkerMutex);
        m_callbackSlotPending[static_cast<size_t>(callbackSlot)].store(true, std::memory_order_release);
    }
    m_callbackWorkerCv.notify_one();

    return true;
}

} // namespace gpa
