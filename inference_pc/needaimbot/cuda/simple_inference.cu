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

// =============================================================================
// RGB Preprocessing Kernels (RGB -> CHW normalized)
// =============================================================================

// Same resolution, no resize -> FP16
__global__ void preprocessKernelFP16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
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
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
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

// =============================================================================
// Bilinear Resize + Preprocessing Kernels (fused for efficiency)
// =============================================================================

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
    const uint8_t* __restrict__ src,
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
    bilinearSample(src, src_w, src_h, sx, sy, r, g, b);

    dst[dst_idx] = __float2half(r * norm_factor);
    dst[dst_idx + hw_size] = __float2half(g * norm_factor);
    dst[dst_idx + 2 * hw_size] = __float2half(b * norm_factor);
}

// Bilinear resize + HWC->CHW + normalize -> FP32
__global__ void resizePreprocessKernel(
    const uint8_t* __restrict__ src,
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
    bilinearSample(src, src_w, src_h, sx, sy, r, g, b);

    dst[dst_idx] = r * norm_factor;
    dst[dst_idx + hw_size] = g * norm_factor;
    dst[dst_idx + 2 * hw_size] = b * norm_factor;
}

// Preprocessing wrapper (handles RGB resize/no-resize, FP16/FP32)
extern "C" cudaError_t cuda_preprocessing(
    const void* src_data,
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
                static_cast<const uint8_t*>(src_data),
                static_cast<__half*>(dst_chw),
                src_width, src_height,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        } else {
            resizePreprocessKernel<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
                static_cast<float*>(dst_chw),
                src_width, src_height,
                target_width, target_height,
                scale_x, scale_y, norm_factor
            );
        }
    } else {
        if (use_fp16) {
            preprocessKernelFP16<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
                static_cast<__half*>(dst_chw),
                target_width, target_height,
                norm_factor
            );
        } else {
            preprocessKernel<<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(src_data),
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
           nearlyEqual(a.p_softness_y, b.p_softness_y);
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
        if (m_callbackEvents[static_cast<size_t>(i)]) {
            cudaEventDestroy(m_callbackEvents[static_cast<size_t>(i)]);
            m_callbackEvents[static_cast<size_t>(i)] = nullptr;
        }
    }

    // Destroy CUDA graphs
    destroyFullGraphs();

    // Free GPU memory
    if (m_d_rawInput) cudaFree(m_d_rawInput);
    if (m_d_chwInput) cudaFree(m_d_chwInput);
    if (m_d_output) cudaFree(m_d_output);

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
        cudaMalloc(&m_d_inferenceResult[i], sizeof(InferenceResult));
        cudaMallocHost(&m_h_inferenceResultPinned[i], sizeof(InferenceResult));
        cudaEventCreateWithFlags(&m_callbackEvents[static_cast<size_t>(i)], cudaEventDisableTiming);
        m_callbackSlotPending[static_cast<size_t>(i)].store(false, std::memory_order_relaxed);
        m_callbackSlotBusy[static_cast<size_t>(i)].store(false, std::memory_order_relaxed);
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
        cuda_preprocessing(m_d_rawInput, m_d_chwInput,
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

// =============================================================================
// OPTIMIZED API: Pinned H2D + Single D2H Transfer + Full CUDA Graph
// =============================================================================

// Pipeline without H2D transfer - for CUDA Graph capture
bool SimpleInference::executeFusedPipelinePostH2D(int width, int height,
                                                  float confThreshold, int headClassId, float headBonus,
                                                  uint32_t allowedClassMask, const AimConfig& aimConfig,
                                                  float iouThreshold, float headYOffset, float bodyYOffset,
                                                  int resultSlot) {
    (void)aimConfig;
    if (resultSlot < 0 || resultSlot >= kMaxCallbacksInFlight) {
        std::cerr << "[SimpleInference] Invalid result slot: " << resultSlot << std::endl;
        return false;
    }
    InferenceResult* dResultSlot = m_d_inferenceResult[static_cast<size_t>(resultSlot)];
    InferenceResult* hResultSlot = m_h_inferenceResultPinned[static_cast<size_t>(resultSlot)];
    if (!dResultSlot || !hResultSlot) {
        std::cerr << "[SimpleInference] Result slot not allocated: " << resultSlot << std::endl;
        return false;
    }

    // GPU preprocessing (RGB + optional resize)
    cudaError_t preprocessErr = cuda_preprocessing(
        m_d_rawInput, m_d_chwInput,
        width, height, m_inputW, m_inputH, m_inputFP16, m_stream
    );
    if (preprocessErr != cudaSuccess) {
        std::cerr << "[SimpleInference] cuda_preprocessing failed: "
                  << cudaGetErrorString(preprocessErr) << std::endl;
        return false;
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

    // One-pass GPU postprocess: decode + target select + movement + result packing
    cudaError_t postErr = postprocessYoloFusedGpu(
        m_d_output, m_outputFP16, m_numBoxes, m_numClasses,
        confThreshold, allowedClassMask, m_maxDetections,
        static_cast<float>(std::max(m_inputW, m_inputH)),
        m_crosshairX, m_crosshairY,
        static_cast<float>(width) / static_cast<float>(m_inputW),
        static_cast<float>(height) / static_cast<float>(m_inputH),
        headClassId, headBonus, m_d_runtimeAimConfig,
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

    // Single D2H transfer (40 bytes)
    cudaError_t d2hErr = cudaMemcpyAsync(hResultSlot, dResultSlot,
                                         sizeof(InferenceResult), cudaMemcpyDeviceToHost, m_stream);
    if (d2hErr != cudaSuccess) {
        std::cerr << "[SimpleInference] cudaMemcpyAsync(result D2H) failed: "
                  << cudaGetErrorString(d2hErr) << std::endl;
        return false;
    }
    return true;
}

bool SimpleInference::executeFusedPipeline(void* rawInput, int width, int height,
                                           float confThreshold, int headClassId, float headBonus,
                                           uint32_t allowedClassMask, const AimConfig& aimConfig,
                                           float iouThreshold, float headYOffset, float bodyYOffset,
                                           int resultSlot) {
    size_t rawSize = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(inputBytesPerPixel());

    // H2D: Upload directly from pinned memory
    cudaError_t h2dErr = cudaMemcpyAsync(m_d_rawInput, rawInput, rawSize, cudaMemcpyHostToDevice, m_stream);
    if (h2dErr != cudaSuccess) {
        std::cerr << "[SimpleInference] cudaMemcpyAsync(m_d_rawInput) failed: "
                  << cudaGetErrorString(h2dErr) << std::endl;
        return false;
    }

    return executeFusedPipelinePostH2D(width, height, confThreshold, headClassId, headBonus,
                                       allowedClassMask, aimConfig, iouThreshold,
                                       headYOffset, bodyYOffset, resultSlot);
}

void SimpleInference::destroyFullGraphs() {
    for (int i = 0; i < kMaxCallbacksInFlight; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (m_graphExecs[idx]) {
            cudaGraphExecDestroy(m_graphExecs[idx]);
            m_graphExecs[idx] = nullptr;
        }
        if (m_graphs[idx]) {
            cudaGraphDestroy(m_graphs[idx]);
            m_graphs[idx] = nullptr;
        }
        m_graphH2DNodes[idx] = nullptr;
        m_graphRawSizes[idx] = 0;
    }
    m_graphSourceW = 0;
    m_graphSourceH = 0;
    m_graphSlotCount = 0;
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

bool SimpleInference::graphParamsMatch(int sourceWidth, int sourceHeight, int requiredGraphSlots,
                                       float confThreshold, int headClassId, float headBonus,
                                       uint32_t allowedClassMask, const AimConfig& aimConfig,
                                       float iouStickinessThreshold, float headYOffset,
                                       float bodyYOffset) const {
    (void)aimConfig;
    if (sourceWidth <= 0 || sourceHeight <= 0) return false;
    if (requiredGraphSlots <= 0 || requiredGraphSlots > kMaxCallbacksInFlight) return false;
    if (sourceWidth != m_graphSourceW || sourceHeight != m_graphSourceH) return false;
    if (requiredGraphSlots > m_graphSlotCount) return false;
    for (int i = 0; i < requiredGraphSlots; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (!m_graphExecs[idx]) {
            return false;
        }
    }
    return (headClassId == m_cachedHeadClassId) &&
           (allowedClassMask == m_cachedAllowedClassMask) &&
           nearlyEqual(confThreshold, m_cachedConfThreshold) &&
           nearlyEqual(headBonus, m_cachedHeadBonus) &&
           nearlyEqual(iouStickinessThreshold, m_cachedIouThreshold) &&
           nearlyEqual(headYOffset, m_cachedHeadYOffset) &&
           nearlyEqual(bodyYOffset, m_cachedBodyYOffset);
}

bool SimpleInference::isFullGraphReadyForShape(int sourceWidth, int sourceHeight,
                                               int graphSlotCount,
                                               float confThreshold, int headClassId, float headBonus,
                                               uint32_t allowedClassMask, const AimConfig& aimConfig,
                                               float iouStickinessThreshold, float headYOffset,
                                               float bodyYOffset) const {
    return graphParamsMatch(sourceWidth, sourceHeight, graphSlotCount,
                            confThreshold, headClassId, headBonus,
                            allowedClassMask, aimConfig, iouStickinessThreshold,
                            headYOffset, bodyYOffset);
}

SimpleInference::LaunchStats SimpleInference::takeLaunchStats() {
    LaunchStats stats;
    stats.graph = m_graphLaunchCount.exchange(0, std::memory_order_relaxed);
    stats.standard = m_standardLaunchCount.exchange(0, std::memory_order_relaxed);
    stats.graphFallback = m_graphFallbackCount.exchange(0, std::memory_order_relaxed);
    return stats;
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
    return true;
}

bool SimpleInference::captureFullGraphForShape(int sourceWidth, int sourceHeight,
                                                int graphSlotCount,
                                                float confThreshold, int headClassId, float headBonus,
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

    if (m_stream) {
        cudaStreamSynchronize(m_stream);
    }
    destroyFullGraphs();

    // Cache parameters
    m_graphSourceW = sourceWidth;
    m_graphSourceH = sourceHeight;
    m_graphSlotCount = graphSlotCount;
    m_cachedConfThreshold = confThreshold;
    m_cachedHeadClassId = headClassId;
    m_cachedHeadBonus = headBonus;
    m_cachedAllowedClassMask = allowedClassMask;
    m_cachedIouThreshold = iouStickinessThreshold;
    m_cachedHeadYOffset = headYOffset;
    m_cachedBodyYOffset = bodyYOffset;

    if (!uploadRuntimeAimConfig(aimConfig, true)) {
        destroyFullGraphs();
        return false;
    }

    // Fill pinned buffer with dummy data. The H2D copy itself is captured into
    // the graph, then its source pointer is patched per frame before launch.
    memset(m_h_rawPinned, 128, rawSize);
    cudaError_t err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to sync before graph capture: "
                  << cudaGetErrorString(err) << std::endl;
        destroyFullGraphs();
        return false;
    }

    for (int slot = 0; slot < graphSlotCount; ++slot) {
        const size_t slotIdx = static_cast<size_t>(slot);
        err = cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] Failed to begin full graph capture for slot "
                      << slot << ": " << cudaGetErrorString(err) << std::endl;
            destroyFullGraphs();
            return false;
        }

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
            destroyFullGraphs();
            return false;
        }

        // Execute pipeline after captured H2D. Each graph writes to its own
        // result slot so callbacks cannot observe overwritten slot-0 results.
        if (!executeFusedPipelinePostH2D(sourceWidth, sourceHeight,
                                         confThreshold, headClassId, headBonus,
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
            destroyFullGraphs();
            return false;
        }

        err = cudaStreamEndCapture(m_stream, &m_graphs[slotIdx]);
        if (err != cudaSuccess || !m_graphs[slotIdx]) {
            std::cerr << "[SimpleInference] Failed to end full graph capture for slot "
                      << slot << ": " << cudaGetErrorString(err) << std::endl;
            destroyFullGraphs();
            return false;
        }

        m_graphH2DNodes[slotIdx] = findGraphH2DMemcpyNode(m_graphs[slotIdx], m_d_rawInput, rawSize);
        m_graphRawSizes[slotIdx] = rawSize;
        if (!m_graphH2DNodes[slotIdx]) {
            std::cerr << "[SimpleInference] Failed to locate captured H2D memcpy node for slot "
                      << slot << std::endl;
            destroyFullGraphs();
            return false;
        }

        err = cudaGraphInstantiate(&m_graphExecs[slotIdx], m_graphs[slotIdx], nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] Failed to instantiate full graph for slot "
                      << slot << ": " << cudaGetErrorString(err) << std::endl;
            destroyFullGraphs();
            return false;
        }
        cudaGraphUpload(m_graphExecs[slotIdx], m_stream);
    }
    cudaStreamSynchronize(m_stream);

    std::cout << "[SimpleInference] Full CUDA graph captured for " << sourceWidth << "x"
              << sourceHeight << " source (" << graphSlotCount
              << " slots, H2D+preprocess+inference+postprocess)" << std::endl;
    return true;
}

// =============================================================================
// Callback Completion Worker
// =============================================================================
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
    if (cpuCount > 2) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(cpuCount - 2), &cpuset);
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
    }
}

bool SimpleInference::runInferenceWithCallback(void* pinnedData, int width, int height,
                                                float confThreshold, int headClassId, float headBonus,
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

    const bool graphAvailable =
        graphParamsMatch(width, height, 1,
                         confThreshold, headClassId, headBonus,
                         allowedClassMask, aimConfig, iouStickinessThreshold,
                         headYOffset, bodyYOffset);

    int callbackSlot = -1;
    if (graphAvailable) {
        for (int attempt = 0; attempt < m_graphSlotCount; ++attempt) {
            const int idx = static_cast<int>((m_callbackSlotCursor + static_cast<uint32_t>(attempt)) %
                                             static_cast<uint32_t>(m_graphSlotCount));
            bool expected = false;
            if (m_callbackSlotBusy[idx].compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                callbackSlot = idx;
                m_callbackSlotCursor =
                    (static_cast<uint32_t>(idx) + 1u) % static_cast<uint32_t>(m_graphSlotCount);
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
    const bool canUseGraph =
        graphAvailable &&
        callbackSlot < m_graphSlotCount &&
        graphParamsMatch(width, height, callbackSlot + 1,
                         confThreshold, headClassId, headBonus,
                         allowedClassMask, aimConfig, iouStickinessThreshold,
                         headYOffset, bodyYOffset) &&
        m_graphExecs[static_cast<size_t>(callbackSlot)] != nullptr &&
        m_graphH2DNodes[static_cast<size_t>(callbackSlot)] != nullptr &&
        m_graphRawSizes[static_cast<size_t>(callbackSlot)] == rawSize;

    if (canUseGraph) {
        // H2D is inside the graph; only patch the source host pointer.
        cudaError_t err = cudaGraphExecMemcpyNodeSetParams1D(
            m_graphExecs[static_cast<size_t>(callbackSlot)],
            m_graphH2DNodes[static_cast<size_t>(callbackSlot)],
            m_d_rawInput,
            pinnedData,
            rawSize,
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[SimpleInference] cudaGraphExecMemcpyNodeSetParams1D(H2D) failed: "
                      << cudaGetErrorString(err) << std::endl;
            if (!executeFusedPipeline(pinnedData, width, height,
                                      confThreshold, headClassId, headBonus,
                                      allowedClassMask, aimConfig,
                                      iouStickinessThreshold, headYOffset, bodyYOffset, callbackSlot)) {
                return clearInFlightAndFail(true);
            }
            m_graphFallbackCount.fetch_add(1, std::memory_order_relaxed);
            m_standardLaunchCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Launch graph (H2D + preprocess + inference + postprocess + D2H)
            err = cudaGraphLaunch(m_graphExecs[static_cast<size_t>(callbackSlot)], m_stream);
            if (err != cudaSuccess) {
                std::cerr << "[SimpleInference] cudaGraphLaunch failed: "
                          << cudaGetErrorString(err) << std::endl;
                return clearInFlightAndFail(true);
            }
            m_graphLaunchCount.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        // Standard pipeline execution
        if (!executeFusedPipeline(pinnedData, width, height,
                                  confThreshold, headClassId, headBonus,
                                  allowedClassMask, aimConfig,
                                  iouStickinessThreshold, headYOffset, bodyYOffset, callbackSlot)) {
            return clearInFlightAndFail(true);
        }
        m_standardLaunchCount.fetch_add(1, std::memory_order_relaxed);
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
