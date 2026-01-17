// Simple TensorRT inference with CUDA Graph optimization
// Supports FP16 and FP32 models natively
// GPU postprocessing for minimal latency
#include "simple_inference.h"
#include "simple_postprocess.h"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

// =============================================================================
// RGB Preprocessing Kernels (inline implementation)
// =============================================================================

// RGB HWC uint8 -> CHW FP16 normalized (same resolution, no resize)
__global__ void rgbPreprocessKernelFP16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;  // RGB HWC
    int hw_size = width * height;
    int dst_idx = y * width + x;

    // RGB HWC -> CHW normalized
    dst[dst_idx] = __float2half(src[src_idx] * scale_factor);                 // R
    dst[dst_idx + hw_size] = __float2half(src[src_idx + 1] * scale_factor);   // G
    dst[dst_idx + 2 * hw_size] = __float2half(src[src_idx + 2] * scale_factor); // B
}

// RGB HWC uint8 -> CHW FP32 normalized (same resolution, no resize)
__global__ void rgbPreprocessKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int width, int height,
    float scale_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;  // RGB HWC
    int hw_size = width * height;
    int dst_idx = y * width + x;

    // RGB HWC -> CHW normalized
    dst[dst_idx] = src[src_idx] * scale_factor;                 // R
    dst[dst_idx + hw_size] = src[src_idx + 1] * scale_factor;   // G
    dst[dst_idx + 2 * hw_size] = src[src_idx + 2] * scale_factor; // B
}

// RGB preprocessing wrapper function
extern "C" cudaError_t cuda_rgb_preprocessing(
    const void* src_rgb_data,
    void* dst_rgb_chw,
    int src_width, int src_height,
    int src_step,  // unused for same-size
    int target_width, int target_height,
    bool use_fp16,
    cudaStream_t stream
) {
    // Same resolution assumed (320x320 -> 320x320)
    dim3 block(16, 16);
    dim3 grid((target_width + block.x - 1) / block.x,
              (target_height + block.y - 1) / block.y);

    const float scale = 1.0f / 255.0f;

    if (use_fp16) {
        rgbPreprocessKernelFP16<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(src_rgb_data),
            static_cast<__half*>(dst_rgb_chw),
            target_width, target_height,
            scale
        );
    } else {
        rgbPreprocessKernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(src_rgb_data),
            static_cast<float*>(dst_rgb_chw),
            target_width, target_height,
            scale
        );
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
    // Destroy CUDA graph
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);

    // Free GPU memory
    if (m_d_rgbInput) cudaFree(m_d_rgbInput);
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

    // Free pinned host memory
    if (m_h_rgbPinned) cudaFreeHost(m_h_rgbPinned);
    if (m_h_outputPinned) cudaFreeHost(m_h_outputPinned);
    if (m_h_bestTargetPinned) cudaFreeHost(m_h_bestTargetPinned);
    if (m_h_hasTargetPinned) cudaFreeHost(m_h_hasTargetPinned);
    if (m_h_mouseMovementPinned) cudaFreeHost(m_h_mouseMovementPinned);

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

    // Get dimensions
    int inputIdx = m_engine->getBindingIndex("images");
    int outputIdx = m_engine->getBindingIndex("output0");

    if (inputIdx < 0 || outputIdx < 0) {
        std::cerr << "[SimpleInference] Invalid binding names" << std::endl;
        return false;
    }

    auto inputDims = m_engine->getBindingDimensions(inputIdx);
    auto outputDims = m_engine->getBindingDimensions(outputIdx);

    // Check data types
    auto inputType = m_engine->getBindingDataType(inputIdx);
    auto outputType = m_engine->getBindingDataType(outputIdx);

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

    // Allocate GPU memory
    size_t rgbInputSize = m_inputH * m_inputW * 3;
    size_t chwInputSize = 1 * 3 * m_inputH * m_inputW * (m_inputFP16 ? sizeof(__half) : sizeof(float));
    size_t outputSizeGPU = 1 * outputDims.d[1] * m_numBoxes * (m_outputFP16 ? sizeof(__half) : sizeof(float));

    cudaMalloc(&m_d_rgbInput, rgbInputSize);
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

    // Allocate pinned host memory - same type as GPU output
    m_outputPinnedSize = outputSizeGPU;
    cudaMallocHost(&m_h_rgbPinned, rgbInputSize);
    cudaMallocHost(&m_h_outputPinned, m_outputPinnedSize);
    cudaMallocHost(&m_h_bestTargetPinned, sizeof(Detection));
    cudaMallocHost(&m_h_hasTargetPinned, sizeof(int));
    cudaMallocHost(&m_h_mouseMovementPinned, sizeof(MouseMovement));

    m_loaded = true;
    std::cout << "[SimpleInference] Engine loaded successfully" << std::endl;

    // Warm up and capture graph
    std::cout << "[SimpleInference] Warming up..." << std::endl;
    memset(m_h_rgbPinned, 128, rgbInputSize);  // Gray image
    for (int i = 0; i < 3; i++) {
        executeStandard();
    }

    std::cout << "[SimpleInference] Capturing CUDA graph..." << std::endl;
    if (captureGraph()) {
        std::cout << "[SimpleInference] CUDA graph captured successfully" << std::endl;
    } else {
        std::cout << "[SimpleInference] CUDA graph capture failed, using standard execution" << std::endl;
    }

    return true;
}

void SimpleInference::executeStandard() {
    size_t rgbSize = m_inputH * m_inputW * 3;

    // Upload RGB to GPU
    cudaMemcpyAsync(m_d_rgbInput, m_h_rgbPinned, rgbSize, cudaMemcpyHostToDevice, m_stream);

    // GPU preprocessing: RGB HWC uint8 -> CHW float/half normalized
    cuda_rgb_preprocessing(
        m_d_rgbInput,
        m_d_chwInput,
        m_inputW, m_inputH,
        m_inputW * 3,
        m_inputW, m_inputH,
        m_inputFP16,  // Use FP16 if model expects it
        m_stream
    );

    // Run inference
    void* bindings[2] = { m_d_chwInput, m_d_output };
    m_context->enqueueV2(bindings, m_stream, nullptr);

    // Download output (FP16 or FP32, no conversion)
    cudaMemcpyAsync(m_h_outputPinned, m_d_output, m_outputPinnedSize, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
}

bool SimpleInference::captureGraph() {
    if (m_graphCaptured) {
        return true;
    }

    size_t rgbSize = m_inputH * m_inputW * 3;

    // Begin graph capture with relaxed mode for better compatibility
    // Relaxed mode allows operations that may not be fully captured, reducing overhead
    cudaError_t err = cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to begin graph capture: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Record the operations
    cudaMemcpyAsync(m_d_rgbInput, m_h_rgbPinned, rgbSize, cudaMemcpyHostToDevice, m_stream);

    cuda_rgb_preprocessing(
        m_d_rgbInput,
        m_d_chwInput,
        m_inputW, m_inputH,
        m_inputW * 3,
        m_inputW, m_inputH,
        m_inputFP16,
        m_stream
    );

    void* bindings[2] = { m_d_chwInput, m_d_output };
    m_context->enqueueV2(bindings, m_stream, nullptr);

    // Download output (same size regardless of FP16/FP32)
    cudaMemcpyAsync(m_h_outputPinned, m_d_output, m_outputPinnedSize, cudaMemcpyDeviceToHost, m_stream);

    // End capture
    err = cudaStreamEndCapture(m_stream, &m_graph);
    if (err != cudaSuccess || !m_graph) {
        std::cerr << "[SimpleInference] Failed to end graph capture: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Instantiate the graph
    err = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "[SimpleInference] Failed to instantiate graph: " << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }

    m_graphCaptured = true;
    return true;
}

void SimpleInference::executeGraph() {
    cudaGraphLaunch(m_graphExec, m_stream);
    cudaStreamSynchronize(m_stream);
}

void SimpleInference::decodeOutput(float confThreshold, std::vector<Detection>& outDetections) {
    outDetections.clear();

    if (m_outputFP16) {
        // FP16 output - read as __half
        const __half* output = static_cast<const __half*>(m_h_outputPinned);

        for (int i = 0; i < m_numBoxes; i++) {
            float cx = __half2float(output[0 * m_numBoxes + i]);
            float cy = __half2float(output[1 * m_numBoxes + i]);
            float w = __half2float(output[2 * m_numBoxes + i]);
            float h = __half2float(output[3 * m_numBoxes + i]);

            // Find best class
            float maxConf = 0;
            int bestClass = 0;
            for (int c = 0; c < m_numClasses; c++) {
                float conf = __half2float(output[(4 + c) * m_numBoxes + i]);
                if (conf > maxConf) {
                    maxConf = conf;
                    bestClass = c;
                }
            }

            if (maxConf > confThreshold) {
                Detection det;
                det.x1 = cx - w / 2;
                det.y1 = cy - h / 2;
                det.x2 = cx + w / 2;
                det.y2 = cy + h / 2;
                det.confidence = maxConf;
                det.classId = bestClass;
                outDetections.push_back(det);
            }
        }
    } else {
        // FP32 output
        const float* output = static_cast<const float*>(m_h_outputPinned);

        for (int i = 0; i < m_numBoxes; i++) {
            float cx = output[0 * m_numBoxes + i];
            float cy = output[1 * m_numBoxes + i];
            float w = output[2 * m_numBoxes + i];
            float h = output[3 * m_numBoxes + i];

            // Find best class
            float maxConf = 0;
            int bestClass = 0;
            for (int c = 0; c < m_numClasses; c++) {
                float conf = output[(4 + c) * m_numBoxes + i];
                if (conf > maxConf) {
                    maxConf = conf;
                    bestClass = c;
                }
            }

            if (maxConf > confThreshold) {
                Detection det;
                det.x1 = cx - w / 2;
                det.y1 = cy - h / 2;
                det.x2 = cx + w / 2;
                det.y2 = cy + h / 2;
                det.confidence = maxConf;
                det.classId = bestClass;
                outDetections.push_back(det);
            }
        }
    }
}

int SimpleInference::runInference(const uint8_t* h_rgbData, int width, int height,
                                   float confThreshold, std::vector<Detection>& outDetections) {
    if (!m_loaded) return 0;

    // Copy to pinned memory
    size_t rgbSize = width * height * 3;
    memcpy(m_h_rgbPinned, h_rgbData, rgbSize);

    // Execute
    if (m_graphCaptured) {
        executeGraph();
    } else {
        executeStandard();
    }

    // Decode (CPU)
    decodeOutput(confThreshold, outDetections);

    return static_cast<int>(outDetections.size());
}

bool SimpleInference::runInferenceGpu(const uint8_t* h_rgbData, int width, int height,
                                       float confThreshold, int headClassId, float headBonus,
                                       Detection& outBestTarget) {
    if (!m_loaded) return false;

    // Copy to pinned memory
    size_t rgbSize = width * height * 3;
    memcpy(m_h_rgbPinned, h_rgbData, rgbSize);

    // Execute inference (H2D + preprocess + inference, no D2H of full output)
    // Use graph path but without the final output D2H copy
    size_t inputRgbSize = m_inputH * m_inputW * 3;
    cudaMemcpyAsync(m_d_rgbInput, m_h_rgbPinned, inputRgbSize, cudaMemcpyHostToDevice, m_stream);

    cuda_rgb_preprocessing(
        m_d_rgbInput,
        m_d_chwInput,
        m_inputW, m_inputH,
        m_inputW * 3,
        m_inputW, m_inputH,
        m_inputFP16,
        m_stream
    );

    void* bindings[2] = { m_d_chwInput, m_d_output };
    m_context->enqueueV2(bindings, m_stream, nullptr);

    // GPU postprocessing: decode on GPU
    decodeYoloGpu(
        m_d_output,
        m_outputFP16,
        m_numBoxes,
        m_numClasses,
        confThreshold,
        m_d_decoded,
        m_d_decodedCount,
        kMaxDetections,
        m_stream
    );

    // GPU target selection
    float crosshairX = m_inputW * 0.5f;
    float crosshairY = m_inputH * 0.5f;

    findBestTargetGpu(
        m_d_decoded,
        m_d_decodedCount,
        crosshairX,
        crosshairY,
        headClassId,
        headBonus,
        m_d_bestTarget,
        m_d_hasTarget,
        m_stream
    );

    // Validate best target before host copy (prevents garbage values)
    validateBestTargetGpu(m_d_bestTarget, m_d_hasTarget, m_stream);

    // Copy only the single best target back (much smaller than full output)
    cudaMemcpyAsync(m_h_bestTargetPinned, m_d_bestTarget, sizeof(Detection), cudaMemcpyDeviceToHost, m_stream);
    cudaMemcpyAsync(m_h_hasTargetPinned, m_d_hasTarget, sizeof(int), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    if (*m_h_hasTargetPinned) {
        outBestTarget = *m_h_bestTargetPinned;
        return true;
    }
    return false;
}

bool SimpleInference::runInferenceFused(const uint8_t* h_rgbData, int width, int height,
                                         float confThreshold, int headClassId, float headBonus,
                                         const PIDConfig& pidConfig,
                                         float iouStickinessThreshold,
                                         float headYOffset, float bodyYOffset,
                                         MouseMovement& outMovement,
                                         Detection* outBestTarget) {
    if (!m_loaded) return false;

    // Copy to pinned memory
    size_t rgbSize = width * height * 3;
    memcpy(m_h_rgbPinned, h_rgbData, rgbSize);

    // Execute inference (H2D + preprocess + inference)
    size_t inputRgbSize = m_inputH * m_inputW * 3;
    cudaMemcpyAsync(m_d_rgbInput, m_h_rgbPinned, inputRgbSize, cudaMemcpyHostToDevice, m_stream);

    cuda_rgb_preprocessing(
        m_d_rgbInput,
        m_d_chwInput,
        m_inputW, m_inputH,
        m_inputW * 3,
        m_inputW, m_inputH,
        m_inputFP16,
        m_stream
    );

    void* bindings[2] = { m_d_chwInput, m_d_output };
    m_context->enqueueV2(bindings, m_stream, nullptr);

    // GPU postprocessing: decode on GPU
    decodeYoloGpu(
        m_d_output,
        m_outputFP16,
        m_numBoxes,
        m_numClasses,
        confThreshold,
        m_d_decoded,
        m_d_decodedCount,
        kMaxDetections,
        m_stream
    );

    // Fused target selection + PID movement on GPU
    float crosshairX = m_inputW * 0.5f;
    float crosshairY = m_inputH * 0.5f;

    fusedTargetSelectionAndMovementGpu(
        m_d_decoded,
        m_d_decodedCount,
        kMaxDetections,
        crosshairX,
        crosshairY,
        headClassId,
        headBonus,
        pidConfig,
        iouStickinessThreshold,
        headYOffset,
        bodyYOffset,
        m_d_selectedTarget,
        m_d_bestTarget,
        m_d_hasTarget,
        m_d_mouseMovement,
        m_d_pidState,
        m_stream
    );

    // Validate best target before host copy
    validateBestTargetGpu(m_d_bestTarget, m_d_hasTarget, m_stream);

    // Copy only the essential data back (mouse movement and has_target flag)
    cudaMemcpyAsync(m_h_mouseMovementPinned, m_d_mouseMovement, sizeof(MouseMovement), cudaMemcpyDeviceToHost, m_stream);
    cudaMemcpyAsync(m_h_hasTargetPinned, m_d_hasTarget, sizeof(int), cudaMemcpyDeviceToHost, m_stream);

    // Optionally copy best target if requested
    if (outBestTarget) {
        cudaMemcpyAsync(m_h_bestTargetPinned, m_d_bestTarget, sizeof(Detection), cudaMemcpyDeviceToHost, m_stream);
    }

    cudaStreamSynchronize(m_stream);

    // Copy output
    outMovement = *m_h_mouseMovementPinned;

    if (outBestTarget && *m_h_hasTargetPinned) {
        *outBestTarget = *m_h_bestTargetPinned;
    }

    return *m_h_hasTargetPinned != 0;
}

} // namespace gpa
