#include "unified_graph_pipeline.h"
#include "detection/cuda_float_processing.h"
#include "simple_cuda_mat.h"
#include "../AppContext.h"
#include "../capture/capture_interface.h"
#include "../core/logger.h"
#include "cuda_error_check.h"
#include "preprocessing.h"
#include "../include/other_tools.h"
#include "../core/constants.h"
#include "detection/postProcess.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <mutex>
#include <atomic>
#include <functional>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Forward declare the mouse control function
extern "C" {
    void executeMouseMovement(int dx, int dy);
}

namespace needaimbot {

// Use blocking CUDA events to avoid CPU yield storms while waiting for GPU work
static constexpr unsigned int kBlockingEventFlags = cudaEventDisableTiming | cudaEventBlockingSync;

void UnifiedGPUArena::initializePointers(uint8_t* basePtr, int maxDetections, int yoloSize) {
    size_t offset = 0;
    
    offset = (offset + alignof(float) - 1) & ~(alignof(float) - 1);
    yoloInput = reinterpret_cast<float*>(basePtr + offset);
    offset += yoloSize * yoloSize * 3 * sizeof(float);
    
    
    offset = (offset + alignof(Target) - 1) & ~(alignof(Target) - 1);
    decodedTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);

    finalTargets = decodedTargets;  // Alias final targets to decoded buffer
    
    
}

size_t UnifiedGPUArena::calculateArenaSize(int maxDetections, int yoloSize) {
    size_t size = 0;
    
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += yoloSize * yoloSize * 3 * sizeof(float);
    
    size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
    size += maxDetections * sizeof(Target);
    
    
    return size;
}


namespace {

int computeTargetSelectionBlockSize(int maxDetections) {
    const int cappedDetections = std::max(1, std::min(maxDetections, 256));

    int pow2 = 1;
    while (pow2 < cappedDetections && pow2 < 256) {
        pow2 <<= 1;
    }

    return pow2;
}

}  // namespace


__device__ float computeBoundingBoxIoU(const Target& a, const Target& b) {
    if (a.classId < 0 || b.classId < 0 || a.width <= 0 || a.height <= 0 ||
        b.width <= 0 || b.height <= 0) {
        return 0.0f;
    }

    int a_x2 = a.x + a.width;
    int a_y2 = a.y + a.height;
    int b_x2 = b.x + b.width;
    int b_y2 = b.y + b.height;

    int inter_x1 = (a.x > b.x) ? a.x : b.x;
    int inter_y1 = (a.y > b.y) ? a.y : b.y;
    int inter_x2 = (a_x2 < b_x2) ? a_x2 : b_x2;
    int inter_y2 = (a_y2 < b_y2) ? a_y2 : b_y2;

    int inter_w = inter_x2 - inter_x1;
    int inter_h = inter_y2 - inter_y1;
    if (inter_w <= 0 || inter_h <= 0) {
        return 0.0f;
    }

    int inter_area = inter_w * inter_h;
    int area_a = a.width * a.height;
    int area_b = b.width * b.height;
    int union_area = area_a + area_b - inter_area;

    if (union_area <= 0) {
        return 0.0f;
    }

    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

__global__ void fusedTargetSelectionAndMovementKernel(
    Target* __restrict__ finalTargets,
    int* __restrict__ finalTargetsCount,
    int maxDetections,
    float screen_center_x,
    float screen_center_y,
    int head_class_id,
    float kp_x,
    float kp_y,
    float ki_x,
    float ki_y,
    float kd_x,
    float kd_y,
    float integral_max,
    float head_y_offset,
    float body_y_offset,
    int detection_resolution,
    Target* __restrict__ selectedTarget,
    int* __restrict__ bestTargetIndex,
    Target* __restrict__ bestTarget,
    needaimbot::MouseMovement* __restrict__ output_movement,
    needaimbot::PIDState* __restrict__ pidState
) {
    extern __shared__ unsigned char sharedMem[];
    float* s_distancesX = reinterpret_cast<float*>(sharedMem);
    int* s_indices = reinterpret_cast<int*>(s_distancesX + blockDim.x);
    float* s_prevIoU = reinterpret_cast<float*>(s_indices + blockDim.x);
    int* s_prevIndices = reinterpret_cast<int*>(s_prevIoU + blockDim.x);

    __shared__ Target s_prevTarget;
    __shared__ bool s_prevValid;

    if (threadIdx.x == 0) {
        Target emptyTarget = {};
        s_prevTarget = emptyTarget;
        s_prevValid = false;

        if (selectedTarget) {
            Target cached = *selectedTarget;
            s_prevTarget = cached;
            s_prevValid = (cached.classId >= 0) && (cached.confidence > 0.0f) &&
                          (cached.width > 0) && (cached.height > 0);
        }

        *bestTargetIndex = -1;
        *bestTarget = emptyTarget;
        output_movement->dx = 0;
        output_movement->dy = 0;
    }
    __syncthreads();

    int count = *finalTargetsCount;
    if (count <= 0 || count > maxDetections) {
        if (threadIdx.x == 0 && selectedTarget) {
            Target emptyTarget = {};
            *selectedTarget = emptyTarget;
        }
        return;
    }

    Target prevTarget = s_prevTarget;
    bool prevValid = s_prevValid;

    int localBestIdx = -1;
    float localBestDistX = 1e9f;
    float localBestIoU = -1.0f;
    int localBestIoUIdx = -1;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        Target& t = finalTargets[i];

        if (t.x < -1000.0f || t.x > 10000.0f ||
            t.y < -1000.0f || t.y > 10000.0f ||
            t.width <= 0 || t.width > detection_resolution ||
            t.height <= 0 || t.height > detection_resolution ||
            t.confidence <= 0.0f || t.confidence > 1.0f) {
            t.confidence = 0.0f;
            continue;
        }

        float centerX = t.x + t.width / 2.0f;
        float dx = fabsf(centerX - screen_center_x);

        if (dx < localBestDistX) {
            localBestDistX = dx;
            localBestIdx = i;
        }

        if (prevValid) {
            float iou = computeBoundingBoxIoU(t, prevTarget);
            if (iou > localBestIoU) {
                localBestIoU = iou;
                localBestIoUIdx = i;
            }
        }
    }

    s_distancesX[threadIdx.x] = localBestDistX;
    s_indices[threadIdx.x] = localBestIdx;
    s_prevIoU[threadIdx.x] = localBestIoU;
    s_prevIndices[threadIdx.x] = localBestIoUIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_distancesX[threadIdx.x + s] < s_distancesX[threadIdx.x]) {
                s_distancesX[threadIdx.x] = s_distancesX[threadIdx.x + s];
                s_indices[threadIdx.x] = s_indices[threadIdx.x + s];
            }

            if (s_prevIoU[threadIdx.x + s] > s_prevIoU[threadIdx.x]) {
                s_prevIoU[threadIdx.x] = s_prevIoU[threadIdx.x + s];
                s_prevIndices[threadIdx.x] = s_prevIndices[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int candidateIndex = s_indices[0];
        bool candidateValid = candidateIndex >= 0;
        Target candidateTarget = candidateValid ? finalTargets[candidateIndex] : Target{};

        // Hysteresis: prefer previous target if IoU stays above threshold
        float bestIoU = s_prevIoU[0];
        int bestIoUIdx = s_prevIndices[0];
        const float iouStickinessThreshold = 0.30f;

        int chosenIndex = candidateIndex;
        Target chosenTarget = candidateTarget;
        bool haveTarget = candidateValid;
        if (prevValid && bestIoUIdx >= 0 && bestIoU > iouStickinessThreshold) {
            chosenIndex = bestIoUIdx;
            chosenTarget = finalTargets[bestIoUIdx];
            haveTarget = true;
        }

        if (haveTarget) {
            *bestTargetIndex = chosenIndex;
            *bestTarget = chosenTarget;
            if (selectedTarget) {
                *selectedTarget = chosenTarget;
            }

            float target_center_x = chosenTarget.x + chosenTarget.width / 2.0f;
            float target_center_y;

            if (chosenTarget.classId == head_class_id) {
                target_center_y = chosenTarget.y + chosenTarget.height * head_y_offset;
            } else {
                target_center_y = chosenTarget.y + chosenTarget.height * body_y_offset;
            }

            // Current error
            float error_x = target_center_x - screen_center_x;
            float error_y = target_center_y - screen_center_y;

            // Load previous PID state
            float prev_error_x = pidState->prev_error_x;
            float prev_error_y = pidState->prev_error_y;
            float integral_x = pidState->integral_x;
            float integral_y = pidState->integral_y;

            // Update integral (with anti-windup clamping)
            integral_x += error_x;
            integral_y += error_y;

            // Clamp integral to prevent windup
            if (integral_x > integral_max) integral_x = integral_max;
            if (integral_x < -integral_max) integral_x = -integral_max;
            if (integral_y > integral_max) integral_y = integral_max;
            if (integral_y < -integral_max) integral_y = -integral_max;

            // Calculate derivative (error change)
            float derivative_x = error_x - prev_error_x;
            float derivative_y = error_y - prev_error_y;

            // PID controller: P + I + D
            float movement_x = kp_x * error_x + ki_x * integral_x + kd_x * derivative_x;
            float movement_y = kp_y * error_y + ki_y * integral_y + kd_y * derivative_y;

            // Save current state for next iteration
            pidState->prev_error_x = error_x;
            pidState->prev_error_y = error_y;
            pidState->integral_x = integral_x;
            pidState->integral_y = integral_y;

            // Round to nearest int
            int emit_dx = static_cast<int>(lroundf(movement_x));
            int emit_dy = static_cast<int>(lroundf(movement_y));

            output_movement->dx = emit_dx;
            output_movement->dy = emit_dy;
        } else {
            Target emptyTarget = {};
            *bestTargetIndex = -1;
            *bestTarget = emptyTarget;
            if (selectedTarget) {
                *selectedTarget = emptyTarget;
            }
            output_movement->dx = 0;
            output_movement->dy = 0;

            // Reset PID state when no target
            pidState->prev_error_x = 0.0f;
            pidState->prev_error_y = 0.0f;
            pidState->integral_x = 0.0f;
            pidState->integral_y = 0.0f;
        }
    }
}


UnifiedGraphPipeline::UnifiedGraphPipeline() {
    m_state.startEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    m_state.endEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    resetMovementFilter();
    m_perfMetrics.reset();
}


bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    int leastPriority, greatestPriority;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (err == cudaSuccess) {
        cudaStream_t priorityStream;
        err = cudaStreamCreateWithPriority(&priorityStream, cudaStreamNonBlocking, greatestPriority);
        if (err == cudaSuccess) {
            m_pipelineStream = std::make_unique<CudaStream>(priorityStream);
        } else {
            m_pipelineStream = std::make_unique<CudaStream>();
        }
    } else {
        m_pipelineStream = std::make_unique<CudaStream>();
    }

    cudaStream_t captureStreamHandle = nullptr;
    cudaError_t captureErr = cudaSuccess;
    if (err == cudaSuccess) {
        captureErr = cudaStreamCreateWithPriority(&captureStreamHandle, cudaStreamNonBlocking, leastPriority);
    }

    if (captureErr == cudaSuccess && captureStreamHandle) {
        m_captureStream = std::make_unique<CudaStream>(captureStreamHandle);
    } else {
        try {
            m_captureStream = std::make_unique<CudaStream>();
        } catch (...) {
            m_captureStream.reset();
        }
    }

    // Create a low-priority stream for preview conversions/copies
    cudaStream_t previewStreamHandle = nullptr;
    cudaError_t previewErr = cudaSuccess;
    if (err == cudaSuccess) {
        previewErr = cudaStreamCreateWithPriority(&previewStreamHandle, cudaStreamNonBlocking, leastPriority);
    }
    if (previewErr == cudaSuccess && previewStreamHandle) {
        m_previewStream = std::make_unique<CudaStream>(previewStreamHandle);
    } else {
        try {
            m_previewStream = std::make_unique<CudaStream>();
        } catch (...) {
            m_previewStream.reset();
        }
    }

    m_previewReadyEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    m_captureReadyEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    
    if (m_config.modelPath.empty()) {
        std::cerr << "[UnifiedGraph] ERROR: Model path is required for TensorRT integration" << std::endl;
        return false;
    }
    
    if (!initializeTensorRT(m_config.modelPath)) {
        std::cerr << "[UnifiedGraph] CRITICAL: TensorRT initialization failed" << std::endl;
        return false;
    }

    // Allocate GPU arenas and small buffers first to avoid duplicate
    // allocations with TensorRT input bindings.
    if (!allocateBuffers()) {
        std::cerr << "[UnifiedGraph] Failed to allocate buffers" << std::endl;
        return false;
    }

    // Allocate TensorRT bindings after arena allocation so we can alias
    // the primary input directly to the unified arena instead of
    // temporarily allocating a duplicate input buffer.
    try {
        getBindings();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Failed to allocate TensorRT bindings: " << e.what() << std::endl;
        return false;
    }
    
    if (m_config.useGraphOptimization) {
        // Auto warmup iterations to stabilize TensorRT/cuBLAS autotuning
        // Use a small fixed number to avoid user misconfiguration
        int warmupIters = 3;

        for (int i = 0; i < warmupIters; i++) {
            auto bindingIt = m_inputBindings.find(m_inputName);
            if (bindingIt != m_inputBindings.end() && bindingIt->second) {
                cudaMemsetAsync(bindingIt->second->get(), 0,
                                bindingIt->second->size(),
                                m_pipelineStream->get());
            }

            if (!bindStaticTensorAddresses()) {
                break;
            }

            if (m_context) {
                m_context->enqueueV3(m_pipelineStream->get());
            }
        }

        m_state.needsRebuild = true;
    }
    
    return true;
}


bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();
    
    if (!stream) stream = m_pipelineStream->get();
    
    
    cleanupGraph();
    
    m_captureNodes.clear();
    m_inferenceNodes.clear();
    m_postprocessNodes.clear();
    
    if (!bindStaticTensorAddresses()) {
        return false;
    }

    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin capture: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Graph ?대??먯꽌??蹂듭궗 遺덊븘??- executeFrame?먯꽌 吏곸젒 ?듯빀 踰꾪띁濡?罹≪쿂??
    // 罹≪쿂??Graph ?몃??먯꽌 performFrameCaptureDirectToUnified()濡?泥섎━
    
    // ?꾩쿂由щ룄 Graph???ы븿
    if (m_unifiedArena.yoloInput && !m_captureBuffer.empty()) {
        int modelRes = getModelInputResolution();
        cuda_unified_preprocessing(
            m_captureBuffer.data(),
            m_unifiedArena.yoloInput,
            m_captureBuffer.cols(),
            m_captureBuffer.rows(),
            static_cast<int>(m_captureBuffer.step()),
            modelRes,
            modelRes,
            stream
        );
        
        ensurePrimaryInputBindingAliased();

        void* inputBinding = (m_primaryInputIndex >= 0 &&
                              m_primaryInputIndex < static_cast<int>(m_inputAddressCache.size()))
                                 ? m_inputAddressCache[m_primaryInputIndex]
                                 : nullptr;
        if (inputBinding && inputBinding != m_unifiedArena.yoloInput) {
            size_t inputSize = modelRes * modelRes * 3 * sizeof(float);
            cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                           cudaMemcpyDeviceToDevice, stream);
        }
    }

    // TensorRT 異붾줎 ?ы븿 (Graph ?명솚 紐⑤뜽留??ъ슜)
    if (m_context && m_config.enableDetection) {
        if (!m_context->enqueueV3(stream)) {
            std::cerr << "Warning: TensorRT enqueue failed during graph capture" << std::endl;
        }
    }
    
    if (m_config.enableDetection) {
        performIntegratedPostProcessing(stream);
        performTargetSelection(stream);
        
        // 寃곌낵瑜??몄뒪?몃줈 蹂듭궗 (Graph ?대?)
        if (!m_mouseMovementUsesMappedMemory) {
            cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                           sizeof(MouseMovement), cudaMemcpyDeviceToHost, stream);
        }
        
        // 留덉슦???대룞 肄쒕갚??Graph???ы븿 - 蹂듭궗 ?꾨즺 ???먮룞 ?ㅽ뻾
        if (!enqueueFrameCompletionCallback(stream)) {
            std::cerr << "[UnifiedGraph] Failed to attach completion callback during graph capture" << std::endl;
        }
    }

    err = cudaStreamEndCapture(stream, &m_graph);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to end capture: " 
                  << cudaGetErrorString(err) << std::endl;
        if (m_graph) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
        }
        return false;
    }
    
    if (!validateGraph()) {
        std::cerr << "[UnifiedGraph] Graph validation failed" << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }
    
    err = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 
                               m_config.graphInstantiateFlags);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to instantiate graph: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }
    
    m_state.graphReady = true;
    m_state.needsRebuild = false;
    m_graphCaptured = true;  // Graph 罹≪쿂 ?꾨즺 ?뚮옒洹??ㅼ젙
    
    return true;
}

bool UnifiedGraphPipeline::updateGraphExec() {
    if (!m_graphExec || !m_graph) {
        std::cerr << "[UnifiedGraph] Cannot update: graph not instantiated" << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();

    // Create a new graph with updated parameters
    cudaGraph_t newGraph = nullptr;
    cudaStream_t stream = m_pipelineStream->get();

    if (!bindStaticTensorAddresses()) {
        return false;
    }

    // Begin capture to create updated graph topology
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin update capture: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Capture same operations as original graph
    if (m_unifiedArena.yoloInput && !m_captureBuffer.empty()) {
        int modelRes = getModelInputResolution();
        cuda_unified_preprocessing(
            m_captureBuffer.data(),
            m_unifiedArena.yoloInput,
            m_captureBuffer.cols(),
            m_captureBuffer.rows(),
            static_cast<int>(m_captureBuffer.step()),
            modelRes,
            modelRes,
            stream
        );

        ensurePrimaryInputBindingAliased();

        void* inputBinding = (m_primaryInputIndex >= 0 &&
                              m_primaryInputIndex < static_cast<int>(m_inputAddressCache.size()))
                                 ? m_inputAddressCache[m_primaryInputIndex]
                                 : nullptr;
        if (inputBinding && inputBinding != m_unifiedArena.yoloInput) {
            size_t inputSize = modelRes * modelRes * 3 * sizeof(float);
            cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                           cudaMemcpyDeviceToDevice, stream);
        }
    }

    if (m_context && m_config.enableDetection) {
        if (!m_context->enqueueV3(stream)) {
            cudaStreamEndCapture(stream, &newGraph);
            if (newGraph) cudaGraphDestroy(newGraph);
            return false;
        }
    }

    if (m_config.enableDetection) {
        performIntegratedPostProcessing(stream);
        performTargetSelection(stream);

        if (!m_mouseMovementUsesMappedMemory) {
            cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                           sizeof(MouseMovement), cudaMemcpyDeviceToHost, stream);
        }

        if (!enqueueFrameCompletionCallback(stream)) {
            cudaStreamEndCapture(stream, &newGraph);
            if (newGraph) cudaGraphDestroy(newGraph);
            return false;
        }
    }

    err = cudaStreamEndCapture(stream, &newGraph);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to end update capture: "
                  << cudaGetErrorString(err) << std::endl;
        if (newGraph) cudaGraphDestroy(newGraph);
        return false;
    }

    // Try to update the existing executable
    cudaGraphExecUpdateResultInfo updateResultInfo{};
    err = cudaGraphExecUpdate(m_graphExec, newGraph, &updateResultInfo);

    if (err == cudaSuccess && updateResultInfo.result == cudaGraphExecUpdateSuccess) {
        // Update succeeded - just clean up the temporary graph
        cudaGraphDestroy(newGraph);
        std::cout << "[UnifiedGraph] Graph exec updated successfully (fast path)" << std::endl;
        m_state.needsRebuild = false;
        return true;
    }

    // Update failed or topology changed - need full reinstantiation
    if (updateResultInfo.result == cudaGraphExecUpdateErrorTopologyChanged) {
        std::cout << "[UnifiedGraph] Topology changed, performing full graph reinstantiation" << std::endl;
    } else if (updateResultInfo.result == cudaGraphExecUpdateErrorNodeTypeChanged) {
        std::cout << "[UnifiedGraph] Node type changed, performing full graph reinstantiation" << std::endl;
    } else if (updateResultInfo.result == cudaGraphExecUpdateErrorUnsupportedFunctionChange) {
        std::cout << "[UnifiedGraph] Function signature changed, performing full graph reinstantiation" << std::endl;
    } else {
        std::cerr << "[UnifiedGraph] Graph update failed: " << cudaGetErrorString(err)
                  << ", result=" << updateResultInfo.result << std::endl;
    }

    // Destroy old exec and reinstantiate with new graph
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }

    if (m_graph) {
        cudaGraphDestroy(m_graph);
    }

    m_graph = newGraph;

    err = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr,
                               m_config.graphInstantiateFlags);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to reinstantiate graph: "
                  << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }

    std::cout << "[UnifiedGraph] Graph reinstantiated successfully (slow path)" << std::endl;
    m_state.needsRebuild = false;
    return true;
}





bool UnifiedGraphPipeline::validateGraph() {
    if (!m_graph) return false;
    
    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(m_graph, nullptr, &numNodes);
    if (err != cudaSuccess || numNodes == 0) {
        std::cerr << "[UnifiedGraph] Graph has no nodes" << std::endl;
        return false;
    }
    
    return true;
}

void UnifiedGraphPipeline::cleanupGraph() {
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }

    if (m_graph) {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }

    if (m_updateGraph) {
        cudaGraphDestroy(m_updateGraph);
        m_updateGraph = nullptr;
    }

    m_state.graphReady = false;
}

bool UnifiedGraphPipeline::allocateBuffers() {
    auto& ctx = AppContext::getInstance();
    const int width = ctx.config.detection_resolution;
    const int height = ctx.config.detection_resolution;
    const int yoloSize = getModelInputResolution();
    const int maxDetections = ctx.config.max_detections;
    
    
    try {
        m_captureBuffer.create(height, width, 4);

        size_t arenaSize = SmallBufferArena::calculateArenaSize();
        m_smallBufferArena.arenaBuffer = std::make_unique<CudaMemory<uint8_t>>(arenaSize);
        m_smallBufferArena.initializePointers(m_smallBufferArena.arenaBuffer->get());
        invalidateSelectedTarget(nullptr);

        size_t unifiedArenaSize = UnifiedGPUArena::calculateArenaSize(maxDetections, yoloSize);
        m_unifiedArena.megaArena = std::make_unique<CudaMemory<uint8_t>>(unifiedArenaSize);
        m_unifiedArena.initializePointers(m_unifiedArena.megaArena->get(), maxDetections, yoloSize);
        
        
        
        m_h_movement = std::make_unique<CudaPinnedMemory<MouseMovement>>(
            1, cudaHostAllocMapped | cudaHostAllocPortable);
        if (m_h_movement && m_h_movement->get()) {
            m_h_movement->get()->dx = 0;
            m_h_movement->get()->dy = 0;
        }
        m_mouseMovementUsesMappedMemory = configureMouseMovementBuffer();
        m_h_allowFlags = std::make_unique<CudaPinnedMemory<unsigned char>>(
            Constants::MAX_CLASSES_FOR_FILTERING);

        if (m_h_allowFlags && m_h_allowFlags->get()) {
            std::fill_n(m_h_allowFlags->get(),
                        Constants::MAX_CLASSES_FOR_FILTERING,
                        static_cast<unsigned char>(0));
        }

        // Defer aliasing until TensorRT bindings are created to avoid spurious warnings
        if (m_unifiedArena.yoloInput && !m_inputBindings.empty()) {
            (void)ensurePrimaryInputBindingAliased();
        }

        ensureFinalTargetAliases();

        m_cachedClassFilter.assign(Constants::MAX_CLASSES_FOR_FILTERING, 0);
        m_classFilterDirty.store(true, std::memory_order_release);
        m_cachedHeadClassId.store(-1, std::memory_order_release);
        m_cachedHeadClassNameHash.store(0, std::memory_order_release);
        m_cachedClassSettingsSize.store(0, std::memory_order_release);
        
        {
            std::lock_guard<std::mutex> previewLock(m_previewMutex);
            // Dynamic preview buffer allocation based on current state
            updatePreviewBufferAllocation();
        }
        
        if (!m_captureBuffer.data()) {
            throw std::runtime_error("Capture buffer allocation failed");
        }
        
        size_t gpuMemory = (width * height * 4 + yoloSize * yoloSize * 3) * sizeof(float);
        gpuMemory += unifiedArenaSize;
        if (m_preview.enabled) {
            gpuMemory += width * height * 4 * sizeof(unsigned char);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Buffer allocation failed: " << e.what() << std::endl;
        deallocateBuffers();
        return false;
    }
}

void UnifiedGraphPipeline::deallocateBuffers() {

    releaseRegisteredCaptureBuffers();

    m_captureBuffer.release();
    // m_unifiedCaptureBuffer removed - using m_captureBuffer
    
    // Always release preview buffer if allocated
    if (m_preview.hostPreviewPinned && m_preview.hostPreview.data()) {
        cudaHostUnregister(m_preview.hostPreview.data());
        m_preview.hostPreviewPinned = false;
        m_preview.hostPreviewPinnedSize = 0;
    }
    m_preview.previewBuffer.release();
    m_preview.hostPreview.release();
    m_preview.finalTargets.clear();
    m_preview.enabled = false;
    m_preview.hasValidHostPreview = false;
    
    m_smallBufferArena.arenaBuffer.reset();
    m_smallBufferArena = SmallBufferArena{};

    m_unifiedArena.megaArena.reset();
    m_unifiedArena = UnifiedGPUArena{};

    m_d_inferenceOutput.reset();
    m_d_outputBuffer.reset();

    m_h_movement.reset();
    m_h_allowFlags.reset();
    m_mouseMovementUsesMappedMemory = false;
    m_cachedClassFilter.clear();
    m_classFilterDirty.store(true, std::memory_order_release);
    m_cachedHeadClassId.store(-1, std::memory_order_release);
    m_cachedHeadClassNameHash.store(0, std::memory_order_release);
    m_cachedClassSettingsSize.store(0, std::memory_order_release);

    m_inputBindings.clear();
    m_outputBindings.clear();
    m_inputAddressCache.clear();
    m_outputAddressCache.clear();
    m_bindingsNeedUpdate = true;
    m_primaryInputIndex = -1;

}

bool UnifiedGraphPipeline::ensurePrimaryInputBindingAliased() {
    if (m_primaryInputIndex < 0 ||
        m_primaryInputIndex >= static_cast<int>(m_inputAddressCache.size()) ||
        !m_unifiedArena.yoloInput) {
        return false;
    }

    void* currentBinding = m_inputAddressCache[m_primaryInputIndex];
    if (currentBinding == m_unifiedArena.yoloInput) {
        return true;
    }

    auto bindingIt = m_inputBindings.find(m_inputName);
    auto sizeIt = m_inputSizes.find(m_inputName);
    if (bindingIt == m_inputBindings.end() ||
        sizeIt == m_inputSizes.end() ||
        sizeIt->second == 0) {
        return false;
    }

    try {
        bindingIt->second = std::make_unique<CudaMemory<uint8_t>>(
            reinterpret_cast<uint8_t*>(m_unifiedArena.yoloInput),
            sizeIt->second,
            false);
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Failed to alias TensorRT input binding: "
                  << e.what() << std::endl;
        return false;
    }

    refreshCachedBindings();

    if (m_primaryInputIndex < static_cast<int>(m_inputAddressCache.size()) &&
        m_inputAddressCache[m_primaryInputIndex] == m_unifiedArena.yoloInput) {
        m_bindingsNeedUpdate = true;
        return true;
    }

    return false;
}

void UnifiedGraphPipeline::ensureFinalTargetAliases() {
    bool updated = false;
    if (m_unifiedArena.decodedTargets &&
        m_unifiedArena.finalTargets != m_unifiedArena.decodedTargets) {
        m_unifiedArena.finalTargets = m_unifiedArena.decodedTargets;
        updated = true;
    }

    if (m_smallBufferArena.decodedCount &&
        m_smallBufferArena.finalTargetsCount != m_smallBufferArena.decodedCount) {
        m_smallBufferArena.finalTargetsCount = m_smallBufferArena.decodedCount;
        updated = true;
    }

    if (updated) {
        std::cerr << "[UnifiedGraph] Realigned post-processing buffers to decoded targets" << std::endl;
    }

}

bool UnifiedGraphPipeline::configureMouseMovementBuffer() {
    if (!m_smallBufferArena.mouseMovement) {
        return false;
    }

    if (!m_h_movement || !m_h_movement->get()) {
        return false;
    }

    MouseMovement* mappedPtr = nullptr;
    cudaError_t mapErr = cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&mappedPtr),
        m_h_movement->get(),
        0);

    if (mapErr == cudaSuccess && mappedPtr) {
        m_smallBufferArena.mouseMovement = mappedPtr;
        return true;
    }

    if (mapErr != cudaErrorInvalidValue && mapErr != cudaErrorNotSupported) {
        std::cerr << "[UnifiedGraph] Failed to map mouse movement buffer: "
                  << cudaGetErrorString(mapErr) << std::endl;
    }

    return false;
}


void UnifiedGraphPipeline::setInputFrame(const SimpleCudaMat& frame) {
    if (frame.empty()) return;
    
    if (!m_pipelineStream || !m_pipelineStream->get()) {
        printf("[ERROR] Pipeline stream not initialized\n");
        return;
    }
    
    // Write into ring at write index and mark ready immediately (no event)
    int writeIdx = m_captureWriteIdx;
    if (writeIdx < 0 || writeIdx >= kCaptureRingSize) {
        writeIdx = 0;
        m_captureWriteIdx = 0;
    }

    SimpleCudaMat& dst = m_captureRing[writeIdx];
    if (dst.empty() || dst.rows() != frame.rows() || dst.cols() != frame.cols() || dst.channels() != frame.channels()) {
        dst.create(frame.rows(), frame.cols(), frame.channels());
    }

    size_t dataSize = static_cast<size_t>(frame.rows()) * static_cast<size_t>(frame.cols()) * static_cast<size_t>(frame.channels()) * sizeof(unsigned char);
    cudaError_t err = cudaMemcpyAsync(dst.data(), frame.data(), dataSize, 
                                      cudaMemcpyDeviceToDevice, m_pipelineStream->get());
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to copy frame to ring buffer: %s\n", cudaGetErrorString(err));
        return;
    }

    // Latest-wins: publish as current and advance write index
    m_captureReadIdx = writeIdx;
    m_captureState.store(CaptureState::READY, std::memory_order_release);
    m_captureWriteIdx = (writeIdx + 1) % kCaptureRingSize;
}


void UnifiedGraphPipeline::handleAimbotDeactivation() {
    auto& ctx = AppContext::getInstance();

    clearCountBuffers();
    clearMovementData();
    clearHostPreviewData(ctx);
    m_allowMovement.store(false, std::memory_order_release);
    m_captureRegionCache = {};
    
    // Reset unified capture state
    m_captureState.store(CaptureState::IDLE, std::memory_order_release);

    // Reset filter state
    {
        std::lock_guard<std::mutex> lock(m_movementFilterMutex);
        m_skipNextMovement = true;
    }
}

void UnifiedGraphPipeline::clearCountBuffers() {
    if (m_smallBufferArena.finalTargetsCount) {
        cudaMemsetAsync(m_smallBufferArena.finalTargetsCount, 0, sizeof(int), m_pipelineStream->get());
    }
    if (m_smallBufferArena.decodedCount &&
        m_smallBufferArena.decodedCount != m_smallBufferArena.finalTargetsCount) {
        cudaMemsetAsync(m_smallBufferArena.decodedCount, 0, sizeof(int), m_pipelineStream->get());
    }
    if (m_smallBufferArena.classFilteredCount) {
        cudaMemsetAsync(m_smallBufferArena.classFilteredCount, 0, sizeof(int), m_pipelineStream->get());
    }
    
    if (m_smallBufferArena.bestTargetIndex) {
        cudaMemsetAsync(m_smallBufferArena.bestTargetIndex, -1, sizeof(int), m_pipelineStream->get());
    }

    invalidateSelectedTarget(m_pipelineStream ? m_pipelineStream->get() : nullptr);
}

void UnifiedGraphPipeline::clearMovementData() {
    if (m_h_movement && m_h_movement->get()) {
        m_h_movement->get()->dx = 0;
        m_h_movement->get()->dy = 0;
    }

    if (m_smallBufferArena.mouseMovement && !m_mouseMovementUsesMappedMemory) {
        cudaError_t resetErr = cudaMemset(m_smallBufferArena.mouseMovement, 0, sizeof(MouseMovement));
        if (resetErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Failed to reset device movement buffer: "
                      << cudaGetErrorString(resetErr) << std::endl;
        }
    }

    // Reset PID state when aimbot is deactivated
    if (m_smallBufferArena.pidState) {
        cudaError_t pidResetErr = cudaMemset(m_smallBufferArena.pidState, 0, sizeof(PIDState));
        if (pidResetErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Failed to reset PID state buffer: "
                      << cudaGetErrorString(pidResetErr) << std::endl;
        }
    }

    invalidateSelectedTarget(m_pipelineStream ? m_pipelineStream->get() : nullptr);
    resetMovementFilter();
}

void UnifiedGraphPipeline::resetMovementFilter() {
    std::lock_guard<std::mutex> lock(m_movementFilterMutex);
    m_skipNextMovement = true;
    m_lastFrameTime = {};
}

void UnifiedGraphPipeline::handleFrameFailure(FrameFailureReason reason, int& consecutiveFails) {
    consecutiveFails++;

    switch (reason) {
    case FrameFailureReason::NO_FRAME_READY:
        // Capture pending - prefer blocking sync to avoid excessive yields
        m_perfMetrics.captureWaitCount++;
        if (m_captureReadyEvent && m_captureReadyEvent->get()) {
            // Event created with cudaEventBlockingSync; this will sleep the thread until ready
            cudaEventSynchronize(m_captureReadyEvent->get());
        } else if (m_capture) {
            // Fallback: wait for next present if provider exposes QPC
            uint64_t cur = m_capture->GetLastPresentQpc();
            if (cur) {
                (void)m_capture->WaitForNewFrameSince(cur + 1, 6);
            } else {
                // Minimal backoff if no timing available
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                m_perfMetrics.sleepCount++;
            }
        }
        break;

    case FrameFailureReason::INPUT_PENDING:
        // Waiting for input to be reflected in frame - block briefly on provider signal
        m_perfMetrics.inputPendingCount++;
        if (m_capture) {
            uint64_t cur = m_capture->GetLastPresentQpc();
            if (cur) {
                (void)m_capture->WaitForNewFrameSince(cur + 1, 6);
                break;
            }
        }
        // Fallback: short sleep instead of yield to avoid skyrocketing yield counts
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        m_perfMetrics.sleepCount++;
        break;

    case FrameFailureReason::GPU_BUSY:
        // Previous frame still processing - wait on a CV signaled on completion
        {
            std::unique_lock<std::mutex> lk(m_inflightMutex);
            m_inflightCv.wait_for(lk, std::chrono::milliseconds(2), [this]() {
                return !m_frameInFlight.load(std::memory_order_acquire);
            });
        }
        break;

    case FrameFailureReason::GRAPH_NOT_READY:
        // Graph initialization - longer wait
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        m_perfMetrics.sleepCount++;
        break;

    case FrameFailureReason::CAPTURE_FAILED:
        // Capture error - exponential backoff without yield storms
        if (consecutiveFails <= 32) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            m_perfMetrics.sleepCount++;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            m_perfMetrics.sleepCount++;
        }
        break;

    case FrameFailureReason::NONE:
    default:
        break;
    }
}

bool UnifiedGraphPipeline::checkQPCSupport() {
    if (m_qpcSupportChecked) {
        return m_qpcSupported;
    }
    
    m_qpcSupportChecked = true;
    
    if (!m_capture) {
        m_qpcSupported = false;
        return false;
    }
    
    // Test if GetLastPresentQpc returns valid data
    uint64_t testQpc = m_capture->GetLastPresentQpc();
    
    // If returns 0, either not supported or no frames yet
    // We'll assume not supported for safety
    m_qpcSupported = (testQpc != 0);
    
    if (m_qpcSupported) {
        std::cout << "[Capture] QPC-based input latency reduction enabled" << std::endl;
    } else {
        std::cout << "[Capture] QPC not available, using timer-based input gating" << std::endl;
    }
    
    return m_qpcSupported;
}

void UnifiedGraphPipeline::invalidateSelectedTarget(cudaStream_t stream) {
    if (!m_smallBufferArena.selectedTarget) {
        return;
    }

    Target invalidTarget{};
    invalidTarget.classId = -1;
    invalidTarget.x = -1;
    invalidTarget.y = -1;
    invalidTarget.width = -1;
    invalidTarget.height = -1;
    invalidTarget.confidence = 0.0f;

    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(
            m_smallBufferArena.selectedTarget,
            &invalidTarget,
            sizeof(Target),
            cudaMemcpyHostToDevice,
            stream);
    } else {
        err = cudaMemcpy(
            m_smallBufferArena.selectedTarget,
            &invalidTarget,
            sizeof(Target),
            cudaMemcpyHostToDevice);
    }

    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to invalidate selected target: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

MouseMovement UnifiedGraphPipeline::filterMouseMovement(const MouseMovement& rawMovement, bool movementEnabled) {
    std::lock_guard<std::mutex> lock(m_movementFilterMutex);
    auto& ctx = AppContext::getInstance();

    if (!movementEnabled) {
        m_skipNextMovement = true;
        return {0, 0};
    }

    if (m_skipNextMovement) {
        // Drop the first real movement after (re)activation to avoid large corrections
        if (rawMovement.dx != 0 || rawMovement.dy != 0) {
            m_skipNextMovement = false;
            return {0, 0};
        }
        return rawMovement;
    }

    return rawMovement;
}

void UnifiedGraphPipeline::clearHostPreviewData(AppContext& ctx) {
    {
        std::lock_guard<std::mutex> lock(m_previewMutex);
        if (m_preview.enabled) {
            m_preview.finalTargets.clear();
            m_preview.finalCount = 0;
            m_preview.copyInProgress = false;
            m_preview.hostPreview.release();
            m_preview.hasValidHostPreview = false;
        }
    }

    ctx.clearTargets();
}

void UnifiedGraphPipeline::handleAimbotActivation() {
    m_state.frameCount = 0;
    m_allowMovement.store(false, std::memory_order_release);
    clearMovementData();
    m_captureRegionCache = {};
    
    // Reset unified capture state
    m_captureState.store(CaptureState::IDLE, std::memory_order_release);

    // Reset movement filter state
    {
        std::lock_guard<std::mutex> lock(m_movementFilterMutex);
        m_skipNextMovement = true;
    }
}

bool UnifiedGraphPipeline::enqueueFrameCompletionCallback(cudaStream_t stream) {
    if (!stream) {
        return false;
    }

    cudaError_t err = cudaLaunchHostFunc(stream,
        [](void* userData) {
            auto* pipeline = static_cast<UnifiedGraphPipeline*>(userData);
            if (!pipeline) {
                return;
            }

            auto& ctx = AppContext::getInstance();

            bool allowMovement = pipeline->m_allowMovement.load(std::memory_order_acquire);
            pipeline->m_allowMovement.store(false, std::memory_order_release);

            if (pipeline->m_h_movement && pipeline->m_h_movement->get()) {
                if (allowMovement && ctx.aiming) {
                    MouseMovement rawMovement = *pipeline->m_h_movement->get();
                    MouseMovement filtered = pipeline->filterMouseMovement(rawMovement, true);
                    pipeline->m_h_movement->get()->dx = filtered.dx;
                    pipeline->m_h_movement->get()->dy = filtered.dy;

                    if (filtered.dx != 0 || filtered.dy != 0) {
                        executeMouseMovement(filtered.dx, filtered.dy);

                        // Record input injection timestamp (QPC)
                        LARGE_INTEGER qpc{};
                        if (QueryPerformanceCounter(&qpc)) {
                            pipeline->m_pendingInputQpc.store(static_cast<uint64_t>(qpc.QuadPart), std::memory_order_release);
                        }
                    }
                } else {
                    pipeline->filterMouseMovement({0, 0}, false);
                    pipeline->m_h_movement->get()->dx = 0;
                    pipeline->m_h_movement->get()->dy = 0;
                }
            }

            // Release frame-in-flight flag
            pipeline->m_frameInFlight.store(false, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(pipeline->m_inflightMutex);
            }
            pipeline->m_inflightCv.notify_all();
        }, this);

    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to enqueue completion callback: "
                  << cudaGetErrorString(err) << std::endl;
        m_allowMovement.store(false, std::memory_order_release);
        return false;
    }

    return true;
}

bool UnifiedGraphPipeline::enqueueMovementResetCallback(cudaStream_t stream) {
    if (!stream) {
        return false;
    }

    cudaError_t err = cudaLaunchHostFunc(stream,
        [](void* userData) {
            auto* pipeline = static_cast<UnifiedGraphPipeline*>(userData);
            if (!pipeline) {
                return;
            }

            pipeline->m_allowMovement.store(false, std::memory_order_release);
            pipeline->clearMovementData();

            // Release frame-in-flight flag
            pipeline->m_frameInFlight.store(false, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(pipeline->m_inflightMutex);
            }
            pipeline->m_inflightCv.notify_all();
        }, this);

    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to enqueue movement reset callback: "
                  << cudaGetErrorString(err) << std::endl;
        m_allowMovement.store(false, std::memory_order_release);
        return false;
    }

    return true;
}

void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();

    bool wasAiming = false;

    while (!m_shouldStop.load(std::memory_order_acquire) && !ctx.should_exit.load()) {
        {
            std::unique_lock<std::mutex> lock(ctx.pipeline_activation_mutex);
            ctx.pipeline_activation_cv.wait(lock, [&ctx, this]() {
                return ctx.aiming.load() || m_shouldStop.load(std::memory_order_acquire) ||
                       ctx.should_exit.load();
            });
        }

        if (m_shouldStop.load(std::memory_order_acquire) || ctx.should_exit.load()) {
            break;
        }

        if (!ctx.aiming.load()) {
            if (wasAiming) {
                handleAimbotDeactivation();
                wasAiming = false;
            }
            continue;
        }

        if (!wasAiming) {
            handleAimbotActivation();
            wasAiming = true;
        }

        // Smart waiting based on failure reason - minimal CPU usage
        int consecutiveFails = 0;
        while (ctx.aiming.load() && !m_shouldStop.load(std::memory_order_acquire) &&
               !ctx.should_exit.load()) {
            
            FrameFailureReason failReason = FrameFailureReason::NONE;
            if (!executeFrame(&failReason)) {
                // Use smart waiting based on failure reason
                handleFrameFailure(failReason, consecutiveFails);
            } else {
                consecutiveFails = 0;
            }
        }

        if (wasAiming && !ctx.aiming.load()) {
            handleAimbotDeactivation();
            wasAiming = false;
        }
    }

    if (wasAiming) {
        handleAimbotDeactivation();
    }
}


void UnifiedGraphPipeline::stopMainLoop() {
    m_shouldStop.store(true, std::memory_order_release);

    auto& ctx = AppContext::getInstance();
    ctx.pipeline_activation_cv.notify_all();
}


bool UnifiedGraphPipeline::initializeTensorRT(const std::string& modelFile) {
    auto& ctx = AppContext::getInstance();
    
    
    if (!loadEngine(modelFile)) {
        std::cerr << "[Pipeline] Failed to load engine" << std::endl;
        return false;
    }
    
    
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[Pipeline] Failed to create execution context" << std::endl;
        return false;
    }
    
    if (m_engine->getNbOptimizationProfiles() > 0) {
        m_context->setOptimizationProfileAsync(0, m_pipelineStream->get());
    }
    
    
    getInputNames();
    getOutputNames();
    
    if (!m_inputNames.empty()) {
        m_inputName = m_inputNames[0];
        m_primaryInputIndex = 0;
        m_inputDims = m_engine->getTensorShape(m_inputName.c_str());
        
        if (m_inputDims.nbDims == 4) {
            m_modelInputResolution = m_inputDims.d[2];
        } else if (m_inputDims.nbDims == 3) {
            m_modelInputResolution = m_inputDims.d[1];
        }
        
        size_t inputSize = 1;
        for (int i = 0; i < m_inputDims.nbDims; ++i) {
            inputSize *= m_inputDims.d[i];
        }
        inputSize *= sizeof(float);
        m_inputSizes[m_inputName] = inputSize;
        
    }
    
    for (const auto& outputName : m_outputNames) {
        nvinfer1::Dims outputDims = m_engine->getTensorShape(outputName.c_str());
        size_t outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; ++i) {
            outputSize *= outputDims.d[i];
        }
        outputSize *= sizeof(float);
        m_outputSizes[outputName] = outputSize;
        
        
    }
    
    // Defer getBindings() until after arena allocation so we can alias
    // the primary input to the unified arena and avoid duplicate memory.
    
    m_imgScale = static_cast<float>(ctx.config.detection_resolution) / getModelInputResolution();
    
    
    const auto& outputShape = m_outputShapes[m_outputNames[0]];
    if (outputShape.size() >= 2) {
        m_numClasses = static_cast<int>(outputShape[1]) - 4;
    } else {
        m_numClasses = 80;
    }
    
    return true;
}

void UnifiedGraphPipeline::getInputNames() {
    auto& ctx = AppContext::getInstance();
    m_inputNames.clear();
    m_inputSizes.clear();

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            m_inputNames.emplace_back(name);
        }
    }
}

void UnifiedGraphPipeline::getOutputNames() {
    auto& ctx = AppContext::getInstance();
    m_outputNames.clear();
    m_outputSizes.clear();
    m_outputShapes.clear();
    m_outputTypes.clear();

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            m_outputNames.emplace_back(name);
            
            auto dims = m_engine->getTensorShape(name);
            std::vector<int64_t> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            m_outputShapes[name] = shape;
            
            auto dataType = m_engine->getTensorDataType(name);
            m_outputTypes[name] = dataType;
            
        }
    }
}

void UnifiedGraphPipeline::getBindings() {
    m_inputBindings.clear();
    m_outputBindings.clear();

    for (const auto& name : m_inputNames) {
        size_t size = m_inputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for input '" << name << "'" << std::endl;
            continue;
        }
        
        try {
            // ?쇰떒 ?먮옒?濡?蹂듦뎄 - 吏곸젒 ?ъ씤???ъ슜? TensorRT 諛붿씤?⑷낵 ?명솚??臾몄젣 ?덉쓬
            m_inputBindings[name] = (name == m_inputName && m_unifiedArena.yoloInput)
                ? std::unique_ptr<CudaMemory<uint8_t>>(new CudaMemory<uint8_t>(
                      reinterpret_cast<uint8_t*>(m_unifiedArena.yoloInput),
                      size,
                      false))
                : std::unique_ptr<CudaMemory<uint8_t>>(new CudaMemory<uint8_t>(size));
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate input memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
    }

    for (const auto& name : m_outputNames) {
        size_t size = m_outputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for output '" << name << "'" << std::endl;
            continue;
        }
        
        try {
            // Simplified - same allocation for all outputs
            m_outputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate output memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
    }
    
    
    if (m_inputBindings.size() != m_inputNames.size()) {
        std::cerr << "[Pipeline] Warning: Input binding count mismatch" << std::endl;
    }
    if (m_outputBindings.size() != m_outputNames.size()) {
        std::cerr << "[Pipeline] Warning: Output binding count mismatch" << std::endl;
    }

    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        if (m_inputNames[i] == m_inputName) {
            m_primaryInputIndex = static_cast<int>(i);
            break;
        }
    }

    refreshCachedBindings();

}

void UnifiedGraphPipeline::refreshCachedBindings() {
    m_inputAddressCache.assign(m_inputNames.size(), nullptr);
    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        const auto& name = m_inputNames[i];
        auto bindingIt = m_inputBindings.find(name);
        if (bindingIt != m_inputBindings.end() && bindingIt->second) {
            m_inputAddressCache[i] = bindingIt->second->get();
        }
    }

    m_outputAddressCache.assign(m_outputNames.size(), nullptr);
    for (size_t i = 0; i < m_outputNames.size(); ++i) {
        const auto& name = m_outputNames[i];
        auto bindingIt = m_outputBindings.find(name);
        if (bindingIt != m_outputBindings.end() && bindingIt->second) {
            m_outputAddressCache[i] = bindingIt->second->get();
        }
    }

    m_bindingsNeedUpdate = true;
}

bool UnifiedGraphPipeline::bindStaticTensorAddresses() {
    if (!m_bindingsNeedUpdate) {
        return true;
    }

    if (!m_context) {
        return false;
    }

    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        void* address = m_inputAddressCache[i];
        if (!address) {
            std::cerr << "[Pipeline] Input binding address missing for: " << m_inputNames[i] << std::endl;
            return false;
        }

        if (!m_context->setTensorAddress(m_inputNames[i].c_str(), address)) {
            std::cerr << "[Pipeline] Failed to bind input tensor: " << m_inputNames[i] << std::endl;
            return false;
        }
    }

    for (size_t i = 0; i < m_outputNames.size(); ++i) {
        void* address = m_outputAddressCache[i];
        if (!address) {
            std::cerr << "[Pipeline] Output binding address missing for: " << m_outputNames[i] << std::endl;
            return false;
        }

        if (!m_context->setTensorAddress(m_outputNames[i].c_str(), address)) {
            std::cerr << "[Pipeline] Failed to bind output tensor: " << m_outputNames[i] << std::endl;
            return false;
        }
    }

    m_bindingsNeedUpdate = false;
    return true;
}


bool UnifiedGraphPipeline::loadEngine(const std::string& modelFile) {
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();
    

    if (extension != ".engine") {
        std::cerr << "[Pipeline] Error: Only .engine files are supported. Please use EngineExport tool to convert ONNX to engine format." << std::endl;
        return false;
    }

    if (!fileExists(modelFile)) {
        std::cerr << "[Pipeline] Engine file does not exist: " << modelFile << std::endl;
        return false;
    }

    std::string engineFilePath = modelFile;

    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
#ifdef _DEBUG
            if (severity <= Severity::kERROR && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
#else
            (void)severity;
            (void)msg;
#endif
        }
    };
    static SimpleLogger logger;

    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Pipeline] Failed to open engine file: " << engineFilePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    if (!m_runtime) {
        m_runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!m_runtime) {
            std::cerr << "[Pipeline] Failed to create runtime" << std::endl;
            return false;
        }
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(buffer.data(), size));
    
    if (m_engine) {
        return true;
    } else {
        std::cerr << "[Pipeline] Failed to load engine from: " << engineFilePath << std::endl;
        return false;
    }
}

int UnifiedGraphPipeline::getModelInputResolution() const {
    return m_modelInputResolution;
}

bool UnifiedGraphPipeline::runInferenceAsync(cudaStream_t stream) {
    if (!m_context || !m_engine) {
        std::cerr << "[Pipeline] TensorRT context or engine not initialized" << std::endl;
        return false;
    }
    
    if (!stream) {
        stream = m_pipelineStream->get();
    }
    
    if (!bindStaticTensorAddresses()) {
        return false;
    }

    
    bool success = m_context->enqueueV3(stream);
    if (!success) {
        cudaError_t cudaErr = cudaGetLastError();
        std::cerr << "[Pipeline] TensorRT inference failed";
        if (cudaErr != cudaSuccess) {
            std::cerr << " - CUDA error: " << cudaGetErrorString(cudaErr);
        }
        std::cerr << std::endl;
        
        
        return false;
    }
     
    // Removed dead code block
    
    return true;
}

void needaimbot::PostProcessingConfig::updateFromContext(const AppContext& ctx, bool graphCaptured) {
    if (!graphCaptured) {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(ctx.configMutex));
        max_detections = ctx.config.max_detections;
        confidence_threshold = ctx.config.confidence_threshold;
        postprocess = ctx.config.postprocess;
    }
}

void UnifiedGraphPipeline::clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream) {
    if (!m_smallBufferArena.decodedCount || config.max_detections <= 0) {
        return;
    }

    cudaMemsetAsync(m_smallBufferArena.decodedCount, 0, sizeof(int), stream);

    if (m_smallBufferArena.finalTargetsCount &&
        m_smallBufferArena.finalTargetsCount != m_smallBufferArena.decodedCount) {
        cudaMemsetAsync(m_smallBufferArena.finalTargetsCount, 0, sizeof(int), stream);
    }

    if (m_smallBufferArena.classFilteredCount) {
        cudaMemsetAsync(m_smallBufferArena.classFilteredCount, 0, sizeof(int), stream);
    }

    

    if (m_smallBufferArena.bestTargetIndex) {
        cudaMemsetAsync(m_smallBufferArena.bestTargetIndex, -1, sizeof(int), stream);
    }

    if (m_smallBufferArena.bestTarget) {
        cudaMemsetAsync(m_smallBufferArena.bestTarget, 0, sizeof(Target), stream);
    }

    invalidateSelectedTarget(stream);
}

cudaError_t UnifiedGraphPipeline::decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType, 
                                                   const std::vector<int64_t>& shape, 
                                                   const PostProcessingConfig& config, cudaStream_t stream) {
    int maxDecodedTargets = config.max_detections;
    
    if (config.postprocess == "yolo_nms") {
        // NMS媛 ?ы븿??紐⑤뜽 - ?대? ?꾩쿂由щ맂 異쒕젰
        // 異쒕젰 ?뺤떇: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
        int num_detections = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        int output_features = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        if (output_features != 6) {
            std::cerr << "[Pipeline] Invalid NMS output format. Expected 6 features [x1,y1,x2,y2,conf,class], got " << output_features << std::endl;
            return cudaErrorInvalidValue;
        }
        
        // Update class filter if needed
        updateClassFilterIfNeeded(stream);
        
        // Use existing decodeYolo10Gpu which handles pre-processed format
        return decodeYolo10Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
            maxDecodedTargets, num_detections,
            m_smallBufferArena.allowFlags, m_numClasses, stream);
            
    } else if (config.postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        // Update class filter if needed
        updateClassFilterIfNeeded(stream);
        
        return decodeYolo10Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
            maxDecodedTargets, max_candidates,
            m_smallBufferArena.allowFlags, m_numClasses, stream);
            
    // NMS removed for performance - not needed for aimbot
    } else if (config.postprocess == "yolo8" || config.postprocess == "yolo9" || 
               config.postprocess == "yolo11" || config.postprocess == "yolo12") {
        int max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        if (!validateYoloDecodeBuffers(maxDecodedTargets, max_candidates)) {
            return cudaErrorInvalidValue;
        }
        
        updateClassFilterIfNeeded(stream);
        
        return decodeYolo11Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
            maxDecodedTargets, max_candidates,
            m_smallBufferArena.allowFlags,
            Constants::MAX_CLASSES_FOR_FILTERING, stream);
    }
    
    std::cerr << "[Pipeline] Unsupported post-processing type: " << config.postprocess << std::endl;
    return cudaErrorNotSupported;
}

bool UnifiedGraphPipeline::validateYoloDecodeBuffers(int maxDecodedTargets, int max_candidates) {
    if (!m_unifiedArena.decodedTargets || !m_smallBufferArena.decodedCount) {
        std::cerr << "[Pipeline] Target buffers not allocated!" << std::endl;
        return false;
    }
    
    if (max_candidates <= 0 || maxDecodedTargets <= 0) {
        std::cerr << "[Pipeline] Invalid buffer sizes: max_candidates=" << max_candidates 
                  << ", maxDecodedTargets=" << maxDecodedTargets << std::endl;
        return false;
    }
    
    return true;
}

void UnifiedGraphPipeline::updateClassFilterIfNeeded(cudaStream_t stream) {
    if (!m_smallBufferArena.allowFlags) {
        return;
    }

    if (!m_classFilterDirty.load(std::memory_order_acquire)) {
        return;
    }

    if (!m_h_allowFlags || m_h_allowFlags->size() < Constants::MAX_CLASSES_FOR_FILTERING) {
        m_h_allowFlags = std::make_unique<CudaPinnedMemory<unsigned char>>(
            Constants::MAX_CLASSES_FOR_FILTERING);
        if (m_h_allowFlags && m_h_allowFlags->get()) {
            std::fill_n(m_h_allowFlags->get(),
                        Constants::MAX_CLASSES_FOR_FILTERING,
                        static_cast<unsigned char>(0));
        }
    }

    unsigned char* hostFlags = m_h_allowFlags ? m_h_allowFlags->get() : nullptr;
    if (!hostFlags) {
        std::cerr << "[Pipeline] Failed to allocate host class filter buffer" << std::endl;
        return;
    }

    auto& ctx = AppContext::getInstance();
    size_t classSettingsSize = 0;
    size_t headNameHash = 0;
    int newHeadId = -1;

    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        classSettingsSize = ctx.config.class_settings.size();
        headNameHash = std::hash<std::string>{}(ctx.config.head_class_name);
        std::fill_n(hostFlags,
                    Constants::MAX_CLASSES_FOR_FILTERING,
                    static_cast<unsigned char>(0));

        for (const auto& setting : ctx.config.class_settings) {
            if (setting.id >= 0 && setting.id < Constants::MAX_CLASSES_FOR_FILTERING) {
                hostFlags[setting.id] = setting.allow ? 1 : 0;
                if (setting.name == ctx.config.head_class_name) {
                    newHeadId = setting.id;
                }
            }
        }
    }

    bool filterChanged = m_cachedClassFilter.size() != Constants::MAX_CLASSES_FOR_FILTERING;
    if (!filterChanged) {
        filterChanged = !std::equal(
            hostFlags,
            hostFlags + Constants::MAX_CLASSES_FOR_FILTERING,
            m_cachedClassFilter.begin());
    }

    bool headChanged = newHeadId != m_cachedHeadClassId.load(std::memory_order_acquire);
    headChanged = headChanged ||
        headNameHash != m_cachedHeadClassNameHash.load(std::memory_order_acquire) ||
        classSettingsSize != m_cachedClassSettingsSize.load(std::memory_order_acquire);

    if (!filterChanged && !headChanged) {
        m_classFilterDirty.store(false, std::memory_order_release);
        return;
    }

    if (filterChanged) {
        m_cachedClassFilter.assign(hostFlags,
                                   hostFlags + Constants::MAX_CLASSES_FOR_FILTERING);

        cudaError_t copyErr = cudaMemcpyAsync(
            m_smallBufferArena.allowFlags,
            hostFlags,
            Constants::MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char),
            cudaMemcpyHostToDevice,
            stream);

        if (copyErr != cudaSuccess) {
            std::cerr << "[Pipeline] Failed to upload class filter flags: "
                      << cudaGetErrorString(copyErr) << std::endl;
            return;
        }
    }

    m_cachedHeadClassId.store(newHeadId, std::memory_order_release);
    m_cachedHeadClassNameHash.store(headNameHash, std::memory_order_release);
    m_cachedClassSettingsSize.store(classSettingsSize, std::memory_order_release);
    m_classFilterDirty.store(false, std::memory_order_release);
}

void UnifiedGraphPipeline::performIntegratedPostProcessing(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    if (m_outputNames.empty()) {
        std::cerr << "[Pipeline] No output names found for post-processing." << std::endl;
        return;
    }

    const std::string& primaryOutputName = m_outputNames[0];
    void* d_rawOutputPtr = m_outputAddressCache.empty() ? nullptr : m_outputAddressCache[0];
    nvinfer1::DataType outputType = m_outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = m_outputShapes[primaryOutputName];

    if (!d_rawOutputPtr) {
        std::cerr << "[Pipeline] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        return;
    }

    static PostProcessingConfig config{Constants::MAX_DETECTIONS, 0.001f, "yolo12"};
    config.updateFromContext(ctx, m_graphCaptured);
    
    clearDetectionBuffers(config, stream);
    
    cudaError_t decodeErr = decodeYoloOutput(d_rawOutputPtr, outputType, shape, config, stream);
    if (decodeErr != cudaSuccess) {
        std::cerr << "[Pipeline] GPU decoding failed: " << cudaGetErrorString(decodeErr) << std::endl;
        return;
    }
    
    // Direct use of decoded targets as final (no NMS needed for aimbot)
    ensureFinalTargetAliases();

}

// Removed redundant NMS and copy functions
// Empty preview functions removed for performance


void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    if (!m_unifiedArena.finalTargets || !m_smallBufferArena.finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }

    if (!m_smallBufferArena.bestTargetIndex || !m_smallBufferArena.bestTarget || !m_smallBufferArena.mouseMovement || !m_smallBufferArena.pidState) {
        std::cerr << "[Pipeline] Target selection buffers not allocated!" << std::endl;
        return;
    }

    // Lock-free config access using cached values
    if (!m_graphCaptured && m_pidConfigDirty.load(std::memory_order_acquire)) {
        // Update cached config only when dirty flag is set
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        m_cachedPIDConfig.max_detections = ctx.config.max_detections;
        m_cachedPIDConfig.kp_x = ctx.config.pid_kp_x;
        m_cachedPIDConfig.kp_y = ctx.config.pid_kp_y;
        m_cachedPIDConfig.ki_x = ctx.config.pid_ki_x;
        m_cachedPIDConfig.ki_y = ctx.config.pid_ki_y;
        m_cachedPIDConfig.kd_x = ctx.config.pid_kd_x;
        m_cachedPIDConfig.kd_y = ctx.config.pid_kd_y;
        m_cachedPIDConfig.integral_max = ctx.config.pid_integral_max;
        m_cachedPIDConfig.head_y_offset = ctx.config.head_y_offset;
        m_cachedPIDConfig.body_y_offset = ctx.config.body_y_offset;
        m_pidConfigDirty.store(false, std::memory_order_release);
    }

    // Use cached values (lock-free)
    int cached_max_detections = m_cachedPIDConfig.max_detections;
    float cached_kp_x = m_cachedPIDConfig.kp_x;
    float cached_kp_y = m_cachedPIDConfig.kp_y;
    float cached_ki_x = m_cachedPIDConfig.ki_x;
    float cached_ki_y = m_cachedPIDConfig.ki_y;
    float cached_kd_x = m_cachedPIDConfig.kd_x;
    float cached_kd_y = m_cachedPIDConfig.kd_y;
    float cached_integral_max = m_cachedPIDConfig.integral_max;
    float cached_head_y_offset = m_cachedPIDConfig.head_y_offset;
    float cached_body_y_offset = m_cachedPIDConfig.body_y_offset;
    
    float crosshairX = ctx.config.detection_resolution / 2.0f;
    float crosshairY = ctx.config.detection_resolution / 2.0f;

    int head_class_id = findHeadClassId(ctx);

    const int blockSize = computeTargetSelectionBlockSize(cached_max_detections);
    const int gridSize = 1;
    const size_t sharedBytes = static_cast<size_t>(blockSize) *
                               (sizeof(float) + sizeof(int) + sizeof(float) + sizeof(int));

    cudaError_t staleError = cudaGetLastError();
    if (staleError != cudaSuccess) {
        std::cerr << "[Pipeline] Clearing stale CUDA error before target selection: "
                  << cudaGetErrorString(staleError) << std::endl;
    }

    fusedTargetSelectionAndMovementKernel<<<gridSize, blockSize, sharedBytes, stream>>>(
        m_unifiedArena.finalTargets,
        m_smallBufferArena.finalTargetsCount,
        cached_max_detections,
        crosshairX,
        crosshairY,
        head_class_id,
        cached_kp_x,
        cached_kp_y,
        cached_ki_x,
        cached_ki_y,
        cached_kd_x,
        cached_kd_y,
        cached_integral_max,
        cached_head_y_offset,
        cached_body_y_offset,
        ctx.config.detection_resolution,
        m_smallBufferArena.selectedTarget,
        m_smallBufferArena.bestTargetIndex,
        m_smallBufferArena.bestTarget,
        m_smallBufferArena.mouseMovement,
        m_smallBufferArena.pidState
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Fused kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

}

bool UnifiedGraphPipeline::updateDDACaptureRegion(const AppContext& ctx) {
    if (!m_capture) {
        return false;
    }

    int detectionRes = ctx.config.detection_resolution;
    if (detectionRes <= 0) {
        return false;
    }

    bool useAimShootOffset = ctx.config.enable_aim_shoot_offset && ctx.aiming && ctx.shooting;
    float offsetX = useAimShootOffset ? ctx.config.aim_shoot_offset_x : ctx.config.crosshair_offset_x;
    float offsetY = useAimShootOffset ? ctx.config.aim_shoot_offset_y : ctx.config.crosshair_offset_y;

    if (m_captureRegionCache.detectionRes == detectionRes &&
        m_captureRegionCache.offsetX == offsetX &&
        m_captureRegionCache.offsetY == offsetY &&
        m_captureRegionCache.usingAimShootOffset == useAimShootOffset) {
        return true;
    }

    int screenW = m_capture->GetScreenWidth();
    int screenH = m_capture->GetScreenHeight();
    if (screenW <= 0 || screenH <= 0) {
        return false;
    }

    int captureSize = std::min(detectionRes, std::min(screenW, screenH));
    if (captureSize <= 0) {
        return false;
    }

    int centerX = screenW / 2 + static_cast<int>(offsetX);
    int centerY = screenH / 2 + static_cast<int>(offsetY);

    int maxLeft = std::max(0, screenW - captureSize);
    int maxTop = std::max(0, screenH - captureSize);

    int left = std::clamp(centerX - captureSize / 2, 0, maxLeft);
    int top = std::clamp(centerY - captureSize / 2, 0, maxTop);

    if (!m_capture->SetCaptureRegion(left, top, captureSize, captureSize)) {
        return false;
    }

    m_captureRegionCache.detectionRes = detectionRes;
    m_captureRegionCache.offsetX = offsetX;
    m_captureRegionCache.offsetY = offsetY;
    m_captureRegionCache.usingAimShootOffset = useAimShootOffset;
    m_captureRegionCache.left = left;
    m_captureRegionCache.top = top;
    m_captureRegionCache.size = captureSize;
    return true;
}

bool UnifiedGraphPipeline::copyFrameToBuffer(
    void* frameData,
    unsigned int width,
    unsigned int height,
    SimpleCudaMat& targetBuffer,
    cudaStream_t stream
) {
    if (!frameData || width == 0 || height == 0 || !stream) {
        return false;
    }

    if (targetBuffer.empty() ||
        targetBuffer.cols() != static_cast<int>(width) ||
        targetBuffer.rows() != static_cast<int>(height) ||
        targetBuffer.channels() != 4) {
        targetBuffer.create(static_cast<int>(height), static_cast<int>(width), 4);
    }

    if (targetBuffer.empty()) {
        return false;
    }

    size_t hostPitch = static_cast<size_t>(width) * 4;
    cudaError_t err = cudaMemcpy2DAsync(
        targetBuffer.data(),
        targetBuffer.step(),
        frameData,
        hostPitch,
        hostPitch,
        height,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        std::cerr << "[Capture] DDA frame copy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

bool UnifiedGraphPipeline::copyDDAFrameToGPU(void* frameData, unsigned int width, unsigned int height) {
    if (!m_pipelineStream) {
        return false;
    }

    size_t totalBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    ensureCaptureBufferRegistered(frameData, totalBytes);

    return copyFrameToBuffer(frameData, width, height, m_captureBuffer, m_pipelineStream->get());
}

bool UnifiedGraphPipeline::ensureCaptureBufferRegistered(void* frameData, size_t size) {
    if (!frameData || size == 0) {
        return false;
    }

    auto it = m_registeredCaptureBuffers.find(frameData);
    if (it != m_registeredCaptureBuffers.end()) {
        if (it->second.registered) {
            return true;
        }
        if (it->second.permanentFailure) {
            return false;
        }
    }

    RegisteredHostBuffer bufferInfo;
    bufferInfo.size = size;

    // Check if this is already pinned memory (from cudaHostAlloc) to skip registration
    cudaPointerAttributes attrs{};
    cudaError_t attrErr = cudaPointerGetAttributes(&attrs, frameData);
    if (attrErr == cudaSuccess && (attrs.type == cudaMemoryTypeHost)) {
        // Already pinned memory, no need to register
        bufferInfo.registered = true;
        bufferInfo.permanentFailure = false;
        m_registeredCaptureBuffers[frameData] = bufferInfo;
        return true;
    }
    // Clear any error from cudaPointerGetAttributes
    if (attrErr != cudaSuccess) {
        cudaGetLastError();
    }

    cudaError_t regErr = cudaHostRegister(frameData, size, cudaHostRegisterPortable);
    if (regErr == cudaSuccess || regErr == cudaErrorHostMemoryAlreadyRegistered) {
        bufferInfo.registered = true;
        bufferInfo.permanentFailure = false;
        m_registeredCaptureBuffers[frameData] = bufferInfo;
        // Trim any stale registrations to keep the map bounded in long runs
        unregisterStaleCaptureBuffers(frameData);
        return true;
    }

    bufferInfo.registered = false;
    bufferInfo.permanentFailure = true;
    m_registeredCaptureBuffers[frameData] = bufferInfo;

    if (regErr != cudaErrorInvalidValue && regErr != cudaErrorNotSupported) {
        std::cerr << "[Capture] Failed to register capture buffer for async copy: "
                  << cudaGetErrorString(regErr) << std::endl;
    }

    return false;
}

void UnifiedGraphPipeline::unregisterStaleCaptureBuffers(void* keepPtr) {
    if (m_registeredCaptureBuffers.size() <= 1) return;
    std::vector<void*> toErase;
    toErase.reserve(m_registeredCaptureBuffers.size());
    for (const auto& entry : m_registeredCaptureBuffers) {
        void* ptr = entry.first;
        const auto& info = entry.second;
        if (ptr == keepPtr) continue;
        if (info.registered) {
            cudaError_t err = cudaHostUnregister(ptr);
            if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
                std::cerr << "[Capture] Failed to unregister stale capture buffer: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        toErase.push_back(ptr);
    }
    for (void* ptr : toErase) {
        m_registeredCaptureBuffers.erase(ptr);
    }
}

void UnifiedGraphPipeline::releaseRegisteredCaptureBuffers() {
    for (auto& entry : m_registeredCaptureBuffers) {
        if (!entry.first || !entry.second.registered) {
            continue;
        }

        cudaError_t err = cudaHostUnregister(entry.first);
        if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
            std::cerr << "[Capture] Failed to unregister capture buffer: "
                      << cudaGetErrorString(err) << std::endl;
        }
    }

    m_registeredCaptureBuffers.clear();
}

bool UnifiedGraphPipeline::waitForCaptureCompletion() {
    CaptureState state = m_captureState.load(std::memory_order_acquire);
    
    if (state != CaptureState::CAPTURING) {
        // Not capturing, return true if we have a ready frame
        return (state == CaptureState::READY);
    }

    if (!m_captureReadyEvent || !m_captureReadyEvent->get()) {
        return false;
    }

    cudaError_t q = cudaEventQuery(m_captureReadyEvent->get());
    if (q == cudaErrorNotReady) {
        // Non-blocking: capture still in flight
        return false;
    }
    if (q != cudaSuccess) {
        std::cerr << "[Capture] Failed to query capture event: "
                  << cudaGetErrorString(q) << std::endl;
        // Mark as IDLE on error
        m_captureState.store(CaptureState::IDLE, std::memory_order_release);
        return false;
    }

    // Capture complete - update read index
    if (m_capturePendingIdx >= 0 && m_capturePendingIdx < kCaptureRingSize) {
        m_captureReadIdx = m_capturePendingIdx; // latest-wins
    }
    m_capturePendingIdx = -1;
    
    // Transition to READY state
    m_captureState.store(CaptureState::READY, std::memory_order_release);

    auto& ctx = AppContext::getInstance();
    if ((m_preview.enabled || ctx.config.show_window) &&
        m_pipelineStream && m_pipelineStream->get() &&
        m_captureReadIdx >= 0) {
        updatePreviewBuffer(m_captureRing[m_captureReadIdx]);
    }

    return true;
}

FrameFailureReason UnifiedGraphPipeline::scheduleNextFrameCapture(bool forceSync) {
    if (!m_config.enableCapture) {
        return FrameFailureReason::NONE;
    }

    if (!m_capture) {
        std::cerr << "[Capture] DDA capture interface not set" << std::endl;
        return FrameFailureReason::CAPTURE_FAILED;
    }

    CaptureState state = m_captureState.load(std::memory_order_acquire);

    // Optimized state transitions
    if (state == CaptureState::CAPTURING) {
        if (!forceSync) {
            // Already capturing - return immediately (most common case in async path)
            return FrameFailureReason::NONE;
        } else {
            // forceSync requested - wait for current capture to complete
            if (!waitForCaptureCompletion()) {
                return FrameFailureReason::NO_FRAME_READY;
            }
            state = m_captureState.load(std::memory_order_acquire);
        }
    }

    // If frame is ready, return immediately (no need to schedule)
    if (state == CaptureState::READY && !forceSync) {
        return FrameFailureReason::NONE;
    }
    
    // Check QPC support on first run
    checkQPCSupport();
    
    // If we issued mouse input last frame, ensure we only process after a frame that includes it
    uint64_t minQpc = m_pendingInputQpc.load(std::memory_order_acquire);
    if (minQpc != 0) {
        if (m_qpcSupported) {
            // Block briefly until a frame with LastPresentQpc >= minQpc is available
            if (!m_capture->WaitForNewFrameSince(minQpc, 6)) {
                m_perfMetrics.frameSkipCount++;
                return FrameFailureReason::INPUT_PENDING;
            }
            // Satisfied - clear flag
            m_pendingInputQpc.store(0, std::memory_order_release);
        } else {
            // Fallback: simple frame count based gating (wait 1 frame minimum)
            static uint64_t lastInputFrame = 0;
            uint64_t currentFrame = m_state.frameCount;
            if (currentFrame <= lastInputFrame + 1) {
                m_perfMetrics.frameSkipCount++;
                return FrameFailureReason::INPUT_PENDING;
            }
            lastInputFrame = currentFrame;
            m_pendingInputQpc.store(0, std::memory_order_release);
        }
    }

    auto& ctx = AppContext::getInstance();
    if (!updateDDACaptureRegion(ctx)) {
        return FrameFailureReason::CAPTURE_FAILED;
    }

    // Try GPU-direct path first (zero-copy CUDA interop)
    cudaArray_t cudaArray = nullptr;
    unsigned int width = 0;
    unsigned int height = 0;

    bool useGPUDirect = m_capture->GetLatestFrameGPU(&cudaArray, &width, &height);

    static bool gpuDirectLoggedOnce = false;
    if (!gpuDirectLoggedOnce && useGPUDirect) {
        std::cout << "[Capture] CUDA Interop enabled - using GPU-direct zero-copy path" << std::endl;
        gpuDirectLoggedOnce = true;
    }

    cudaStream_t copyStream = nullptr;
    if (m_captureStream && m_captureStream->get()) {
        copyStream = m_captureStream->get();
    } else if (m_pipelineStream && m_pipelineStream->get()) {
        copyStream = m_pipelineStream->get();
    }

    if (!copyStream) {
        return FrameFailureReason::CAPTURE_FAILED;
    }

    // Select ring write slot for capture
    int writeIdx = m_captureWriteIdx;
    if (writeIdx < 0 || writeIdx >= kCaptureRingSize) {
        writeIdx = 0;
        m_captureWriteIdx = 0;
    }
    SimpleCudaMat& targetBuffer = m_captureRing[writeIdx];

    if (useGPUDirect && cudaArray) {
        // Zero-copy GPU path: copy directly from CUDA array
        int heightInt = (int)height;
        int widthInt = (int)width;

        // Call member functions to extract values
        uint8_t* bufferData = targetBuffer.data();
        int bufferRows = targetBuffer.rows();
        int bufferCols = targetBuffer.cols();
        int bufferChannels = targetBuffer.channels();
        size_t bufferStep = targetBuffer.step();

        // Auto-reallocate buffer if size mismatches to maintain GPU-direct path
        if (!bufferData || bufferRows != heightInt || bufferCols != widthInt || bufferChannels != 4) {
            try {
                targetBuffer.create(heightInt, widthInt, 4);
                bufferData = targetBuffer.data();
                bufferStep = targetBuffer.step();
                std::cout << "[Capture] Reallocated GPU-direct buffer: " << widthInt << "x" << heightInt << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Capture] Failed to reallocate GPU buffer: " << e.what() << std::endl;
                useGPUDirect = false;
            }
        }

        if (useGPUDirect && bufferData) {
            cudaError_t err = cudaMemcpy2DFromArrayAsync(
                bufferData,
                bufferStep,
                cudaArray,
                0,
                0,
                width * 4,  // 4 bytes per pixel (BGRA)
                height,
                cudaMemcpyDeviceToDevice,
                copyStream
            );

            if (err != cudaSuccess) {
                std::cerr << "[Capture] GPU-direct copy failed: " << cudaGetErrorString(err) << std::endl;
                useGPUDirect = false;  // Fall back to CPU path
            }
        }
    }

    // CPU fallback path
    if (!useGPUDirect) {
        void* frameData = nullptr;
        unsigned int size = 0;

        if (!m_capture->GetLatestFrame(&frameData, &width, &height, &size)) {
            return FrameFailureReason::CAPTURE_FAILED;
        }

        if (!frameData || width == 0 || height == 0) {
            return FrameFailureReason::CAPTURE_FAILED;
        }

        ensureCaptureBufferRegistered(frameData, size);

        if (!copyFrameToBuffer(frameData, width, height, targetBuffer, copyStream)) {
            return FrameFailureReason::CAPTURE_FAILED;
        }
    }

    if (!m_captureReadyEvent) {
        m_captureReadyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    }

    if (!m_captureReadyEvent || !m_captureReadyEvent->get()) {
        return FrameFailureReason::CAPTURE_FAILED;
    }

    cudaError_t recordErr = cudaEventRecord(m_captureReadyEvent->get(), copyStream);
    if (recordErr != cudaSuccess) {
        std::cerr << "[Capture] Failed to record capture completion event: "
                  << cudaGetErrorString(recordErr) << std::endl;
        return FrameFailureReason::CAPTURE_FAILED;
    }

    // Capture is now in flight into ring[writeIdx]
    m_captureState.store(CaptureState::CAPTURING, std::memory_order_release);
    m_lastCaptureW = static_cast<int>(width);
    m_lastCaptureH = static_cast<int>(height);
    m_lastGpuDirect = useGPUDirect;
    m_capturePendingIdx = writeIdx;
    // Mark shape dirty if upcoming frame dims differ from stable buffer
    if (m_stableCaptureRows != static_cast<int>(height) ||
        m_stableCaptureCols != static_cast<int>(width) ||
        m_stableCaptureChannels != 4) {
        m_captureBufferShapeDirty = true;
    }
    // Advance write index to keep a free slot available
    m_captureWriteIdx = (writeIdx + 1) % kCaptureRingSize;

    if (forceSync) {
        bool success = waitForCaptureCompletion();
        return success ? FrameFailureReason::NONE : FrameFailureReason::NO_FRAME_READY;
    }

    return FrameFailureReason::NONE;
}

FrameFailureReason UnifiedGraphPipeline::ensureFrameReady() {
    if (!m_config.enableCapture) {
        return FrameFailureReason::NONE;
    }

    CaptureState state = m_captureState.load(std::memory_order_acquire);
    
    // If capturing, wait for completion
    if (state == CaptureState::CAPTURING) {
        if (!waitForCaptureCompletion()) {
            return FrameFailureReason::NO_FRAME_READY;
        }
        state = m_captureState.load(std::memory_order_acquire);
    }

    // If no frame ready, schedule capture
    if (state != CaptureState::READY) {
        return scheduleNextFrameCapture(true);
    }

    return FrameFailureReason::NONE;
}

bool UnifiedGraphPipeline::performFrameCapture() {
    if (!m_config.enableCapture) {
        return true;
    }
    
    CaptureState state = m_captureState.load(std::memory_order_acquire);
    if (state == CaptureState::READY) {
        return true; // Already have a frame
    }

    FrameFailureReason reason = scheduleNextFrameCapture(true);
    return (reason == FrameFailureReason::NONE);
}

bool UnifiedGraphPipeline::performFrameCaptureDirectToUnified() {
    return performFrameCapture();
}

bool UnifiedGraphPipeline::performPreprocessing() {
    auto& ctx = AppContext::getInstance();
    
    // Preview buffer is now updated in performFrameCapture() instead
    
    // Select source from ring (non-graph path uses ring directly)
    if (!m_unifiedArena.yoloInput || m_captureReadIdx < 0) {
        return false;
    }
    const SimpleCudaMat& src = m_captureRing[m_captureReadIdx];
    if (src.empty()) {
        return false;
    }
    
    int modelRes = getModelInputResolution();
    cudaError_t err = cuda_unified_preprocessing(
        src.data(),
        m_unifiedArena.yoloInput,
        src.cols(),
        src.rows(),
        static_cast<int>(src.step()),
        modelRes,
        modelRes,
        m_pipelineStream->get()
    );
    
    if (err != cudaSuccess) {
        printf("[ERROR] Unified preprocessing failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

void UnifiedGraphPipeline::updatePreviewBufferAllocation() {
    auto& ctx = AppContext::getInstance();

    // Lazy allocation/deallocation - keep buffers allocated for fast re-enable
    if (ctx.config.show_window && !m_preview.enabled) {
        int width = ctx.config.detection_resolution;
        int height = ctx.config.detection_resolution;

        // Reuse existing buffers if they match dimensions
        bool needRealloc = m_preview.previewBuffer.empty() ||
                          m_preview.previewBuffer.rows() != height ||
                          m_preview.previewBuffer.cols() != width;

        if (needRealloc) {
            // Release old buffers first if they exist
            if (m_preview.hostPreviewPinned && m_preview.hostPreview.data()) {
                cudaHostUnregister(m_preview.hostPreview.data());
                m_preview.hostPreviewPinned = false;
            }
            m_preview.previewBuffer.release();
            m_preview.hostPreview.release();

            // Allocate new buffers
            m_preview.previewBuffer.create(height, width, 4);
            m_preview.hostPreview.create(height, width, 4);

            // Pin host preview buffer for faster async copies
            size_t bytes = static_cast<size_t>(m_preview.hostPreview.step()) * m_preview.hostPreview.rows();
            if (bytes > 0) {
                if (cudaHostRegister(m_preview.hostPreview.data(), bytes, cudaHostRegisterPortable) == cudaSuccess) {
                    m_preview.hostPreviewPinned = true;
                    m_preview.hostPreviewPinnedSize = bytes;
                } else {
                    m_preview.hostPreviewPinned = false;
                    m_preview.hostPreviewPinnedSize = 0;
                }
            }
        }

        m_preview.hasValidHostPreview = false;
        m_preview.finalTargets.reserve(ctx.config.max_detections);
        m_preview.enabled = true;
        m_preview.lastCopyTime = {};
    } else if (!ctx.config.show_window && m_preview.enabled) {
        // Lazy deallocation - just disable without releasing memory
        // Buffers stay allocated for instant re-enable
        m_preview.enabled = false;
        m_preview.hasValidHostPreview = false;
        // NOTE: Actual deallocation happens only in shutdown() to avoid
        // allocation churn when user rapidly toggles preview window
    }
}

void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    auto& ctx = AppContext::getInstance();

    std::lock_guard<std::mutex> lock(m_previewMutex);

    // First ensure preview buffer allocation is correct
    updatePreviewBufferAllocation();

    // m_preview.enabled reflects show_window state; avoid redundant check
    if (!m_preview.enabled || m_preview.previewBuffer.empty()) {
        return;
    }

    if (currentBuffer.channels() != 4) {
        std::cerr << "[Preview] Unexpected channel count for capture buffer: "
                  << currentBuffer.channels() << std::endl;
        return;
    }

    if (currentBuffer.empty() || !currentBuffer.data()) {
        return;
    }

    if (m_preview.previewBuffer.rows() != currentBuffer.rows() ||
        m_preview.previewBuffer.cols() != currentBuffer.cols() ||
        m_preview.previewBuffer.channels() != currentBuffer.channels()) {
        // If host preview was pinned, unregister before reallocating
        if (m_preview.hostPreviewPinned && m_preview.hostPreview.data()) {
            cudaHostUnregister(m_preview.hostPreview.data());
            m_preview.hostPreviewPinned = false;
            m_preview.hostPreviewPinnedSize = 0;
        }
        m_preview.previewBuffer.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
        m_preview.hostPreview.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
        // Re-pin host preview buffer after reallocation
        size_t bytes = static_cast<size_t>(m_preview.hostPreview.step()) * m_preview.hostPreview.rows();
        if (bytes > 0 && m_preview.enabled) {
            if (cudaHostRegister(m_preview.hostPreview.data(), bytes, cudaHostRegisterPortable) == cudaSuccess) {
                m_preview.hostPreviewPinned = true;
                m_preview.hostPreviewPinnedSize = bytes;
            }
        }
    }

    if (m_preview.previewBuffer.empty() || !m_preview.previewBuffer.data()) {
        return;
    }

    if (!m_pipelineStream || !m_pipelineStream->get()) {
        return;
    }

    size_t srcStep = currentBuffer.step();
    size_t dstStep = m_preview.previewBuffer.step();

    if (srcStep > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        dstStep > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "[Preview] Capture pitch exceeds supported range" << std::endl;
        return;
    }

    cudaError_t convertErr = cuda_bgra2rgba(
        currentBuffer.data(),
        m_preview.previewBuffer.data(),
        currentBuffer.cols(),
        currentBuffer.rows(),
        static_cast<int>(srcStep),
        static_cast<int>(dstStep),
        (m_previewStream && m_previewStream->get()) ? m_previewStream->get() : m_pipelineStream->get());

    if (convertErr != cudaSuccess) {
        std::cerr << "[Preview] Failed to convert capture buffer for preview: "
                  << cudaGetErrorString(convertErr) << std::endl;
        return;
    }
}

bool UnifiedGraphPipeline::getPreviewSnapshot(SimpleMat& outFrame) {
    auto& ctx = AppContext::getInstance();

    if (!m_preview.enabled) {
        return false;
    }

    if (!m_pipelineStream || !m_pipelineStream->get()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_previewMutex);

    if (m_preview.previewBuffer.empty() || m_preview.previewBuffer.data() == nullptr) {
        return false;
    }

    if (m_preview.hostPreview.empty() ||
        m_preview.hostPreview.rows() != m_preview.previewBuffer.rows() ||
        m_preview.hostPreview.cols() != m_preview.previewBuffer.cols() ||
        m_preview.hostPreview.channels() != m_preview.previewBuffer.channels()) {
        m_preview.hostPreview.create(m_preview.previewBuffer.rows(),
                                     m_preview.previewBuffer.cols(),
                                     m_preview.previewBuffer.channels());
        m_preview.hasValidHostPreview = false;
    }

    if (m_preview.copyInProgress) {
        if (m_previewReadyEvent && m_previewReadyEvent->get()) {
            cudaError_t queryStatus = cudaEventQuery(m_previewReadyEvent->get());
            if (queryStatus == cudaSuccess) {
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = true;
            } else if (queryStatus != cudaErrorNotReady) {
                std::cerr << "[Preview] Event query failed: " << cudaGetErrorString(queryStatus) << std::endl;
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = false;
            }
        } else if (m_pipelineStream && m_pipelineStream->get()) {
            cudaError_t queryStatus = cudaStreamQuery(m_pipelineStream->get());
            if (queryStatus == cudaSuccess) {
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = true;
            } else if (queryStatus != cudaErrorNotReady) {
                std::cerr << "[Preview] Stream query failed: " << cudaGetErrorString(queryStatus) << std::endl;
                m_preview.copyInProgress = false;
                m_preview.hasValidHostPreview = false;
            }
        }
        if (m_preview.copyInProgress) {
            if (m_preview.hasValidHostPreview) {
                outFrame = m_preview.hostPreview;
                return true;
            }
            return false;
        }
    }

    bool hasFrameToReturn = false;
    if (m_preview.hasValidHostPreview && m_preview.hostPreview.data()) {
        outFrame = m_preview.hostPreview;
        hasFrameToReturn = true;
    }

    // Throttle preview copies to ~30 FPS to reduce overhead
    auto now = std::chrono::steady_clock::now();
    if (m_preview.lastCopyTime.time_since_epoch().count() != 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_preview.lastCopyTime).count();
        if (elapsed < 33) {
            return hasFrameToReturn;
        }
    }

    size_t rowBytes = static_cast<size_t>(m_preview.previewBuffer.cols()) * m_preview.previewBuffer.channels();
    cudaStream_t pstream = (m_previewStream && m_previewStream->get()) ? m_previewStream->get() : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!pstream) {
        return hasFrameToReturn;
    }
    cudaError_t copyErr = cudaMemcpy2DAsync(
        m_preview.hostPreview.data(),
        m_preview.hostPreview.step(),
        m_preview.previewBuffer.data(),
        m_preview.previewBuffer.step(),
        rowBytes,
        m_preview.previewBuffer.rows(),
        cudaMemcpyDeviceToHost,
        pstream);
    if (copyErr != cudaSuccess) {
        std::cerr << "[Preview] Failed to copy preview to host: " << cudaGetErrorString(copyErr) << std::endl;
        m_preview.hasValidHostPreview = false;
        return hasFrameToReturn;
    }

    if (m_previewReadyEvent && m_previewReadyEvent->get()) {
        m_previewReadyEvent->record(pstream);
    }

    m_preview.copyInProgress = true;
    m_preview.lastCopyTime = now;
    return hasFrameToReturn;
}

bool UnifiedGraphPipeline::performInference() {
    if (m_primaryInputIndex < 0 ||
        m_primaryInputIndex >= static_cast<int>(m_inputAddressCache.size()) ||
        !m_unifiedArena.yoloInput) {
        return false;
    }

    void* inputBinding = m_inputAddressCache[m_primaryInputIndex];
    if (!inputBinding) {
        return false;
    }
    
    // Aliasing should be set during initialization - no runtime retry
    // If not aliased, copy is required (logged once during init)
    if (inputBinding != m_unifiedArena.yoloInput) {
        size_t inputSize = getModelInputResolution() * getModelInputResolution() * 3 * sizeof(float);
        cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                       cudaMemcpyDeviceToDevice, m_pipelineStream->get());
    } else {
        // Successfully aliased - skip memcpy
        m_perfMetrics.memcpySkipCount++;
    }
    
    if (!runInferenceAsync(m_pipelineStream->get())) {
        std::cerr << "[UnifiedGraph] TensorRT inference failed" << std::endl;
        return false;
    }

    return true;
}

int UnifiedGraphPipeline::findHeadClassId(AppContext& ctx) {
    const size_t headNameHash = std::hash<std::string>{}(ctx.config.head_class_name);
    const size_t classSettingsSize = ctx.config.class_settings.size();

    const size_t cachedSize = m_cachedClassSettingsSize.load(std::memory_order_acquire);
    const size_t cachedHash = m_cachedHeadClassNameHash.load(std::memory_order_acquire);

    if (classSettingsSize != cachedSize || headNameHash != cachedHash) {
        int resolvedId = -1;
        {
            std::lock_guard<std::mutex> lock(ctx.configMutex);
            for (const auto& cs : ctx.config.class_settings) {
                if (cs.name == ctx.config.head_class_name) {
                    resolvedId = cs.id;
                    break;
                }
            }
        }

        m_cachedHeadClassId.store(resolvedId, std::memory_order_release);
        m_cachedHeadClassNameHash.store(headNameHash, std::memory_order_release);
        m_cachedClassSettingsSize.store(classSettingsSize, std::memory_order_release);
        m_classFilterDirty.store(true, std::memory_order_release);
        return resolvedId;
    }

    return m_cachedHeadClassId.load(std::memory_order_acquire);
}

// performResultCopy?????댁긽 ?꾩슂 ?놁쓬 - Graph? 肄쒕갚?먯꽌 吏곸젒 泥섎━

bool UnifiedGraphPipeline::executeFrame(FrameFailureReason* outReason, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    // Latest-only semantic: skip if previous frame still in flight
    bool expected = false;
    if (!m_frameInFlight.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        // Frame already in flight, skip this one (not an error)
        if (outReason) *outReason = FrameFailureReason::GPU_BUSY;
        return false;
    }

    // Performance tracking with new metrics
    auto frameStart = std::chrono::steady_clock::now();

    if (ctx.config.use_cuda_graph && !m_state.graphReady) {
        if (!captureGraph(stream)) {
            m_frameInFlight.store(false, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(m_inflightMutex);
            }
            m_inflightCv.notify_all();
            if (outReason) *outReason = FrameFailureReason::GRAPH_NOT_READY;
            return false;
        }
    }

    // Check if graph needs rebuild (e.g., config changed)
    if (m_state.graphReady && m_state.needsRebuild && ctx.config.use_cuda_graph) {
        // Try fast update first, falls back to full recapture if needed
        if (!updateGraphExec()) {
            std::cerr << "[UnifiedGraph] Graph update failed, attempting full recapture" << std::endl;
            if (!captureGraph(stream)) {
                std::cerr << "[UnifiedGraph] Graph recapture failed" << std::endl;
                m_state.graphReady = false;
            }
        }
    }

    cudaStream_t launchStream = stream ? stream : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!launchStream) {
        m_frameInFlight.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> g(m_inflightMutex);
        }
        m_inflightCv.notify_all();
        return false;
    }

    if (ctx.config.use_cuda_graph && m_state.graphReady && m_graphExec) {
        // Prime buffers on first entry
        if (!m_graphPrimed) {
            FrameFailureReason primeReason = scheduleNextFrameCapture(false);
            if (primeReason != FrameFailureReason::NONE) {
                m_allowMovement.store(false, std::memory_order_release);
                m_frameInFlight.store(false, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> g(m_inflightMutex);
                }
                m_inflightCv.notify_all();
                if (outReason) *outReason = primeReason;
                return false;
            }
            // Single synchronization point - waitForCaptureCompletion handles event sync internally
            if (!waitForCaptureCompletion()) {
                m_allowMovement.store(false, std::memory_order_release);
                m_frameInFlight.store(false, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> g(m_inflightMutex);
                }
                m_inflightCv.notify_all();
                return false;
            }
            // Primed: m_captureBuffer now contains valid frame
            m_graphPrimed = true;
        }

        // Schedule next capture if nothing in flight
        (void)scheduleNextFrameCapture(false);

        // If a fresh frame is pending/ready in the ring, wait on event and copy it into the stable graph input buffer
        if (m_captureReadyEvent && m_captureReadyEvent->get()) {
            cudaStreamWaitEvent(launchStream, m_captureReadyEvent->get(), 0);

            int srcIdx = (m_capturePendingIdx >= 0) ? m_capturePendingIdx : m_captureReadIdx;
            if (srcIdx >= 0 && srcIdx < kCaptureRingSize) {
                const SimpleCudaMat& srcBuf = m_captureRing[srcIdx];

                // Ensure destination allocation matches only when shape flagged dirty
                if (m_captureBufferShapeDirty) {
                    m_captureBuffer.create(srcBuf.rows(), srcBuf.cols(), srcBuf.channels());
                    m_stableCaptureRows = srcBuf.rows();
                    m_stableCaptureCols = srcBuf.cols();
                    m_stableCaptureChannels = srcBuf.channels();
                    m_captureBufferShapeDirty = false;
                }

                size_t rowBytes = static_cast<size_t>(srcBuf.cols()) * srcBuf.channels();
                cudaMemcpy2DAsync(
                    m_captureBuffer.data(), m_captureBuffer.step(),
                    srcBuf.data(), srcBuf.step(),
                    rowBytes, srcBuf.rows(),
                    cudaMemcpyDeviceToDevice, launchStream);

                if (m_preview.enabled && !m_captureBuffer.empty()) {
                    updatePreviewBuffer(m_captureBuffer);
                }
            }
        }

        bool shouldDispatchMovement = m_config.enableDetection && !ctx.detection_paused.load();
        m_allowMovement.store(shouldDispatchMovement, std::memory_order_release);

        // Launch the graph using the stabilized m_captureBuffer pointer
        cudaError_t launchErr = cudaGraphLaunch(m_graphExec, launchStream);
        if (launchErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Graph launch failed: "
                      << cudaGetErrorString(launchErr) << std::endl;
            m_allowMovement.store(false, std::memory_order_release);
            m_frameInFlight.store(false, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(m_inflightMutex);
            }
            m_inflightCv.notify_all();
            return false;
        }

        // Mark frame as consumed; the next capture will update the ring
        m_captureState.store(CaptureState::CONSUMED, std::memory_order_release);
    } else {
        if (!executeNormalPipeline(launchStream)) {
            m_allowMovement.store(false, std::memory_order_release);
            m_frameInFlight.store(false, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(m_inflightMutex);
            }
            m_inflightCv.notify_all();
            if (outReason) *outReason = FrameFailureReason::NO_FRAME_READY;
            return false;
        }
    }

    // Update performance metrics
    auto frameEnd = std::chrono::steady_clock::now();
    auto latencyMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
    
    m_perfMetrics.totalFrames++;
    m_perfMetrics.totalLatencyMs += latencyMs;
    m_perfMetrics.logIfNeeded("[Perf]");
    
    if (outReason) *outReason = FrameFailureReason::NONE;

    // Periodic CUDA memory maintenance for long-running sessions
    static auto lastTrim = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::minutes>(now - lastTrim).count() >= 10) {
        cudaMemPoolTrimTo(nullptr, 0);
        // Trim graph memory pool for the current device (0 used elsewhere in code)
        cudaDeviceGraphMemTrim(0);
        lastTrim = now;
    }

    return true;
}

bool UnifiedGraphPipeline::executeNormalPipeline(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    cudaStream_t activeStream = stream ? stream : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!activeStream) {
        return false;
    }

    FrameFailureReason ensureReason = ensureFrameReady();
    if (ensureReason != FrameFailureReason::NONE) {
        return false; // Frame not ready
    }

    bool shouldRunDetection = m_config.enableDetection && !ctx.detection_paused.load();

    if (shouldRunDetection) {
        // Ensure any pending capture copy is visible to the pipeline stream
        if (m_captureReadyEvent && m_captureReadyEvent->get()) {
            cudaStreamWaitEvent(activeStream, m_captureReadyEvent->get(), 0);
        }
        if (!performPreprocessing()) {
            return false;
        }

        (void)scheduleNextFrameCapture(false); // Non-blocking schedule

        if (!performInference()) {
            return false;
        }

        performIntegratedPostProcessing(activeStream);
        performTargetSelection(activeStream);

        if (!m_mouseMovementUsesMappedMemory) {
            cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                           sizeof(MouseMovement), cudaMemcpyDeviceToHost, activeStream);
        }
    }

    m_allowMovement.store(shouldRunDetection, std::memory_order_release);
    if (shouldRunDetection) {
        if (!enqueueFrameCompletionCallback(activeStream)) {
            m_allowMovement.store(false, std::memory_order_release);
            return false;
        }
    } else {
        (void)scheduleNextFrameCapture(false); // Non-blocking schedule

        cudaError_t streamState = cudaStreamQuery(activeStream);
        if (streamState == cudaSuccess) {
            clearMovementData();
            m_frameInFlight.store(false, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(m_inflightMutex);
            }
            m_inflightCv.notify_all();
        } else {
            if (streamState != cudaErrorNotReady) {
                std::cerr << "[UnifiedGraph] Stream query failed while resetting movement: "
                          << cudaGetErrorString(streamState) << std::endl;
            }

            if (!enqueueMovementResetCallback(activeStream)) {
                return false;
            }
        }
    }

    // Mark frame as consumed
    m_captureState.store(CaptureState::CONSUMED, std::memory_order_release);

    return true;
}


// processMouseMovement???댁젣 Graph? 肄쒕갚 ?대??먯꽌 ?몃씪?몄쑝濡?泥섎━??

}
void needaimbot::UnifiedGraphPipeline::getCaptureStats(needaimbot::UnifiedGraphPipeline::CaptureStats& out) const {
    auto& ctx = AppContext::getInstance();
    out.lastWidth = m_lastCaptureW;
    out.lastHeight = m_lastCaptureH;
    out.roiLeft = m_captureRegionCache.left;
    out.roiTop = m_captureRegionCache.top;
    out.roiSize = m_captureRegionCache.size;
    out.gpuDirect = m_lastGpuDirect;
    out.hasFrame = (m_captureState.load(std::memory_order_acquire) == CaptureState::READY);
    out.previewEnabled = m_preview.enabled;
    out.previewHasHost = m_preview.hasValidHostPreview;
    out.backend = ctx.config.capture_method.c_str();
}
