#include "unified_graph_pipeline.h"
#include "detection/cuda_float_processing.h"
#include "simple_cuda_mat.h"
#include "../AppContext.h"
#include "../capture/dda_capture.h"
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

// Forward declare the mouse control function
extern "C" {
    void executeMouseMovement(int dx, int dy);
}

namespace needaimbot {

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


__global__ void fusedTargetSelectionAndMovementKernel(
    Target* __restrict__ finalTargets,
    int* __restrict__ finalTargetsCount,
    int maxDetections,
    float screen_center_x,
    float screen_center_y,
    int head_class_id,
    float kp_x,
    float kp_y,
    float head_y_offset,
    float body_y_offset,
    int detection_resolution,
    int* __restrict__ bestTargetIndex,
    Target* __restrict__ bestTarget,
    float* __restrict__ previousErrorX,
    float* __restrict__ previousErrorY,
    needaimbot::MouseMovement* __restrict__ output_movement
) {
    extern __shared__ unsigned char sharedMem[];
    float* s_distancesX = reinterpret_cast<float*>(sharedMem);
    int* s_indices = reinterpret_cast<int*>(s_distancesX + blockDim.x);

    if (threadIdx.x == 0) {
        *bestTargetIndex = -1;
        output_movement->dx = 0;
        output_movement->dy = 0;

        Target emptyTarget = {};
        *bestTarget = emptyTarget;
    }
    __syncthreads();

    int count = *finalTargetsCount;
    if (count <= 0 || count > maxDetections) {
        return;
    }

    int localBestIdx = -1;
    float localBestDistX = 1e9f;  // X축 거리만 사용

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        Target& t = finalTargets[i];

        if (t.x < -1000.0f || t.x > 10000.0f ||
            t.y < -1000.0f || t.y > 10000.0f ||
            t.width <= 0 || t.width > detection_resolution ||
            t.height <= 0 || t.height > detection_resolution ||
            t.confidence <= 0 || t.confidence > 1.0f) {
            t.confidence = 0;
            continue;
        }

        float centerX = t.x + t.width / 2.0f;
        float dx = fabsf(centerX - screen_center_x);  // X축 거리만 계산 (절댓값)

        if (dx < localBestDistX) {
            localBestDistX = dx;
            localBestIdx = i;
        }
    }

    s_distancesX[threadIdx.x] = localBestDistX;
    s_indices[threadIdx.x] = localBestIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_distancesX[threadIdx.x + s] < s_distancesX[threadIdx.x]) {
                s_distancesX[threadIdx.x] = s_distancesX[threadIdx.x + s];
                s_indices[threadIdx.x] = s_indices[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (s_indices[0] >= 0) {
            *bestTargetIndex = s_indices[0];
            *bestTarget = finalTargets[s_indices[0]];

            Target& best = *bestTarget;
            float target_center_x = best.x + best.width / 2.0f;
            float target_center_y;

            if (best.classId == head_class_id) {
                target_center_y = best.y + best.height * head_y_offset;
            } else {
                target_center_y = best.y + best.height * body_y_offset;
            }

            float error_x = target_center_x - screen_center_x;
            float error_y = target_center_y - screen_center_y;

            float movement_x = kp_x * error_x;
            float movement_y = kp_y * error_y;

            if (previousErrorX) {
                previousErrorX[0] = error_x;
            }
            if (previousErrorY) {
                previousErrorY[0] = error_y;
            }

            output_movement->dx = __float2int_rn(movement_x);
            output_movement->dy = __float2int_rn(movement_y);
        }
    }
}


UnifiedGraphPipeline::UnifiedGraphPipeline() {
    constexpr unsigned int kBlockingEventFlags = cudaEventDisableTiming | cudaEventBlockingSync;
    m_state.startEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    m_state.endEvent = std::make_unique<CudaEvent>(kBlockingEventFlags);
    resetMovementFilter();
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
    
    m_previewReadyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    
    if (m_config.modelPath.empty()) {
        std::cerr << "[UnifiedGraph] ERROR: Model path is required for TensorRT integration" << std::endl;
        return false;
    }
    
    if (!initializeTensorRT(m_config.modelPath)) {
        std::cerr << "[UnifiedGraph] CRITICAL: TensorRT initialization failed" << std::endl;
        return false;
    }
    
    if (!allocateBuffers()) {
        std::cerr << "[UnifiedGraph] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    if (m_config.useGraphOptimization) {
        
        for (int i = 0; i < 3; i++) {
            if (m_inputBindings.find(m_inputName) != m_inputBindings.end()) {
                auto inputBuffer = m_inputBindings[m_inputName].get();
                cudaMemsetAsync(inputBuffer->get(), 0, inputBuffer->size() * sizeof(uint8_t), m_pipelineStream->get());
            }
            
            for (const auto& [name, buffer] : m_inputBindings) {
                if (buffer && buffer->get()) {
                    m_context->setTensorAddress(name.c_str(), buffer->get());
                }
            }
            for (const auto& [name, buffer] : m_outputBindings) {
                if (buffer && buffer->get()) {
                    m_context->setTensorAddress(name.c_str(), buffer->get());
                }
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
    
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin capture: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Graph 내부에서는 복사 불필요 - executeFrame에서 직접 통합 버퍼로 캡처함
    // 캡처는 Graph 외부에서 performFrameCaptureDirectToUnified()로 처리
    
    // 전처리도 Graph에 포함
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
        
        // 전처리된 데이터를 TensorRT 입력 버퍼로 복사
        if (m_inputBindings.find(m_inputName) != m_inputBindings.end()) {
            void* inputBinding = m_inputBindings[m_inputName]->get();
            if (inputBinding != m_unifiedArena.yoloInput) {
                size_t inputSize = modelRes * modelRes * 3 * sizeof(float);
                cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize,
                               cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
    
    // TensorRT 추론 포함 (Graph 호환 모델만 사용)
    if (m_context && m_config.enableDetection) {
        for (const auto& [name, buffer] : m_inputBindings) {
            if (buffer && buffer->get()) {
                m_context->setTensorAddress(name.c_str(), buffer->get());
            }
        }
        for (const auto& [name, buffer] : m_outputBindings) {
            if (buffer && buffer->get()) {
                m_context->setTensorAddress(name.c_str(), buffer->get());
            }
        }
        
        if (!m_context->enqueueV3(stream)) {
            std::cerr << "Warning: TensorRT enqueue failed during graph capture" << std::endl;
        }
    }
    
    if (m_config.enableDetection) {
        performIntegratedPostProcessing(stream);
        performTargetSelection(stream);
        
        // 결과를 호스트로 복사 (Graph 내부)
        if (!m_mouseMovementUsesMappedMemory) {
            cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                           sizeof(MouseMovement), cudaMemcpyDeviceToHost, stream);
        }
        
        // 마우스 이동 콜백을 Graph에 포함 - 복사 완료 후 자동 실행
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
    m_graphCaptured = true;  // Graph 캡처 완료 플래그 설정
    
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

        if (m_smallBufferArena.previousErrorX && m_smallBufferArena.previousErrorY) {
            cudaError_t zeroErr = cudaMemset(m_smallBufferArena.previousErrorX, 0, sizeof(float));
            if (zeroErr != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Failed to zero previousErrorX: "
                          << cudaGetErrorString(zeroErr) << std::endl;
                return false;
            }
            zeroErr = cudaMemset(m_smallBufferArena.previousErrorY, 0, sizeof(float));
            if (zeroErr != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Failed to zero previousErrorY: "
                          << cudaGetErrorString(zeroErr) << std::endl;
                return false;
            }
        }

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

        if (m_unifiedArena.yoloInput) {
            auto bindingIt = m_inputBindings.find(m_inputName);
            auto sizeIt = m_inputSizes.find(m_inputName);
            if (bindingIt != m_inputBindings.end() && sizeIt != m_inputSizes.end()) {
                size_t inputSize = sizeIt->second;
                auto arenaAlias = std::make_unique<CudaMemory<uint8_t>>(
                    reinterpret_cast<uint8_t*>(m_unifiedArena.yoloInput),
                    inputSize,
                    false);
                bindingIt->second = std::move(arenaAlias);
            }
        }

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
    m_d_preprocessBuffer.reset();
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
    
    if (m_captureBuffer.empty() || 
        m_captureBuffer.rows() != frame.rows() || 
        m_captureBuffer.cols() != frame.cols() || 
        m_captureBuffer.channels() != frame.channels()) {
        m_captureBuffer.create(frame.rows(), frame.cols(), frame.channels());
    }
    
    size_t dataSize = frame.rows() * frame.cols() * frame.channels() * sizeof(unsigned char);
    cudaError_t err = cudaMemcpyAsync(m_captureBuffer.data(), frame.data(), dataSize, 
                                      cudaMemcpyDeviceToDevice, m_pipelineStream->get());
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to copy frame to buffer: %s\n", cudaGetErrorString(err));
        return;
    }
    
    m_hasFrameData = true;
}


void UnifiedGraphPipeline::handleAimbotDeactivation() {
    auto& ctx = AppContext::getInstance();

    clearCountBuffers();
    clearMovementData();
    clearHostPreviewData(ctx);
    m_allowMovement.store(false, std::memory_order_release);
    m_captureRegionCache = {};
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
}

void UnifiedGraphPipeline::clearMovementData() {
    if (m_h_movement && m_h_movement->get()) {
        m_h_movement->get()->dx = 0;
        m_h_movement->get()->dy = 0;
    }
    resetMovementFilter();
}

void UnifiedGraphPipeline::resetMovementFilter() {
    std::lock_guard<std::mutex> lock(m_movementFilterMutex);

    m_lastFilteredMovement = {0, 0};
    m_lastMovementTimestamp = std::chrono::steady_clock::now();
    m_hasFilteredMovement = true;
}

MouseMovement UnifiedGraphPipeline::filterMouseMovement(const MouseMovement& rawMovement, bool movementEnabled) {
    std::lock_guard<std::mutex> lock(m_movementFilterMutex);

    auto now = std::chrono::steady_clock::now();

    if (!movementEnabled) {
        m_lastFilteredMovement = {0, 0};
        m_lastMovementTimestamp = now;
        m_hasFilteredMovement = true;
        return m_lastFilteredMovement;
    }

    MouseMovement filtered = rawMovement;

    if (!m_hasFilteredMovement) {
        m_lastMovementTimestamp = now;
        m_lastFilteredMovement = {0, 0};
        m_hasFilteredMovement = true;
    }

    const float deltaSeconds = std::max(0.0f, std::chrono::duration<float>(now - m_lastMovementTimestamp).count());
    constexpr float kSmoothingRate = 6.5f;
    float alpha = std::clamp(deltaSeconds * kSmoothingRate, 0.12f, 1.0f);

    auto computeStepLimit = [&](float dt) {
        constexpr int kBaseStep = 8;
        constexpr float kStepPerSecond = 140.0f;
        int dynamic = static_cast<int>(std::lround(dt * kStepPerSecond));
        int maxStep = kBaseStep + dynamic;
        if (maxStep > 64) {
            maxStep = 64;
        }
        if (maxStep < kBaseStep) {
            maxStep = kBaseStep;
        }
        return maxStep;
    };

    const int maxStep = computeStepLimit(deltaSeconds);

    auto smoothAxis = [&](int previous, int desired) {
        float blended = static_cast<float>(previous) + alpha * (static_cast<float>(desired - previous));
        int proposed = static_cast<int>(std::lround(blended));
        int delta = proposed - previous;
        if (delta > maxStep) {
            delta = maxStep;
        } else if (delta < -maxStep) {
            delta = -maxStep;
        }
        return previous + delta;
    };

    filtered.dx = smoothAxis(m_lastFilteredMovement.dx, rawMovement.dx);
    filtered.dy = smoothAxis(m_lastFilteredMovement.dy, rawMovement.dy);

    if (std::abs(filtered.dx) <= 1 && rawMovement.dx == 0) {
        filtered.dx = 0;
    }
    if (std::abs(filtered.dy) <= 1 && rawMovement.dy == 0) {
        filtered.dy = 0;
    }

    m_lastFilteredMovement = filtered;
    m_lastMovementTimestamp = now;
    m_hasFilteredMovement = true;

    return filtered;
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
    resetMovementFilter();
    m_captureRegionCache = {};

    if (m_smallBufferArena.previousErrorX && m_smallBufferArena.previousErrorY) {
        cudaError_t resetErr = cudaMemset(m_smallBufferArena.previousErrorX, 0, sizeof(float));
        if (resetErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Failed to reset previousErrorX: "
                      << cudaGetErrorString(resetErr) << std::endl;
        }
        resetErr = cudaMemset(m_smallBufferArena.previousErrorY, 0, sizeof(float));
        if (resetErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Failed to reset previousErrorY: "
                      << cudaGetErrorString(resetErr) << std::endl;
        }
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
                    }
                } else {
                    pipeline->filterMouseMovement({0, 0}, false);
                    pipeline->m_h_movement->get()->dx = 0;
                    pipeline->m_h_movement->get()->dy = 0;
                }
            }
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

        while (ctx.aiming.load() && !m_shouldStop.load(std::memory_order_acquire) &&
               !ctx.should_exit.load()) {
            if (!executeFrame()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    
    try {
        getBindings();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Failed to allocate TensorRT bindings: " << e.what() << std::endl;
        return false;
    }
    
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
    auto& ctx = AppContext::getInstance();
    
    
    
    std::unordered_map<std::string, void*> reusableInputs;
    std::unordered_map<std::string, void*> reusableOutputs;
    
    for (const auto& binding : m_inputBindings) {
        if (binding.second && binding.second->get()) {
            reusableInputs[binding.first] = binding.second->get();
        }
    }
    for (const auto& binding : m_outputBindings) {
        if (binding.second && binding.second->get()) {
            reusableOutputs[binding.first] = binding.second->get();
        }
    }
    
    m_inputBindings.clear();
    m_outputBindings.clear();

    for (const auto& name : m_inputNames) {
        size_t size = m_inputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for input '" << name << "'" << std::endl;
            continue;
        }
        
        try {
            // 일단 원래대로 복구 - 직접 포인터 사용은 TensorRT 바인딩과 호환성 문제 있음
            m_inputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
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
    
    // Removed previous error check for performance
    
    for (const auto& inputName : m_inputNames) {
        auto bindingIt = m_inputBindings.find(inputName);
        if (bindingIt == m_inputBindings.end() || bindingIt->second == nullptr) {
            std::cerr << "[Pipeline] Input binding not found or null for: " << inputName << std::endl;
            return false;
        }
        
        
        if (!m_context->setTensorAddress(inputName.c_str(), bindingIt->second->get())) {
            std::cerr << "[Pipeline] Failed to set input tensor address for: " << inputName << std::endl;
            return false;
        }
    }
    
    for (const auto& outputName : m_outputNames) {
        auto bindingIt = m_outputBindings.find(outputName);
        if (bindingIt == m_outputBindings.end() || bindingIt->second == nullptr) {
            std::cerr << "[Pipeline] Output binding not found or null for: " << outputName << std::endl;
            return false;
        }
        
        void* tensorAddress = bindingIt->second->get();
        
        if (!m_context->setTensorAddress(outputName.c_str(), tensorAddress)) {
            std::cerr << "[Pipeline] Failed to set output tensor address for: " << outputName << std::endl;
            return false;
        }
    }
    
    // Removed memory error check for performance
    
    
    
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

    if (m_smallBufferArena.colorFilteredCount) {
        cudaMemsetAsync(m_smallBufferArena.colorFilteredCount, 0, sizeof(int), stream);
    }

    if (m_smallBufferArena.bestTargetIndex) {
        cudaMemsetAsync(m_smallBufferArena.bestTargetIndex, -1, sizeof(int), stream);
    }

    if (m_smallBufferArena.bestTarget) {
        cudaMemsetAsync(m_smallBufferArena.bestTarget, 0, sizeof(Target), stream);
    }
}

cudaError_t UnifiedGraphPipeline::decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType, 
                                                   const std::vector<int64_t>& shape, 
                                                   const PostProcessingConfig& config, cudaStream_t stream) {
    int maxDecodedTargets = config.max_detections;
    
    if (config.postprocess == "yolo_nms") {
        // NMS가 포함된 모델 - 이미 후처리된 출력
        // 출력 형식: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
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
    void* d_rawOutputPtr = m_outputBindings[primaryOutputName]->get();
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
    if (m_unifiedArena.finalTargets && m_smallBufferArena.finalTargetsCount &&
        m_unifiedArena.decodedTargets && m_smallBufferArena.decodedCount) {

        if (m_smallBufferArena.finalTargetsCount != m_smallBufferArena.decodedCount) {
            cudaMemcpyAsync(m_smallBufferArena.finalTargetsCount, m_smallBufferArena.decodedCount, sizeof(int),
                           cudaMemcpyDeviceToDevice, stream);
        }

        if (m_unifiedArena.finalTargets != m_unifiedArena.decodedTargets) {
            cudaMemcpyAsync(m_unifiedArena.finalTargets, m_unifiedArena.decodedTargets,
                           config.max_detections * sizeof(Target), cudaMemcpyDeviceToDevice, stream);
        }
    }

}

// Removed redundant NMS and copy functions
// Empty preview functions removed for performance


void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    if (!m_unifiedArena.finalTargets || !m_smallBufferArena.finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }
    
    if (!m_smallBufferArena.bestTargetIndex || !m_smallBufferArena.bestTarget || !m_smallBufferArena.mouseMovement) {
        std::cerr << "[Pipeline] Target selection buffers not allocated!" << std::endl;
        return;
    }
    
    static int cached_max_detections = Constants::MAX_DETECTIONS;
    static float cached_kp_x = 0.1f;
    static float cached_kp_y = 0.1f;
    static float cached_head_y_offset = 0.2f;
    static float cached_body_y_offset = 0.5f;

    if (!m_graphCaptured) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_max_detections = ctx.config.max_detections;
        cached_kp_x = ctx.config.pd_kp_x;
        cached_kp_y = ctx.config.pd_kp_y;
        cached_head_y_offset = ctx.config.head_y_offset;
        cached_body_y_offset = ctx.config.body_y_offset;
    }
    
    float crosshairX = ctx.config.detection_resolution / 2.0f;
    float crosshairY = ctx.config.detection_resolution / 2.0f;

    int head_class_id = findHeadClassId(ctx);

    const int blockSize = computeTargetSelectionBlockSize(cached_max_detections);
    const int gridSize = 1;
    const size_t sharedBytes = static_cast<size_t>(blockSize) * (sizeof(float) + sizeof(int));

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
        cached_head_y_offset,
        cached_body_y_offset,
        ctx.config.detection_resolution,
        m_smallBufferArena.bestTargetIndex,
        m_smallBufferArena.bestTarget,
        m_smallBufferArena.previousErrorX,
        m_smallBufferArena.previousErrorY,
        m_smallBufferArena.mouseMovement
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Fused kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

}

bool UnifiedGraphPipeline::updateDDACaptureRegion(const AppContext& ctx) {
    if (!m_ddaCapture) {
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

    int screenW = m_ddaCapture->GetScreenWidth();
    int screenH = m_ddaCapture->GetScreenHeight();
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

    if (!m_ddaCapture->SetCaptureRegion(left, top, captureSize, captureSize)) {
        return false;
    }

    m_captureRegionCache.detectionRes = detectionRes;
    m_captureRegionCache.offsetX = offsetX;
    m_captureRegionCache.offsetY = offsetY;
    m_captureRegionCache.usingAimShootOffset = useAimShootOffset;
    return true;
}

bool UnifiedGraphPipeline::copyDDAFrameToGPU(void* frameData, unsigned int width, unsigned int height) {
    if (!frameData || width == 0 || height == 0) {
        return false;
    }

    if (m_captureBuffer.empty() ||
        m_captureBuffer.cols() != static_cast<int>(width) ||
        m_captureBuffer.rows() != static_cast<int>(height) ||
        m_captureBuffer.channels() != 4) {
        m_captureBuffer.create(static_cast<int>(height), static_cast<int>(width), 4);
    }

    if (m_captureBuffer.empty() || !m_pipelineStream) {
        return false;
    }

    size_t totalBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    ensureCaptureBufferRegistered(frameData, totalBytes);

    size_t hostPitch = static_cast<size_t>(width) * 4;
    cudaError_t err = cudaMemcpy2DAsync(
        m_captureBuffer.data(),
        m_captureBuffer.step(),
        frameData,
        hostPitch,
        hostPitch,
        height,
        cudaMemcpyHostToDevice,
        m_pipelineStream->get()
    );

    if (err != cudaSuccess) {
        std::cerr << "[Capture] DDA frame copy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
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

    cudaError_t regErr = cudaHostRegister(frameData, size, cudaHostRegisterPortable);
    if (regErr == cudaSuccess || regErr == cudaErrorHostMemoryAlreadyRegistered) {
        bufferInfo.registered = true;
        bufferInfo.permanentFailure = false;
        m_registeredCaptureBuffers[frameData] = bufferInfo;
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

bool UnifiedGraphPipeline::performFrameCapture() {
    auto& ctx = AppContext::getInstance();

    if (!m_config.enableCapture || m_hasFrameData) {
        return true;
    }

    if (!m_ddaCapture) {
        std::cerr << "[Capture] DDA capture interface not set" << std::endl;
        return false;
    }

    if (!updateDDACaptureRegion(ctx)) {
        return false;
    }

    void* frameData = nullptr;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int size = 0;

    if (!m_ddaCapture->GetLatestFrame(&frameData, &width, &height, &size)) {
        return false;
    }

    if (!copyDDAFrameToGPU(frameData, width, height)) {
        return false;
    }

    if ((m_preview.enabled || ctx.config.show_window) && m_pipelineStream && m_pipelineStream->get()) {
        updatePreviewBuffer(m_captureBuffer);
    }

    m_hasFrameData = true;
    return true;
}

bool UnifiedGraphPipeline::performFrameCaptureDirectToUnified() {
    return performFrameCapture();
}

bool UnifiedGraphPipeline::performPreprocessing() {
    auto& ctx = AppContext::getInstance();
    
    // Preview buffer is now updated in performFrameCapture() instead
    
    if (!m_unifiedArena.yoloInput || m_captureBuffer.empty()) {
        return false;
    }
    
    int modelRes = getModelInputResolution();
    cudaError_t err = cuda_unified_preprocessing(
        m_captureBuffer.data(),
        m_unifiedArena.yoloInput,
        m_captureBuffer.cols(),
        m_captureBuffer.rows(),
        static_cast<int>(m_captureBuffer.step()),
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
    
    // Dynamically allocate/deallocate preview buffer based on show_window state
    if (ctx.config.show_window && !m_preview.enabled) {
        int width = ctx.config.detection_resolution;
        int height = ctx.config.detection_resolution;
        m_preview.previewBuffer.create(height, width, 4);
        m_preview.hostPreview.create(height, width, 4);
        m_preview.hasValidHostPreview = false;
        m_preview.finalTargets.reserve(ctx.config.max_detections);
        m_preview.enabled = true;
    } else if (!ctx.config.show_window && m_preview.enabled) {
        m_preview.previewBuffer.release();
        m_preview.hostPreview.release();
        m_preview.hasValidHostPreview = false;
        m_preview.finalTargets.clear();
        m_preview.enabled = false;
    }
}

void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    auto& ctx = AppContext::getInstance();

    std::lock_guard<std::mutex> lock(m_previewMutex);

    // First ensure preview buffer allocation is correct
    updatePreviewBufferAllocation();

    // Check both m_preview.enabled AND current show_window state
    if (!m_preview.enabled || !ctx.config.show_window || m_preview.previewBuffer.empty()) {
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
        m_preview.previewBuffer.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
        m_preview.hostPreview.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
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
        m_pipelineStream->get());

    if (convertErr != cudaSuccess) {
        std::cerr << "[Preview] Failed to convert capture buffer for preview: "
                  << cudaGetErrorString(convertErr) << std::endl;
        return;
    }
}

bool UnifiedGraphPipeline::getPreviewSnapshot(SimpleMat& outFrame) {
    auto& ctx = AppContext::getInstance();

    if (!m_preview.enabled || !ctx.config.show_window) {
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

    size_t rowBytes = static_cast<size_t>(m_preview.previewBuffer.cols()) * m_preview.previewBuffer.channels();
    cudaError_t copyErr = cudaMemcpy2DAsync(
        m_preview.hostPreview.data(),
        m_preview.hostPreview.step(),
        m_preview.previewBuffer.data(),
        m_preview.previewBuffer.step(),
        rowBytes,
        m_preview.previewBuffer.rows(),
        cudaMemcpyDeviceToHost,
        m_pipelineStream->get());
    if (copyErr != cudaSuccess) {
        std::cerr << "[Preview] Failed to copy preview to host: " << cudaGetErrorString(copyErr) << std::endl;
        m_preview.hasValidHostPreview = false;
        return hasFrameToReturn;
    }

    if (m_previewReadyEvent && m_previewReadyEvent->get()) {
        m_previewReadyEvent->record(m_pipelineStream->get());
    }

    m_preview.copyInProgress = true;
    return hasFrameToReturn;
}

bool UnifiedGraphPipeline::performInference() {
    if (m_inputBindings.find(m_inputName) == m_inputBindings.end() || !m_unifiedArena.yoloInput) {
        return false;
    }
    
    void* inputBinding = m_inputBindings[m_inputName]->get();
    
    // 복사 최적화: 동일한 포인터면 복사 생략
    if (inputBinding != m_unifiedArena.yoloInput) {
        size_t inputSize = getModelInputResolution() * getModelInputResolution() * 3 * sizeof(float);
        cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize, 
                       cudaMemcpyDeviceToDevice, m_pipelineStream->get());
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

// performResultCopy는 더 이상 필요 없음 - Graph와 콜백에서 직접 처리

bool UnifiedGraphPipeline::executeFrame(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    static bool graphInitialized = false;
    if (!graphInitialized && ctx.config.use_cuda_graph) {
        captureGraph(stream);
        graphInitialized = true;
    }

    cudaStream_t launchStream = stream ? stream : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!launchStream) {
        return false;
    }

    if (ctx.config.use_cuda_graph && m_state.graphReady && m_graphExec) {
        if (!performFrameCaptureDirectToUnified()) {
            m_allowMovement.store(false, std::memory_order_release);
            return false;
        }

        bool shouldDispatchMovement = m_config.enableDetection && !ctx.detection_paused.load();
        m_allowMovement.store(shouldDispatchMovement, std::memory_order_release);

        cudaError_t launchErr = cudaGraphLaunch(m_graphExec, launchStream);
        if (launchErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Graph launch failed: "
                      << cudaGetErrorString(launchErr) << std::endl;
            m_allowMovement.store(false, std::memory_order_release);
            return false;
        }

        if (m_preview.enabled && ctx.config.show_window && !m_captureBuffer.empty()) {
            updatePreviewBuffer(m_captureBuffer);
        }

        m_hasFrameData = false;
    } else {
        if (!executeNormalPipeline(launchStream)) {
            m_allowMovement.store(false, std::memory_order_release);
            return false;
        }
    }

    return true;
}

bool UnifiedGraphPipeline::executeNormalPipeline(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();

    cudaStream_t activeStream = stream ? stream : (m_pipelineStream ? m_pipelineStream->get() : nullptr);
    if (!activeStream) {
        return false;
    }

    if (!performFrameCapture()) {
        return false;
    }

    bool shouldRunDetection = m_config.enableDetection && !ctx.detection_paused.load();

    if (shouldRunDetection) {
        if (!performPreprocessing()) {
            return false;
        }

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
        cudaError_t streamState = cudaStreamQuery(activeStream);
        if (streamState == cudaSuccess) {
            clearMovementData();
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

    m_hasFrameData = false;

    return true;
}


// processMouseMovement는 이제 Graph와 콜백 내부에서 인라인으로 처리됨

}
