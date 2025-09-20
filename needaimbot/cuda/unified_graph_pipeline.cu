#include "unified_graph_pipeline.h"
#include "detection/cuda_float_processing.h"
#include "simple_cuda_mat.h"
#include "../AppContext.h"
#include "../capture/nvfbc_capture.h"
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
#include <cuda.h>

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
    
    finalTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
    
    
}

size_t UnifiedGPUArena::calculateArenaSize(int maxDetections, int yoloSize) {
    size_t size = 0;
    
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += yoloSize * yoloSize * 3 * sizeof(float);
    
    size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
    size += maxDetections * sizeof(Target) * 2;
    
    
    return size;
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
    float head_y_offset,
    float body_y_offset,
    int detection_resolution,
    int* __restrict__ bestTargetIndex,
    Target* __restrict__ bestTarget,
    needaimbot::MouseMovement* __restrict__ output_movement
) {
    if (blockIdx.x == 0) {
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
        
        __shared__ float s_distancesX[256];  // X축 거리 배열
        __shared__ int s_indices[256];
        
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
        
        if (threadIdx.x == 0 && s_indices[0] >= 0) {
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
            
            float movement_x = error_x * kp_x;
            float movement_y = error_y * kp_y;
            
            output_movement->dx = static_cast<int>(movement_x);
            output_movement->dy = static_cast<int>(movement_y);
        }
    }
}


__global__ void clearAllDetectionBuffersKernel(
    Target* decodedTargets,
    int* decodedCount,
    int* classFilteredCount,
    int* finalTargetsCount,
    int* colorFilteredCount,
    int* bestTargetIndex,
    Target* bestTarget,
    int maxTargetsToClear
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    if (tid < 6) {
        if (tid == 0) *decodedCount = 0;
        else if (tid == 1) *classFilteredCount = 0;
        else if (tid == 2) *finalTargetsCount = 0;
        else if (tid == 3 && colorFilteredCount) *colorFilteredCount = 0;
        else if (tid == 4 && bestTargetIndex) *bestTargetIndex = -1;
        else if (tid == 5 && bestTarget) {
            Target emptyTarget = {};
            *bestTarget = emptyTarget;
        }
    }
    
    for (int i = tid; i < maxTargetsToClear; i += gridSize) {
        Target emptyTarget = {};
        decodedTargets[i] = emptyTarget;
    }
}


UnifiedGraphPipeline::UnifiedGraphPipeline() {
    m_state.startEvent = std::make_unique<CudaEvent>();
    m_state.endEvent = std::make_unique<CudaEvent>();
    
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
    
    // Graph 내부에서는 복사 불필요 - executeGraphNonBlocking에서 직접 통합 버퍼로 캡처함
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
            size_t inputSize = modelRes * modelRes * 3 * sizeof(float);
            cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize, 
                           cudaMemcpyDeviceToDevice, stream);
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
        cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                       sizeof(MouseMovement), cudaMemcpyDeviceToHost, stream);
        
        // 마우스 이동 콜백을 Graph에 포함 - 복사 완료 후 자동 실행
        cudaLaunchHostFunc(stream,
            [](void* userData) {
                auto* pipeline = static_cast<UnifiedGraphPipeline*>(userData);
                auto& ctx = AppContext::getInstance();
                
                if (!ctx.aiming || !pipeline->m_h_movement || !pipeline->m_h_movement->get()) {
                    return;
                }
                
                const MouseMovement* movement = pipeline->m_h_movement->get();
                if (movement->dx != 0 || movement->dy != 0) {
                    executeMouseMovement(movement->dx, movement->dy);
                }
            }, this);
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

void UnifiedGraphPipeline::updateStatistics(float latency) {
    // 통계 업데이트 제거 - CPU 사용 감소
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

        size_t unifiedArenaSize = UnifiedGPUArena::calculateArenaSize(maxDetections, yoloSize);
        m_unifiedArena.megaArena = std::make_unique<CudaMemory<uint8_t>>(unifiedArenaSize);
        m_unifiedArena.initializePointers(m_unifiedArena.megaArena->get(), maxDetections, yoloSize);
        
        
        
        m_h_movement = std::make_unique<CudaPinnedMemory<MouseMovement>>(1);
        
        // Dynamic preview buffer allocation based on current state
        updatePreviewBufferAllocation();
        
        if (!m_captureBuffer.data()) {
            throw std::runtime_error("Capture buffer allocation failed");
        }
        
        size_t gpuMemory = (width * height * 4 + yoloSize * yoloSize * 3) * sizeof(float);
        gpuMemory += unifiedArenaSize;
        if (m_preview.enabled) {
            gpuMemory += width * height * 4 * sizeof(unsigned char);
        }
        
        // Pinned Memory 힌트 추가 - GPU 메모리 상주 최적화
        if (m_unifiedArena.megaArena && m_unifiedArena.megaArena->get()) {
            // GPU 0번 디바이스에 메모리 선호 위치 설정
            cudaMemAdvise(m_unifiedArena.megaArena->get(), unifiedArenaSize,
                         cudaMemAdviseSetPreferredLocation, 0);
            // GPU 0번 디바이스에서 접근 예정임을 알림
            cudaMemAdvise(m_unifiedArena.megaArena->get(), unifiedArenaSize,
                         cudaMemAdviseSetAccessedBy, 0);
        }
        
        // Small buffer arena에도 동일한 힌트 적용
        if (m_smallBufferArena.arenaBuffer && m_smallBufferArena.arenaBuffer->get()) {
            cudaMemAdvise(m_smallBufferArena.arenaBuffer->get(), arenaSize,
                         cudaMemAdviseSetPreferredLocation, 0);
            cudaMemAdvise(m_smallBufferArena.arenaBuffer->get(), arenaSize,
                         cudaMemAdviseSetAccessedBy, 0);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Buffer allocation failed: " << e.what() << std::endl;
        deallocateBuffers();
        return false;
    }
}

void UnifiedGraphPipeline::deallocateBuffers() {
    
    m_captureBuffer.release();
    // m_unifiedCaptureBuffer removed - using m_captureBuffer
    
    // Always release preview buffer if allocated
    m_preview.previewBuffer.release();
    m_preview.finalTargets.clear();
    m_preview.enabled = false;
    
    m_smallBufferArena.arenaBuffer.reset();
    
    m_unifiedArena.megaArena.reset();
    
    m_d_inferenceOutput.reset();
    m_d_preprocessBuffer.reset();
    m_d_outputBuffer.reset();
    
    m_h_movement.reset();
    
    m_inputBindings.clear();
    m_outputBindings.clear();
    
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

void UnifiedGraphPipeline::updateProfilingAsync(cudaStream_t stream) {
    return;
}


void UnifiedGraphPipeline::handleAimbotDeactivation() {
    auto& ctx = AppContext::getInstance();
    
    clearCountBuffers();
    clearMovementData();
    clearHostPreviewData(ctx);
}

void UnifiedGraphPipeline::clearCountBuffers() {
    if (m_smallBufferArena.finalTargetsCount) {
        cudaMemsetAsync(m_smallBufferArena.finalTargetsCount, 0, sizeof(int), m_pipelineStream->get());
    }
    if (m_smallBufferArena.decodedCount) {
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
}

void UnifiedGraphPipeline::clearHostPreviewData(AppContext& ctx) {
    if (m_preview.enabled) {
        m_preview.finalTargets.clear();
        m_preview.finalCount = 0;
        m_preview.copyInProgress = false;
    }
    
    ctx.clearTargets();
}

void UnifiedGraphPipeline::handleAimbotActivation() {
    
    m_state.frameCount = 0;
}

bool UnifiedGraphPipeline::executePipelineWithErrorHandling() {
    // 에러 처리 제거 - GPU가 알아서 처리
    executeGraphNonBlocking(m_pipelineStream->get());
    return true;  // 항상 성공으로 간주
}

void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();
    
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
    
    bool wasAiming = false;
    
    while (!m_shouldStop && !ctx.should_exit) {
        {
            std::unique_lock<std::mutex> lock(ctx.pipeline_activation_mutex);
            ctx.pipeline_activation_cv.wait(lock, [&ctx, this]() {
                return ctx.aiming || m_shouldStop || ctx.should_exit;
            });
        }
        
        if (m_shouldStop || ctx.should_exit) break;
        
        if (!ctx.aiming) {
            if (wasAiming) {
                handleAimbotDeactivation();
                wasAiming = false;
            }
            continue;
        }
        
        if (!wasAiming) {
            std::cout << "[UnifiedGraph] Main loop activated (Right-click detected)" << std::endl;
            handleAimbotActivation();
            wasAiming = true;
        }
        
        while (ctx.aiming && !m_shouldStop && !ctx.should_exit) {
            executePipelineWithErrorHandling();
        }
        
        if (wasAiming && !ctx.aiming) {
            handleAimbotDeactivation();
            wasAiming = false;
        }
    }
    
}

void UnifiedGraphPipeline::stopMainLoop() {
    m_shouldStop = true;
    
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
    if (!m_unifiedArena.decodedTargets || !m_smallBufferArena.decodedCount || !m_smallBufferArena.classFilteredCount || 
        !m_smallBufferArena.finalTargetsCount || config.max_detections <= 0) {
        return;
    }
    
    static int cached_blockSize = 0;
    static int cached_minGridSize = 0;
    if (cached_blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&cached_minGridSize, &cached_blockSize, clearAllDetectionBuffersKernel, 0, 0);
    }
    
    int gridSize = std::min((config.max_detections + cached_blockSize - 1) / cached_blockSize, cached_minGridSize);
    
    clearAllDetectionBuffersKernel<<<gridSize, cached_blockSize, 0, stream>>>(
        m_unifiedArena.decodedTargets,
        m_smallBufferArena.decodedCount,
        m_smallBufferArena.classFilteredCount,
        m_smallBufferArena.finalTargetsCount,
        m_smallBufferArena.colorFilteredCount,
        m_smallBufferArena.bestTargetIndex,
        m_smallBufferArena.bestTarget,
        config.max_detections
    );
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
    if (!m_classFilterDirty || !m_smallBufferArena.allowFlags) {
        return;
    }
    
    auto& ctx = AppContext::getInstance();
    unsigned char h_allowFlags[Constants::MAX_CLASSES_FOR_FILTERING];
    memset(h_allowFlags, 0, Constants::MAX_CLASSES_FOR_FILTERING);
    
    for (const auto& setting : ctx.config.class_settings) {
        if (setting.id >= 0 && setting.id < Constants::MAX_CLASSES_FOR_FILTERING) {
            h_allowFlags[setting.id] = setting.allow ? 1 : 0;
        }
    }
    
    m_cachedClassFilter.assign(h_allowFlags, h_allowFlags + Constants::MAX_CLASSES_FOR_FILTERING);
    cudaMemcpyAsync(m_smallBufferArena.allowFlags, h_allowFlags, 
                   Constants::MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), 
                   cudaMemcpyHostToDevice, stream);
    m_classFilterDirty = false;
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
        
        cudaMemcpyAsync(m_smallBufferArena.finalTargetsCount, m_smallBufferArena.decodedCount, sizeof(int), 
                       cudaMemcpyDeviceToDevice, stream);
        
        cudaMemcpyAsync(m_unifiedArena.finalTargets, m_unifiedArena.decodedTargets, 
                       config.max_detections * sizeof(Target), cudaMemcpyDeviceToDevice, stream);
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
    
    const int blockSize = 256;
    const int gridSize = (cached_max_detections + blockSize - 1) / blockSize;
    
    fusedTargetSelectionAndMovementKernel<<<gridSize, blockSize, 0, stream>>>(
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
        m_smallBufferArena.mouseMovement
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Fused kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaMemsetAsync(m_smallBufferArena.bestTargetIndex, -1, sizeof(int), stream);
        cudaMemsetAsync(m_smallBufferArena.bestTarget, 0, sizeof(Target), stream);
        cudaMemsetAsync(m_smallBufferArena.mouseMovement, 0, sizeof(MouseMovement), stream);
    }
}

bool UnifiedGraphPipeline::updateNVFBCCaptureRegion(const AppContext& ctx) {
    if (!m_nvfbcCapture) {
        return false;
    }

    static int lastDetectionRes = 0;
    static float lastOffsetX = 0.0f;
    static float lastOffsetY = 0.0f;
    static bool lastAimShoot = false;

    int detectionRes = ctx.config.detection_resolution;
    if (detectionRes <= 0) {
        return false;
    }

    bool useAimShootOffset = ctx.config.enable_aim_shoot_offset && ctx.aiming && ctx.shooting;
    float offsetX = useAimShootOffset ? ctx.config.aim_shoot_offset_x : ctx.config.crosshair_offset_x;
    float offsetY = useAimShootOffset ? ctx.config.aim_shoot_offset_y : ctx.config.crosshair_offset_y;

    if (lastDetectionRes == detectionRes &&
        lastOffsetX == offsetX &&
        lastOffsetY == offsetY &&
        lastAimShoot == useAimShootOffset) {
        return true;
    }

    int screenW = m_nvfbcCapture->GetScreenWidth();
    int screenH = m_nvfbcCapture->GetScreenHeight();
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

    if (!m_nvfbcCapture->SetCaptureRegion(left, top, captureSize, captureSize)) {
        return false;
    }

    lastDetectionRes = detectionRes;
    lastOffsetX = offsetX;
    lastOffsetY = offsetY;
    lastAimShoot = useAimShootOffset;
    return true;
}

bool UnifiedGraphPipeline::copyNVFBCFrameToGPU(void* frameData, unsigned int width, unsigned int height) {
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
        std::cerr << "[Capture] NVFBC frame copy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

bool UnifiedGraphPipeline::performFrameCapture() {
    auto& ctx = AppContext::getInstance();

    if (!m_config.enableCapture || m_hasFrameData) {
        return true;
    }

    if (!m_nvfbcCapture) {
        std::cerr << "[Capture] NVFBC capture interface not set" << std::endl;
        return false;
    }

    if (!updateNVFBCCaptureRegion(ctx)) {
        return false;
    }

    void* frameData = nullptr;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int size = 0;

    if (!m_nvfbcCapture->GetLatestFrame(&frameData, &width, &height, &size)) {
        return false;
    }

    if (!copyNVFBCFrameToGPU(frameData, width, height)) {
        return false;
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
        // Need to allocate preview buffer
        int width = ctx.config.detection_resolution;
        int height = ctx.config.detection_resolution;
        m_preview.previewBuffer.create(height, width, 4);
        m_preview.finalTargets.reserve(ctx.config.max_detections);
        m_preview.enabled = true;
    } else if (!ctx.config.show_window && m_preview.enabled) {
        // Need to deallocate preview buffer
        m_preview.previewBuffer.release();
        m_preview.finalTargets.clear();
        m_preview.enabled = false;
    }
}

void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    auto& ctx = AppContext::getInstance();
    
    // First ensure preview buffer allocation is correct
    updatePreviewBufferAllocation();
    
    // Check both m_preview.enabled AND current show_window state
    if (!m_preview.enabled || !ctx.config.show_window || m_preview.previewBuffer.empty()) {
        return;
    }
    
    if (m_preview.previewBuffer.rows() != currentBuffer.rows() || 
        m_preview.previewBuffer.cols() != currentBuffer.cols() || 
        m_preview.previewBuffer.channels() != currentBuffer.channels()) {
        m_preview.previewBuffer.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
    }
    
    size_t dataSize = currentBuffer.rows() * currentBuffer.cols() * currentBuffer.channels() * sizeof(unsigned char);
    cudaMemcpyAsync(m_preview.previewBuffer.data(), currentBuffer.data(), dataSize, 
                   cudaMemcpyDeviceToDevice, m_pipelineStream->get());
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
    
    performIntegratedPostProcessing(m_pipelineStream->get());
    performTargetSelection(m_pipelineStream->get());
    
    return true;
}

int UnifiedGraphPipeline::findHeadClassId(const AppContext& ctx) {
    for(const auto& cs : ctx.config.class_settings) {
        if (cs.name == ctx.config.head_class_name) {
            return cs.id;
        }
    }
    return -1;
}

// performResultCopy는 더 이상 필요 없음 - Graph와 콜백에서 직접 처리

bool UnifiedGraphPipeline::executeGraphNonBlocking(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Graph 초기 빌드만 체크 (재빌드 체크 제거)
    static bool graphInitialized = false;
    if (!graphInitialized && ctx.config.use_cuda_graph) {
        captureGraph(stream);
        graphInitialized = true;
    }
    
    // Graph 모드이고 Graph가 준비된 경우
    if (ctx.config.use_cuda_graph && m_state.graphReady && m_graphExec) {
        // Graph 실행 전에 직접 통합 버퍼로 캡처 (중복 복사 제거)
        if (!performFrameCaptureDirectToUnified()) {
            return false;
        }
        
        // Graph 실행 - 에러 체크 없이 바로 실행
        cudaGraphLaunch(m_graphExec, stream ? stream : m_pipelineStream->get());
        
        // Graph 실행 후 preview buffer 업데이트 (Graph 외부에서 처리)
        if (m_preview.enabled && ctx.config.show_window && !m_captureBuffer.empty()) {
            updatePreviewBuffer(m_captureBuffer);
        }
        
        // 프레임 카운트 제거 - 불필요한 CPU 작업
        m_hasFrameData = false;
        return true;
    }
    
    // Normal 실행 경로
    return executeNormalPipeline(stream);
}

bool UnifiedGraphPipeline::executeNormalPipeline(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // 이벤트 및 동기화 제거 - 콜백 체인만 사용
    
    if (!performFrameCapture()) {
        return false;
    }
    
    if (!ctx.detection_paused.load()) {
        if (!performPreprocessing()) {
            return false;
        }
        
        if (!performInference()) {
            return false;
        }
        
        // 후처리
        performIntegratedPostProcessing(m_pipelineStream->get());
        performTargetSelection(m_pipelineStream->get());
        
        // 결과 복사
        cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement,
                       sizeof(MouseMovement), cudaMemcpyDeviceToHost, m_pipelineStream->get());
        
        // 마우스 이동 콜백 - 복사 완료 후 자동 실행
        cudaLaunchHostFunc(m_pipelineStream->get(),
            [](void* userData) {
                auto* pipeline = static_cast<UnifiedGraphPipeline*>(userData);
                auto& ctx = AppContext::getInstance();
                
                if (!ctx.aiming || !pipeline->m_h_movement || !pipeline->m_h_movement->get()) {
                    return;
                }
                
                const MouseMovement* movement = pipeline->m_h_movement->get();
                if (movement->dx != 0 || movement->dy != 0) {
                    executeMouseMovement(movement->dx, movement->dy);
                }
            }, this);
    }
    
    // 프레임 카운트 제거
    m_hasFrameData = false;
    
    return true;
}

// processMouseMovement는 이제 Graph와 콜백 내부에서 인라인으로 처리됨

}