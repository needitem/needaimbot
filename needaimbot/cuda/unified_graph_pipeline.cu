#include "unified_graph_pipeline.h"
#include "detection/cuda_float_processing.h"
#include "tracking/gpu_kalman_filter.h"
#include "detection/filterGpu.h"
#include "simple_cuda_mat.h"
#include "mouse_interface.h"
// #include "pd_controller_shared.h"  // Removed - using GPU-based PD controller
#include "../AppContext.h"
#include "../core/logger.h"
#include "cuda_error_check.h"
#include "preprocessing.h"  // Use existing CUDA error checking macros
#include <d3d11.h>
#include <dxgi1_2.h>
#include "../include/other_tools.h"  // For fileExists function
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


namespace needaimbot {

// All complex graph and coordinator code removed - using simple single stream

// ============================================================================
// OPTIMIZED CUDA KERNELS
// ============================================================================

// fusedPreprocessKernel removed - using CudaImageProcessing pipeline instead

// Helper kernel for copying D3D11 texture to CUDA buffer without CPU mapping
__global__ void copyTextureToBuffer(cudaSurfaceObject_t surface, 
                                   unsigned char* output,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        uchar4 pixel;
        surf2Dread(&pixel, surface, x * sizeof(uchar4), y);
        
        int idx = (y * width + x) * 4;
        output[idx] = pixel.x;
        output[idx + 1] = pixel.y;
        output[idx + 2] = pixel.z;
        output[idx + 3] = pixel.w;
    }
}

UnifiedGraphPipeline::UnifiedGraphPipeline() {
    // Initialize events for profiling
    cudaEventCreate(&m_state.startEvent);
    cudaEventCreate(&m_state.endEvent);
    
    // Simple initialization - no complex graph management needed
    m_tripleBuffer = nullptr;
}

UnifiedGraphPipeline::~UnifiedGraphPipeline() {
    shutdown();
    
    if (m_state.startEvent) cudaEventDestroy(m_state.startEvent);
    if (m_state.endEvent) cudaEventDestroy(m_state.endEvent);
    // Events removed - using simple pipeline
    if (m_previewReadyEvent) cudaEventDestroy(m_previewReadyEvent);
}

bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    // Create single CUDA stream for all operations
    cudaStreamCreateWithFlags(&m_primaryStream, cudaStreamNonBlocking);
    
    // Initialize Triple Buffer for async pipeline
    if (m_config.enableCapture) {
        m_tripleBuffer = std::make_unique<TripleBuffer>();
        // Initialize pinned memory for zero-copy transfers
        m_tripleBuffer->initPinnedMemory();
        // Triple buffer initialization will be done in allocateBuffers()
    }
    // No complex events needed
    cudaEventCreateWithFlags(&m_previewReadyEvent, cudaEventDisableTiming);
    
    // Initialize TensorRT (Phase 1 integration - now required)
    if (m_config.modelPath.empty()) {
        std::cerr << "[UnifiedGraph] ERROR: Model path is required for TensorRT integration" << std::endl;
        return false;
    }
    
    std::cout << "[UnifiedGraph] Initializing TensorRT with model: " << m_config.modelPath << std::endl;
    if (!initializeTensorRT(m_config.modelPath)) {
        std::cerr << "[UnifiedGraph] CRITICAL: TensorRT initialization failed" << std::endl;
        return false;
    }
    std::cout << "[UnifiedGraph] TensorRT integration completed successfully" << std::endl;
    
    // Allocate pipeline buffers
    if (!allocateBuffers()) {
        std::cerr << "[UnifiedGraph] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    // Initial graph capture if enabled
    if (m_config.useGraphOptimization) {
        std::cout << "[DEBUG] CUDA Graph optimization enabled with proper initialization" << std::endl;
        // Capture detection graph with proper input initialization
        if (!captureGraph(m_primaryStream)) {
            std::cerr << "[UnifiedGraph] Warning: Failed to capture detection graph" << std::endl;
        }
        
        // We'll capture the full graph on first execution with real data
        m_state.needsRebuild = true;
    }
    
    std::cout << "[UnifiedGraph] Pipeline initialized with:" << std::endl;
    std::cout << "  - TensorRT integration: " << (!m_config.modelPath.empty() ? "Yes" : "No") << std::endl;
    std::cout << "  - Multi-stream coordinator: Yes" << std::endl;
    std::cout << "  - Dynamic graph updates: Yes" << std::endl;
    std::cout << "  - Triple buffering: " << (m_tripleBuffer ? "Yes" : "No") << std::endl;
    std::cout << "  - Graph optimization: " << (m_config.useGraphOptimization ? "Yes" : "No") << std::endl;
    return true;
}

void UnifiedGraphPipeline::shutdown() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    
    // Clean up TensorRT resources (Phase 1 integration)
    for (auto& binding : m_inputBindings) {
        if (binding.second) cudaFree(binding.second);
    }
    m_inputBindings.clear();
    
    for (auto& binding : m_outputBindings) {
        if (binding.second) cudaFree(binding.second);
    }
    m_outputBindings.clear();
    
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
    
    cleanupGraph();
    deallocateBuffers();
    if (m_primaryStream) {
        cudaStreamDestroy(m_primaryStream);
        m_primaryStream = nullptr;
    }
}

bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();
    
    if (!stream) stream = m_primaryStream;
    
    std::cout << "[UnifiedGraph] Starting graph capture..." << std::endl;
    
    // Clean up existing graph
    cleanupGraph();
    
    // Clear node tracking vectors
    m_captureNodes.clear();
    m_inferenceNodes.clear();
    m_postprocessNodes.clear();
    m_trackingNodes.clear();
    m_pidNodes.clear();
    
    // Begin graph capture with relaxed mode to handle TensorRT stream dependencies
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin capture: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Execute the entire pipeline once to capture all operations
    // The graph will record all CUDA operations performed in this stream
    
    // 1. Input copy (if we have pinned host memory)
    if (m_config.enableCapture && m_h_inputBuffer) {
        cudaMemcpyAsync(m_captureBuffer.data(), m_h_inputBuffer,
                       m_captureBuffer.sizeInBytes(), cudaMemcpyHostToDevice, stream);
    }
    
    // Preprocessing removed - using CudaImageProcessing pipeline in executeGraph instead
    
    // 3. TensorRT Inference  
    if (false && m_config.enableDetection) {  // Temporarily disabled for Graph capture
        // Initialize input buffer with valid data for graph capture
        if (m_d_yoloInput) {
            // std::cout << "[DEBUG] Initializing input buffer for graph capture..." << std::endl;
            
            // Create host buffer with dummy normalized values
            size_t inputSize = 640 * 640 * 3 * sizeof(float);
            std::vector<float> dummyData(640 * 640 * 3, 0.5f); // Fill with 0.5 (normalized pixel value)
            
            // Copy from host to device
            cudaMemcpyAsync(m_d_yoloInput, dummyData.data(), inputSize, cudaMemcpyHostToDevice, stream);
            
            // CRITICAL FIX: Also initialize TensorRT binding buffers
            auto bindingIt = m_inputBindings.find(m_inputName);
            if (bindingIt != m_inputBindings.end() && bindingIt->second != nullptr) {
                std::cout << "[DEBUG] Found TensorRT binding for '" << m_inputName << "' at: " << bindingIt->second << std::endl;
                if (bindingIt->second != m_d_yoloInput) {
                    std::cout << "[DEBUG] Initializing TensorRT binding buffer (different from yolo buffer)" << std::endl;
                    // First clear with zeros, then set dummy data
                    cudaMemsetAsync(bindingIt->second, 0, inputSize, stream);
                    cudaMemcpyAsync(bindingIt->second, dummyData.data(), inputSize, cudaMemcpyHostToDevice, stream);
                } else {
                    std::cout << "[DEBUG] TensorRT binding uses same buffer as yolo input" << std::endl;
                }
            } else {
                std::cout << "[DEBUG] WARNING: TensorRT binding not found for '" << m_inputName << "'" << std::endl;
            }
            
            // Wait for copy to complete before graph capture
            cudaStreamSynchronize(stream);
            
            std::cout << "[DEBUG] Input buffer initialized with dummy data (0.5)" << std::endl;
        }
        
        // Use integrated TensorRT inference (Phase 1)
        if (!runInferenceAsync(stream)) {
            std::cerr << "[UnifiedGraph] TensorRT inference failed during graph capture" << std::endl;
        }
    }
    
    // 4. Postprocessing (NMS, filtering, target selection)
    // Note: NMS is skipped during graph capture to avoid kernel errors
    // NMS will be executed outside the graph in the actual execution
    if (m_config.enableDetection && m_d_selectedTarget && m_d_detections) {
        // Just initialize the target buffer during graph capture
        cudaMemsetAsync(m_d_selectedTarget, 0, sizeof(Target), stream);
        cudaMemsetAsync(m_d_numDetections, 0, sizeof(int), stream);
    }
    
    // 5. Tracking (Kalman filter)
    if (m_config.enableTracking && m_tracker && m_d_selectedTarget) {
        // Process tracking with GPU Kalman filter using pre-allocated buffer
        processKalmanFilter(
            m_tracker,
            m_d_selectedTarget,  // Input measurement
            1,                   // Single target for now
            m_d_trackedTarget,   // Output tracked target
            m_d_outputCount,     // Output count (pre-allocated)
            stream,
            false,              // Don't use graph within graph capture
            1.0f                // Lookahead frames
        );
    }
    
    // 6. Bezier Control (already handled in the main pipeline)
    // The actual Bezier control is executed in graph method
    
    // 7. Final output copy (only 2 floats for mouse X,Y)
    if (m_d_outputBuffer && m_h_outputBuffer) {
        // Copy final movement to output buffer if needed
        // This is mainly for debugging/monitoring purposes
        cudaMemcpyAsync(m_h_outputBuffer, m_d_outputBuffer,
                       2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    
    // End capture
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
    
    // Validate graph
    if (!validateGraph()) {
        std::cerr << "[UnifiedGraph] Graph validation failed" << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }
    
    // Instantiate graph for execution
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
    
    // Get node count for debugging
    size_t numNodes = 0;
    cudaGraphGetNodes(m_graph, nullptr, &numNodes);
    
    std::cout << "[UnifiedGraph] Graph captured successfully with " 
              << numNodes << " nodes" << std::endl;
    
    return true;
}

void UnifiedGraphPipeline::checkTargetsAsync(cudaStream_t stream) {
    // Target checking removed - simplified pipeline
}





bool UnifiedGraphPipeline::validateGraph() {
    if (!m_graph) return false;
    
    // Get graph nodes
    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(m_graph, nullptr, &numNodes);
    if (err != cudaSuccess || numNodes == 0) {
        std::cerr << "[UnifiedGraph] Graph has no nodes" << std::endl;
        return false;
    }
    
    std::cout << "[UnifiedGraph] Graph validated with " << numNodes << " nodes" << std::endl;
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
    // Simple moving average
    const float alpha = 0.1f;
    m_state.avgLatency = (1.0f - alpha) * m_state.avgLatency + alpha * latency;
}

bool UnifiedGraphPipeline::allocateBuffers() {
    auto& ctx = AppContext::getInstance();
    const int width = ctx.config.detection_resolution;   // Use actual detection resolution
    const int height = ctx.config.detection_resolution;  // Square capture region
    const int yoloSize = ctx.config.onnx_input_resolution;
    const int maxDetections = 100;
    
    try {
        // Initialize Triple Buffer System for async pipeline
        if (m_config.enableCapture && m_tripleBuffer) {
            for (int i = 0; i < 3; i++) {
                m_tripleBuffer->buffers[i].create(height, width, 4);  // BGRA
                CUDA_CHECK(cudaEventCreateWithFlags(&m_tripleBuffer->events[i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&m_tripleBuffer->target_ready_events[i], cudaEventDisableTiming));
                m_tripleBuffer->isReady[i] = false;
                m_tripleBuffer->target_data_valid[i] = false;
                m_tripleBuffer->target_count[i] = 0;
            }
            std::cout << "[UnifiedGraph] Triple buffer system initialized" << std::endl;
        }
        
        // Allocate GPU buffers
        m_captureBuffer.create(height, width, 4);  // BGRA
        m_preprocessBuffer.create(height, width, 3);  // BGR
        
        // YOLO input buffer (configurable size x3 in CHW format)
        CUDA_CHECK(cudaMalloc(&m_d_yoloInput, yoloSize * yoloSize * 3 * sizeof(float)));
        if (!m_d_yoloInput) throw std::runtime_error("m_d_yoloInput allocation failed");
        
        // Detection pipeline buffers
        CUDA_CHECK(cudaMalloc(&m_d_inferenceOutput, maxDetections * 6 * sizeof(float)));
        if (!m_d_inferenceOutput) throw std::runtime_error("m_d_inferenceOutput allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_nmsOutput, maxDetections * 6 * sizeof(float)));
        if (!m_d_nmsOutput) throw std::runtime_error("m_d_nmsOutput allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_filteredOutput, maxDetections * 6 * sizeof(float)));
        if (!m_d_filteredOutput) throw std::runtime_error("m_d_filteredOutput allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_detections, maxDetections * sizeof(Target)));
        if (!m_d_detections) throw std::runtime_error("m_d_detections allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_selectedTarget, sizeof(Target)));
        if (!m_d_selectedTarget) throw std::runtime_error("m_d_selectedTarget allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_trackedTarget, sizeof(Target)));
        if (!m_d_trackedTarget) throw std::runtime_error("m_d_trackedTarget allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_trackedTargets, maxDetections * sizeof(Target)));
        if (!m_d_trackedTargets) throw std::runtime_error("m_d_trackedTargets allocation failed");
        
        // Allocate NMS temporary buffers
        CUDA_CHECK(cudaMalloc(&m_d_numDetections, sizeof(int)));
        if (!m_d_numDetections) throw std::runtime_error("m_d_numDetections allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_x1, maxDetections * sizeof(int)));
        if (!m_d_x1) throw std::runtime_error("m_d_x1 allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_y1, maxDetections * sizeof(int)));
        if (!m_d_y1) throw std::runtime_error("m_d_y1 allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_x2, maxDetections * sizeof(int)));
        if (!m_d_x2) throw std::runtime_error("m_d_x2 allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_y2, maxDetections * sizeof(int)));
        if (!m_d_y2) throw std::runtime_error("m_d_y2 allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_areas, maxDetections * sizeof(float)));
        if (!m_d_areas) throw std::runtime_error("m_d_areas allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_scores_nms, maxDetections * sizeof(float)));
        if (!m_d_scores_nms) throw std::runtime_error("m_d_scores_nms allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_classIds_nms, maxDetections * sizeof(int)));
        if (!m_d_classIds_nms) throw std::runtime_error("m_d_classIds_nms allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_iou_matrix, maxDetections * maxDetections * sizeof(float)));
        if (!m_d_iou_matrix) throw std::runtime_error("m_d_iou_matrix allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_keep, maxDetections * sizeof(bool)));
        if (!m_d_keep) throw std::runtime_error("m_d_keep allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_indices, maxDetections * sizeof(int)));
        if (!m_d_indices) throw std::runtime_error("m_d_indices allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_outputCount, sizeof(int)));
        if (!m_d_outputCount) throw std::runtime_error("m_d_outputCount allocation failed");
        
        // Allocate post-processing buffers (Phase 3 integration)
        CUDA_CHECK(cudaMalloc(&m_d_decodedTargets, maxDetections * sizeof(Target)));
        if (!m_d_decodedTargets) throw std::runtime_error("m_d_decodedTargets allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_decodedCount, sizeof(int)));
        if (!m_d_decodedCount) throw std::runtime_error("m_d_decodedCount allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_finalTargets, maxDetections * sizeof(Target)));
        if (!m_d_finalTargets) throw std::runtime_error("m_d_finalTargets allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_finalTargetsCount, sizeof(int)));
        if (!m_d_finalTargetsCount) throw std::runtime_error("m_d_finalTargetsCount allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_classFilteredTargets, maxDetections * sizeof(Target)));
        if (!m_d_classFilteredTargets) throw std::runtime_error("m_d_classFilteredTargets allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_classFilteredCount, sizeof(int)));
        if (!m_d_classFilteredCount) throw std::runtime_error("m_d_classFilteredCount allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_colorFilteredTargets, maxDetections * sizeof(Target)));
        if (!m_d_colorFilteredTargets) throw std::runtime_error("m_d_colorFilteredTargets allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_colorFilteredCount, sizeof(int)));
        if (!m_d_colorFilteredCount) throw std::runtime_error("m_d_colorFilteredCount allocation failed");
        
        // Target selection buffers
        CUDA_CHECK(cudaMalloc(&m_d_bestTargetIndex, sizeof(int)));
        if (!m_d_bestTargetIndex) throw std::runtime_error("m_d_bestTargetIndex allocation failed");
        
        CUDA_CHECK(cudaMalloc(&m_d_bestTarget, sizeof(Target)));
        if (!m_d_bestTarget) throw std::runtime_error("m_d_bestTarget allocation failed");
        
        // Class filtering control buffer (64 classes max)
        CUDA_CHECK(cudaMalloc(&m_d_allowFlags, Constants::MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char)));
        if (!m_d_allowFlags) throw std::runtime_error("m_d_allowFlags allocation failed");
        
        // Allocate pinned host memory for zero-copy transfers
        CUDA_CHECK(cudaHostAlloc(&m_h_inputBuffer, width * height * 4, cudaHostAllocDefault));
        if (!m_h_inputBuffer) throw std::runtime_error("m_h_inputBuffer allocation failed");
        
        CUDA_CHECK(cudaHostAlloc(&m_h_outputBuffer, 2 * sizeof(float), cudaHostAllocMapped));
        if (!m_h_outputBuffer) throw std::runtime_error("m_h_outputBuffer allocation failed");
        
        // Additional validation for critical buffers
        if (!m_captureBuffer.data() || !m_preprocessBuffer.data()) {
            throw std::runtime_error("SimpleCudaMat buffer allocation failed");
        }
        
        std::cout << "[UnifiedGraph] Allocated buffers: "
                  << "GPU: " << ((width * height * 7 + yoloSize * yoloSize * 3 + 
                                 maxDetections * 20) * sizeof(float) / (1024 * 1024)) 
                  << " MB, Pinned: " << ((width * height * 4 + 8) / (1024 * 1024)) 
                  << " MB" << std::endl;
        
        // Debug: Print NMS buffer pointers
        std::cout << "[UnifiedGraph] NMS buffer pointers:" << std::endl;
        std::cout << "  m_d_x1=" << m_d_x1 << " m_d_y1=" << m_d_y1 << std::endl;
        std::cout << "  m_d_x2=" << m_d_x2 << " m_d_y2=" << m_d_y2 << std::endl;
        std::cout << "  m_d_areas=" << m_d_areas << " m_d_scores_nms=" << m_d_scores_nms << std::endl;
        std::cout << "  m_d_classIds_nms=" << m_d_classIds_nms << std::endl;
        std::cout << "  m_d_iou_matrix=" << m_d_iou_matrix << " m_d_keep=" << m_d_keep << std::endl;
        std::cout << "  m_d_indices=" << m_d_indices << " m_d_numDetections=" << m_d_numDetections << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Buffer allocation failed: " << e.what() << std::endl;
        deallocateBuffers();
        return false;
    }
}

void UnifiedGraphPipeline::deallocateBuffers() {
    // Synchronize all CUDA operations before deallocating
    cudaDeviceSynchronize();
    
    // Release SimpleCudaMat buffers
    m_captureBuffer.release();
    m_preprocessBuffer.release();
    
    // Clean up triple buffer events
    if (m_tripleBuffer) {
        for (int i = 0; i < 3; i++) {
            if (m_tripleBuffer->events[i]) {
                cudaEventDestroy(m_tripleBuffer->events[i]);
            }
            if (m_tripleBuffer->target_ready_events[i]) {
                cudaEventDestroy(m_tripleBuffer->target_ready_events[i]);
            }
        }
    }
    
    // Helper lambda to safely free CUDA memory
    auto safeCudaFree = [](void*& ptr, const char* name) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
                std::cerr << "[UnifiedGraph] Warning: Failed to free " << name 
                         << ": " << cudaGetErrorString(err) << std::endl;
            }
            ptr = nullptr;
        }
    };
    
    auto safeCudaFreeHost = [](void*& ptr, const char* name) {
        if (ptr) {
            cudaError_t err = cudaFreeHost(ptr);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
                std::cerr << "[UnifiedGraph] Warning: Failed to free host " << name 
                         << ": " << cudaGetErrorString(err) << std::endl;
            }
            ptr = nullptr;
        }
    };
    
    // Free GPU buffers with error checking
    safeCudaFree(reinterpret_cast<void*&>(m_d_yoloInput), "m_d_yoloInput");
    safeCudaFree(reinterpret_cast<void*&>(m_d_inferenceOutput), "m_d_inferenceOutput");
    safeCudaFree(reinterpret_cast<void*&>(m_d_nmsOutput), "m_d_nmsOutput");
    safeCudaFree(reinterpret_cast<void*&>(m_d_filteredOutput), "m_d_filteredOutput");
    safeCudaFree(reinterpret_cast<void*&>(m_d_detections), "m_d_detections");
    safeCudaFree(reinterpret_cast<void*&>(m_d_selectedTarget), "m_d_selectedTarget");
    safeCudaFree(reinterpret_cast<void*&>(m_d_trackedTarget), "m_d_trackedTarget");
    safeCudaFree(reinterpret_cast<void*&>(m_d_trackedTargets), "m_d_trackedTargets");
    
    // Free NMS temporary buffers
    safeCudaFree(reinterpret_cast<void*&>(m_d_numDetections), "m_d_numDetections");
    safeCudaFree(reinterpret_cast<void*&>(m_d_x1), "m_d_x1");
    safeCudaFree(reinterpret_cast<void*&>(m_d_y1), "m_d_y1");
    safeCudaFree(reinterpret_cast<void*&>(m_d_x2), "m_d_x2");
    safeCudaFree(reinterpret_cast<void*&>(m_d_y2), "m_d_y2");
    safeCudaFree(reinterpret_cast<void*&>(m_d_areas), "m_d_areas");
    safeCudaFree(reinterpret_cast<void*&>(m_d_scores_nms), "m_d_scores_nms");
    safeCudaFree(reinterpret_cast<void*&>(m_d_classIds_nms), "m_d_classIds_nms");
    safeCudaFree(reinterpret_cast<void*&>(m_d_iou_matrix), "m_d_iou_matrix");
    safeCudaFree(reinterpret_cast<void*&>(m_d_keep), "m_d_keep");
    safeCudaFree(reinterpret_cast<void*&>(m_d_indices), "m_d_indices");
    safeCudaFree(reinterpret_cast<void*&>(m_d_outputCount), "m_d_outputCount");
    
    // Free post-processing buffers (Phase 3 integration)
    safeCudaFree(reinterpret_cast<void*&>(m_d_decodedTargets), "m_d_decodedTargets");
    safeCudaFree(reinterpret_cast<void*&>(m_d_decodedCount), "m_d_decodedCount");
    safeCudaFree(reinterpret_cast<void*&>(m_d_finalTargets), "m_d_finalTargets");
    safeCudaFree(reinterpret_cast<void*&>(m_d_finalTargetsCount), "m_d_finalTargetsCount");
    safeCudaFree(reinterpret_cast<void*&>(m_d_classFilteredTargets), "m_d_classFilteredTargets");
    safeCudaFree(reinterpret_cast<void*&>(m_d_classFilteredCount), "m_d_classFilteredCount");
    safeCudaFree(reinterpret_cast<void*&>(m_d_colorFilteredTargets), "m_d_colorFilteredTargets");
    safeCudaFree(reinterpret_cast<void*&>(m_d_colorFilteredCount), "m_d_colorFilteredCount");
    safeCudaFree(reinterpret_cast<void*&>(m_d_bestTargetIndex), "m_d_bestTargetIndex");
    safeCudaFree(reinterpret_cast<void*&>(m_d_bestTarget), "m_d_bestTarget");
    safeCudaFree(reinterpret_cast<void*&>(m_d_allowFlags), "m_d_allowFlags");
    
    // Free pinned host memory
    safeCudaFreeHost(reinterpret_cast<void*&>(m_h_inputBuffer), "m_h_inputBuffer");
    safeCudaFreeHost(reinterpret_cast<void*&>(m_h_outputBuffer), "m_h_outputBuffer");
    
    // Reset all pointers
    m_d_yoloInput = nullptr;
    m_d_inferenceOutput = nullptr;
    m_d_nmsOutput = nullptr;
    m_d_filteredOutput = nullptr;
    m_d_detections = nullptr;
    m_d_selectedTarget = nullptr;
    m_d_trackedTarget = nullptr;
    m_d_trackedTargets = nullptr;
    m_d_numDetections = nullptr;
    m_d_x1 = nullptr;
    m_d_y1 = nullptr;
    m_d_x2 = nullptr;
    m_d_y2 = nullptr;
    m_d_areas = nullptr;
    m_d_scores_nms = nullptr;
    m_d_classIds_nms = nullptr;
    m_d_iou_matrix = nullptr;
    m_d_keep = nullptr;
    m_d_indices = nullptr;
    m_d_outputCount = nullptr;
    
    // Reset post-processing buffer pointers (Phase 3 integration)
    m_d_decodedTargets = nullptr;
    m_d_decodedCount = nullptr;
    m_d_finalTargets = nullptr;
    m_d_finalTargetsCount = nullptr;
    m_d_classFilteredTargets = nullptr;
    m_d_classFilteredCount = nullptr;
    m_d_colorFilteredTargets = nullptr;
    m_d_colorFilteredCount = nullptr;
    m_d_bestTargetIndex = nullptr;
    m_d_bestTarget = nullptr;
    m_d_allowFlags = nullptr;
    
    m_h_inputBuffer = nullptr;
    m_h_outputBuffer = nullptr;
}

// ============================================================================
// DYNAMIC PARAMETER UPDATE METHODS (No Graph Recapture Needed!)
// ============================================================================

void UnifiedGraphPipeline::setInputFrame(const SimpleCudaMat& frame) {
    // Copy frame data to capture buffer
    if (frame.empty()) return;
    
    // Check if we have a valid stream
    if (!m_primaryStream) {
        printf("[ERROR] Primary stream not initialized\n");
        return;
    }
    
    // Ensure buffer is the right size
    if (m_captureBuffer.empty() || 
        m_captureBuffer.rows() != frame.rows() || 
        m_captureBuffer.cols() != frame.cols() || 
        m_captureBuffer.channels() != frame.channels()) {
        m_captureBuffer.create(frame.rows(), frame.cols(), frame.channels());
    }
    
    // Copy data
    size_t dataSize = frame.rows() * frame.cols() * frame.channels() * sizeof(unsigned char);
    cudaError_t err = cudaMemcpyAsync(m_captureBuffer.data(), frame.data(), dataSize, 
                                      cudaMemcpyDeviceToDevice, m_primaryStream);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to copy frame to buffer: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Mark that we have frame data
    m_hasFrameData = true;
}

// Two-stage pipeline helper implementations
void UnifiedGraphPipeline::updateProfilingAsync(cudaStream_t stream) {
    // Async profiling update without blocking
    static cudaEvent_t lastFrameEnd = nullptr;
    if (!lastFrameEnd) {
        cudaEventCreateWithFlags(&lastFrameEnd, cudaEventDefault);
    }
    
    if (m_state.frameCount > 0) {
        // Check if last frame's profiling is ready
        if (cudaEventQuery(lastFrameEnd) == cudaSuccess) {
            float latency;
            cudaEventElapsedTime(&latency, m_state.startEvent, lastFrameEnd);
            updateStatistics(latency);
        }
    }
    
    // Record current frame end
    cudaEventRecord(m_state.startEvent, stream);
    cudaEventRecord(lastFrameEnd, stream);
}

// ============================================================================
// MAIN LOOP IMPLEMENTATION
// ============================================================================

void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();
    std::cout << "[UnifiedPipeline] Starting main loop - MAXIMUM PERFORMANCE MODE (No FPS Limit)" << std::endl;
    
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
    auto cycleStartTime = m_lastFrameTime;  // 1000사이클 시작 시간
    int cycleStartFrame = m_state.frameCount;  // 1000사이클 시작 프레임
    
    // NO FPS LIMITING - Maximum performance when active
    // Frame limiter removed for maximum responsiveness
    
    // Track aimbot state changes
    bool wasAiming = false;
    
    while (!m_shouldStop && !ctx.should_exit) {
        // Only run pipeline when aimbot is active
        if (!ctx.aiming) {
            // Log state change
            if (wasAiming) {
                std::cout << "[UnifiedPipeline] Aimbot deactivated - suspending pipeline" << std::endl;
                
                // Synchronize to ensure all GPU work is complete
                cudaStreamSynchronize(m_primaryStream);
                
                // Clear any pending targets
                if (m_d_bestTargetIndex) {
                    cudaMemsetAsync(m_d_bestTargetIndex, -1, sizeof(int), m_primaryStream);
                }
                if (m_d_bestTarget) {
                    cudaMemsetAsync(m_d_bestTarget, 0, sizeof(Target), m_primaryStream);
                }
                
                wasAiming = false;
            }
            
            // When aimbot is inactive, sleep with 1ms polling for ultra-fast response
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Log activation
        if (!wasAiming) {
            std::cout << "[UnifiedPipeline] Aimbot activated - MAXIMUM PERFORMANCE MODE" << std::endl;
            wasAiming = true;
            
            // Reset statistics
            m_state.frameCount = 0;
            cycleStartTime = std::chrono::high_resolution_clock::now();
            cycleStartFrame = 0;
        }
        
        // NO FRAME LIMITING - Run at maximum speed for lowest latency
        
        // Execute full pipeline only when aiming
        bool success = false;
        try {
            success = executeGraphNonBlocking(m_primaryStream);
        } catch (const std::exception& e) {
            std::cerr << "[UnifiedPipeline] Exception in pipeline: " << e.what() << std::endl;
            success = false;
        }
        
        if (!success) {
            // 에러 시 짧은 대기 후 재시도
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // 성능 통계 업데이트 (매 1000사이클마다)
        if (m_state.frameCount % 1000 == 0 && m_state.frameCount > 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto cycleElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - cycleStartTime);
            int framesProcessed = m_state.frameCount - cycleStartFrame;
            
            if (cycleElapsed.count() > 0 && framesProcessed > 0) {
                double averageFPS = (framesProcessed * 1000.0) / cycleElapsed.count();
                
                // Get GPU utilization (simplified - in production use NVML)
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                float gpu_memory_usage = ((total_mem - free_mem) * 100.0f) / total_mem;
                
                std::cout << "[UnifiedPipeline] Frame " << m_state.frameCount 
                          << " - Avg FPS: " << std::fixed << std::setprecision(1) << averageFPS
                          << " | GPU Mem: " << std::setprecision(1) << gpu_memory_usage << "%"
                          << " | Latency: " << m_state.avgLatency << "ms"
                          << " | Mode: MAX PERFORMANCE" << std::endl;
            }
            
            // 다음 1000사이클을 위한 시작점 리셋
            cycleStartTime = currentTime;
            cycleStartFrame = m_state.frameCount;
        }
    }
    
    std::cout << "[UnifiedPipeline] Main loop stopped after " << m_state.frameCount << " frames" << std::endl;
}

void UnifiedGraphPipeline::stopMainLoop() {
    std::cout << "[UnifiedPipeline] Stop requested" << std::endl;
    m_shouldStop = true;
}

// ============================================================================
// TENSORRT ENGINE MANAGEMENT (Phase 1 Integration)
// ============================================================================

bool UnifiedGraphPipeline::initializeTensorRT(const std::string& modelFile) {
    auto& ctx = AppContext::getInstance();
    
    std::cout << "[Pipeline] Initializing TensorRT with model: " << modelFile << std::endl;
    
    // Load the engine
    if (!loadEngine(modelFile)) {
        std::cerr << "[Pipeline] Failed to load engine" << std::endl;
        return false;
    }
    
    std::cout << "[Pipeline] Engine loaded successfully. Engine ptr: " << m_engine.get() << std::endl;
    
    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[Pipeline] Failed to create execution context" << std::endl;
        return false;
    }
    
    // Set optimization profile for dynamic shapes if needed
    if (m_engine->getNbOptimizationProfiles() > 0) {
        m_context->setOptimizationProfileAsync(0, m_primaryStream);
    }
    
    std::cout << "[Pipeline] Execution context created successfully. Context ptr: " << m_context.get() << std::endl;
    
    // Get input and output information
    getInputNames();
    getOutputNames();
    
    // Set up input dimensions and calculate sizes
    if (!m_inputNames.empty()) {
        m_inputName = m_inputNames[0];
        m_inputDims = m_engine->getTensorShape(m_inputName.c_str());
        
        // Calculate input size
        size_t inputSize = 1;
        for (int i = 0; i < m_inputDims.nbDims; ++i) {
            inputSize *= m_inputDims.d[i];
        }
        inputSize *= sizeof(float);  // Assuming float32 input
        m_inputSizes[m_inputName] = inputSize;
        
        std::cout << "[Pipeline] Input '" << m_inputName << "' dimensions: ";
        for (int i = 0; i < m_inputDims.nbDims; ++i) {
            std::cout << m_inputDims.d[i];
            if (i < m_inputDims.nbDims - 1) std::cout << "x";
        }
        std::cout << " (size: " << inputSize << " bytes)" << std::endl;
    }
    
    // Calculate output sizes
    for (const auto& outputName : m_outputNames) {
        nvinfer1::Dims outputDims = m_engine->getTensorShape(outputName.c_str());
        size_t outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; ++i) {
            outputSize *= outputDims.d[i];
        }
        outputSize *= sizeof(float);  // Assuming float32 output
        m_outputSizes[outputName] = outputSize;
        
        std::cout << "[Pipeline] Output '" << outputName << "' dimensions: ";
        for (int i = 0; i < outputDims.nbDims; ++i) {
            std::cout << outputDims.d[i];
            if (i < outputDims.nbDims - 1) std::cout << "x";
        }
        std::cout << " (size: " << outputSize << " bytes)" << std::endl;
    }
    
    // Allocate GPU memory bindings with error handling
    try {
        getBindings();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Failed to allocate TensorRT bindings: " << e.what() << std::endl;
        return false;
    }
    
    // Set up model-specific parameters
    // m_imgScale is used in post-processing to scale model output coordinates back to original image size
    // Model outputs coordinates in model input resolution space, need to scale to actual detection_resolution
    // ctx is already declared at line 1735, so just use it here
    m_imgScale = static_cast<float>(ctx.config.detection_resolution) / static_cast<float>(ctx.config.onnx_input_resolution);
    
    // Determine number of classes from output shape
    // Output shape is typically [batch, rows, boxes] where rows = 4 + num_classes
    const auto& outputShape = m_outputShapes[m_outputNames[0]];
    if (outputShape.size() >= 2) {
        m_numClasses = static_cast<int>(outputShape[1]) - 4;  // rows - 4 (bbox coords)
        std::cout << "[Pipeline] Detected " << m_numClasses << " classes from model output shape" << std::endl;
    } else {
        m_numClasses = 80;  // Default COCO classes
        std::cout << "[Pipeline] Using default 80 classes (COCO)" << std::endl;
    }
    
    std::cout << "[Pipeline] TensorRT initialization completed successfully" << std::endl;
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
            
            // Get and store output shape information
            auto dims = m_engine->getTensorShape(name);
            std::vector<int64_t> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            m_outputShapes[name] = shape;
            
            // Get and store output data type
            auto dataType = m_engine->getTensorDataType(name);
            m_outputTypes[name] = dataType;
            
            std::cout << "[Pipeline] Output '" << name << "' dimensions: ";
            for (int j = 0; j < dims.nbDims; ++j) {
                std::cout << dims.d[j];
                if (j < dims.nbDims - 1) std::cout << "x";
            }
            std::cout << std::endl;
        }
    }
}

void UnifiedGraphPipeline::getBindings() {
    auto& ctx = AppContext::getInstance();
    
    std::cout << "[Pipeline] Setting up optimized TensorRT bindings..." << std::endl;
    
    // Enhanced binding management with memory reuse optimization
    
    // Check for existing bindings that can be reused
    std::unordered_map<std::string, void*> reusableInputs;
    std::unordered_map<std::string, void*> reusableOutputs;
    
    // Store existing bindings for potential reuse
    for (const auto& binding : m_inputBindings) {
        if (binding.second) {
            reusableInputs[binding.first] = binding.second;
        }
    }
    for (const auto& binding : m_outputBindings) {
        if (binding.second) {
            reusableOutputs[binding.first] = binding.second;
        }
    }
    
    m_inputBindings.clear();
    m_outputBindings.clear();

    // Optimized input binding allocation with reuse
    for (const auto& name : m_inputNames) {
        size_t size = m_inputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for input '" << name << "'" << std::endl;
            continue;
        }
        
        void* ptr = nullptr;
        
        // Try to reuse existing allocation if size matches
        auto reusableIt = reusableInputs.find(name);
        if (reusableIt != reusableInputs.end()) {
            ptr = reusableIt->second;
            reusableInputs.erase(reusableIt); // Remove from reusable list
            std::cout << "[Pipeline] Reusing input '" << name << "': " << size << " bytes" << std::endl;
        } else {
            // Allocate new memory with alignment optimization
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess && ptr != nullptr) {
                // Initialize allocated memory to zero to prevent garbage values
                cudaMemset(ptr, 0, size);
                std::cout << "[Pipeline] Allocated and initialized input '" << name << "': " << size << " bytes" << std::endl;
            } else {
                std::cerr << "[Pipeline] Failed to allocate input memory for '" << name << "': " << cudaGetErrorString(err) << std::endl;
                
                // Clean up reusable memory before throwing
                for (auto& reusable : reusableInputs) {
                    cudaFree(reusable.second);
                }
                for (auto& reusable : reusableOutputs) {
                    cudaFree(reusable.second);
                }
                throw std::runtime_error("Failed to allocate TensorRT input memory");
            }
        }
        
        m_inputBindings[name] = ptr;
        
        // Connect to existing pipeline buffers where possible
        if (name == m_inputName && m_d_yoloInput) {
            // Use existing preprocessing buffer for YOLO input
            if (ptr != m_d_yoloInput) {
                std::cout << "[Pipeline] Input binding will use existing preprocessing buffer" << std::endl;
            }
        }
    }

    // Optimized output binding allocation with reuse
    for (const auto& name : m_outputNames) {
        size_t size = m_outputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for output '" << name << "'" << std::endl;
            continue;
        }
        
        void* ptr = nullptr;
        
        // Try to reuse existing allocation if size matches
        auto reusableIt = reusableOutputs.find(name);
        if (reusableIt != reusableOutputs.end()) {
            ptr = reusableIt->second;
            reusableOutputs.erase(reusableIt); // Remove from reusable list
            std::cout << "[Pipeline] Reusing output '" << name << "': " << size << " bytes" << std::endl;
        } else {
            // Allocate new memory with alignment optimization
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess && ptr != nullptr) {
                // Initialize allocated memory to zero to prevent garbage values
                cudaMemset(ptr, 0, size);
                std::cout << "[Pipeline] Allocated and initialized output '" << name << "': " << size << " bytes" << std::endl;
            } else {
                std::cerr << "[Pipeline] Failed to allocate output memory for '" << name << "': " << cudaGetErrorString(err) << std::endl;
                
                // Clean up reusable memory before throwing
                for (auto& reusable : reusableInputs) {
                    cudaFree(reusable.second);
                }
                for (auto& reusable : reusableOutputs) {
                    cudaFree(reusable.second);
                }
                throw std::runtime_error("Failed to allocate TensorRT output memory");
            }
        }
        
        m_outputBindings[name] = ptr;
        
        // Connect to existing pipeline buffers where possible
        if (m_d_inferenceOutput == nullptr) {
            m_d_inferenceOutput = (float*)ptr;
            std::cout << "[Pipeline] Output binding connected to inference output buffer" << std::endl;
        }
    }
    
    // Clean up any unused reusable memory
    for (auto& reusable : reusableInputs) {
        cudaError_t err = cudaFree(reusable.second);
        if (err != cudaSuccess) {
            std::cerr << "[Pipeline] Warning: Failed to free unused input binding: " << cudaGetErrorString(err) << std::endl;
        }
    }
    for (auto& reusable : reusableOutputs) {
        cudaError_t err = cudaFree(reusable.second);
        if (err != cudaSuccess) {
            std::cerr << "[Pipeline] Warning: Failed to free unused output binding: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    // Validate all bindings are properly set
    if (m_inputBindings.size() != m_inputNames.size()) {
        std::cerr << "[Pipeline] Warning: Input binding count mismatch" << std::endl;
    }
    if (m_outputBindings.size() != m_outputNames.size()) {
        std::cerr << "[Pipeline] Warning: Output binding count mismatch" << std::endl;
    }
    
    std::cout << "[Pipeline] Optimized TensorRT bindings setup completed" << std::endl;
}

nvinfer1::ICudaEngine* UnifiedGraphPipeline::buildEngineFromOnnx(const std::string& onnxPath) {
    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Suppress TensorRT internal errors
            if (severity <= Severity::kERROR && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    static SimpleLogger logger;

    auto& ctx = AppContext::getInstance();
    
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) return nullptr;

    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network) return nullptr;

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) return nullptr;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[Pipeline] Failed to parse ONNX file: " << onnxPath << std::endl;
        return nullptr;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return nullptr;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB

    // More aggressive TensorRT optimizations
    if (ctx.config.tensorrt_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[Pipeline] FP16 optimization enabled" << std::endl;
    }
    
    // Additional optimization flags
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    
    // Enable tactics sources for better kernel selection
    config->setTacticSources(
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS) |
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT) |
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN)
    );
    
    // Profiling for optimal kernel selection
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // Create optimization profile for dynamic inputs
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "[Pipeline] Failed to create optimization profile" << std::endl;
        return nullptr;
    }

    // Set optimization profile for the input (assuming batch size = 1, channels = 3, and dynamic height/width)
    const char* inputName = network->getInput(0)->getName();
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    
    // For YOLO models, typically the input is [1, 3, height, width]
    // Set min, opt, and max dimensions - using the config's input resolution
    int resolution = ctx.config.onnx_input_resolution;
    nvinfer1::Dims minDims = nvinfer1::Dims4{1, 3, resolution, resolution};
    nvinfer1::Dims optDims = nvinfer1::Dims4{1, 3, resolution, resolution};
    nvinfer1::Dims maxDims = nvinfer1::Dims4{1, 3, resolution, resolution};
    
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

bool UnifiedGraphPipeline::loadEngine(const std::string& modelFile) {
    auto& ctx = AppContext::getInstance();
    std::string engineFilePath;
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();
    
    std::cout << "[Pipeline] loadEngine called with: " << modelFile << std::endl;
    std::cout << "[Pipeline] File extension: " << extension << std::endl;

    if (extension == ".engine") {
        engineFilePath = modelFile;
        std::cout << "[Pipeline] Using engine file directly: " << engineFilePath << std::endl;
    } else if (extension == ".onnx") {
        // generate engine filename with resolution and precision suffixes
        std::string baseName = modelPath.stem().string();
        baseName += "_" + std::to_string(ctx.config.onnx_input_resolution);
        if (ctx.config.export_enable_fp16) baseName += "_fp16";
        if (ctx.config.export_enable_fp8)  baseName += "_fp8";
        std::string engineFilename = baseName + ".engine";
        engineFilePath = (modelPath.parent_path() / engineFilename).string();

        if (!fileExists(engineFilePath)) {
            std::cout << "[Pipeline] Building engine from ONNX model" << std::endl;

            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile);
            if (builtEngine) {
                nvinfer1::IHostMemory* serializedEngine = builtEngine->serialize();

                if (serializedEngine) {
                    std::ofstream engineFile(engineFilePath, std::ios::binary);
                    if (engineFile) {
                        engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
                        engineFile.close();
                        
                        std::cout << "[Pipeline] Engine saved to: " << engineFilePath << std::endl;
                    }
                    delete serializedEngine;
                }
                delete builtEngine;
            }
        }
    } else {
        std::cerr << "[Pipeline] Unsupported model format: " << extension << std::endl;
        return false;
    }

    // Load engine from file
    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Suppress TensorRT internal errors
            if (severity <= Severity::kERROR && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
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
        std::cout << "[Pipeline] Engine loaded successfully!" << std::endl;
        return true;
    } else {
        std::cerr << "[Pipeline] Failed to load engine from: " << engineFilePath << std::endl;
        return false;
    }
}

bool UnifiedGraphPipeline::runInferenceAsync(cudaStream_t stream) {
    // Validate TensorRT components
    if (!m_context || !m_engine) {
        std::cerr << "[Pipeline] TensorRT context or engine not initialized" << std::endl;
        return false;
    }
    
    // Use default stream if none provided
    if (!stream) {
        stream = m_primaryStream;
    }
    
    // Clear any previous CUDA errors and check for success
    cudaError_t prevErr = cudaGetLastError();
    if (prevErr != cudaSuccess) {
        std::cerr << "[Pipeline] Previous CUDA error detected: " << cudaGetErrorString(prevErr) << std::endl;
        // Continue execution but log the error
    }
    
    // Optimized tensor address setting with validation
    // Pre-check input bindings before setting addresses
    for (const auto& inputName : m_inputNames) {
        auto bindingIt = m_inputBindings.find(inputName);
        if (bindingIt == m_inputBindings.end() || bindingIt->second == nullptr) {
            std::cerr << "[Pipeline] Input binding not found or null for: " << inputName << std::endl;
            return false;
        }
        
        // Pointer validation removed for performance - validation done at allocation time
        
        if (!m_context->setTensorAddress(inputName.c_str(), bindingIt->second)) {
            std::cerr << "[Pipeline] Failed to set input tensor address for: " << inputName << std::endl;
            return false;
        }
    }
    
    // Set output tensor addresses with validation
    for (const auto& outputName : m_outputNames) {
        auto bindingIt = m_outputBindings.find(outputName);
        if (bindingIt == m_outputBindings.end() || bindingIt->second == nullptr) {
            std::cerr << "[Pipeline] Output binding not found or null for: " << outputName << std::endl;
            return false;
        }
        
        // Pointer validation removed for performance - validation done at allocation time
        
        if (!m_context->setTensorAddress(outputName.c_str(), bindingIt->second)) {
            std::cerr << "[Pipeline] Failed to set output tensor address for: " << outputName << std::endl;
            return false;
        }
    }
    
    // Validate input data integrity before inference
    if (m_inputBindings.find(m_inputName) != m_inputBindings.end()) {
        cudaError_t memErr = cudaGetLastError();
        if (memErr != cudaSuccess) {
            std::cerr << "[Pipeline] CUDA memory error before inference: " << cudaGetErrorString(memErr) << std::endl;
            return false;
        }
    }
    
    
    // Execute TensorRT inference with enhanced error handling
    
    bool success = m_context->enqueueV3(stream);
    if (!success) {
        // Get more detailed error information
        cudaError_t cudaErr = cudaGetLastError();
        std::cerr << "[Pipeline] TensorRT inference failed";
        if (cudaErr != cudaSuccess) {
            std::cerr << " - CUDA error: " << cudaGetErrorString(cudaErr);
        }
        std::cerr << std::endl;
        
        std::cout << "[DEBUG] TensorRT inference failed with unknown error" << std::endl;
        
        return false;
    }
     
    // Record inference completion event for pipeline synchronization
    // Event recording removed - simple pipeline
    if (false) {
        cudaError_t eventErr = cudaSuccess;
        if (eventErr != cudaSuccess) {
            std::cerr << "[Pipeline] Failed to record detection event: " << cudaGetErrorString(eventErr) << std::endl;
            // Continue execution as this is not critical
        }
    }
    
    return true;
}

void UnifiedGraphPipeline::performIntegratedPostProcessing(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Ensure we have valid output names from TensorRT inference
    if (m_outputNames.empty()) {
        std::cerr << "[Pipeline] No output names found for post-processing." << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    // Get primary output from TensorRT inference
    const std::string& primaryOutputName = m_outputNames[0];
    void* d_rawOutputPtr = m_outputBindings[primaryOutputName];
    nvinfer1::DataType outputType = m_outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = m_outputShapes[primaryOutputName];

    if (!d_rawOutputPtr) {
        std::cerr << "[Pipeline] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    // Clear all detection buffers at the start of processing
    if (m_d_decodedCount) {
        cudaMemsetAsync(m_d_decodedCount, 0, sizeof(int), stream);
    }
    if (m_d_classFilteredCount) {
        cudaMemsetAsync(m_d_classFilteredCount, 0, sizeof(int), stream);
    }
    if (m_d_finalTargetsCount) {
        cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
    }

    // Use cached config values for CUDA Graph compatibility
    static int cached_max_detections = Constants::MAX_DETECTIONS;
    static float cached_nms_threshold = 0.45f;
    static float cached_confidence_threshold = 0.001f;
    static std::string cached_postprocess = "yolo12";
    
    // Update cache from config when not in graph capture mode
    if (!m_graphCaptured) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_max_detections = ctx.config.max_detections;
        cached_nms_threshold = ctx.config.nms_threshold;
        cached_confidence_threshold = ctx.config.confidence_threshold;
        cached_postprocess = ctx.config.postprocess;
    }

    // Step 1: Decode YOLO output based on model type
    int maxDecodedTargets = 300;  // Reasonable buffer for detections
    cudaError_t decodeErr = cudaSuccess;
    
    
    if (cached_postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        decodeErr = decodeYolo10Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            m_numClasses,
            cached_confidence_threshold,
            m_imgScale,
            m_d_decodedTargets,
            m_d_decodedCount,
            maxDecodedTargets,
            max_candidates,
            stream);
    } else if (cached_postprocess == "yolo8" || cached_postprocess == "yolo9" || 
               cached_postprocess == "yolo11" || cached_postprocess == "yolo12") {
        int max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        // Validate parameters before calling
        if (!m_d_decodedTargets || !m_d_decodedCount) {
            std::cerr << "[Pipeline] Target buffers not allocated!" << std::endl;
            if (m_d_finalTargetsCount) {
                cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
            }
            return;
        }
        
        if (max_candidates <= 0 || maxDecodedTargets <= 0) {
            std::cerr << "[Pipeline] Invalid buffer sizes: max_candidates=" << max_candidates 
                      << ", maxDecodedTargets=" << maxDecodedTargets << std::endl;
            if (m_d_finalTargetsCount) {
                cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
            }
            return;
        }
        
        decodeErr = decodeYolo11Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            m_numClasses,
            cached_confidence_threshold,
            m_imgScale,
            m_d_decodedTargets,
            m_d_decodedCount,
            maxDecodedTargets,
            max_candidates,
            stream);
    } else {
        std::cerr << "[Pipeline] Unsupported post-processing type: " << cached_postprocess << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    if (decodeErr != cudaSuccess) {
        std::cerr << "[Pipeline] GPU decoding failed: " << cudaGetErrorString(decodeErr) << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    // Step 2: Continue processing - early exit checks will be done by GPU kernels

    // Step 3: Class ID filtering
    if (m_d_classFilteredTargets && m_d_classFilteredCount && m_d_allowFlags) {
        // Update class filtering flags from config
        unsigned char h_allowFlags[Constants::MAX_CLASSES_FOR_FILTERING];
        memset(h_allowFlags, 0, Constants::MAX_CLASSES_FOR_FILTERING);
        
        // Copy class settings to allow flags
        for (const auto& setting : ctx.config.class_settings) {
            if (setting.id >= 0 && setting.id < Constants::MAX_CLASSES_FOR_FILTERING) {
                h_allowFlags[setting.id] = setting.allow ? 1 : 0;
            }
        }
        
        // Copy to GPU
        cudaMemcpyAsync(m_d_allowFlags, h_allowFlags, Constants::MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
        
        cudaError_t filterErr = filterTargetsByClassIdGpu(
            m_d_decodedTargets,
            m_d_decodedCount,
            m_d_classFilteredTargets,
            m_d_classFilteredCount,
            m_d_allowFlags,
            Constants::MAX_CLASSES_FOR_FILTERING,
            300,  // max output buffer
            stream
        );
        
        if (filterErr != cudaSuccess) {
            std::cerr << "[Pipeline] Class ID filtering failed: " << cudaGetErrorString(filterErr) << std::endl;
            if (m_d_finalTargetsCount) {
                cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
            }
            return;
        }
    } else {
        // No class filtering - copy decoded targets directly (max buffer size)
        if (m_d_classFilteredTargets && m_d_classFilteredCount) {
            cudaMemcpyAsync(m_d_classFilteredTargets, m_d_decodedTargets, 
                          300 * sizeof(Target), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(m_d_classFilteredCount, m_d_decodedCount, 
                          sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Step 4: Determine NMS input (class filtered targets or decoded targets)
    Target* nmsInputTargets = m_d_classFilteredTargets ? m_d_classFilteredTargets : m_d_decodedTargets;
    int* nmsInputCount = m_d_classFilteredCount ? m_d_classFilteredCount : m_d_decodedCount;

    // TODO: Color filtering will be implemented when color mask integration is complete
    // For now, we skip color filtering step

    // Step 5: NMS (Non-Maximum Suppression)
    if (m_d_finalTargets && m_d_finalTargetsCount && 
        m_d_x1 && m_d_y1 && m_d_x2 && m_d_y2 && m_d_areas && 
        m_d_scores_nms && m_d_classIds_nms && m_d_iou_matrix && 
        m_d_keep && m_d_indices) {
        
        // Use cached frame dimensions for CUDA Graph compatibility
        static int cached_frame_width = ctx.config.onnx_input_resolution;
        static int cached_frame_height = ctx.config.onnx_input_resolution;
        if (!m_graphCaptured) {
            cached_frame_width = ctx.config.detection_resolution;
            cached_frame_height = ctx.config.detection_resolution;
        }
        
        // Get count for NMS input - use pinned memory for async transfer
        static int* h_inputCount_pinned = nullptr;
        if (!h_inputCount_pinned) {
            cudaHostAlloc(&h_inputCount_pinned, sizeof(int), cudaHostAllocDefault);
        }
        
        // Copy count async without blocking
        cudaMemcpyAsync(h_inputCount_pinned, nmsInputCount, sizeof(int), cudaMemcpyDeviceToHost, stream);
        
        // Continue processing without waiting - use previous frame's count
        static int last_input_count = 0;
        
        try {
            // Use previous frame's count to avoid sync
            // The actual count will be ready for next frame
            NMSGpu(
                nmsInputTargets,
                last_input_count,  // Use previous frame's count to avoid sync
                m_d_finalTargets,
                m_d_finalTargetsCount,
                cached_max_detections,
                cached_nms_threshold,
                cached_frame_width,
                cached_frame_height,
                m_d_x1,
                m_d_y1,
                m_d_x2,
                m_d_y2,
                m_d_areas,
                m_d_scores_nms,
                m_d_classIds_nms,
                m_d_iou_matrix,
                m_d_keep,
                m_d_indices,
                stream
            );
            
            // Validate detections on GPU without sync
            validateTargetsGpu(m_d_finalTargets, cached_max_detections, stream);
            
            // Only copy to CPU if preview window is enabled
            if (ctx.config.show_window) {
                // Use static variables to persist between frames
                static int h_finalCount = 0;
                static std::vector<Target> h_finalTargets(cached_max_detections);
                static cudaEvent_t copyEvent = nullptr;
                static bool copyInProgress = false;
                
                // Initialize event once
                if (!copyEvent) {
                    cudaEventCreateWithFlags(&copyEvent, cudaEventDisableTiming);
                }
                
                // Check if previous copy is complete
                if (copyInProgress) {
                    if (cudaEventQuery(copyEvent) == cudaSuccess) {
                        // Previous copy completed, update preview
                        if (h_finalCount > 0 && h_finalCount <= cached_max_detections) {
                            h_finalTargets.resize(h_finalCount);
                            ctx.getDetectionState().updateTargets(h_finalTargets);
                        }
                        copyInProgress = false;
                    }
                }
                
                // Start new copy if not already in progress
                if (!copyInProgress) {
                    // Copy count first (small, fast)
                    cudaMemcpyAsync(&h_finalCount, m_d_finalTargetsCount, sizeof(int), 
                                   cudaMemcpyDeviceToHost, stream);
                    
                    // Copy targets (only up to max_detections)
                    cudaMemcpyAsync(h_finalTargets.data(), m_d_finalTargets, 
                                  cached_max_detections * sizeof(Target), cudaMemcpyDeviceToHost, stream);
                    
                    // Record event for this copy
                    cudaEventRecord(copyEvent, stream);
                    copyInProgress = true;
                }
            }
            
            // Always update input count for next frame (regardless of preview)
            last_input_count = *h_inputCount_pinned;
            
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Exception during NMSGpu: " << e.what() << std::endl;
            if (m_d_finalTargetsCount) {
                cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
            }
            ctx.getDetectionState().clearTargets();
        }
    } else {
        std::cerr << "[Pipeline] NMS buffers not properly allocated!" << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        ctx.getDetectionState().clearTargets();
    }
}

void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Check if we have final targets to select from
    if (!m_d_finalTargets || !m_d_finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }
    
    // Use the new GPU-only version to avoid synchronization
    // The kernel will check count internally
    
    // Get crosshair position (center of screen)
    float crosshairX, crosshairY;
    if (!m_graphCaptured) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        crosshairX = ctx.config.detection_resolution / 2.0f;
        crosshairY = ctx.config.detection_resolution / 2.0f;
    } else {
        // Use cached values for graph mode
        static float cached_crosshair_x = 320.0f;
        static float cached_crosshair_y = 320.0f;
        crosshairX = cached_crosshair_x;
        crosshairY = cached_crosshair_y;
    }
    
    // Ensure target selection buffers are allocated
    if (!m_d_bestTargetIndex || !m_d_bestTarget) {
        std::cerr << "[Pipeline] Target selection buffers not allocated!" << std::endl;
        return;
    }
    
    // Call new GPU target selection with device pointer for count
    cudaError_t selectErr = findClosestTargetGpu(
        m_d_finalTargets,
        m_d_finalTargetsCount,  // Pass device pointer directly
        crosshairX,
        crosshairY,
        m_d_bestTargetIndex,
        m_d_bestTarget,
        stream
    );
    
    if (selectErr != cudaSuccess) {
        std::cerr << "[Pipeline] Target selection failed: " << cudaGetErrorString(selectErr) << std::endl;
        // Clear best target on failure
        if (m_d_bestTargetIndex) {
            cudaMemsetAsync(m_d_bestTargetIndex, -1, sizeof(int), stream);
        }
        if (m_d_bestTarget) {
            cudaMemsetAsync(m_d_bestTarget, 0, sizeof(Target), stream);
        }
    }
}

// Non-blocking pipeline execution with improved performance
bool UnifiedGraphPipeline::executeGraphNonBlocking(cudaStream_t stream) {
    if (!stream) stream = m_primaryStream;
    auto& ctx = AppContext::getInstance();
    
    // Get current and previous pipeline indices
    int currentIdx = m_currentPipelineIdx.load();
    int prevIdx = m_prevPipelineIdx.load();
    
    if (!m_tripleBuffer) {
        // Fallback to original blocking execution if triple buffer not available
        std::cerr << "[UnifiedGraph] Warning: Triple buffer not available, falling back to blocking mode" << std::endl;
        return false;
    }
    
    // Step 1: Process mouse movement from PREVIOUS frame results (non-blocking)
    processMouseMovementAsync();
    
    // Step 2: Launch GPU work for CURRENT frame
    // First, capture the frame from D3D11 texture (skip if frame already set by setInputFrame)
    if (m_config.enableCapture && m_cudaResource && !m_hasFrameData) {
        // Use current buffer for capture
        SimpleCudaMat& currentBuffer = m_tripleBuffer->buffers[currentIdx];
        
        // Capture current desktop frame using Desktop Duplication
        if (m_desktopDuplication && m_d3dDevice && m_d3dContext && m_captureTextureD3D) {
            auto* duplication = static_cast<IDXGIOutputDuplication*>(m_desktopDuplication);
            auto* d3dContext = static_cast<ID3D11DeviceContext*>(m_d3dContext);
            auto* captureTexture = static_cast<ID3D11Texture2D*>(m_captureTextureD3D);
            
            DXGI_OUTDUPL_FRAME_INFO frameInfo;
            IDXGIResource* desktopResource = nullptr;
            HRESULT hr = duplication->AcquireNextFrame(1, &frameInfo, &desktopResource);
            
            if (SUCCEEDED(hr) && desktopResource) {
                ID3D11Texture2D* desktopTexture = nullptr;
                hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&desktopTexture);
                if (SUCCEEDED(hr) && desktopTexture) {
                    D3D11_TEXTURE2D_DESC desktopDesc;
                    desktopTexture->GetDesc(&desktopDesc);
                    
                    // Use aim_shoot_offset when both aiming and shooting, otherwise use crosshair_offset
                    int centerX, centerY;
                    if (ctx.config.enable_aim_shoot_offset && ctx.aiming && ctx.shooting) {
                        centerX = desktopDesc.Width / 2 + static_cast<int>(ctx.config.aim_shoot_offset_x);
                        centerY = desktopDesc.Height / 2 + static_cast<int>(ctx.config.aim_shoot_offset_y);
                    } else {
                        centerX = desktopDesc.Width / 2 + static_cast<int>(ctx.config.crosshair_offset_x);
                        centerY = desktopDesc.Height / 2 + static_cast<int>(ctx.config.crosshair_offset_y);
                    }
                    int captureSize = ctx.config.detection_resolution;
                    
                    int cropX = std::max(0, centerX - captureSize / 2);
                    int cropY = std::max(0, centerY - captureSize / 2);
                    cropX = std::min(cropX, static_cast<int>(desktopDesc.Width) - captureSize);
                    cropY = std::min(cropY, static_cast<int>(desktopDesc.Height) - captureSize);
                    
                    D3D11_BOX srcBox;
                    srcBox.left = cropX;
                    srcBox.top = cropY;
                    srcBox.right = cropX + captureSize;
                    srcBox.bottom = cropY + captureSize;
                    srcBox.front = 0;
                    srcBox.back = 1;
                    
                    d3dContext->CopySubresourceRegion(captureTexture, 0, 0, 0, 0, 
                                                     desktopTexture, 0, &srcBox);
                    desktopTexture->Release();
                }
                desktopResource->Release();
                duplication->ReleaseFrame();
            }
        }
        
        cudaGetLastError();
        
        cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, stream);
        if (err == cudaSuccess) {
            cudaArray_t array;
            err = cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0);
            if (err == cudaSuccess) {
                err = cudaMemcpy2DFromArrayAsync(
                    currentBuffer.data(),
                    currentBuffer.step(),
                    array,
                    0, 0,
                    currentBuffer.cols() * sizeof(uchar4),
                    currentBuffer.rows(),
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            }
            cudaGraphicsUnmapResources(1, &m_cudaResource, stream);
        }
        
        if (err != cudaSuccess) {
            printf("[ERROR] Capture failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        m_hasFrameData = true;
    }
    
    // Step 3: Execute detection and inference pipeline on current frame
    if (!ctx.getDetectionState().isPaused()) {
        // Copy current buffer to main capture buffer for processing
        if (!m_tripleBuffer->buffers[currentIdx].empty()) {
            size_t dataSize = m_tripleBuffer->buffers[currentIdx].rows() * 
                             m_tripleBuffer->buffers[currentIdx].cols() * 
                             m_tripleBuffer->buffers[currentIdx].channels() * sizeof(unsigned char);
            
            cudaMemcpyAsync(m_captureBuffer.data(), 
                           m_tripleBuffer->buffers[currentIdx].data(), 
                           dataSize, cudaMemcpyDeviceToDevice, stream);
        }
        
        // Unified Preprocessing: BGRA → RGB + Resize + Normalize + HWC→CHW
        if (m_d_yoloInput && !m_captureBuffer.empty()) {
            cudaError_t err = cuda_unified_preprocessing(
                m_captureBuffer.data(),
                m_d_yoloInput,
                m_captureBuffer.cols(),
                m_captureBuffer.rows(),
                static_cast<int>(m_captureBuffer.step()),
                ctx.config.onnx_input_resolution,
                ctx.config.onnx_input_resolution,
                stream
            );
            
            if (err != cudaSuccess) {
                printf("[ERROR] Unified preprocessing failed: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        
        // TensorRT Inference
        if (m_inputBindings.find(m_inputName) != m_inputBindings.end() && m_d_yoloInput) {
            void* inputBinding = m_inputBindings[m_inputName];
            
            if (inputBinding != m_d_yoloInput) {
                size_t inputSize = ctx.config.onnx_input_resolution * ctx.config.onnx_input_resolution * 3 * sizeof(float);
                cudaMemcpyAsync(inputBinding, m_d_yoloInput, inputSize, 
                               cudaMemcpyDeviceToDevice, stream);
            }
            
            if (!runInferenceAsync(stream)) {
                std::cerr << "[UnifiedGraph] TensorRT inference failed in executeGraphNonBlocking" << std::endl;
                return false;
            }
            
            // Connect TensorRT output to inference buffer
            if (m_d_inferenceOutput && !m_outputNames.empty()) {
                auto bindingIt = m_outputBindings.find(m_outputNames[0]);
                if (bindingIt != m_outputBindings.end() && bindingIt->second != nullptr) {
                    size_t outputSize = m_outputSizes[m_outputNames[0]];
                    cudaMemcpyAsync(m_d_inferenceOutput, bindingIt->second, outputSize, 
                                   cudaMemcpyDeviceToDevice, stream);
                }
            }
            
            // Integrated post-processing (decode, filter, NMS, target selection)
            performIntegratedPostProcessing(stream);
            performTargetSelection(stream);
            
            // Step 4: Copy target results to host memory (async) using pinned memory
            Target* finalTarget = m_d_bestTarget;
            if (finalTarget && m_tripleBuffer->h_target_coords_pinned[currentIdx]) {
                cudaMemcpyAsync(m_tripleBuffer->h_target_coords_pinned[currentIdx], finalTarget, sizeof(Target), 
                               cudaMemcpyDeviceToHost, stream);
                
                // Record event when target data is ready
                cudaEventRecord(m_tripleBuffer->target_ready_events[currentIdx], stream);
                m_tripleBuffer->target_data_valid[currentIdx] = true;
                
                // Check if we have valid targets
                if (m_d_finalTargetsCount) {
                    int targetCount = 0;
                    cudaMemcpyAsync(&targetCount, m_d_finalTargetsCount, sizeof(int), 
                                   cudaMemcpyDeviceToHost, stream);
                    m_tripleBuffer->target_count[currentIdx] = targetCount;
                } else {
                    m_tripleBuffer->target_count[currentIdx] = 1;
                }
            }
        }
    }
    
    // Step 5: Advance pipeline indices for next frame
    int nextIdx = (currentIdx + 1) % 3;
    m_prevPipelineIdx.store(currentIdx);
    m_currentPipelineIdx.store(nextIdx);
    
    // Step 6: Check if next buffer is ready without blocking
    int nextNextIdx = (nextIdx + 1) % 3;
    if (m_tripleBuffer->target_data_valid[nextNextIdx]) {
        // Use non-blocking query instead of synchronize
        cudaError_t status = cudaEventQuery(m_tripleBuffer->target_ready_events[nextNextIdx]);
        if (status != cudaSuccess && status != cudaErrorNotReady) {
            // Only sync if there's an actual error, not if just not ready
            cudaEventSynchronize(m_tripleBuffer->target_ready_events[nextNextIdx]);
        }
    }
    
    m_state.frameCount++;
    m_hasFrameData = false;
    
    return true;
}

void UnifiedGraphPipeline::processMouseMovementAsync() {
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.aiming || !m_tripleBuffer) {
        return;
    }
    
    int prevIdx = m_prevPipelineIdx.load();
    
    // Check if previous frame's target data is ready (non-blocking)
    if (!m_tripleBuffer->target_data_valid[prevIdx] || 
        m_tripleBuffer->target_count[prevIdx] == 0) {
        return;
    }
    
    cudaError_t eventStatus = cudaEventQuery(m_tripleBuffer->target_ready_events[prevIdx]);
    if (eventStatus != cudaSuccess) {
        return; // Data not ready yet, skip this frame
    }
    
    // Target data is ready, use pinned memory
    Target* h_target_ptr = m_tripleBuffer->h_target_coords_pinned[prevIdx];
    if (!h_target_ptr) {
        return; // Pinned memory not initialized
    }
    Target& h_target = *h_target_ptr;
    
    // Ensure center is calculated
    h_target.updateCenter();
    
    // Apply head/body offset to target center
    float targetCenterX = h_target.center_x;
    float targetCenterY;
    
    // Find head class ID from class_settings
    int head_class_id = -1;
    for(const auto& cs : ctx.config.class_settings) {
        if (cs.name == ctx.config.head_class_name) {
            head_class_id = cs.id;
            break;
        }
    }
    
    // Check if this is a head or body target and apply appropriate offset
    if (h_target.classId == head_class_id) {
        targetCenterY = h_target.y + h_target.height * ctx.config.head_y_offset;
    } else {
        targetCenterY = h_target.y + h_target.height * ctx.config.body_y_offset;
    }
    
    // Calculate mouse movement
    float screenCenterX = ctx.config.detection_resolution / 2.0f;
    float screenCenterY = ctx.config.detection_resolution / 2.0f;
    
    float error_x = targetCenterX - screenCenterX;
    float error_y = targetCenterY - screenCenterY;
    
    // Use PD controller for precise, stable movement
    float movement_x, movement_y;
    
    // Simple PD control parameters
    float kp_x = ctx.config.pd_kp_x;
    float kp_y = ctx.config.pd_kp_y;
    // kd_x and kd_y removed - not used in simple proportional control
    float deadzone_x = ctx.config.min_movement_threshold_x;
    float deadzone_y = ctx.config.min_movement_threshold_y;
    
    // Apply deadzone
    if (abs(error_x) < deadzone_x) error_x = 0;
    if (abs(error_y) < deadzone_y) error_y = 0;
    
    // Simple proportional control (temporary until GPU PD controller is integrated)
    movement_x = error_x * kp_x;
    movement_y = error_y * kp_y;
    
    int dx = static_cast<int>(movement_x);
    int dy = static_cast<int>(movement_y);
    
    // Execute mouse movement (non-blocking)
    if (dx != 0 || dy != 0) {
        cuda::executeMouseMovementFromGPU(dx, dy);
    }
    
    // Mark this buffer's data as processed
    m_tripleBuffer->target_data_valid[prevIdx] = false;
}

} // namespace needaimbot