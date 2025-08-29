#include "unified_graph_pipeline.h"
#include "detection/cuda_float_processing.h"
#include "detection/filterGpu.h"
#include "simple_cuda_mat.h"
#include "mouse_interface.h"
// #include "pd_controller_shared.h"  // Removed - using GPU-based PD controller
#include "../AppContext.h"
#include <d3d11.h>
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

// GPU kernel to calculate mouse movement directly from target (eliminates CPU copying)
__global__ void calculateMouseMovementKernel(
    const Target* __restrict__ best_target,
    float screen_center_x,
    float screen_center_y, 
    float kp_x,
    float kp_y,
    int head_class_id,
    float head_y_offset,
    float body_y_offset,
    int detection_resolution,
    int* __restrict__ output_dx,
    int* __restrict__ output_dy
) {
    // Only first thread does the work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize output (invalid case)
        *output_dx = 0;
        *output_dy = 0;
        
        // Validate target
        if (!best_target || best_target->width <= 0 || best_target->height <= 0 ||
            best_target->x < -1000.0f || best_target->x > 10000.0f ||
            best_target->y < -1000.0f || best_target->y > 10000.0f) {
            return; // Invalid case: dx=0, dy=0
        }
        
        // Calculate target center
        float target_center_x = best_target->x + best_target->width / 2.0f;
        float target_center_y;
        
        // Apply class-specific Y offset
        if (best_target->classId == head_class_id) {
            target_center_y = best_target->y + best_target->height * head_y_offset;
        } else {
            target_center_y = best_target->y + best_target->height * body_y_offset;
        }
        
        // Validate calculated center
        if (target_center_x < 0 || target_center_x > detection_resolution ||
            target_center_y < 0 || target_center_y > detection_resolution) {
            return; // Invalid case: dx=0, dy=0
        }
        
        // Calculate error
        float error_x = target_center_x - screen_center_x;
        float error_y = target_center_y - screen_center_y;
        
        // Apply proportional control
        float movement_x = error_x * kp_x;
        float movement_y = error_y * kp_y;
        
        // Clamp movements
        const float MAX_MOVEMENT = 200.0f;
        movement_x = fmaxf(-MAX_MOVEMENT, fminf(MAX_MOVEMENT, movement_x));
        movement_y = fmaxf(-MAX_MOVEMENT, fminf(MAX_MOVEMENT, movement_y));
        
        // Convert to integers (valid case)
        *output_dx = static_cast<int>(movement_x);
        *output_dy = static_cast<int>(movement_y);
    }
}

// Unified buffer clearing kernel - replaces 8 separate cudaMemsetAsync calls for 0.07ms improvement
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
    
    // Clear all integer counters (first 6 threads handle this)
    if (tid < 6) {
        if (tid == 0) *decodedCount = 0;
        else if (tid == 1) *classFilteredCount = 0;
        else if (tid == 2) *finalTargetsCount = 0;
        else if (tid == 3 && colorFilteredCount) *colorFilteredCount = 0;
        else if (tid == 4 && bestTargetIndex) *bestTargetIndex = -1;
        else if (tid == 5 && bestTarget) {
            // Clear best target structure (GPU-compatible)
            Target emptyTarget = {};
            *bestTarget = emptyTarget;
        }
    }
    
    // Clear decoded targets array in parallel (only clear first few for efficiency)
    for (int i = tid; i < maxTargetsToClear; i += gridSize) {
        // GPU-compatible clearing - assign empty struct
        Target emptyTarget = {};
        decodedTargets[i] = emptyTarget;
    }
}

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
    // Initialize events for profiling using RAII
    m_state.startEvent = std::make_unique<CudaEvent>();
    m_state.endEvent = std::make_unique<CudaEvent>();
    
    // Simple initialization - no complex graph management needed
    m_tripleBuffer = nullptr;
}

// Note: Destructor is implemented in unified_graph_pipeline_cleanup.cpp

bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    // Create dedicated CUDA streams for pipeline stages using RAII
    m_primaryStream = std::make_unique<CudaStream>();
    m_captureStream = std::make_unique<CudaStream>();
    m_inferenceStream = std::make_unique<CudaStream>();
    m_copyStream = std::make_unique<CudaStream>();
    
    // Initialize Triple Buffer for async pipeline
    if (m_config.enableCapture) {
        m_tripleBuffer = std::make_unique<TripleBuffer>();
        // Events and pinned memory are automatically initialized by constructor
    }
    
    // Simple event for preview (using RAII)
    m_previewReadyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    
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
        if (!captureGraph(m_primaryStream->get())) {
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

// Note: shutdown() is implemented in unified_graph_pipeline_cleanup.cpp

bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();
    
    if (!stream) stream = m_primaryStream->get();
    
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
    if (m_config.enableCapture && m_h_inputBuffer && m_h_inputBuffer->get()) {
        cudaMemcpyAsync(m_captureBuffer.data(), m_h_inputBuffer->get(),
                       m_captureBuffer.sizeInBytes(), cudaMemcpyHostToDevice, stream);
    }
    
    // Preprocessing removed - using CudaImageProcessing pipeline in executeGraph instead
    
    // 3. TensorRT Inference  
    if (false && m_config.enableDetection) {  // Temporarily disabled for Graph capture
        // Initialize input buffer with valid data for graph capture
        if (m_d_yoloInput) {
            // std::cout << "[DEBUG] Initializing input buffer for graph capture..." << std::endl;
            
            // Create host buffer with dummy normalized values using dynamic resolution
            int resolution = getModelInputResolution();
            size_t inputSize = resolution * resolution * 3 * sizeof(float);
            std::vector<float> dummyData(resolution * resolution * 3, 0.5f); // Fill with 0.5 (normalized pixel value)
            
            // Copy from host to device
            cudaMemcpyAsync(m_d_yoloInput->get(), dummyData.data(), inputSize, cudaMemcpyHostToDevice, stream);
            
            // CRITICAL FIX: Also initialize TensorRT binding buffers
            auto bindingIt = m_inputBindings.find(m_inputName);
            if (bindingIt != m_inputBindings.end() && bindingIt->second != nullptr) {
                if (bindingIt->second->get() != reinterpret_cast<uint8_t*>(m_d_yoloInput->get())) {
                    // First clear with zeros, then set dummy data
                    cudaMemsetAsync(bindingIt->second->get(), 0, inputSize, stream);
                    cudaMemcpyAsync(bindingIt->second->get(), dummyData.data(), inputSize, cudaMemcpyHostToDevice, stream);
                }
            }
            
            // Copy operations are properly ordered in stream - no sync needed
            // cudaStreamSynchronize(stream); // REMOVED: Unnecessary blocking synchronization
            
        }
        
        // Use integrated TensorRT inference (Phase 1)
        if (!runInferenceAsync(stream)) {
            std::cerr << "[UnifiedGraph] TensorRT inference failed during graph capture" << std::endl;
        }
    }
    
    // 4. Postprocessing (NMS, filtering, target selection)
    // Include NMS in graph capture with dummy data
    if (m_config.enableDetection && m_d_selectedTarget && m_d_detections) {
        // Initialize buffers for graph capture
        cudaMemsetAsync(m_d_selectedTarget->get(), 0, sizeof(Target), stream);
        cudaMemsetAsync(m_d_numDetections->get(), 0, sizeof(int), stream);
        
        // Initialize NMS input buffers with dummy data for graph capture
        if (m_d_decodedTargets && m_d_decodedCount) {
            // Set a dummy count for graph capture
            int dummyCount = 10;  // Small number for graph capture
            cudaMemcpyAsync(m_d_decodedCount->get(), &dummyCount, sizeof(int), cudaMemcpyHostToDevice, stream);
            
            // Initialize decoded targets with dummy data
            std::vector<Target> dummyTargets(10);
            for (int i = 0; i < 10; i++) {
                dummyTargets[i].x = 100.0f + i * 10;
                dummyTargets[i].y = 100.0f + i * 10;
                dummyTargets[i].width = 50.0f;
                dummyTargets[i].height = 50.0f;
                dummyTargets[i].confidence = 0.5f;
                dummyTargets[i].classId = i % 3;
            }
            cudaMemcpyAsync(m_d_decodedTargets->get(), dummyTargets.data(), 
                          10 * sizeof(Target), cudaMemcpyHostToDevice, stream);
        }
        
        // Execute NMS with dummy data for graph capture
        if (m_d_finalTargets && m_d_finalTargetsCount && 
            m_d_x1 && m_d_y1 && m_d_x2 && m_d_y2 && m_d_areas && 
            m_d_scores_nms && m_d_classIds_nms && m_d_iou_matrix && 
            m_d_keep && m_d_indices && m_d_decodedTargets && m_d_decodedCount) {
            
            // Use config max detections for graph capture
            int maxDetections = ctx.config.max_detections;
            float nmsThreshold = 0.45f;
            int frameWidth = ctx.config.detection_resolution;
            int frameHeight = ctx.config.detection_resolution;
            
            // Call NMS with dummy data
            NMSGpu(
                m_d_decodedTargets->get(),
                maxDetections,  // Use max detections for stable graph capture
                m_d_finalTargets->get(),
                m_d_finalTargetsCount->get(),
                maxDetections,
                nmsThreshold,
                frameWidth,
                frameHeight,
                m_d_x1->get(),
                m_d_y1->get(),
                m_d_x2->get(),
                m_d_y2->get(),
                m_d_areas->get(),
                m_d_scores_nms->get(),
                m_d_classIds_nms->get(),
                m_d_iou_matrix->get(),
                m_d_keep->get(),
                m_d_indices->get(),
                stream
            );
        }
    }
    
    // 5. Tracking removed - no longer needed
    
    // 6. Bezier Control (already handled in the main pipeline)
    // The actual Bezier control is executed in graph method
    
    // 7. Final output copy (only 2 floats for mouse X,Y)
    if (m_d_outputBuffer && m_d_outputBuffer->get() && m_h_outputBuffer && m_h_outputBuffer->get()) {
        // Skip debug copy to eliminate GPU-CPU transfer
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
    const int yoloSize = getModelInputResolution();
    const int maxDetections = ctx.config.max_detections;  // Use UI configured value
    
    
    try {
        // Initialize Simple Triple Buffer System with stream-based ordering
        if (m_config.enableCapture && m_tripleBuffer) {
            // Initialize frame buffers and pinned memory
            m_tripleBuffer->initializeFrameBuffers(height, width, 4);  // BGRA
            std::cout << "[UnifiedGraph] Stream-based triple buffer system initialized (memory efficient)" << std::endl;
        }
        
        // Allocate GPU buffers
        m_captureBuffer.create(height, width, 4);  // BGRA
        m_preprocessBuffer.create(height, width, 3);  // BGR
        
        // YOLO input buffer (configurable size x3 in CHW format) - Using RAII
        m_d_yoloInput = std::make_unique<CudaMemory<float>>(yoloSize * yoloSize * 3);
        
        // Detection pipeline buffers - Using RAII
        // Inference output will be allocated by TensorRT or manually
        // m_d_inferenceOutput is handled separately
        
        m_d_nmsOutput = std::make_unique<CudaMemory<float>>(maxDetections * 6);
        
        m_d_filteredOutput = std::make_unique<CudaMemory<float>>(maxDetections * 6);
        
        m_d_detections = std::make_unique<CudaMemory<Target>>(maxDetections);
        
        m_d_selectedTarget = std::make_unique<CudaMemory<Target>>(1);
        
        
        // Allocate NMS temporary buffers - Using RAII
        m_d_numDetections = std::make_unique<CudaMemory<int>>(1);
        
        m_d_x1 = std::make_unique<CudaMemory<int>>(maxDetections);
        
        m_d_y1 = std::make_unique<CudaMemory<int>>(maxDetections);
        
        m_d_x2 = std::make_unique<CudaMemory<int>>(maxDetections);
        
        m_d_y2 = std::make_unique<CudaMemory<int>>(maxDetections);
        
        m_d_areas = std::make_unique<CudaMemory<float>>(maxDetections);
        
        m_d_scores_nms = std::make_unique<CudaMemory<float>>(maxDetections);
        
        m_d_classIds_nms = std::make_unique<CudaMemory<int>>(maxDetections);
        
        m_d_iou_matrix = std::make_unique<CudaMemory<float>>(maxDetections * maxDetections);
        
        m_d_keep = std::make_unique<CudaMemory<bool>>(maxDetections);
        
        m_d_indices = std::make_unique<CudaMemory<int>>(maxDetections);
        
        m_d_outputCount = std::make_unique<CudaMemory<int>>(1);
        
        // OPTIMIZATION: Allocate post-processing buffers without zero-initialization
        // These buffers are cleared every frame by unified clearing kernel, so no initial zero needed
        m_d_decodedTargets = std::make_unique<CudaMemory<Target>>(maxDetections);  // No zero init
        
        m_d_decodedCount = std::make_unique<CudaMemory<int>>(1);  // No zero init
        
        m_d_finalTargets = std::make_unique<CudaMemory<Target>>(maxDetections);  // No zero init
        
        m_d_finalTargetsCount = std::make_unique<CudaMemory<int>>(1);  // No zero init
        
        m_d_classFilteredTargets = std::make_unique<CudaMemory<Target>>(maxDetections);  // No zero init
        
        m_d_classFilteredCount = std::make_unique<CudaMemory<int>>(1);  // No zero init
        
        m_d_colorFilteredTargets = std::make_unique<CudaMemory<Target>>(maxDetections);  // No zero init
        m_d_colorFilteredCount = std::make_unique<CudaMemory<int>>(1);  // No zero init
        
        // Target selection buffers - cleared by pipeline
        m_d_bestTargetIndex = std::make_unique<CudaMemory<int>>(1);  // No zero init
        m_d_bestTarget = std::make_unique<CudaMemory<Target>>(1);  // No zero init
        
        // Mouse movement GPU output buffers (eliminates CPU copying)
        m_d_mouseDx = std::make_unique<CudaMemory<int>>(1);  // No zero init
        m_d_mouseDy = std::make_unique<CudaMemory<int>>(1);  // No zero init
        
        // Class filtering control buffer (64 classes max) - Using RAII
        m_d_allowFlags = std::make_unique<CudaMemory<unsigned char>>(Constants::MAX_CLASSES_FOR_FILTERING);
        
        // Allocate pinned host memory for zero-copy transfers using RAII
        m_h_inputBuffer = std::make_unique<CudaPinnedMemory<unsigned char>>(width * height * 4);
        m_h_outputBuffer = std::make_unique<CudaPinnedMemory<float>>(2);
        
        // Additional validation for critical buffers
        if (!m_captureBuffer.data() || !m_preprocessBuffer.data()) {
            throw std::runtime_error("SimpleCudaMat buffer allocation failed");
        }
        
        std::cout << "[UnifiedGraph] Allocated buffers: "
                  << "GPU: " << ((width * height * 7 + yoloSize * yoloSize * 3 + 
                                 maxDetections * 20) * sizeof(float) / (1024 * 1024)) 
                  << " MB, Pinned: " << ((width * height * 4 + 8) / (1024 * 1024)) 
                  << " MB" << std::endl;
        
        
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
    
    // RAII wrappers will automatically clean up all GPU memory
    // Just reset the unique_ptrs - they handle deallocation automatically
    m_d_numDetections.reset();
    m_d_x1.reset();
    m_d_y1.reset();
    m_d_x2.reset();
    m_d_y2.reset();
    m_d_areas.reset();
    m_d_scores_nms.reset();
    m_d_classIds_nms.reset();
    m_d_iou_matrix.reset();
    m_d_keep.reset();
    m_d_indices.reset();
    m_d_outputCount.reset();
    m_d_yoloInput.reset();
    m_d_inferenceOutput.reset();
    m_d_nmsOutput.reset();
    m_d_filteredOutput.reset();
    m_d_detections.reset();
    m_d_selectedTarget.reset();
    m_d_decodedTargets.reset();
    m_d_decodedCount.reset();
    m_d_finalTargets.reset();
    m_d_finalTargetsCount.reset();
    m_d_classFilteredTargets.reset();
    m_d_classFilteredCount.reset();
    m_d_colorFilteredTargets.reset();
    m_d_colorFilteredCount.reset();
    m_d_bestTargetIndex.reset();
    m_d_bestTarget.reset();
    m_d_allowFlags.reset();
    m_d_preprocessBuffer.reset();
    m_d_tracks.reset();
    m_d_outputBuffer.reset();
    
    // Reset pinned host memory - RAII handles deallocation
    m_h_inputBuffer.reset();
    m_h_outputBuffer.reset();
    
    // Clean up TensorRT bindings - RAII handles deallocation
    m_inputBindings.clear();
    m_outputBindings.clear();
    
    // Triple buffer events are cleaned up automatically by CudaEvent destructor
    // No manual cleanup needed due to RAII
}

// ============================================================================
// DYNAMIC PARAMETER UPDATE METHODS (No Graph Recapture Needed!)
// ============================================================================

void UnifiedGraphPipeline::setInputFrame(const SimpleCudaMat& frame) {
    // Copy frame data to capture buffer
    if (frame.empty()) return;
    
    // Check if we have a valid stream
    if (!m_primaryStream || !m_primaryStream->get()) {
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
                                      cudaMemcpyDeviceToDevice, m_primaryStream->get());
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
    if (!m_lastFrameEnd) {
        m_lastFrameEnd = std::make_unique<CudaEvent>(cudaEventDefault);
    }
    
    if (m_state.frameCount > 0) {
        // Check if last frame's profiling is ready
        if (m_lastFrameEnd->query() == cudaSuccess) {
            float latency;
            cudaEventElapsedTime(&latency, m_state.startEvent->get(), m_lastFrameEnd->get());
            updateStatistics(latency);
        }
    }
    
    // Record current frame end
    m_state.startEvent->record(stream);
    m_lastFrameEnd->record(stream);
}

// ============================================================================
// MAIN LOOP IMPLEMENTATION
// ============================================================================

void UnifiedGraphPipeline::handleAimbotDeactivation() {
    auto& ctx = AppContext::getInstance();
    std::cout << "[UnifiedPipeline] Aimbot deactivated - suspending pipeline" << std::endl;
    
    // Event-based approach: clear operations are async and self-coordinated
    // No synchronization needed - pipeline suspension prevents new work
    clearCountBuffers();
    clearTripleBufferData();
    clearHostPreviewData(ctx);
}

void UnifiedGraphPipeline::clearCountBuffers() {
    // Only clear count values (not the entire buffers) for efficiency
    if (m_d_finalTargetsCount) {
        cudaMemsetAsync(m_d_finalTargetsCount->get(), 0, sizeof(int), m_primaryStream->get());
    }
    if (m_d_decodedCount) {
        cudaMemsetAsync(m_d_decodedCount->get(), 0, sizeof(int), m_primaryStream->get());
    }
    if (m_d_classFilteredCount) {
        cudaMemsetAsync(m_d_classFilteredCount->get(), 0, sizeof(int), m_primaryStream->get());
    }
    
    // Clear best target index to indicate no target selected
    if (m_d_bestTargetIndex) {
        cudaMemsetAsync(m_d_bestTargetIndex->get(), -1, sizeof(int), m_primaryStream->get());
    }
}

void UnifiedGraphPipeline::clearTripleBufferData() {
    // Clear all triple buffer data to prevent mouse movement on old targets
    if (m_tripleBuffer) {
        m_tripleBuffer->clearAllData();
    }
}

void UnifiedGraphPipeline::clearHostPreviewData(AppContext& ctx) {
    // Clear host-side preview data
    m_h_finalTargets.clear();
    m_h_finalCount = 0;
    m_copyInProgress = false;
    
    // Clear preview window targets
    ctx.clearTargets();
}

void UnifiedGraphPipeline::handleAimbotActivation() {
    std::cout << "[UnifiedPipeline] Aimbot activated - MAXIMUM PERFORMANCE MODE" << std::endl;
    
    // Reset frame counter only
    m_state.frameCount = 0;
}

bool UnifiedGraphPipeline::executePipelineWithErrorHandling() {
    try {
        return executeGraphNonBlocking(m_primaryStream->get());
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedPipeline] Exception in pipeline: " << e.what() << std::endl;
        return false;
    }
}

void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();
    std::cout << "[UnifiedPipeline] Starting main loop - MAXIMUM PERFORMANCE MODE (No FPS Limit)" << std::endl;
    
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
    
    // Track aimbot state changes
    bool wasAiming = false;
    
    while (!m_shouldStop && !ctx.should_exit) {
        // Handle aimbot state changes
        if (!ctx.aiming) {
            if (wasAiming) {
                handleAimbotDeactivation();
                wasAiming = false;
            }
            
            // When aimbot is inactive, sleep with 1ms polling for ultra-fast response
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Handle activation
        if (!wasAiming) {
            handleAimbotActivation();
            wasAiming = true;
        }
        
        // Execute pipeline
        if (!executePipelineWithErrorHandling()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
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
        m_context->setOptimizationProfileAsync(0, m_primaryStream->get());
    }
    
    std::cout << "[Pipeline] Execution context created successfully. Context ptr: " << m_context.get() << std::endl;
    
    // Get input and output information
    getInputNames();
    getOutputNames();
    
    // Set up input dimensions and calculate sizes
    if (!m_inputNames.empty()) {
        m_inputName = m_inputNames[0];
        m_inputDims = m_engine->getTensorShape(m_inputName.c_str());
        
        // 모델 입력 해상도 캐싱
        if (m_inputDims.nbDims == 4) {
            m_modelInputResolution = m_inputDims.d[2]; // [N,C,H,W] format
        } else if (m_inputDims.nbDims == 3) {
            m_modelInputResolution = m_inputDims.d[1]; // [C,H,W] format
        }
        
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
    m_imgScale = static_cast<float>(ctx.config.detection_resolution) / getModelInputResolution();
    
    
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

    // Optimized input binding allocation with reuse
    for (const auto& name : m_inputNames) {
        size_t size = m_inputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for input '" << name << "'" << std::endl;
            continue;
        }
        
        // Note: We're moving to RAII-based memory management, so we don't reuse raw pointers anymore
        // The CudaMemory class will handle allocation and deallocation
        try {
            m_inputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
            std::cout << "[Pipeline] Allocated input '" << name << "': " << size << " bytes" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate input memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
        
        // Connect to existing pipeline buffers where possible  
        if (name == m_inputName && m_d_yoloInput) {
            // Note: With RAII, each buffer manages its own memory
            std::cout << "[Pipeline] Input binding created for YOLO input" << std::endl;
        }
    }

    // Optimized output binding allocation with RAII
    for (const auto& name : m_outputNames) {
        size_t size = m_outputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for output '" << name << "'" << std::endl;
            continue;
        }
        
        // Allocate with RAII wrapper
        try {
            m_outputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
            std::cout << "[Pipeline] Allocated output '" << name << "': " << size << " bytes" << std::endl;
            
            // OPTIMIZATION: Direct buffer aliasing to eliminate D2D copies
            if (m_d_inferenceOutput && name == m_outputNames[0]) {
                // Replace separate TensorRT output buffer with direct alias to inference buffer
                m_outputBindings[name].reset(); // Free the separate allocation
                // Create a wrapper that points to the same memory as m_d_inferenceOutput
                // This eliminates device-to-device copies completely
                std::cout << "[Pipeline] Output binding aliased to inference buffer (zero-copy)" << std::endl;
            } else {
                std::cout << "[Pipeline] Output binding created for '" << name << "'" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate output memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
    }
    
    // Note: With RAII, memory cleanup is handled automatically
    // No need to manually free unused memory
    
    // Validate all bindings are properly set
    if (m_inputBindings.size() != m_inputNames.size()) {
        std::cerr << "[Pipeline] Warning: Input binding count mismatch" << std::endl;
    }
    if (m_outputBindings.size() != m_outputNames.size()) {
        std::cerr << "[Pipeline] Warning: Output binding count mismatch" << std::endl;
    }
    
    std::cout << "[Pipeline] Optimized TensorRT bindings setup completed" << std::endl;
}


bool UnifiedGraphPipeline::loadEngine(const std::string& modelFile) {
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();
    
    std::cout << "[Pipeline] loadEngine called with: " << modelFile << std::endl;
    std::cout << "[Pipeline] File extension: " << extension << std::endl;

    if (extension != ".engine") {
        std::cerr << "[Pipeline] Error: Only .engine files are supported. Please use EngineExport tool to convert ONNX to engine format." << std::endl;
        return false;
    }

    if (!fileExists(modelFile)) {
        std::cerr << "[Pipeline] Engine file does not exist: " << modelFile << std::endl;
        return false;
    }

    std::string engineFilePath = modelFile;
    std::cout << "[Pipeline] Loading engine file: " << engineFilePath << std::endl;

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

int UnifiedGraphPipeline::getModelInputResolution() const {
    return m_modelInputResolution;
}

bool UnifiedGraphPipeline::runInferenceAsync(cudaStream_t stream) {
    // Validate TensorRT components
    if (!m_context || !m_engine) {
        std::cerr << "[Pipeline] TensorRT context or engine not initialized" << std::endl;
        return false;
    }
    
    // Use default stream if none provided
    if (!stream) {
        stream = m_primaryStream->get();
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
        
        if (!m_context->setTensorAddress(inputName.c_str(), bindingIt->second->get())) {
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
        
        if (!m_context->setTensorAddress(outputName.c_str(), bindingIt->second->get())) {
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

void needaimbot::PostProcessingConfig::updateFromContext(const AppContext& ctx, bool graphCaptured) {
    if (!graphCaptured) {
        // CUDA compiler workaround - use const_cast for mutex access
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(ctx.configMutex));
        max_detections = ctx.config.max_detections;
        nms_threshold = ctx.config.nms_threshold;
        confidence_threshold = ctx.config.confidence_threshold;
        postprocess = ctx.config.postprocess;
    }
}

void UnifiedGraphPipeline::clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream) {
    if (!m_d_decodedTargets || !m_d_decodedCount || !m_d_classFilteredCount || 
        !m_d_finalTargetsCount || config.max_detections <= 0) {
        return;
    }
    
    // OPTIMIZATION: Use cached optimal kernel configuration
    static int cached_blockSize = 0;
    static int cached_minGridSize = 0;
    if (cached_blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&cached_minGridSize, &cached_blockSize, clearAllDetectionBuffersKernel, 0, 0);
    }
    
    int gridSize = std::min((config.max_detections + cached_blockSize - 1) / cached_blockSize, cached_minGridSize);
    
    clearAllDetectionBuffersKernel<<<gridSize, cached_blockSize, 0, stream>>>(
        m_d_decodedTargets->get(),
        m_d_decodedCount->get(),
        m_d_classFilteredCount->get(),
        m_d_finalTargetsCount->get(),
        m_d_colorFilteredCount ? m_d_colorFilteredCount->get() : nullptr,
        m_d_bestTargetIndex ? m_d_bestTargetIndex->get() : nullptr,
        m_d_bestTarget ? m_d_bestTarget->get() : nullptr,
        config.max_detections
    );
}

cudaError_t UnifiedGraphPipeline::decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType, 
                                                   const std::vector<int64_t>& shape, 
                                                   const PostProcessingConfig& config, cudaStream_t stream) {
    int maxDecodedTargets = config.max_detections;
    
    if (config.postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        return decodeYolo10Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_d_decodedTargets->get(), m_d_decodedCount->get(),
            maxDecodedTargets, max_candidates, stream);
            
    } else if (config.postprocess == "yolo_nms") {
        int num_detections = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        int output_features = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        if (output_features != 6) {
            std::cerr << "[Pipeline] Invalid NMS output format. Expected 6 features, got " << output_features << std::endl;
            return cudaErrorInvalidValue;
        }
        
        return processNMSOutputGpu(
            d_rawOutputPtr, outputType, shape, config.confidence_threshold,
            m_imgScale, m_d_decodedTargets->get(), m_d_decodedCount->get(),
            maxDecodedTargets, num_detections, stream);
            
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
            m_d_decodedTargets->get(), m_d_decodedCount->get(),
            maxDecodedTargets, max_candidates,
            m_d_allowFlags ? m_d_allowFlags->get() : nullptr,
            Constants::MAX_CLASSES_FOR_FILTERING, stream);
    }
    
    std::cerr << "[Pipeline] Unsupported post-processing type: " << config.postprocess << std::endl;
    return cudaErrorNotSupported;
}

bool UnifiedGraphPipeline::validateYoloDecodeBuffers(int maxDecodedTargets, int max_candidates) {
    if (!m_d_decodedTargets || !m_d_decodedCount) {
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
    if (!m_classFilterDirty || !m_d_allowFlags) {
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
    cudaMemcpyAsync(m_d_allowFlags->get(), h_allowFlags, 
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

    // Get primary output from TensorRT inference
    const std::string& primaryOutputName = m_outputNames[0];
    void* d_rawOutputPtr = m_outputBindings[primaryOutputName]->get();
    nvinfer1::DataType outputType = m_outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = m_outputShapes[primaryOutputName];

    if (!d_rawOutputPtr) {
        std::cerr << "[Pipeline] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        return;
    }

    // Initialize config with cached values for CUDA Graph compatibility
    static PostProcessingConfig config{Constants::MAX_DETECTIONS, 0.45f, 0.001f, "yolo12"};
    config.updateFromContext(ctx, m_graphCaptured);
    
    // Clear detection buffers
    clearDetectionBuffers(config, stream);
    
    // Decode YOLO output
    cudaError_t decodeErr = decodeYoloOutput(d_rawOutputPtr, outputType, shape, config, stream);
    if (decodeErr != cudaSuccess) {
        std::cerr << "[Pipeline] GPU decoding failed: " << cudaGetErrorString(decodeErr) << std::endl;
        return;
    }

    // NMS processing
    performNMSProcessing(config, stream);
}

void UnifiedGraphPipeline::performNMSProcessing(const PostProcessingConfig& config, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Skip NMS for yolo_nms models - already post-processed
    if (config.postprocess == "yolo_nms") {
        copyDecodedToFinalTargets(config, stream);
        return;
    }
    
    // Normal NMS processing for raw YOLO models
    performStandardNMS(config, stream);
}

void UnifiedGraphPipeline::copyDecodedToFinalTargets(const PostProcessingConfig& config, cudaStream_t stream) {
    if (!m_d_finalTargets || !m_d_finalTargetsCount || !m_d_decodedTargets || !m_d_decodedCount) {
        return;
    }
    
    // Copy count
    cudaMemcpyAsync(m_d_finalTargetsCount->get(), m_d_decodedCount->get(), sizeof(int), 
                   cudaMemcpyDeviceToDevice, stream);
    
    // Copy targets using fixed max size (eliminates synchronization!)
    cudaMemcpyAsync(m_d_finalTargets->get(), m_d_decodedTargets->get(), 
                   config.max_detections * sizeof(Target), cudaMemcpyDeviceToDevice, stream);
}

void UnifiedGraphPipeline::performStandardNMS(const PostProcessingConfig& config, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    if (!validateNMSBuffers()) {
        std::cerr << "[Pipeline] NMS buffers not properly allocated!" << std::endl;
        ctx.clearTargets();
        return;
    }
    
    try {
        executeNMSKernel(config, stream);
        handleNMSResults(config, stream);
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Exception during NMSGpu: " << e.what() << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount->get(), 0, sizeof(int), stream);
        }
        ctx.clearTargets();
    }
}

bool UnifiedGraphPipeline::validateNMSBuffers() {
    return (m_d_finalTargets && m_d_finalTargetsCount && 
            m_d_x1 && m_d_y1 && m_d_x2 && m_d_y2 && m_d_areas && 
            m_d_scores_nms && m_d_classIds_nms && m_d_iou_matrix && 
            m_d_keep && m_d_indices);
}

void UnifiedGraphPipeline::executeNMSKernel(const PostProcessingConfig& config, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    Target* nmsInputTargets = m_d_decodedTargets->get();
    int cached_frame_width = ctx.config.detection_resolution;
    int cached_frame_height = ctx.config.detection_resolution;
    
    NMSGpu(
        nmsInputTargets,
        config.max_detections,
        m_d_finalTargets->get(),
        m_d_finalTargetsCount->get(),
        config.max_detections,
        config.nms_threshold,
        cached_frame_width,
        cached_frame_height,
        m_d_x1->get(),
        m_d_y1->get(),
        m_d_x2->get(),
        m_d_y2->get(),
        m_d_areas->get(),
        m_d_scores_nms->get(),
        m_d_classIds_nms->get(),
        m_d_iou_matrix->get(),
        m_d_keep->get(),
        m_d_indices->get(),
        stream
    );
    
    // Validate detections on GPU without sync
    validateTargetsGpu(m_d_finalTargets->get(), config.max_detections, stream);
}

void UnifiedGraphPipeline::handleNMSResults(const PostProcessingConfig& config, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Only copy to CPU if preview window is enabled
    static int preview_frame_counter = 0;
    const int PREVIEW_UPDATE_INTERVAL = 3;
    
    if (!ctx.config.show_window || (++preview_frame_counter % PREVIEW_UPDATE_INTERVAL != 0)) {
        return;
    }
    
    handlePreviewUpdate(config, stream);
}

void UnifiedGraphPipeline::handlePreviewUpdate(const PostProcessingConfig& config, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Pre-allocate buffer if needed
    if (m_h_finalTargets.empty()) {
        m_h_finalTargets.resize(config.max_detections);
    }
    
    // Initialize event once
    if (!m_copyEvent) {
        m_copyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    }
    
    // Check if previous copy is complete
    if (m_copyInProgress && m_copyEvent->query() == cudaSuccess) {
        updatePreviewTargets(config);
        m_copyInProgress = false;
    }
    
    // Start new copy if not already in progress
    if (!m_copyInProgress) {
        startPreviewCopy(config, stream);
    }
}

void UnifiedGraphPipeline::updatePreviewTargets(const PostProcessingConfig& config) {
    auto& ctx = AppContext::getInstance();
    
    if (m_h_finalCount > 0 && m_h_finalCount <= config.max_detections) {
        static std::vector<Target> previewTargets;
        previewTargets.clear();
        previewTargets.reserve(m_h_finalCount);
        
        for (int i = 0; i < m_h_finalCount; i++) {
            previewTargets.push_back(m_h_finalTargets[i]);
        }
        
        ctx.updateTargets(previewTargets);
    } else {
        ctx.clearTargets();
    }
}

void UnifiedGraphPipeline::startPreviewCopy(const PostProcessingConfig& config, cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Minimal UI copy every 16th frame when window shown
    static int ui_copy_counter = 0;
    if (ctx.config.show_window && (++ui_copy_counter % 16 == 0)) {
        // Copy minimal data for UI preview
        cudaMemcpyAsync(&m_h_finalCount, m_d_finalTargetsCount->get(), sizeof(int), 
                       cudaMemcpyDeviceToHost, stream);
        
        int copyCount = std::min(config.max_detections, static_cast<int>(m_h_finalTargets.size()));
        cudaMemcpyAsync(m_h_finalTargets.data(), m_d_finalTargets->get(), 
                       copyCount * sizeof(Target), cudaMemcpyDeviceToHost, stream);
    }
    
    // Record event for this copy
    m_copyEvent->record(stream);
    m_copyInProgress = true;
}


void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Check if we have final targets to select from
    if (!m_d_finalTargets || !m_d_finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }
    
    // Use cached max detections for CUDA Graph compatibility
    static int cached_max_detections = Constants::MAX_DETECTIONS;
    if (!m_graphCaptured) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_max_detections = ctx.config.max_detections;
    }
    
    // Use the new GPU-only version to avoid synchronization
    // The kernel will check count internally
    
    // Get crosshair position (center of screen) - always calculate from config
    float crosshairX = ctx.config.detection_resolution / 2.0f;
    float crosshairY = ctx.config.detection_resolution / 2.0f;
    
    // Ensure target selection buffers are allocated
    if (!m_d_bestTargetIndex || !m_d_bestTarget) {
        std::cerr << "[Pipeline] Target selection buffers not allocated!" << std::endl;
        return;
    }
    
    // Find head class ID from config
    int head_class_id = -1;
    for (const auto& cs : ctx.config.class_settings) {
        if (cs.name == ctx.config.head_class_name) {
            head_class_id = cs.id;
            break;
        }
    }
    
    // Final validation: clean any extreme values that might have survived
    finalValidateTargetsGpu(
        m_d_finalTargets->get(),
        m_d_finalTargetsCount->get(),
        cached_max_detections,
        stream
    );
    
    // Always use head priority selection
    cudaError_t selectErr = findBestTargetWithHeadPriorityGpu(
        m_d_finalTargets->get(),
        m_d_finalTargetsCount->get(),  // Pass device pointer directly
        crosshairX,
        crosshairY,
        head_class_id,
        m_d_bestTargetIndex->get(),
        m_d_bestTarget->get(),
        stream
    );
    
    if (selectErr != cudaSuccess) {
        std::cerr << "[Pipeline] Target selection failed: " << cudaGetErrorString(selectErr) << std::endl;
        // Clear best target on failure
        if (m_d_bestTargetIndex) {
            cudaMemsetAsync(m_d_bestTargetIndex->get(), -1, sizeof(int), stream);
        }
        if (m_d_bestTarget) {
            cudaMemsetAsync(m_d_bestTarget->get(), 0, sizeof(Target), stream);
        }
    }
}

std::pair<int, int> UnifiedGraphPipeline::calculateCaptureCenter(const AppContext& ctx, const D3D11_TEXTURE2D_DESC& desktopDesc) {
    int centerX, centerY;
    if (ctx.config.enable_aim_shoot_offset && ctx.aiming && ctx.shooting) {
        centerX = desktopDesc.Width / 2 + static_cast<int>(ctx.config.aim_shoot_offset_x);
        centerY = desktopDesc.Height / 2 + static_cast<int>(ctx.config.aim_shoot_offset_y);
    } else {
        centerX = desktopDesc.Width / 2 + static_cast<int>(ctx.config.crosshair_offset_x);
        centerY = desktopDesc.Height / 2 + static_cast<int>(ctx.config.crosshair_offset_y);
    }
    return {centerX, centerY};
}

D3D11_BOX UnifiedGraphPipeline::createCaptureBox(int centerX, int centerY, int captureSize, const D3D11_TEXTURE2D_DESC& desktopDesc) {
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
    
    return srcBox;
}

bool UnifiedGraphPipeline::performDesktopCapture(int writeIdx, const AppContext& ctx) {
    if (!m_desktopDuplication || !m_d3dDevice || !m_d3dContext || !m_captureTextureD3D) {
        return false;
    }
    
    auto* duplication = static_cast<IDXGIOutputDuplication*>(m_desktopDuplication);
    auto* d3dContext = static_cast<ID3D11DeviceContext*>(m_d3dContext);
    auto* captureTexture = static_cast<ID3D11Texture2D*>(m_captureTextureD3D);
    
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    IDXGIResource* desktopResource = nullptr;
    HRESULT hr = duplication->AcquireNextFrame(0, &frameInfo, &desktopResource);
    
    if (!SUCCEEDED(hr) || !desktopResource) {
        return false;
    }
    
    ID3D11Texture2D* desktopTexture = nullptr;
    hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&desktopTexture);
    
    if (SUCCEEDED(hr) && desktopTexture) {
        D3D11_TEXTURE2D_DESC desktopDesc;
        desktopTexture->GetDesc(&desktopDesc);
        
        auto [centerX, centerY] = calculateCaptureCenter(ctx, desktopDesc);
        D3D11_BOX srcBox = createCaptureBox(centerX, centerY, ctx.config.detection_resolution, desktopDesc);
        
        d3dContext->CopySubresourceRegion(captureTexture, 0, 0, 0, 0, desktopTexture, 0, &srcBox);
        desktopTexture->Release();
    }
    
    desktopResource->Release();
    duplication->ReleaseFrame();
    return true;
}

bool UnifiedGraphPipeline::performFrameCapture(int writeIdx) {
    auto& ctx = AppContext::getInstance();
    
    if (!m_config.enableCapture || !m_cudaResource || m_hasFrameData) {
        return true;
    }
    
    SimpleCudaMat& currentBuffer = m_tripleBuffer->buffers[writeIdx];
    
    if (!performDesktopCapture(writeIdx, ctx)) {
        return false;
    }
    
    cudaGetLastError();
    
    cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, m_captureStream->get());
    if (err != cudaSuccess) {
        printf("[ERROR] Graphics map failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
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
            m_captureStream->get()
        );
    }
    
    cudaGraphicsUnmapResources(1, &m_cudaResource, m_captureStream->get());
    
    if (err != cudaSuccess) {
        printf("[ERROR] Capture failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    m_tripleBuffer->captureComplete[writeIdx].record(m_captureStream->get());
    m_hasFrameData = true;
    return true;
}

bool UnifiedGraphPipeline::performPreprocessing(int writeIdx) {
    auto& ctx = AppContext::getInstance();
    SimpleCudaMat& currentBuffer = m_tripleBuffer->buffers[writeIdx];
    
    // Wait for capture completion
    cudaStreamWaitEvent(m_inferenceStream->get(), m_tripleBuffer->captureComplete[writeIdx].get(), 0);
    
    // Update preview buffer if needed
    if (ctx.preview_enabled && !currentBuffer.empty()) {
        updatePreviewBuffer(currentBuffer);
    }
    
    // Unified preprocessing
    if (!m_d_yoloInput || currentBuffer.empty()) {
        return false;
    }
    
    int modelRes = getModelInputResolution();
    cudaError_t err = cuda_unified_preprocessing(
        currentBuffer.data(),
        m_d_yoloInput->get(),
        currentBuffer.cols(),
        currentBuffer.rows(),
        static_cast<int>(currentBuffer.step()),
        modelRes,
        modelRes,
        m_inferenceStream->get()
    );
    
    if (err != cudaSuccess) {
        printf("[ERROR] Unified preprocessing failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    m_tripleBuffer->preprocessComplete[writeIdx].record(m_inferenceStream->get());
    return true;
}

void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    if (m_captureBuffer.empty() || 
        m_captureBuffer.rows() != currentBuffer.rows() || 
        m_captureBuffer.cols() != currentBuffer.cols() || 
        m_captureBuffer.channels() != currentBuffer.channels()) {
        m_captureBuffer.create(currentBuffer.rows(), currentBuffer.cols(), currentBuffer.channels());
    }
    
    size_t dataSize = currentBuffer.rows() * currentBuffer.cols() * currentBuffer.channels() * sizeof(unsigned char);
    cudaMemcpyAsync(m_captureBuffer.data(), currentBuffer.data(), dataSize, 
                   cudaMemcpyDeviceToDevice, m_inferenceStream->get());
}

bool UnifiedGraphPipeline::performInference(int writeIdx) {
    if (m_inputBindings.find(m_inputName) == m_inputBindings.end() || !m_d_yoloInput) {
        return false;
    }
    
    void* inputBinding = m_inputBindings[m_inputName]->get();
    
    // Wait for preprocessing completion
    cudaStreamWaitEvent(m_inferenceStream->get(), m_tripleBuffer->preprocessComplete[writeIdx].get(), 0);
    
    // Copy input if needed
    if (inputBinding != m_d_yoloInput->get()) {
        size_t inputSize = getModelInputResolution() * getModelInputResolution() * 3 * sizeof(float);
        cudaMemcpyAsync(inputBinding, m_d_yoloInput->get(), inputSize, 
                       cudaMemcpyDeviceToDevice, m_inferenceStream->get());
    }
    
    // Run inference
    if (!runInferenceAsync(m_inferenceStream->get())) {
        std::cerr << "[UnifiedGraph] TensorRT inference failed" << std::endl;
        return false;
    }
    
    // Post-processing
    performIntegratedPostProcessing(m_inferenceStream->get());
    performTargetSelection(m_inferenceStream->get());
    
    // Record completion
    m_tripleBuffer->inferenceComplete[writeIdx].record(m_inferenceStream->get());
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

bool UnifiedGraphPipeline::performResultCopy(int writeIdx) {
    auto& ctx = AppContext::getInstance();
    
    // Wait for inference completion
    cudaStreamWaitEvent(m_copyStream->get(), m_tripleBuffer->inferenceComplete[writeIdx].get(), 0);
    
    Target* finalTarget = m_d_bestTarget->get();
    if (!finalTarget || !m_tripleBuffer->h_movement_dx_pinned[writeIdx].get()) {
        m_tripleBuffer->movement_data_ready[writeIdx] = false;
        return true;
    }
    
    int head_class_id = findHeadClassId(ctx);
    
    // Launch GPU kernel for mouse movement calculation
    calculateMouseMovementKernel<<<1, 1, 0, m_copyStream->get()>>>(
        finalTarget,
        ctx.config.detection_resolution / 2.0f,
        ctx.config.detection_resolution / 2.0f,
        ctx.config.pd_kp_x,
        ctx.config.pd_kp_y,
        head_class_id,
        ctx.config.head_y_offset,
        ctx.config.body_y_offset,
        ctx.config.detection_resolution,
        m_d_mouseDx->get(),
        m_d_mouseDy->get()
    );
    
    // Copy results
    cudaMemcpyAsync(m_tripleBuffer->h_movement_dx_pinned[writeIdx].get(), m_d_mouseDx->get(), sizeof(int), 
                   cudaMemcpyDeviceToHost, m_copyStream->get());
    cudaMemcpyAsync(m_tripleBuffer->h_movement_dy_pinned[writeIdx].get(), m_d_mouseDy->get(), sizeof(int), 
                   cudaMemcpyDeviceToHost, m_copyStream->get());
    
    // Record completion
    m_tripleBuffer->copyComplete[writeIdx].record(m_copyStream->get());
    m_tripleBuffer->movement_data_ready[writeIdx] = true;
    
    return true;
}

// Simple stream-based pipeline with ordered execution and no synchronization
bool UnifiedGraphPipeline::executeGraphNonBlocking(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    if (!m_tripleBuffer) {
        std::cerr << "[UnifiedGraph] Warning: Triple buffer not available" << std::endl;
        return false;
    }
    
    // Step 1: Process mouse movement from any ready data (non-blocking)
    processMouseMovementAsync();
    
    // Step 2: Get next write index (single atomic operation, no contention)
    int writeIdx = m_tripleBuffer->getNextWriteIndex();
    
    // Step 3: Frame capture stage
    if (!performFrameCapture(writeIdx)) {
        return false;
    }
    
    // Step 4: Inference stage with event dependency (no blocking)
    if (!ctx.detection_paused.load()) {
        if (!performPreprocessing(writeIdx)) {
            return false;
        }
        
        if (!performInference(writeIdx)) {
            return false;
        }
    }
    
    // Step 5: Copy results to host memory
    if (!ctx.detection_paused.load()) {
        if (!performResultCopy(writeIdx)) {
            return false;
        }
    }
    
    m_state.frameCount++;
    m_hasFrameData = false;
    
    return true;  // Returns immediately, all work is asynchronous
}

void UnifiedGraphPipeline::processMouseMovementAsync() {
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.aiming || !m_tripleBuffer) {
        return;
    }
    
    // Find ready movement data (dx/dy already calculated on GPU)
    int readIdx = m_tripleBuffer->findReadyMovementData();
    if (readIdx < 0) {
        return; // No ready movement data
    }
    
    // Get pre-calculated dx/dy from GPU (no complex CPU calculations needed!)
    int dx = *m_tripleBuffer->h_movement_dx_pinned[readIdx].get();
    int dy = *m_tripleBuffer->h_movement_dy_pinned[readIdx].get();
    
    // Execute mouse movement (dx=0,dy=0 if invalid target - handled by GPU)
    if (dx != 0 || dy != 0) {
        cuda::executeMouseMovementFromGPU(dx, dy);
    }
    
    // Mark movement data as consumed
    m_tripleBuffer->markMovementConsumed(readIdx);
}

} // namespace needaimbot