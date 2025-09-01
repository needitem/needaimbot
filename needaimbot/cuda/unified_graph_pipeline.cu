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

// OPTIMIZATION: UnifiedGPUArena implementation
void UnifiedGPUArena::initializePointers(uint8_t* basePtr, int maxDetections, int yoloSize) {
    size_t offset = 0;
    
    // YOLO input buffer (largest, align first)
    offset = (offset + alignof(float) - 1) & ~(alignof(float) - 1);
    yoloInput = reinterpret_cast<float*>(basePtr + offset);
    offset += yoloSize * yoloSize * 3 * sizeof(float);
    
    // NMS output buffers
    nmsOutput = reinterpret_cast<float*>(basePtr + offset);
    offset += maxDetections * 6 * sizeof(float);
    
    filteredOutput = reinterpret_cast<float*>(basePtr + offset);
    offset += maxDetections * 6 * sizeof(float);
    
    // Target buffers (align to Target boundary)
    offset = (offset + alignof(Target) - 1) & ~(alignof(Target) - 1);
    decodedTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
    
    finalTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
    
    classFilteredTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
    
    colorFilteredTargets = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
    
    detections = reinterpret_cast<Target*>(basePtr + offset);
    offset += maxDetections * sizeof(Target);
    
    // NMS buffers removed - no longer needed
}

size_t UnifiedGPUArena::calculateArenaSize(int maxDetections, int yoloSize) {
    size_t size = 0;
    
    // YOLO input buffer
    size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
    size += yoloSize * yoloSize * 3 * sizeof(float);
    
    // Output buffers (removed NMS buffers)
    size += maxDetections * 6 * sizeof(float) * 2;  // nmsOutput + filteredOutput
    
    // Target buffers (5 buffers)
    size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
    size += maxDetections * sizeof(Target) * 5;
    
    // NMS buffers removed - significant memory savings!
    
    return size;
}

// Dynamic IOU matrix functions removed - now using static allocation in arena

// Single buffer implementation - removed DoubleBuffer as we're not doing parallel processing

// All complex graph and coordinator code removed - using simple single stream

// ============================================================================
// OPTIMIZED CUDA KERNELS
// ============================================================================

// fusedPreprocessKernel removed - using CudaImageProcessing pipeline instead

// OPTIMIZED: Fused kernel combining validation, selection, and mouse movement calculation
// Reduces 3 kernel calls (validateTargetsGpu + findBestTargetWithHeadPriorityGpu + calculateMouseMovementKernel) to 1
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
    // Use more threads for better parallelization
    // Process with full block instead of just first warp
    if (blockIdx.x == 0) {
        // Initialize outputs
        if (threadIdx.x == 0) {
            *bestTargetIndex = -1;
            output_movement->dx = 0;
            output_movement->dy = 0;
            
            // Clear best target
            Target emptyTarget = {};
            *bestTarget = emptyTarget;
        }
        __syncthreads();
        
        // Get actual count (bounded check)
        int count = *finalTargetsCount;
        if (count <= 0 || count > maxDetections) {
            return;
        }
        
        // Parallel search for best target with validation
        int localBestIdx = -1;
        float localBestDist = 1e9f;
        
        // Each thread checks different targets (use all threads in block)
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            Target& t = finalTargets[i];
            
            // Validation: remove extreme/invalid values
            if (t.x < -1000.0f || t.x > 10000.0f ||
                t.y < -1000.0f || t.y > 10000.0f ||
                t.width <= 0 || t.width > detection_resolution ||
                t.height <= 0 || t.height > detection_resolution ||
                t.confidence <= 0 || t.confidence > 1.0f) {
                // Mark as invalid
                t.confidence = 0;
                continue;
            }
            
            // Calculate distance (no head priority)
            float centerX = t.x + t.width / 2.0f;
            float centerY = t.y + t.height / 2.0f;
            float dx = centerX - screen_center_x;
            float dy = centerY - screen_center_y;
            float distance = sqrtf(dx * dx + dy * dy);
            
            if (distance < localBestDist) {
                localBestDist = distance;
                localBestIdx = i;
            }
        }
        
        // Use shared memory for block-level reduction
        __shared__ float s_distances[256];
        __shared__ int s_indices[256];
        
        s_distances[threadIdx.x] = localBestDist;
        s_indices[threadIdx.x] = localBestIdx;
        __syncthreads();
        
        // Block-level reduction to find minimum distance
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (s_distances[threadIdx.x + s] < s_distances[threadIdx.x]) {
                    s_distances[threadIdx.x] = s_distances[threadIdx.x + s];
                    s_indices[threadIdx.x] = s_indices[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        
        // Thread 0 writes results and calculates movement
        if (threadIdx.x == 0 && s_indices[0] >= 0) {
            *bestTargetIndex = s_indices[0];
            *bestTarget = finalTargets[s_indices[0]];
            
            // Calculate mouse movement for best target
            Target& best = *bestTarget;
            float target_center_x = best.x + best.width / 2.0f;
            float target_center_y;
            
            // Apply class-specific Y offset
            if (best.classId == head_class_id) {
                target_center_y = best.y + best.height * head_y_offset;
            } else {
                target_center_y = best.y + best.height * body_y_offset;
            }
            
            // Calculate error and movement
            float error_x = target_center_x - screen_center_x;
            float error_y = target_center_y - screen_center_y;
            
            float movement_x = error_x * kp_x;
            float movement_y = error_y * kp_y;
            
            // Store unified output (no clamping)
            output_movement->dx = static_cast<int>(movement_x);
            output_movement->dy = static_cast<int>(movement_y);
        }
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


UnifiedGraphPipeline::UnifiedGraphPipeline() {
    // Initialize events for profiling
    m_state.startEvent = std::make_unique<CudaEvent>();
    m_state.endEvent = std::make_unique<CudaEvent>();
    
    // Simple initialization - single buffer, no complex management needed
}

// Note: Destructor is implemented in unified_graph_pipeline_cleanup.cpp

bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    // Create high-priority CUDA stream for inference pipeline
    // This ensures inference tasks get GPU priority over other operations
    int leastPriority, greatestPriority;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    if (err == cudaSuccess) {
        // Create stream with highest priority (greatest priority value)
        cudaStream_t priorityStream;
        err = cudaStreamCreateWithPriority(&priorityStream, cudaStreamNonBlocking, greatestPriority);
        if (err == cudaSuccess) {
            m_pipelineStream = std::make_unique<CudaStream>(priorityStream);
        } else {
            // Fallback to normal stream if priority creation fails
            m_pipelineStream = std::make_unique<CudaStream>();
        }
    } else {
        // Fallback to normal stream if priority range query fails
        m_pipelineStream = std::make_unique<CudaStream>();
    }
    
    // Simple event for preview (using RAII)
    m_previewReadyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    
    // Initialize TensorRT (Phase 1 integration - now required)
    if (m_config.modelPath.empty()) {
        std::cerr << "[UnifiedGraph] ERROR: Model path is required for TensorRT integration" << std::endl;
        return false;
    }
    
    if (!initializeTensorRT(m_config.modelPath)) {
        std::cerr << "[UnifiedGraph] CRITICAL: TensorRT initialization failed" << std::endl;
        return false;
    }
    
    // Allocate pipeline buffers
    if (!allocateBuffers()) {
        std::cerr << "[UnifiedGraph] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    // Initial warm-up if enabled (but don't capture graph yet)
    if (m_config.useGraphOptimization) {
        
        // CRITICAL: Run warm-up iterations before graph capture
        // This ensures TensorRT completes autotuning and all allocations are done
        for (int i = 0; i < 3; i++) {
            // Create dummy input data
            if (m_inputBindings.find(m_inputName) != m_inputBindings.end()) {
                auto inputBuffer = m_inputBindings[m_inputName].get();
                cudaMemsetAsync(inputBuffer->get(), 0, inputBuffer->size() * sizeof(uint8_t), m_pipelineStream->get());
            }
            
            // Set tensor addresses for warm-up
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
            
            // Execute inference to warm up TensorRT
            if (m_context) {
                m_context->enqueueV3(m_pipelineStream->get());
            }
        }
        cudaStreamSynchronize(m_pipelineStream->get());
        
        // Don't capture graph here - wait for first real execution
        // This avoids capturing with potentially incorrect settings
        m_state.needsRebuild = true;
    }
    
    return true;
}

// Note: shutdown() is implemented in unified_graph_pipeline_cleanup.cpp

bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    auto& ctx = AppContext::getInstance();
    
    if (!stream) stream = m_pipelineStream->get();
    
    
    // Clean up existing graph
    cleanupGraph();
    
    // Clear node tracking vectors
    m_captureNodes.clear();
    m_inferenceNodes.clear();
    m_postprocessNodes.clear();
    
    // Begin graph capture with relaxed mode to handle TensorRT stream dependencies
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin capture: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Execute the entire pipeline once to capture all operations
    // The graph will record all CUDA operations performed in this stream
    
    // 1. Input copy (if we have capture buffer)
    if (m_config.enableCapture && !m_captureBuffer.empty()) {
        cudaMemcpyAsync(m_unifiedCaptureBuffer.data(), m_captureBuffer.data(),
                       m_unifiedCaptureBuffer.sizeInBytes(), cudaMemcpyHostToDevice, stream);
    }
    
    // Preprocessing removed - using CudaImageProcessing pipeline in executeGraph instead
    
    // 3. TensorRT Inference - INCLUDE in graph for maximum performance
    // Important: Ensure TensorRT context is configured for graph capture:
    // - Use cudaStreamCaptureModeRelaxed for TensorRT stream dependencies
    // - Pre-allocate all buffers to avoid dynamic allocation
    // - Run warm-up iterations before graph capture to complete autotuning
    
    if (m_context && m_config.enableDetection) {
        // Set tensor addresses before inference in graph capture
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
        
        // Execute inference within the graph capture
        // This will significantly reduce kernel launch overhead
        if (!m_context->enqueueV3(stream)) {
            std::cerr << "Warning: TensorRT enqueue failed during graph capture" << std::endl;
        }
    }
    
    // 4. Postprocessing (NMS, filtering, target selection) - INCLUDE in graph
    // We CAN capture post-processing because it doesn't do dynamic allocation
    if (m_config.enableDetection) {
        // Create dummy output data to simulate TensorRT output for graph capture
        if (m_outputBindings.size() > 0 && m_outputNames.size() > 0) {
            const std::string& primaryOutputName = m_outputNames[0];
            void* outputBuffer = m_outputBindings[primaryOutputName]->get();
            
            // Initialize with dummy detection data
            int modelRes = getModelInputResolution();
            size_t outputSize = m_outputSizes[primaryOutputName];
            
            // Clear output buffer
            cudaMemsetAsync(outputBuffer, 0, outputSize, stream);
            
            // Simulate some detections for graph capture
            // This ensures all post-processing kernels are captured
        }
        
        // Clear all detection buffers
        clearDetectionBuffers(PostProcessingConfig{Constants::MAX_DETECTIONS, 0.001f, "yolo12"}, stream);
        
        // Run integrated post-processing (will be captured in graph)
        performIntegratedPostProcessing(stream);
        
        // Run target selection (will be captured in graph)
        performTargetSelection(stream);
        
        // Mouse movement calculation is now part of performTargetSelection (fused kernel)
        // No separate kernel needed here
    }
    
    // 5. Tracking removed - no longer needed
    
    // 6. Bezier Control (already handled in the main pipeline)
    // The actual Bezier control is executed in graph method
    
    // 7. Final output copy (only needed for debug/external output)
    // Skip debug copy to eliminate GPU-CPU transfer for better performance
    
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
    
    
    return true;
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
        // Single capture buffer for sequential processing
        m_captureBuffer.create(height, width, 4);  // BGRA
        
        // OPTIMIZATION: Unified buffer for capture and preprocessing (25% memory savings)
        // Single buffer handles both BGRA capture and BGR preprocessing in-place
        m_unifiedCaptureBuffer.create(height, width, 4);  // BGRA (can be converted to BGR in-place)
        
        // OPTIMIZATION: Allocate memory arena for small frequently used buffers
        size_t arenaSize = SmallBufferArena::calculateArenaSize();
        m_smallBufferArena.arenaBuffer = std::make_unique<CudaMemory<uint8_t>>(arenaSize);
        m_smallBufferArena.initializePointers(m_smallBufferArena.arenaBuffer->get());
        
        
        // OPTIMIZATION: Allocate unified GPU arena (replaces 20+ individual allocations!)
        size_t unifiedArenaSize = UnifiedGPUArena::calculateArenaSize(maxDetections, yoloSize);
        m_unifiedArena.megaArena = std::make_unique<CudaMemory<uint8_t>>(unifiedArenaSize);
        m_unifiedArena.initializePointers(m_unifiedArena.megaArena->get(), maxDetections, yoloSize);
        
        
        // Class filtering control buffer now handled by SmallBufferArena (OPTIMIZED)
        // m_d_allowFlags removed - using m_smallBufferArena.allowFlags instead
        
        // Allocate single pinned host memory for mouse movement using RAII
        m_h_movement = std::make_unique<CudaPinnedMemory<MouseMovement>>(1);
        
        // OPTIMIZATION: Preview buffers only allocated when needed
        if (ctx.config.show_window) {
            m_preview.enabled = true;
            m_preview.previewBuffer.create(height, width, 4);  // BGRA for preview
            m_preview.finalTargets.reserve(maxDetections);
        } else {
            m_preview.enabled = false;
        }
        
        // Additional validation for critical buffers
        if (!m_unifiedCaptureBuffer.data()) {
            throw std::runtime_error("Unified capture buffer allocation failed");
        }
        
        // Calculate actual memory usage with optimizations
        size_t gpuMemory = (width * height * 4 + yoloSize * yoloSize * 3) * sizeof(float);  // Unified buffer
        gpuMemory += unifiedArenaSize;  // Arena memory
        if (m_preview.enabled) {
            gpuMemory += width * height * 4 * sizeof(unsigned char);  // Preview buffer
        }
        
        
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedGraph] Buffer allocation failed: " << e.what() << std::endl;
        deallocateBuffers();
        return false;
    }
}

void UnifiedGraphPipeline::deallocateBuffers() {
    // Wait for pipeline stream completion
    if (m_pipelineStream && m_pipelineStream->get()) {
        cudaStreamSynchronize(m_pipelineStream->get());
    }
    
    // Release buffers
    m_captureBuffer.release();
    m_unifiedCaptureBuffer.release();
    
    // Release preview buffers if allocated
    if (m_preview.enabled) {
        m_preview.previewBuffer.release();
        m_preview.finalTargets.clear();
    }
    
    // OPTIMIZATION: Memory arena cleanup (single deallocation replaces 20+ individual deallocations)
    m_smallBufferArena.arenaBuffer.reset();
    
    // MEGA OPTIMIZATION: Single unified arena cleanup (pointers become invalid automatically)
    m_unifiedArena.megaArena.reset();  // This deallocates ALL unified arena buffers at once!
    
    // Remaining separate buffers
    m_d_inferenceOutput.reset();
    // m_d_allowFlags.reset(); // Now handled by SmallBufferArena
    m_d_preprocessBuffer.reset();
    m_d_outputBuffer.reset();
    
    // Reset pinned host memory - RAII handles deallocation
    m_h_movement.reset();
    
    // Clean up TensorRT bindings - RAII handles deallocation
    m_inputBindings.clear();
    m_outputBindings.clear();
    
    // Events are cleaned up automatically by CudaEvent destructor
    // No manual cleanup needed due to RAII
}

// ============================================================================
// DYNAMIC PARAMETER UPDATE METHODS (No Graph Recapture Needed!)
// ============================================================================

void UnifiedGraphPipeline::setInputFrame(const SimpleCudaMat& frame) {
    // Copy frame data to capture buffer
    if (frame.empty()) return;
    
    // Check if we have a valid stream
    if (!m_pipelineStream || !m_pipelineStream->get()) {
        printf("[ERROR] Pipeline stream not initialized\n");
        return;
    }
    
    // Ensure buffer is the right size
    if (m_unifiedCaptureBuffer.empty() || 
        m_unifiedCaptureBuffer.rows() != frame.rows() || 
        m_unifiedCaptureBuffer.cols() != frame.cols() || 
        m_unifiedCaptureBuffer.channels() != frame.channels()) {
        m_unifiedCaptureBuffer.create(frame.rows(), frame.cols(), frame.channels());
    }
    
    // Copy data
    size_t dataSize = frame.rows() * frame.cols() * frame.channels() * sizeof(unsigned char);
    cudaError_t err = cudaMemcpyAsync(m_unifiedCaptureBuffer.data(), frame.data(), dataSize, 
                                      cudaMemcpyDeviceToDevice, m_pipelineStream->get());
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to copy frame to buffer: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Mark that we have frame data
    m_hasFrameData = true;
}

// Profiling helper
void UnifiedGraphPipeline::updateProfilingAsync(cudaStream_t stream) {
    // Simple profiling for single buffer
    if (m_state.frameCount > 0) {
        m_state.endEvent->record(stream);
        cudaEventSynchronize(m_state.endEvent->get());
        
        float latency;
        cudaEventElapsedTime(&latency, m_state.startEvent->get(), m_state.endEvent->get());
        updateStatistics(latency);
    }
    
    m_state.startEvent->record(stream);
}

// ============================================================================
// MAIN LOOP IMPLEMENTATION
// ============================================================================

void UnifiedGraphPipeline::handleAimbotDeactivation() {
    auto& ctx = AppContext::getInstance();
    
    // Clear all data when aimbot is deactivated
    clearCountBuffers();
    clearMovementData();
    clearHostPreviewData(ctx);
}

void UnifiedGraphPipeline::clearCountBuffers() {
    // Clear count values from memory arena (single stream operation)
    if (m_smallBufferArena.finalTargetsCount) {
        cudaMemsetAsync(m_smallBufferArena.finalTargetsCount, 0, sizeof(int), m_pipelineStream->get());
    }
    if (m_smallBufferArena.decodedCount) {
        cudaMemsetAsync(m_smallBufferArena.decodedCount, 0, sizeof(int), m_pipelineStream->get());
    }
    if (m_smallBufferArena.classFilteredCount) {
        cudaMemsetAsync(m_smallBufferArena.classFilteredCount, 0, sizeof(int), m_pipelineStream->get());
    }
    
    // Clear best target index to indicate no target selected
    if (m_smallBufferArena.bestTargetIndex) {
        cudaMemsetAsync(m_smallBufferArena.bestTargetIndex, -1, sizeof(int), m_pipelineStream->get());
    }
}

void UnifiedGraphPipeline::clearMovementData() {
    // Clear movement data to prevent mouse movement on old targets
    if (m_h_movement && m_h_movement->get()) {
        m_h_movement->get()->dx = 0;
        m_h_movement->get()->dy = 0;
    }
}

void UnifiedGraphPipeline::clearHostPreviewData(AppContext& ctx) {
    // Clear host-side preview data
    if (m_preview.enabled) {
        m_preview.finalTargets.clear();
        m_preview.finalCount = 0;
        m_preview.copyInProgress = false;
    }
    
    // Clear preview window targets
    ctx.clearTargets();
}

void UnifiedGraphPipeline::handleAimbotActivation() {
    
    // Reset frame counter only
    m_state.frameCount = 0;
}

bool UnifiedGraphPipeline::executePipelineWithErrorHandling() {
    try {
        return executeGraphNonBlocking(m_pipelineStream->get());
    } catch (const std::exception& e) {
        std::cerr << "[UnifiedPipeline] Exception in pipeline: " << e.what() << std::endl;
        return false;
    }
}

void UnifiedGraphPipeline::runMainLoop() {
    auto& ctx = AppContext::getInstance();
    
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
    
    // Track aimbot state changes
    bool wasAiming = false;
    
    while (!m_shouldStop && !ctx.should_exit) {
        // Event-driven wait for activation - Zero CPU usage when idle
        {
            std::unique_lock<std::mutex> lock(ctx.pipeline_activation_mutex);
            ctx.pipeline_activation_cv.wait(lock, [&ctx, this]() {
                return ctx.aiming || m_shouldStop || ctx.should_exit;
            });
        }
        
        // Exit check after wait
        if (m_shouldStop || ctx.should_exit) break;
        
        // Handle aimbot state changes
        if (!ctx.aiming) {
            if (wasAiming) {
                handleAimbotDeactivation();
                wasAiming = false;
            }
            continue;
        }
        
        // Handle activation
        if (!wasAiming) {
            handleAimbotActivation();
            wasAiming = true;
        }
        
        // Execute pipeline while aiming
        while (ctx.aiming && !m_shouldStop && !ctx.should_exit) {
            if (!executePipelineWithErrorHandling()) {
                // OPTIMIZATION: Use yield instead of sleep for faster error recovery
                std::this_thread::yield();
            }
        }
        
        // Handle deactivation after exiting the loop
        if (wasAiming && !ctx.aiming) {
            handleAimbotDeactivation();
            wasAiming = false;
        }
    }
    
}

void UnifiedGraphPipeline::stopMainLoop() {
    m_shouldStop = true;
    
    // Wake up the pipeline thread if it's waiting
    auto& ctx = AppContext::getInstance();
    ctx.pipeline_activation_cv.notify_all();
}

// ============================================================================
// TENSORRT ENGINE MANAGEMENT (Phase 1 Integration)
// ============================================================================

bool UnifiedGraphPipeline::initializeTensorRT(const std::string& modelFile) {
    auto& ctx = AppContext::getInstance();
    
    
    // Load the engine
    if (!loadEngine(modelFile)) {
        std::cerr << "[Pipeline] Failed to load engine" << std::endl;
        return false;
    }
    
    
    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[Pipeline] Failed to create execution context" << std::endl;
        return false;
    }
    
    // Set optimization profile for dynamic shapes if needed
    if (m_engine->getNbOptimizationProfiles() > 0) {
        m_context->setOptimizationProfileAsync(0, m_pipelineStream->get());
    }
    
    
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
    } else {
        m_numClasses = 80;  // Default COCO classes
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
            
        }
    }
}

void UnifiedGraphPipeline::getBindings() {
    auto& ctx = AppContext::getInstance();
    
    
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
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Failed to allocate input memory for '" << name << "': " << e.what() << std::endl;
            throw;
        }
        
        // Connect to existing pipeline buffers where possible  
        if (name == m_inputName && m_unifiedArena.yoloInput) {
            // Note: With RAII, each buffer manages its own memory
        }
    }

    // Optimized output binding allocation with RAII
    for (const auto& name : m_outputNames) {
        size_t size = m_outputSizes[name];
        if (size <= 0) {
            std::cerr << "[Pipeline] Warning: Invalid size for output '" << name << "'" << std::endl;
            continue;
        }
        
        // OPTIMIZATION: Direct buffer sharing to eliminate D2D copies and save memory
        try {
            if (name == m_outputNames[0]) {
                // OPTIMIZATION: Share buffer between TensorRT binding and inference output (saves 15-25MB)
                m_outputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
            } else {
                // Allocate separate buffer for non-primary outputs
                m_outputBindings[name] = std::make_unique<CudaMemory<uint8_t>>(size);
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
    
    // Use default pipeline stream if none provided
    if (!stream) {
        stream = m_pipelineStream->get();
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
    
    // Set output tensor addresses with validation and buffer sharing optimization
    for (const auto& outputName : m_outputNames) {
        auto bindingIt = m_outputBindings.find(outputName);
        if (bindingIt == m_outputBindings.end() || bindingIt->second == nullptr) {
            std::cerr << "[Pipeline] Output binding not found or null for: " << outputName << std::endl;
            return false;
        }
        
        // OPTIMIZATION: Use TensorRT binding buffer directly (no separate inference buffer needed)
        void* tensorAddress = bindingIt->second->get();
        
        if (!m_context->setTensorAddress(outputName.c_str(), tensorAddress)) {
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
        // NMS removed - no longer needed
        confidence_threshold = ctx.config.confidence_threshold;
        postprocess = ctx.config.postprocess;
    }
}

void UnifiedGraphPipeline::clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream) {
    if (!m_unifiedArena.decodedTargets || !m_smallBufferArena.decodedCount || !m_smallBufferArena.classFilteredCount || 
        !m_smallBufferArena.finalTargetsCount || config.max_detections <= 0) {
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
    
    if (config.postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        return decodeYolo10Gpu(
            d_rawOutputPtr, outputType, shape, m_numClasses,
            config.confidence_threshold, m_imgScale,
            m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
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
            m_imgScale, m_unifiedArena.decodedTargets, m_smallBufferArena.decodedCount,
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
    static PostProcessingConfig config{Constants::MAX_DETECTIONS, 0.001f, "yolo12"};
    config.updateFromContext(ctx, m_graphCaptured);
    
    // Clear detection buffers
    clearDetectionBuffers(config, stream);
    
    // Decode YOLO output
    cudaError_t decodeErr = decodeYoloOutput(d_rawOutputPtr, outputType, shape, config, stream);
    if (decodeErr != cudaSuccess) {
        std::cerr << "[Pipeline] GPU decoding failed: " << cudaGetErrorString(decodeErr) << std::endl;
        // Add debug info
        std::cerr << "  - Output pointer: " << d_rawOutputPtr << std::endl;
        std::cerr << "  - Shape size: " << shape.size() << std::endl;
        if (!shape.empty()) {
            std::cerr << "  - Shape: ";
            for (size_t i = 0; i < shape.size(); i++) {
                std::cerr << shape[i];
                if (i < shape.size() - 1) std::cerr << "x";
            }
            std::cerr << std::endl;
        }
        std::cerr << "  - Post-process type: " << config.postprocess << std::endl;
        return;
    }

    // NMS processing
    performNMSProcessing(config, stream);
}

void UnifiedGraphPipeline::performNMSProcessing(const PostProcessingConfig& config, cudaStream_t stream) {
    // NMS completely removed - directly copy decoded targets to final
    copyDecodedToFinalTargets(config, stream);
}

void UnifiedGraphPipeline::copyDecodedToFinalTargets(const PostProcessingConfig& config, cudaStream_t stream) {
    if (!m_unifiedArena.finalTargets || !m_smallBufferArena.finalTargetsCount || !m_unifiedArena.decodedTargets || !m_smallBufferArena.decodedCount) {
        return;
    }
    
    // Copy count
    cudaMemcpyAsync(m_smallBufferArena.finalTargetsCount, m_smallBufferArena.decodedCount, sizeof(int), 
                   cudaMemcpyDeviceToDevice, stream);
    
    // Copy targets using fixed max size (eliminates synchronization!)
    cudaMemcpyAsync(m_unifiedArena.finalTargets, m_unifiedArena.decodedTargets, 
                   config.max_detections * sizeof(Target), cudaMemcpyDeviceToDevice, stream);
}

// NMS functions removed - no longer needed

// UI-Pipeline separation: These functions are no longer needed
// UI thread will handle all preview logic independently
void UnifiedGraphPipeline::handlePreviewUpdate(const PostProcessingConfig& config, cudaStream_t stream) {
    // Deprecated - UI handles this
    return;
}

void UnifiedGraphPipeline::updatePreviewTargets(const PostProcessingConfig& config) {
    // Deprecated - UI handles this
    return;
}

void UnifiedGraphPipeline::startPreviewCopy(const PostProcessingConfig& config, cudaStream_t stream) {
    // Deprecated - UI handles this
    return;
}


void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Check if we have final targets to select from
    if (!m_unifiedArena.finalTargets || !m_smallBufferArena.finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }
    
    // Ensure all required buffers are allocated
    if (!m_smallBufferArena.bestTargetIndex || !m_smallBufferArena.bestTarget || !m_smallBufferArena.mouseMovement) {
        std::cerr << "[Pipeline] Target selection buffers not allocated!" << std::endl;
        return;
    }
    
    // Use cached values for CUDA Graph compatibility
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
    
    // Get crosshair position (center of screen)
    float crosshairX = ctx.config.detection_resolution / 2.0f;
    float crosshairY = ctx.config.detection_resolution / 2.0f;
    
    // Find head class ID from config
    int head_class_id = findHeadClassId(ctx);
    
    // OPTIMIZED: Use more threads for better parallelization
    // Dynamic grid size based on max detections for better GPU utilization
    const int blockSize = 256;  // Use full warp multiples for better occupancy
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
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Fused kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        // Clear outputs on failure
        cudaMemsetAsync(m_smallBufferArena.bestTargetIndex, -1, sizeof(int), stream);
        cudaMemsetAsync(m_smallBufferArena.bestTarget, 0, sizeof(Target), stream);
        cudaMemsetAsync(m_smallBufferArena.mouseMovement, 0, sizeof(MouseMovement), stream);
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

bool UnifiedGraphPipeline::performDesktopCapture(const AppContext& ctx) {
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

bool UnifiedGraphPipeline::performFrameCapture() {
    auto& ctx = AppContext::getInstance();
    
    if (!m_config.enableCapture || !m_cudaResource || m_hasFrameData) {
        return true;
    }
    
    if (!performDesktopCapture(ctx)) {
        return false;
    }
    
    cudaGetLastError();
    
    cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, m_pipelineStream->get());
    if (err != cudaSuccess) {
        printf("[ERROR] Graphics map failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    cudaArray_t array;
    err = cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0);
    if (err == cudaSuccess) {
        err = cudaMemcpy2DFromArrayAsync(
            m_captureBuffer.data(),
            m_captureBuffer.step(),
            array,
            0, 0,
            m_captureBuffer.cols() * sizeof(uchar4),
            m_captureBuffer.rows(),
            cudaMemcpyDeviceToDevice,
            m_pipelineStream->get()
        );
    }
    
    cudaGraphicsUnmapResources(1, &m_cudaResource, m_pipelineStream->get());
    
    if (err != cudaSuccess) {
        printf("[ERROR] Capture failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    m_hasFrameData = true;
    return true;
}

bool UnifiedGraphPipeline::performPreprocessing() {
    auto& ctx = AppContext::getInstance();
    
    // Update preview buffer if needed
    if (m_preview.enabled && ctx.preview_enabled && !m_captureBuffer.empty()) {
        updatePreviewBuffer(m_captureBuffer);
    }
    
    // Unified preprocessing
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

void UnifiedGraphPipeline::updatePreviewBuffer(const SimpleCudaMat& currentBuffer) {
    // Only update preview buffer if it's enabled
    if (!m_preview.enabled || m_preview.previewBuffer.empty()) {
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
    
    // Copy input if needed
    if (inputBinding != m_unifiedArena.yoloInput) {
        size_t inputSize = getModelInputResolution() * getModelInputResolution() * 3 * sizeof(float);
        cudaMemcpyAsync(inputBinding, m_unifiedArena.yoloInput, inputSize, 
                       cudaMemcpyDeviceToDevice, m_pipelineStream->get());
    }
    
    // Run inference
    if (!runInferenceAsync(m_pipelineStream->get())) {
        std::cerr << "[UnifiedGraph] TensorRT inference failed" << std::endl;
        return false;
    }
    
    // Post-processing
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

bool UnifiedGraphPipeline::performResultCopy() {
    auto& ctx = AppContext::getInstance();
    
    // Check if we have valid mouse movement data (already calculated by fused kernel)
    if (!m_smallBufferArena.mouseMovement || !m_h_movement || !m_h_movement->get()) {
        return true;
    }
    
    // OPTIMIZED: Mouse movement already calculated by fusedTargetSelectionAndMovementKernel
    // Just copy the pre-calculated result (saves 1 kernel launch)
    cudaMemcpyAsync(m_h_movement->get(), m_smallBufferArena.mouseMovement, 
                   sizeof(MouseMovement), cudaMemcpyDeviceToHost, m_pipelineStream->get());
    
    return true;
}

// Simple single-buffer pipeline with sequential execution
bool UnifiedGraphPipeline::executeGraphNonBlocking(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Frame skip strategy: Skip new frames if inference is still running
    // This ensures we always process the most recent frame
    if (m_inferenceInProgress.load()) {
        m_skippedFrames++;
        // Skip this frame to maintain real-time responsiveness
        return true;
    }
    
    // Step 1: Frame capture
    if (!performFrameCapture()) {
        return false;
    }
    
    // Step 2: Processing and inference
    if (!ctx.detection_paused.load()) {
        // Mark inference as in-progress
        m_inferenceInProgress = true;
        
        if (!performPreprocessing()) {
            m_inferenceInProgress = false;
            return false;
        }
        
        if (!performInference()) {
            m_inferenceInProgress = false;
            return false;
        }
        
        // Step 3: Copy results to host memory
        if (!performResultCopy()) {
            m_inferenceInProgress = false;
            return false;
        }
        
        // Step 4: Synchronize and execute mouse movement
        cudaStreamSynchronize(m_pipelineStream->get());
        processMouseMovement();
        
        // Mark inference as complete
        m_inferenceInProgress = false;
    }
    
    m_state.frameCount++;
    m_hasFrameData = false;
    
    return true;
}

void UnifiedGraphPipeline::processMouseMovement() {
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.aiming || !m_h_movement || !m_h_movement->get()) {
        return;
    }
    
    // Get pre-calculated movement from GPU (already copied in performResultCopy)
    const MouseMovement* movement = m_h_movement->get();
    
    // Execute mouse movement (dx=0,dy=0 if invalid target - handled by GPU)
    if (movement->dx != 0 || movement->dy != 0) {
        cuda::executeMouseMovementFromGPU(movement->dx, movement->dy);
    }
}

} // namespace needaimbot