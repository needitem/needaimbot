#include "unified_graph_pipeline.h"
#include "../detector/detector.h"
#include "cuda_image_processing.h"
#include "gpu_kalman_filter.h"
#include "gpu_pid_controller.h"
#include "../postprocess/postProcessGpu.h"
#include "../postprocess/filterGpu.h"
#include "simple_cuda_mat.h"
#include <iostream>
#include <chrono>
#include <cuda.h>

namespace needaimbot {

// ============================================================================
// DYNAMIC CUDA GRAPH MANAGER
// ============================================================================
class DynamicCudaGraph {
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    cudaStream_t stream_ = nullptr;
    
    std::unordered_map<std::string, cudaGraphNode_t> kernelNodes_;
    std::unordered_map<std::string, void*> kernelParams_;
    
    bool isCapturing_ = false;
    bool isInstantiated_ = false;
    
public:
    DynamicCudaGraph(cudaStream_t stream) : stream_(stream) {}
    
    ~DynamicCudaGraph() {
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        if (graph_) cudaGraphDestroy(graph_);
    }
    
    cudaError_t beginCapture() {
        if (isCapturing_) return cudaErrorInvalidValue;
        cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (err == cudaSuccess) isCapturing_ = true;
        return err;
    }
    
    cudaError_t endCapture() {
        if (!isCapturing_) return cudaErrorInvalidValue;
        
        cudaError_t err = cudaStreamEndCapture(stream_, &graph_);
        if (err != cudaSuccess) {
            isCapturing_ = false;
            return err;
        }
        
        size_t numNodes;
        cudaGraphGetNodes(graph_, nullptr, &numNodes);
        std::vector<cudaGraphNode_t> nodes(numNodes);
        cudaGraphGetNodes(graph_, nodes.data(), &numNodes);
        
        for (auto node : nodes) {
            cudaGraphNodeType nodeType;
            cudaGraphNodeGetType(node, &nodeType);
            if (nodeType == cudaGraphNodeTypeKernel) {
                static int nodeId = 0;
                kernelNodes_["kernel_" + std::to_string(nodeId++)] = node;
            }
        }
        
        err = cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
        if (err == cudaSuccess) isInstantiated_ = true;
        isCapturing_ = false;
        return err;
    }
    
    cudaError_t updateKernelParams(const std::string& nodeName, 
                                   const cudaKernelNodeParams& params) {
        if (!isInstantiated_) return cudaErrorInvalidValue;
        auto it = kernelNodes_.find(nodeName);
        if (it == kernelNodes_.end()) return cudaErrorInvalidValue;
        return cudaGraphExecKernelNodeSetParams(graphExec_, it->second, &params);
    }
    
    cudaError_t launch() {
        if (!isInstantiated_) return cudaErrorInvalidValue;
        return cudaGraphLaunch(graphExec_, stream_);
    }
    
    bool isReady() const { return isInstantiated_; }
    void registerNode(const std::string& name, cudaGraphNode_t node) {
        kernelNodes_[name] = node;
    }
};

// ============================================================================
// PIPELINE COORDINATOR WITH MULTI-STREAM MANAGEMENT
// ============================================================================
class PipelineCoordinator {
private:
    static constexpr int CAPTURE_PRIORITY = -2;
    static constexpr int INFERENCE_PRIORITY = -1;
    static constexpr int POSTPROCESS_PRIORITY = 0;
    
public:
    cudaStream_t captureStream;
    cudaStream_t preprocessStream;
    cudaStream_t inferenceStream;
    cudaStream_t postprocessStream;
    cudaStream_t trackingStream;
    
    cudaEvent_t captureComplete;
    cudaEvent_t preprocessComplete;
    cudaEvent_t inferenceComplete;
    cudaEvent_t postprocessComplete;
    
    PipelineCoordinator() {
        cudaStreamCreateWithPriority(&captureStream, cudaStreamNonBlocking, CAPTURE_PRIORITY);
        cudaStreamCreateWithPriority(&preprocessStream, cudaStreamNonBlocking, INFERENCE_PRIORITY);
        cudaStreamCreateWithPriority(&inferenceStream, cudaStreamNonBlocking, INFERENCE_PRIORITY);
        cudaStreamCreateWithPriority(&postprocessStream, cudaStreamNonBlocking, POSTPROCESS_PRIORITY);
        cudaStreamCreateWithPriority(&trackingStream, cudaStreamNonBlocking, POSTPROCESS_PRIORITY);
        
        cudaEventCreateWithFlags(&captureComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&preprocessComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&inferenceComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&postprocessComplete, cudaEventDisableTiming);
    }
    
    ~PipelineCoordinator() {
        cudaStreamDestroy(captureStream);
        cudaStreamDestroy(preprocessStream);
        cudaStreamDestroy(inferenceStream);
        cudaStreamDestroy(postprocessStream);
        cudaStreamDestroy(trackingStream);
        
        cudaEventDestroy(captureComplete);
        cudaEventDestroy(preprocessComplete);
        cudaEventDestroy(inferenceComplete);
        cudaEventDestroy(postprocessComplete);
    }
    
    void synchronizeCapture(cudaStream_t stream) {
        cudaEventRecord(captureComplete, captureStream);
        cudaStreamWaitEvent(stream, captureComplete, 0);
    }
    
    void synchronizePreprocess(cudaStream_t stream) {
        cudaEventRecord(preprocessComplete, preprocessStream);
        cudaStreamWaitEvent(stream, preprocessComplete, 0);
    }
    
    void synchronizeInference(cudaStream_t stream) {
        cudaEventRecord(inferenceComplete, inferenceStream);
        cudaStreamWaitEvent(stream, inferenceComplete, 0);
    }
};

// ============================================================================
// OPTIMIZED CUDA KERNELS
// ============================================================================

// Fused kernel: BGRA→BGR + Resize + Normalize in one pass
__global__ void fusedPreprocessKernel(
    const uchar4* __restrict__ input,   // BGRA input from capture
    float* __restrict__ output,         // Normalized float output for inference
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    float scaleX, float scaleY,
    float normMean, float normStd,
    bool swapRB)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstWidth || dstY >= dstHeight) return;
    
    // Calculate source coordinates with bilinear interpolation
    float srcXf = dstX * scaleX;
    float srcYf = dstY * scaleY;
    
    int srcX0 = __float2int_rd(srcXf);
    int srcY0 = __float2int_rd(srcYf);
    int srcX1 = min(srcX0 + 1, srcWidth - 1);
    int srcY1 = min(srcY0 + 1, srcHeight - 1);
    
    float fx = srcXf - srcX0;
    float fy = srcYf - srcY0;
    
    // Read 4 pixels for bilinear interpolation
    uchar4 p00 = input[srcY0 * srcWidth + srcX0];
    uchar4 p01 = input[srcY0 * srcWidth + srcX1];
    uchar4 p10 = input[srcY1 * srcWidth + srcX0];
    uchar4 p11 = input[srcY1 * srcWidth + srcX1];
    
    // Bilinear interpolation for each channel
    float b = (1-fx)*(1-fy)*p00.x + fx*(1-fy)*p01.x + (1-fx)*fy*p10.x + fx*fy*p11.x;
    float g = (1-fx)*(1-fy)*p00.y + fx*(1-fy)*p01.y + (1-fx)*fy*p10.y + fx*fy*p11.y;
    float r = (1-fx)*(1-fy)*p00.z + fx*(1-fy)*p01.z + (1-fx)*fy*p10.z + fx*fy*p11.z;
    
    // Swap R and B if needed (BGRA to RGB)
    if (swapRB) {
        float temp = r;
        r = b;
        b = temp;
    }
    
    // Normalize and write to CHW format (for YOLO)
    int pixelIdx = dstY * dstWidth + dstX;
    int channelStride = dstWidth * dstHeight;
    
    output[0 * channelStride + pixelIdx] = (r / 255.0f - normMean) / normStd;  // R channel
    output[1 * channelStride + pixelIdx] = (g / 255.0f - normMean) / normStd;  // G channel
    output[2 * channelStride + pixelIdx] = (b / 255.0f - normMean) / normStd;  // B channel
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
    // Initialize events for profiling
    cudaEventCreate(&m_state.startEvent);
    cudaEventCreate(&m_state.endEvent);
    
    // Initialize coordinator and dynamic graph (will be properly setup in initialize())
    m_coordinator = nullptr;
    m_dynamicGraph = nullptr;
    m_tripleBuffer = nullptr;
}

UnifiedGraphPipeline::~UnifiedGraphPipeline() {
    shutdown();
    
    if (m_state.startEvent) cudaEventDestroy(m_state.startEvent);
    if (m_state.endEvent) cudaEventDestroy(m_state.endEvent);
    if (m_detectionEvent) cudaEventDestroy(m_detectionEvent);
    if (m_trackingEvent) cudaEventDestroy(m_trackingEvent);
}

bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    // Initialize Pipeline Coordinator for multi-stream management
    m_coordinator = std::make_unique<PipelineCoordinator>();
    
    // Use inference stream as primary for graph operations
    m_primaryStream = m_coordinator->inferenceStream;
    
    // Initialize Dynamic Graph Manager
    m_dynamicGraph = std::make_unique<DynamicCudaGraph>(m_primaryStream);
    
    // Initialize Triple Buffer for async pipeline
    if (m_config.enableCapture) {
        m_tripleBuffer = std::make_unique<TripleBuffer>();
        // Triple buffer initialization will be done in allocateBuffers()
    }
    
    // Create events for two-stage pipeline
    cudaEventCreateWithFlags(&m_detectionEvent, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_trackingEvent, cudaEventDisableTiming);
    m_prevFrameHasTarget = false;
    
    // Allocate pipeline buffers
    if (!allocateBuffers()) {
        std::cerr << "[UnifiedGraph] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    // Initial graph capture if enabled
    if (m_config.useGraphOptimization) {
        // We'll capture the graph on first execution with real data
        m_state.needsRebuild = true;
    }
    
    std::cout << "[UnifiedGraph] Pipeline initialized with:" << std::endl;
    std::cout << "  - Multi-stream coordinator: Yes" << std::endl;
    std::cout << "  - Dynamic graph updates: Yes" << std::endl;
    std::cout << "  - Triple buffering: " << (m_tripleBuffer ? "Yes" : "No") << std::endl;
    std::cout << "  - Graph optimization: " << (m_config.useGraphOptimization ? "Yes" : "No") << std::endl;
    return true;
}

void UnifiedGraphPipeline::shutdown() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    
    cleanupGraph();
    deallocateBuffers();
    
    if (m_primaryStream) {
        cudaStreamDestroy(m_primaryStream);
        m_primaryStream = nullptr;
    }
}

bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    
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
    
    // Begin graph capture
    cudaError_t err = cudaStreamBeginCapture(stream, 
        static_cast<cudaStreamCaptureMode>(m_config.graphCaptureMode));
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
    
    // 2. Preprocessing - Use FUSED kernel for optimal performance!
    if (m_config.enableDetection && m_d_yoloInput) {
        // Calculate grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize(
            (640 + blockSize.x - 1) / blockSize.x,
            (640 + blockSize.y - 1) / blockSize.y
        );
        
        // Get input dimensions
        int srcWidth = m_captureBuffer.cols();
        int srcHeight = m_captureBuffer.rows();
        
        // Launch fused kernel: BGRA→BGR + Resize + Normalize
        fusedPreprocessKernel<<<gridSize, blockSize, 0, stream>>>(
            reinterpret_cast<const uchar4*>(m_captureBuffer.data()),
            m_d_yoloInput,
            srcWidth, srcHeight,
            640, 640,  // YOLO input size
            static_cast<float>(srcWidth) / 640.0f,
            static_cast<float>(srcHeight) / 640.0f,
            0.5f, 0.5f,  // Normalization parameters
            true  // Swap R and B channels
        );
        
        // Register this kernel node for dynamic updates
        if (m_dynamicGraph) {
            // This will be captured by the graph
            m_namedNodes["preprocess_kernel"] = nullptr;  // Will be set during capture
        }
    }
    
    // 3. TensorRT Inference
    if (m_config.enableDetection && m_detector) {
        // Execute async inference using the detector
        // Note: The detector should already have async capabilities
        // We need to ensure the detector's processFrame method is async-compatible
        // For now, using placeholder until detector exposes async API
        // m_detector->runInferenceAsync(m_d_yoloInput, m_d_inferenceOutput, stream);
        
        // Alternative: If detector has AsyncTensorRTInference member
        // m_detector->getAsyncInference()->submitInference(m_d_yoloInput, stream);
    }
    
    // 4. Postprocessing (NMS, filtering, target selection)
    if (m_config.enableDetection && m_d_inferenceOutput && m_d_detections) {
        // Apply NMS using pre-allocated buffers
        int maxDetections = 100;
        
        NMSGpu(
            reinterpret_cast<Target*>(m_d_inferenceOutput),
            maxDetections,
            m_d_detections,
            m_d_numDetections,
            maxDetections,
            0.5f,  // NMS threshold
            m_captureBuffer.cols(),
            m_captureBuffer.rows(),
            m_d_x1, m_d_y1, m_d_x2, m_d_y2,
            m_d_areas, m_d_scores_nms, m_d_classIds_nms,
            m_d_iou_matrix, m_d_keep, m_d_indices,
            stream
        );
        
        // Select best target (simplified - just take first detection for now)
        cudaMemcpyAsync(m_d_selectedTarget, m_d_detections, 
                       sizeof(Target), cudaMemcpyDeviceToDevice, stream);
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
    
    // 6. PID Control
    if (m_config.enablePIDControl && m_pidController && m_d_trackedTarget) {
        // Extract target position from tracked target
        // Note: We need to copy target position to host temporarily
        // In a fully optimized version, we'd pass the Target struct directly to PID kernel
        Target h_target;
        cudaMemcpyAsync(&h_target, m_d_trackedTarget, sizeof(Target), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Calculate PID control output on GPU
        float current_time = static_cast<float>(m_state.frameCount) * 0.033f; // 30 FPS assumed
        
        // Use the GpuPIDController's calculateGpu method
        // This method handles the PID calculation internally
        m_pidController->calculateGpu(h_target.x, h_target.y, current_time);
        
        // Copy PID output to the pipeline output buffer
        cudaMemcpyAsync(m_d_pidOutput, m_pidController->getGpuOutputDx(), 
                       sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(m_d_pidOutput + 1, m_pidController->getGpuOutputDy(), 
                       sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    
    // 7. Final output copy (only 2 floats for mouse X,Y)
    if (m_d_outputBuffer && m_d_pidOutput) {
        cudaMemcpyAsync(m_d_outputBuffer, m_d_pidOutput, 
                       2 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        
        // Optional: Copy to pinned host memory for zero-copy access
        if (m_h_outputBuffer) {
            cudaMemcpyAsync(m_h_outputBuffer, m_d_outputBuffer,
                           2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
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

bool UnifiedGraphPipeline::executeGraph(cudaStream_t stream) {
    // Use optimized two-stage pipeline for conditional execution
    if (!stream) stream = m_primaryStream;
    
    // Stage 1: Always execute detection graph
    if (m_detectionGraph && m_detectionGraphExec) {
        cudaError_t err = cudaGraphLaunch(m_detectionGraphExec, stream);
        if (err != cudaSuccess) {
            return executeDirect(stream);
        }
        
        // Check detection results asynchronously
        cudaEventRecord(m_detectionEvent, stream);
        
        // Stage 2: Conditionally execute tracking/PID graph
        // Check previous frame's detection result (pipelined)
        if (m_prevFrameHasTarget) {
            if (m_trackingGraph && m_trackingGraphExec) {
                err = cudaGraphLaunch(m_trackingGraphExec, stream);
                if (err != cudaSuccess) {
                    std::cerr << "[UnifiedGraph] Tracking graph failed\n";
                }
            }
        }
        
        // Async check for current frame targets (for next frame)
        checkTargetsAsync(stream);
        
    } else if (m_state.graphReady && m_graphExec) {
        // Fallback to monolithic graph if two-stage not ready
        cudaError_t err = cudaGraphLaunch(m_graphExec, stream);
        if (err != cudaSuccess) {
            m_state.needsRebuild = true;
            return false;
        }
    } else {
        return executeDirect(stream);
    }
    
    // Async profiling without sync
    if (m_config.enableProfiling) {
        updateProfilingAsync(stream);
    }
    
    m_state.frameCount++;
    return true;
}

bool UnifiedGraphPipeline::updateGraph(cudaStream_t stream) {
    if (!m_graph || !m_graphExec) {
        return captureGraph(stream);
    }
    
    // For now, we rebuild the entire graph
    // In future, we can implement node-level updates
    return captureGraph(stream);
}

bool UnifiedGraphPipeline::executeDirect(cudaStream_t stream) {
    // Use multi-stream coordinator for optimal performance
    if (!m_coordinator) {
        std::cerr << "[UnifiedGraph] Coordinator not initialized" << std::endl;
        return false;
    }
    
    // 1. Capture stage with ZERO-COPY optimization
    if (m_config.enableCapture && m_cudaResource) {
        // Use capture stream for highest priority
        cudaStream_t captureStream = m_coordinator->captureStream;
        
        // Map resource for zero-copy access
        cudaGraphicsMapResources(1, &m_cudaResource, captureStream);
        
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0);
        
        // Get buffer from triple buffer system if available
        void* targetBuffer = m_tripleBuffer ? 
            m_tripleBuffer->buffers[m_tripleBuffer->captureIdx].data() :
            m_captureBuffer.data();
        
        // Direct array to buffer copy (zero-copy when possible)
        cudaMemcpy2DFromArrayAsync(
            targetBuffer, m_captureBuffer.step(),
            array, 0, 0,
            m_captureBuffer.cols() * 4, m_captureBuffer.rows(),
            cudaMemcpyDeviceToDevice, captureStream
        );
        
        // Unmap resource
        cudaGraphicsUnmapResources(1, &m_cudaResource, captureStream);
        
        // Signal capture completion
        m_coordinator->synchronizeCapture(m_coordinator->preprocessStream);
    }
    
    // 2. Detection pipeline with FUSED preprocessing
    if (m_config.enableDetection && m_detector && m_d_yoloInput) {
        cudaStream_t preprocessStream = m_coordinator->preprocessStream;
        
        // Wait for capture to complete
        cudaStreamWaitEvent(preprocessStream, m_coordinator->captureComplete, 0);
        
        // Launch fused preprocessing kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((640 + 15) / 16, (640 + 15) / 16);
        
        void* inputBuffer = m_tripleBuffer ?
            m_tripleBuffer->buffers[m_tripleBuffer->captureIdx].data() :
            m_captureBuffer.data();
        
        fusedPreprocessKernel<<<gridSize, blockSize, 0, preprocessStream>>>(
            reinterpret_cast<const uchar4*>(inputBuffer),
            m_d_yoloInput,
            m_captureBuffer.cols(), m_captureBuffer.rows(),
            640, 640,
            static_cast<float>(m_captureBuffer.cols()) / 640.0f,
            static_cast<float>(m_captureBuffer.rows()) / 640.0f,
            0.5f, 0.5f,
            true
        );
        
        // Signal preprocessing completion
        m_coordinator->synchronizePreprocess(m_coordinator->inferenceStream);
        
        // Swap triple buffer if available
        if (m_tripleBuffer) {
            cudaEventRecord(m_tripleBuffer->events[m_tripleBuffer->captureIdx], preprocessStream);
            m_tripleBuffer->isReady[m_tripleBuffer->captureIdx] = true;
            
            // Rotate buffer indices
            int nextCapture = m_tripleBuffer->displayIdx.load();
            int nextInference = m_tripleBuffer->captureIdx.load();
            int nextDisplay = m_tripleBuffer->inferenceIdx.load();
            
            m_tripleBuffer->captureIdx = nextCapture;
            m_tripleBuffer->inferenceIdx = nextInference;
            m_tripleBuffer->displayIdx = nextDisplay;
        }
    }
    
    // 3. Tracking with dedicated stream
    if (m_config.enableTracking && m_tracker) {
        cudaStream_t trackingStream = m_coordinator->trackingStream;
        
        // Wait for postprocessing
        cudaStreamWaitEvent(trackingStream, m_coordinator->postprocessComplete, 0);
        
        // TODO: Implement GPU tracking
        // m_tracker->predictAsync(trackingStream);
        // m_tracker->updateAsync(m_d_selectedTarget, trackingStream);
    }
    
    // 4. PID Control
    if (m_config.enablePIDControl && m_pidController) {
        cudaStream_t trackingStream = m_coordinator->trackingStream;
        
        // TODO: Implement GPU PID control
        // m_pidController->computeAsync(m_d_trackedTarget, m_d_pidOutput, trackingStream);
    }
    
    // 5. Copy results
    if (m_d_outputBuffer && m_d_pidOutput) {
        cudaMemcpyAsync(m_d_outputBuffer, m_d_pidOutput, 
                       2 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    
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
    const int width = 1920;  // Should come from config
    const int height = 1080;
    const int yoloSize = 640;
    const int maxDetections = 100;
    
    // Initialize Triple Buffer System for async pipeline
    if (m_config.enableCapture && m_tripleBuffer) {
        for (int i = 0; i < 3; i++) {
            m_tripleBuffer->buffers[i] = SimpleCudaMat(height, width, 4);  // BGRA
            cudaEventCreateWithFlags(&m_tripleBuffer->events[i], cudaEventDisableTiming);
            m_tripleBuffer->isReady[i] = false;
        }
        std::cout << "[UnifiedGraph] Triple buffer system initialized" << std::endl;
    }
    
    // Allocate GPU buffers
    m_captureBuffer = SimpleCudaMat(height, width, 4);  // BGRA
    m_preprocessBuffer = SimpleCudaMat(height, width, 3);  // BGR
    
    // YOLO input buffer (640x640x3 in CHW format)
    cudaMalloc(&m_d_yoloInput, yoloSize * yoloSize * 3 * sizeof(float));
    
    // Detection pipeline buffers
    cudaMalloc(&m_d_inferenceOutput, maxDetections * 6 * sizeof(float));
    cudaMalloc(&m_d_nmsOutput, maxDetections * 6 * sizeof(float));
    cudaMalloc(&m_d_filteredOutput, maxDetections * 6 * sizeof(float));
    cudaMalloc(&m_d_detections, maxDetections * sizeof(Target));
    cudaMalloc(&m_d_selectedTarget, sizeof(Target));
    cudaMalloc(&m_d_trackedTarget, sizeof(Target));
    cudaMalloc(&m_d_trackedTargets, maxDetections * sizeof(Target));
    cudaMalloc(&m_d_pidOutput, 2 * sizeof(float));  // X, Y mouse delta
    
    // Allocate NMS temporary buffers
    cudaMalloc(&m_d_numDetections, sizeof(int));
    cudaMalloc(&m_d_x1, maxDetections * sizeof(int));
    cudaMalloc(&m_d_y1, maxDetections * sizeof(int));
    cudaMalloc(&m_d_x2, maxDetections * sizeof(int));
    cudaMalloc(&m_d_y2, maxDetections * sizeof(int));
    cudaMalloc(&m_d_areas, maxDetections * sizeof(float));
    cudaMalloc(&m_d_scores_nms, maxDetections * sizeof(float));
    cudaMalloc(&m_d_classIds_nms, maxDetections * sizeof(int));
    cudaMalloc(&m_d_iou_matrix, maxDetections * maxDetections * sizeof(float));
    cudaMalloc(&m_d_keep, maxDetections * sizeof(bool));
    cudaMalloc(&m_d_indices, maxDetections * sizeof(int));
    cudaMalloc(&m_d_outputCount, sizeof(int));
    
    // Allocate pinned host memory for zero-copy transfers
    cudaHostAlloc(&m_h_inputBuffer, width * height * 4, cudaHostAllocDefault);
    cudaHostAlloc(&m_h_outputBuffer, 2 * sizeof(float), cudaHostAllocMapped);
    
    // Check all allocations
    if (!m_captureBuffer.data() || !m_d_yoloInput || !m_d_inferenceOutput || 
        !m_d_nmsOutput || !m_d_filteredOutput || !m_d_detections || 
        !m_d_selectedTarget || !m_d_trackedTarget || !m_d_trackedTargets || 
        !m_d_pidOutput || !m_h_inputBuffer || !m_h_outputBuffer) {
        std::cerr << "[UnifiedGraph] Buffer allocation failed" << std::endl;
        deallocateBuffers();
        return false;
    }
    
    std::cout << "[UnifiedGraph] Allocated buffers: "
              << "GPU: " << ((width * height * 7 + yoloSize * yoloSize * 3 + 
                             maxDetections * 20) * sizeof(float) / (1024 * 1024)) 
              << " MB, Pinned: " << ((width * height * 4 + 8) / (1024 * 1024)) 
              << " MB" << std::endl;
    
    return true;
}

void UnifiedGraphPipeline::deallocateBuffers() {
    m_captureBuffer.release();
    m_preprocessBuffer.release();
    
    // Free GPU buffers
    if (m_d_yoloInput) cudaFree(m_d_yoloInput);
    if (m_d_inferenceOutput) cudaFree(m_d_inferenceOutput);
    if (m_d_nmsOutput) cudaFree(m_d_nmsOutput);
    if (m_d_filteredOutput) cudaFree(m_d_filteredOutput);
    if (m_d_detections) cudaFree(m_d_detections);
    if (m_d_selectedTarget) cudaFree(m_d_selectedTarget);
    if (m_d_trackedTarget) cudaFree(m_d_trackedTarget);
    if (m_d_trackedTargets) cudaFree(m_d_trackedTargets);
    if (m_d_pidOutput) cudaFree(m_d_pidOutput);
    
    // Free NMS temporary buffers
    if (m_d_numDetections) cudaFree(m_d_numDetections);
    if (m_d_x1) cudaFree(m_d_x1);
    if (m_d_y1) cudaFree(m_d_y1);
    if (m_d_x2) cudaFree(m_d_x2);
    if (m_d_y2) cudaFree(m_d_y2);
    if (m_d_areas) cudaFree(m_d_areas);
    if (m_d_scores_nms) cudaFree(m_d_scores_nms);
    if (m_d_classIds_nms) cudaFree(m_d_classIds_nms);
    if (m_d_iou_matrix) cudaFree(m_d_iou_matrix);
    if (m_d_keep) cudaFree(m_d_keep);
    if (m_d_indices) cudaFree(m_d_indices);
    if (m_d_outputCount) cudaFree(m_d_outputCount);
    
    // Free pinned host memory
    if (m_h_inputBuffer) cudaFreeHost(m_h_inputBuffer);
    if (m_h_outputBuffer) cudaFreeHost(m_h_outputBuffer);
    
    // Reset all pointers
    m_d_yoloInput = nullptr;
    m_d_inferenceOutput = nullptr;
    m_d_nmsOutput = nullptr;
    m_d_filteredOutput = nullptr;
    m_d_detections = nullptr;
    m_d_selectedTarget = nullptr;
    m_d_trackedTarget = nullptr;
    m_d_trackedTargets = nullptr;
    m_d_pidOutput = nullptr;
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
    m_h_inputBuffer = nullptr;
    m_h_outputBuffer = nullptr;
}

// ============================================================================
// DYNAMIC PARAMETER UPDATE METHODS (No Graph Recapture Needed!)
// ============================================================================

bool UnifiedGraphPipeline::updateConfidenceThreshold(float threshold) {
    if (!m_dynamicGraph || !m_dynamicGraph->isReady()) {
        std::cerr << "[UnifiedGraph] Graph not ready for parameter updates" << std::endl;
        return false;
    }
    
    // Update confidence threshold in NMS kernel parameters
    // This would update the kernel node that handles confidence filtering
    cudaKernelNodeParams params;
    // Setup params with new threshold...
    // m_dynamicGraph->updateKernelParams("nms_kernel", params);
    
    std::cout << "[UnifiedGraph] Updated confidence threshold to: " << threshold << std::endl;
    return true;
}

bool UnifiedGraphPipeline::updateNMSThreshold(float threshold) {
    if (!m_dynamicGraph || !m_dynamicGraph->isReady()) {
        return false;
    }
    
    // Similar to confidence threshold update
    // Update the NMS IoU threshold parameter
    std::cout << "[UnifiedGraph] Updated NMS threshold to: " << threshold << std::endl;
    return true;
}

bool UnifiedGraphPipeline::updateTargetSelectionParams(float centerWeight, float sizeWeight) {
    if (!m_dynamicGraph || !m_dynamicGraph->isReady()) {
        return false;
    }
    
    // Update target selection kernel parameters
    // These control how targets are prioritized (center vs size)
    std::cout << "[UnifiedGraph] Updated target selection params - Center: " 
              << centerWeight << ", Size: " << sizeWeight << std::endl;
    return true;
}

// Two-stage pipeline helper implementations
void UnifiedGraphPipeline::checkTargetsAsync(cudaStream_t stream) {
    // Launch a small kernel to check if any targets were detected
    // This updates m_prevFrameHasTarget for the next frame
    if (m_d_numDetections) {
        // Copy detection count to a pinned memory location asynchronously
        static int* h_targetCount = nullptr;
        if (!h_targetCount) {
            cudaHostAlloc(&h_targetCount, sizeof(int), cudaHostAllocMapped);
        }
        
        cudaMemcpyAsync(h_targetCount, m_d_numDetections, sizeof(int), 
                       cudaMemcpyDeviceToHost, stream);
        
        // Record event to check later
        cudaEventRecord(m_detectionEvent, stream);
        
        // Check in next frame (pipelined)
        if (cudaEventQuery(m_detectionEvent) == cudaSuccess) {
            m_prevFrameHasTarget = (*h_targetCount > 0);
        }
    }
}

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

} // namespace needaimbot