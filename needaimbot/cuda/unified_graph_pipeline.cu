#include "unified_graph_pipeline.h"
#include "detection/cuda_image_processing.h"
#include "tracking/gpu_kalman_filter.h"
#include "detection/postProcessGpu.h"
#include "detection/filterGpu.h"
#include "simple_cuda_mat.h"
#include "mouse_interface.h"
#include "pd_controller_shared.h"
#include "../AppContext.h"
#include "cuda_error_check.h"  // Use existing CUDA error checking macros
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
    // Use LOWEST priority to minimize game interference
    // Higher number = lower priority = less GPU time
    static constexpr int CAPTURE_PRIORITY = 1;  // Lowest priority
    static constexpr int INFERENCE_PRIORITY = 1;
    static constexpr int POSTPROCESS_PRIORITY = 1;
    
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
    
    // Match stable branch: simple division by 255.0f (ignore normMean/normStd)
    output[0 * channelStride + pixelIdx] = r / 255.0f;  // R channel
    output[1 * channelStride + pixelIdx] = g / 255.0f;  // G channel
    output[2 * channelStride + pixelIdx] = b / 255.0f;  // B channel
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
    m_coordinator = new PipelineCoordinator();
    
    // Use inference stream as primary for graph operations
    m_primaryStream = m_coordinator->inferenceStream;
    
    // Initialize Dynamic Graph Manager
    m_dynamicGraph = new DynamicCudaGraph(m_primaryStream);
    
    // Initialize Triple Buffer for async pipeline
    if (m_config.enableCapture) {
        m_tripleBuffer = std::make_unique<TripleBuffer>();
        // Triple buffer initialization will be done in allocateBuffers()
    }
    // Create events for two-stage pipeline
    cudaEventCreateWithFlags(&m_detectionEvent, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_trackingEvent, cudaEventDisableTiming);
    m_prevFrameHasTarget = false;
    
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
        // Capture detection graph immediately for preprocessing
        if (!captureDetectionGraph(m_primaryStream)) {
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
    
    // 2. Preprocessing only for graph capture (detector has dynamic allocations)
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
            false  // Keep BGR order (match stable branch)
        );
        
        // Register this kernel node for dynamic updates
        if (m_dynamicGraph) {
            // This will be captured by the graph
            m_namedNodes["preprocess_kernel"] = nullptr;  // Will be set during capture
        }
    }
    
    // 3. TensorRT Inference
    if (m_config.enableDetection) {
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
    // The actual Bezier control is executed in executeDirect method
    
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

bool UnifiedGraphPipeline::captureDetectionGraph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    
    if (!stream) stream = m_primaryStream;
    
    
    // Clean up existing detection graph
    if (m_detectionGraphExec) {
        cudaGraphExecDestroy(m_detectionGraphExec);
        m_detectionGraphExec = nullptr;
    }
    if (m_detectionGraph) {
        cudaGraphDestroy(m_detectionGraph);
        m_detectionGraph = nullptr;
    }
    
    // Begin graph capture for detection only
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to begin detection graph capture: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Capture only preprocessing operations
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
            false  // Keep BGR order (match stable branch)
        );
    }
    
    // End capture
    err = cudaStreamEndCapture(stream, &m_detectionGraph);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to end detection graph capture: " 
                  << cudaGetErrorString(err) << std::endl;
        if (m_detectionGraph) {
            cudaGraphDestroy(m_detectionGraph);
            m_detectionGraph = nullptr;
        }
        return false;
    }
    
    // Instantiate graph for execution
    err = cudaGraphInstantiate(&m_detectionGraphExec, m_detectionGraph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to instantiate detection graph: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_detectionGraph);
        m_detectionGraph = nullptr;
        return false;
    }
    
    // Get node count for debugging
    size_t numNodes = 0;
    cudaGraphGetNodes(m_detectionGraph, nullptr, &numNodes);
    
    
    return true;
}

void UnifiedGraphPipeline::checkTargetsAsync(cudaStream_t stream) {
    // Check if TensorRT has targets from previous frame (Phase 1)
    // TODO: Implement target checking from integrated TensorRT results
    m_prevFrameHasTarget = false;  // Placeholder until full integration
}

bool UnifiedGraphPipeline::executeGraph(cudaStream_t stream) {
    // Use optimized two-stage pipeline for conditional execution
    if (!stream) stream = m_primaryStream;
    
    // First, capture the frame from D3D11 texture (skip if frame already set by setInputFrame)
    // Note: gpu_only_capture.cpp calls setInputFrame() before executeGraph()
    
    
    if (m_config.enableCapture && m_cudaResource && !m_hasFrameData) {
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
                    // Get desktop texture description to calculate center crop
                    D3D11_TEXTURE2D_DESC desktopDesc;
                    desktopTexture->GetDesc(&desktopDesc);
                    
                    // Calculate center crop coordinates with offset
                    auto& ctx = AppContext::getInstance();
                    int centerX = desktopDesc.Width / 2 + static_cast<int>(ctx.config.crosshair_offset_x);
                    int centerY = desktopDesc.Height / 2 + static_cast<int>(ctx.config.crosshair_offset_y);
                    int captureSize = ctx.config.detection_resolution;  // 320x320
                    
                    // Calculate crop region (ensure it's within bounds)
                    int cropX = std::max(0, centerX - captureSize / 2);
                    int cropY = std::max(0, centerY - captureSize / 2);
                    cropX = std::min(cropX, static_cast<int>(desktopDesc.Width) - captureSize);
                    cropY = std::min(cropY, static_cast<int>(desktopDesc.Height) - captureSize);
                    
                    // Copy only the center region
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
        
        // Clear any previous CUDA errors
        cudaGetLastError();
        
        cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, stream);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(err));
            // Reset CUDA resource to prevent further errors
            m_cudaResource = nullptr;
            return false;
        }
        
        
        cudaArray_t array;
        err = cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaGraphicsSubResourceGetMappedArray failed: %s\n", cudaGetErrorString(err));
            cudaGraphicsUnmapResources(1, &m_cudaResource, stream);
            return false;
        }
        
        
        // Copy from D3D11 texture to our capture buffer
        err = cudaMemcpy2DFromArrayAsync(
            m_captureBuffer.data(),
            m_captureBuffer.step(),
            array,
            0, 0,
            m_captureBuffer.cols() * sizeof(uchar4),
            m_captureBuffer.rows(),
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMemcpy2DFromArrayAsync failed: %s\n", cudaGetErrorString(err));
            cudaGraphicsUnmapResources(1, &m_cudaResource, stream);
            return false;
        }
        
        
        cudaGraphicsUnmapResources(1, &m_cudaResource, stream);
        
        // Mark that we have frame data now
        m_hasFrameData = true;
    } else {
        if (!m_cudaResource) {
            static int warnCounter = 0;
            if (++warnCounter % 300 == 0) {
                printf("[WARNING] m_cudaResource is NULL!\n");
            }
        }
        if (!m_config.enableCapture) {
            static int warnCounter2 = 0;
            if (++warnCounter2 % 300 == 0) {
                printf("[WARNING] enableCapture is false!\n");
            }
        }
    }
    
    // Stage 1: Execute preprocessing graph (if available)
    if (m_detectionGraph && m_detectionGraphExec) {
        cudaError_t err = cudaGraphLaunch(m_detectionGraphExec, stream);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaGraphLaunch failed: %s\n", cudaGetErrorString(err));
            // Reset CUDA error state
            cudaGetLastError();
            return executeDirect(stream);
        }
        
        // Log model input/output after graph execution
        static int graph_log_count = 0;
        if (graph_log_count < 5) {
            cudaStreamSynchronize(stream);
            
            // Log model input
            if (m_d_yoloInput) {
                float first_pixel[3];
                cudaMemcpy(first_pixel, m_d_yoloInput, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "[MODEL INPUT] First pixel BGR: (" 
                          << first_pixel[0] << ", " << first_pixel[1] << ", " << first_pixel[2] 
                          << ") Frame: " << graph_log_count << std::endl;
            }
            
            // Log model output if available
            if (m_d_finalTargets && m_d_finalTargetsCount) {
                int finalCount = 0;
                cudaMemcpy(&finalCount, m_d_finalTargetsCount, sizeof(int), cudaMemcpyDeviceToHost);
                if (finalCount > 0) {
                    Target first_detection;
                    cudaMemcpy(&first_detection, m_d_finalTargets, sizeof(Target), cudaMemcpyDeviceToHost);
                    std::cout << "[MODEL OUTPUT] First detection - x:" << first_detection.x 
                              << " y:" << first_detection.y << " w:" << first_detection.width 
                              << " h:" << first_detection.height << " conf:" << first_detection.confidence 
                              << " class:" << first_detection.classId << " Frame: " << graph_log_count << std::endl;
                }
            }
            
            // Log final selected target if available
            if (m_d_bestTarget && m_d_bestTargetIndex) {
                Target final_target;
                int target_index;
                cudaMemcpy(&final_target, m_d_bestTarget, sizeof(Target), cudaMemcpyDeviceToHost);
                cudaMemcpy(&target_index, m_d_bestTargetIndex, sizeof(int), cudaMemcpyDeviceToHost);
                if (target_index >= 0) {
                    std::cout << "[FINAL TARGET] Selected target - x:" << final_target.x 
                              << " y:" << final_target.y << " w:" << final_target.width 
                              << " h:" << final_target.height << " conf:" << final_target.confidence 
                              << " class:" << final_target.classId << " index:" << target_index 
                              << " Frame: " << graph_log_count << std::endl;
                }
            }
            
            graph_log_count++;
        }
    }
    
    // Always run detector after graph (or without graph)
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.getDetectionState().isPaused()) {
        
        // Phase 2.3: Use integrated TensorRT pipeline with asynchronous execution
        
        // 1. Preprocessing: Convert capture buffer to YOLO input
        if (m_d_yoloInput && !m_captureBuffer.empty()) {
            // Phase 2.3: Optimized preprocessing pipeline
            SimpleCudaMat tempResize, tempFloat;
            tempResize.create(640, 640, 3);
            tempFloat.create(640, 640, 3);
            
            // Convert and resize in pipeline
            SimpleCudaMat bgrBuffer;
            bgrBuffer.create(m_captureBuffer.rows(), m_captureBuffer.cols(), 3);
            CudaImageProcessing::bgra2bgr(m_captureBuffer, bgrBuffer, stream);
            CudaImageProcessing::resize(bgrBuffer, tempResize, 640, 640, stream);
            CudaImageProcessing::convertTo(tempResize, tempFloat, m_imgScale, 0.0f, stream);
            
            size_t inputSize = 640 * 640 * 3 * sizeof(float);
            cudaMemcpyAsync(m_d_yoloInput, tempFloat.data(), inputSize, 
                           cudaMemcpyDeviceToDevice, stream);
        }
        
        // 2. TensorRT Inference
        if (m_inputBindings.find(m_inputName) != m_inputBindings.end() && m_d_yoloInput) {
            void* inputBinding = m_inputBindings[m_inputName];
            if (inputBinding != m_d_yoloInput) {
                size_t inputSize = 640 * 640 * 3 * sizeof(float);
                cudaMemcpyAsync(inputBinding, m_d_yoloInput, inputSize, 
                               cudaMemcpyDeviceToDevice, stream);
            }
            
            if (!runInferenceAsync(stream)) {
                std::cerr << "[UnifiedGraph] TensorRT inference failed in executeGraph" << std::endl;
                return false;
            }
            
            // 3. Integrated post-processing (Phase 3: decode, filter, NMS, target selection)
            performIntegratedPostProcessing(stream);
            
            // 4. Target selection from post-processed results
            performTargetSelection(stream);
        }
        
        // 5. Get final processed results for mouse control
        Target* finalTarget = m_d_bestTarget;  // Use selected target from post-processing
        int targetCount = 0;
        if (m_d_finalTargetsCount) {
            cudaMemcpyAsync(&targetCount, m_d_finalTargetsCount, sizeof(int), 
                           cudaMemcpyDeviceToHost, stream);
        }
        
        // Process mouse movement if target detected AND aiming is active
        if (targetCount > 0 && finalTarget && ctx.aiming) {
            // Get the selected target from integrated post-processing
            Target h_target;
            cudaMemcpyAsync(&h_target, finalTarget, sizeof(Target), 
                           cudaMemcpyDeviceToHost, stream);
            // Sync only for mouse movement - critical for accuracy
            cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                printf("[ERROR] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(syncErr));
                cudaGetLastError(); // Reset error state
                return false;
            }
            
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
                // Head target - apply head offset
                targetCenterY = h_target.y + h_target.height * ctx.config.head_y_offset;
            } else {
                // Body target - apply body offset  
                targetCenterY = h_target.y + h_target.height * ctx.config.body_y_offset;
            }
            
            // Calculate mouse movement with simple percentage approach
            float screenCenterX = ctx.config.detection_resolution / 2.0f;
            float screenCenterY = ctx.config.detection_resolution / 2.0f;
            
            // Calculate error (distance from center)
            float error_x = targetCenterX - screenCenterX;
            float error_y = targetCenterY - screenCenterY;
            
            
            // Use PD controller for precise, stable movement
            float movement_x, movement_y;
            
            // Create PD config from application settings
            cuda::PDConfig pd_config = {
                ctx.config.pd_kp_x,                      // kp_x
                ctx.config.pd_kp_y,                      // kp_y
                ctx.config.pd_kd_x,                      // kd_x
                ctx.config.pd_kd_y,                      // kd_y
                ctx.config.min_movement_threshold_x,     // deadzone_x
                ctx.config.min_movement_threshold_y,     // deadzone_y
                ctx.config.pd_derivative_filter,         // derivative_filter_alpha
                100.0f,                                   // max_output_x
                100.0f                                    // max_output_y
            };
            
            // Calculate movement using PD controller with target ID if available
            int target_id = h_target.id;  // Use tracking ID from Target struct
            
            // Calculate dt based on frame rate (approximate)
            float dt = 1.0f / ctx.config.target_fps;
            if (dt <= 0.0f || dt > 1.0f) dt = 0.016f;  // Default to 60 FPS if invalid
            
            cuda::calculatePDControlWithID(
                target_id, 
                error_x, error_y, 
                movement_x, movement_y, 
                pd_config, 
                dt
            );
            
            int dx = static_cast<int>(movement_x);
            int dy = static_cast<int>(movement_y);
                        
            // Execute mouse movement only when aiming
            if (dx != 0 || dy != 0) {
                cuda::executeMouseMovementFromGPU(dx, dy);
            }
        }
    }
    
    // Check detection results asynchronously
    cudaEventRecord(m_detectionEvent, stream);
    
    // Stage 2: Conditionally execute tracking/PID graph
    // Check previous frame's detection result (pipelined)
    if (m_prevFrameHasTarget) {
        if (m_trackingGraph && m_trackingGraphExec) {
            cudaError_t err = cudaGraphLaunch(m_trackingGraphExec, stream);
            if (err != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Tracking graph failed\n";
            }
        }
    }
    
    // Async check for current frame targets (for next frame)
    checkTargetsAsync(stream);
    
    // Handle fallback cases if detection graph is not available
    if (!m_detectionGraph) {
        if (m_state.graphReady && m_graphExec) {
            // Fallback to monolithic graph if two-stage not ready
            cudaError_t err = cudaGraphLaunch(m_graphExec, stream);
            if (err != cudaSuccess) {
                m_state.needsRebuild = true;
                return false;
            }
            
            // Log model input/output after monolithic graph
            static int mono_graph_log_count = 0;
            if (mono_graph_log_count < 5 && m_d_yoloInput) {
                cudaStreamSynchronize(stream);
                float first_pixel[3];
                cudaMemcpy(first_pixel, m_d_yoloInput, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "[MODEL INPUT] First pixel BGR: (" 
                          << first_pixel[0] << ", " << first_pixel[1] << ", " << first_pixel[2] 
                          << ") Frame: " << mono_graph_log_count << std::endl;
                
                // Also try to log output if available
                if (m_d_finalTargets && m_d_finalTargetsCount) {
                    int finalCount = 0;
                    cudaMemcpy(&finalCount, m_d_finalTargetsCount, sizeof(int), cudaMemcpyDeviceToHost);
                    if (finalCount > 0) {
                        Target first_detection;
                        cudaMemcpy(&first_detection, m_d_finalTargets, sizeof(Target), cudaMemcpyDeviceToHost);
                        std::cout << "[MODEL OUTPUT] First detection - x:" << first_detection.x 
                                  << " y:" << first_detection.y << " w:" << first_detection.width 
                                  << " h:" << first_detection.height << " conf:" << first_detection.confidence 
                                  << " class:" << first_detection.classId << " Frame: " << mono_graph_log_count << std::endl;
                    }
                }
                mono_graph_log_count++;
            }
        } else {
            // No graph available, use direct execution
            return executeDirect(stream);
        }
    }
    
    // Async profiling without sync
    if (m_config.enableProfiling) {
        cudaStreamSynchronize(stream);  // Wait for all operations to complete
        updateProfilingAsync(stream);
    }
    
    m_state.frameCount++;
    
    // Reset frame data flag for next frame
    m_hasFrameData = false;
    
    return true;
}

bool UnifiedGraphPipeline::executeDirect(cudaStream_t stream) {
    static int executeCount = 0;
    executeCount++;
    
    if (executeCount <= 10 || executeCount % 100 == 0) {
        std::cout << "[UnifiedGraph::executeDirect] Call #" << executeCount << std::endl;
    }
    
    // Check for CUDA errors periodically
    if (executeCount % 50 == 0) {  // Check more frequently
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[UnifiedGraph] CUDA error detected at frame " << executeCount 
                      << ": " << cudaGetErrorString(err) << std::endl;
            
            // Get more detailed error info
            const char* errorName = cudaGetErrorName(err);
            std::cerr << "[UnifiedGraph] Error name: " << errorName << std::endl;
            
            // DO NOT reset device - it will invalidate all resources!
            // Just clear the error and continue
            cudaGetLastError(); // Clear error state
            return false;
        }
        
        // Don't do full device sync in production - it kills performance
        // Just check stream status instead
        err = cudaStreamQuery(stream ? stream : m_primaryStream);
        if (err != cudaSuccess && err != cudaErrorNotReady) {
            std::cerr << "[UnifiedGraph] CUDA stream error at frame " << executeCount 
                      << ": " << cudaGetErrorString(err) << std::endl;
            cudaGetLastError(); // Clear error state
            return false;
        }
    }
    
    // Use multi-stream coordinator for optimal performance
    if (!m_coordinator) {
        std::cerr << "[UnifiedGraph] ERROR: Coordinator not initialized!" << std::endl;
        return false;
    }
    
    // 1. Capture stage with ZERO-COPY optimization
    if (m_config.enableCapture && m_cudaResource) {
        // Ensure capture buffer is initialized with correct dimensions
        if (m_captureBuffer.empty() || m_captureBuffer.channels() != 4) {
            // Get capture dimensions from config (passed to GameCapture constructor)
            // These values are set in detector.cpp when creating GameCapture
            int captureWidth = m_config.detectionWidth;  // Using detection width/height as capture size
            int captureHeight = m_config.detectionHeight;
            m_captureBuffer.create(captureHeight, captureWidth, 4);  // BGRA format
            std::cout << "[UnifiedGraph] Initialized capture buffer: " << captureWidth << "x" << captureHeight << " with 4 channels" << std::endl;
        }
        
        // Debug: Check capture buffer state
        if (executeCount <= 5 || executeCount % 100 == 0) {
            std::cout << "[UnifiedGraph] Capture buffer - rows: " << m_captureBuffer.rows() 
                      << ", cols: " << m_captureBuffer.cols() 
                      << ", channels: " << m_captureBuffer.channels() 
                      << ", empty: " << m_captureBuffer.empty() << std::endl;
        }
        
        try {
            // Use capture stream for highest priority
            cudaStream_t captureStream = m_coordinator->captureStream;
            
            // Debug: Log before mapping
            if (executeCount <= 5 || executeCount % 100 == 0) {
                std::cout << "[UnifiedGraph] Mapping CUDA resource (frame #" << executeCount << ")" << std::endl;
            }
            
            // Map resource for zero-copy access
            cudaError_t mapErr = cudaGraphicsMapResources(1, &m_cudaResource, captureStream);
            if (mapErr != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Failed to map CUDA resource at frame #" << executeCount 
                          << ": " << cudaGetErrorString(mapErr) << std::endl;
                return false;
            }
            
            cudaArray_t array;
            cudaError_t arrayErr = cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0);
            if (arrayErr != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Failed to get mapped array: " << cudaGetErrorString(arrayErr) << std::endl;
                cudaGraphicsUnmapResources(1, &m_cudaResource, captureStream);
                return false;
            }
        
        // Get buffer from triple buffer system if available
        void* targetBuffer = m_tripleBuffer ? 
            m_tripleBuffer->buffers[m_tripleBuffer->captureIdx].data() :
            m_captureBuffer.data();
        
            // Direct array to buffer copy (zero-copy when possible)
            cudaError_t copyErr = cudaMemcpy2DFromArrayAsync(
                targetBuffer, m_captureBuffer.step(),
                array, 0, 0,
                m_captureBuffer.cols() * 4, m_captureBuffer.rows(),
                cudaMemcpyDeviceToDevice, captureStream
            );
            
            if (copyErr != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Memcpy2D failed: " << cudaGetErrorString(copyErr) << std::endl;
            }
            
            // Debug: Log before unmapping
            if (executeCount <= 5 || executeCount % 100 == 0) {
                std::cout << "[UnifiedGraph] Unmapping CUDA resource (frame #" << executeCount << ")" << std::endl;
            }
            
            // Always unmap resource, even on error
            cudaError_t unmapErr = cudaGraphicsUnmapResources(1, &m_cudaResource, captureStream);
            if (unmapErr != cudaSuccess) {
                std::cerr << "[UnifiedGraph] Failed to unmap resource at frame #" << executeCount 
                          << ": " << cudaGetErrorString(unmapErr) << std::endl;
            }
            
            if (copyErr != cudaSuccess) {
                return false;
            }
            
            // Signal capture completion
            m_coordinator->synchronizeCapture(m_coordinator->preprocessStream);
        }
        catch (const std::exception& e) {
            std::cerr << "[UnifiedGraph] Exception in capture stage: " << e.what() << std::endl;
            // Try to unmap resource if still mapped
            cudaGraphicsUnmapResources(1, &m_cudaResource, m_coordinator->captureStream);
            return false;
        }
    }
    
    // 2. Detection pipeline with FUSED preprocessing
    if (m_config.enableDetection && m_d_yoloInput) {
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
            false
        );
        
        // Check for kernel launch errors
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            std::cerr << "[UnifiedGraph] Preprocessing kernel launch failed: " << cudaGetErrorString(kernelErr) << std::endl;
            return false;
        }
        
        // Log first pixel of model input data for debugging
        static int main_log_count = 0;
        if (main_log_count < 5) {  // Log first 5 frames only
            cudaStreamSynchronize(preprocessStream);  // Ensure preprocessing is done
            float first_pixel[3];
            cudaMemcpy(first_pixel, m_d_yoloInput, 3 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "[MODEL INPUT] First pixel BGR: (" 
                      << first_pixel[0] << ", " << first_pixel[1] << ", " << first_pixel[2] 
                      << ") Frame: " << main_log_count << std::endl;
            main_log_count++;
        }
        
        // Signal preprocessing completion
        m_coordinator->synchronizePreprocess(m_coordinator->inferenceStream);
        
        // 2.5. YOLO 추론 실행 - Use existing processFrame for now
        auto& ctx = AppContext::getInstance();
        if (!ctx.getDetectionState().isPaused()) {
            // Use integrated TensorRT pipeline (Phase 1)
            static int processCount = 0;
            processCount++;
            if (processCount <= 10 || processCount % 100 == 0) {
                std::cout << "[UnifiedGraph] Running TensorRT inference, count: " << processCount << std::endl;
            }
            
            try {
                // Phase 2.3: Integrated TensorRT preprocessing and inference pipeline
                
                // 1. Preprocessing: Copy and convert capture to YOLO input format  
                if (m_d_yoloInput && !m_captureBuffer.empty()) {
                    // Phase 2.3: Optimized preprocessing with existing CudaImageProcessing functions
                    // Create temporary buffers for processing pipeline
                    SimpleCudaMat tempResize, tempFloat;
                    tempResize.create(640, 640, 3);  // BGR format for YOLO
                    tempFloat.create(640, 640, 3);   // Float format
                    
                    // Step 1: Convert BGRA to BGR and resize
                    SimpleCudaMat bgrBuffer;
                    bgrBuffer.create(m_captureBuffer.rows(), m_captureBuffer.cols(), 3);
                    CudaImageProcessing::bgra2bgr(m_captureBuffer, bgrBuffer, preprocessStream);
                    
                    // Step 2: Resize to YOLO input dimensions
                    CudaImageProcessing::resize(bgrBuffer, tempResize, 640, 640, preprocessStream);
                    
                    // Step 3: Convert to float and normalize (0-1 range)
                    CudaImageProcessing::convertTo(tempResize, tempFloat, m_imgScale, 0.0f, preprocessStream);
                    
                    // Step 4: Copy to YOLO input buffer
                    size_t inputSize = 640 * 640 * 3 * sizeof(float);
                    cudaMemcpyAsync(m_d_yoloInput, tempFloat.data(), inputSize, 
                                   cudaMemcpyDeviceToDevice, preprocessStream);
                }
                
                // 2. TensorRT Inference with optimized stream handling
                if (m_inputBindings.find(m_inputName) != m_inputBindings.end() && m_d_yoloInput) {
                    // Connect preprocessing output to TensorRT input
                    void* inputBinding = m_inputBindings[m_inputName];
                    if (inputBinding != m_d_yoloInput) {
                        // Copy to input binding if not using shared memory
                        size_t inputSize = 640 * 640 * 3 * sizeof(float);
                        cudaMemcpyAsync(inputBinding, m_d_yoloInput, inputSize, 
                                       cudaMemcpyDeviceToDevice, preprocessStream);
                    }
                    
                    // Execute TensorRT inference asynchronously
                    if (!runInferenceAsync(preprocessStream)) {
                        std::cerr << "[UnifiedGraph] TensorRT inference failed in executeDirect" << std::endl;
                        return false;
                    }
                    
                    // 3. Integrated post-processing (Phase 3: decode, filter, NMS, target selection)
                    performIntegratedPostProcessing(preprocessStream);
                    
                    // 4. Target selection from post-processed results
                    performTargetSelection(preprocessStream);
                }
                
                // 5. Signal inference completion for pipeline synchronization
                cudaEventRecord(m_coordinator->inferenceComplete, preprocessStream);
                m_coordinator->synchronizeInference(m_coordinator->postprocessStream);
            }
            catch (const std::exception& e) {
                std::cerr << "[UnifiedGraph] Exception in detector->processFrame: " << e.what() << std::endl;
                return false;
            }
        }
        
        // Swap triple buffer if available
        if (m_tripleBuffer) {
            cudaEventRecord(m_tripleBuffer->events[m_tripleBuffer->captureIdx], preprocessStream);
            m_tripleBuffer->isReady[m_tripleBuffer->captureIdx] = true;
            
            // Rotate buffer indices with atomic operations to prevent race conditions
            int nextCapture = m_tripleBuffer->displayIdx.load();
            int nextInference = m_tripleBuffer->captureIdx.load();
            int nextDisplay = m_tripleBuffer->inferenceIdx.load();
            
            m_tripleBuffer->captureIdx.store(nextCapture);
            m_tripleBuffer->inferenceIdx.store(nextInference);
            m_tripleBuffer->displayIdx.store(nextDisplay);
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
    
    // 4. Get detection results and calculate mouse movement with Bezier curve
    auto& ctx = AppContext::getInstance();
    if (!ctx.getDetectionState().isPaused() && ctx.aiming) {
        try {
            // Phase 2.3: Get latest detections from integrated TensorRT
            // The detections should now be available from the asynchronous inference
            void* rawOutput = nullptr;
            if (!m_outputBindings.empty()) {
                rawOutput = m_outputBindings.begin()->second;
            }
            std::pair<void*, int> detections = std::make_pair(rawOutput, rawOutput ? 1 : 0);
            
            if (detections.second > 0 && detections.first) {
                // Get the first/best target
                Target h_target;
                cudaMemcpyAsync(&h_target, detections.first, sizeof(Target), 
                               cudaMemcpyDeviceToHost, stream);
                // Sync needed for accurate target data
                cudaStreamSynchronize(stream);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[UnifiedGraph] Exception in mouse movement: " << e.what() << std::endl;
        }
    }
    
    // Memory health check every 1000 frames
    if (executeCount % 1000 == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        float used_gb = (total_mem - free_mem) / (1024.0f * 1024.0f * 1024.0f);
        float free_gb = free_mem / (1024.0f * 1024.0f * 1024.0f);
        
        std::cout << "[UnifiedGraph] GPU Memory - Used: " << used_gb << " GB, Free: " << free_gb << " GB" << std::endl;
        
        // Warning if memory is low
        if (free_gb < 0.5f) {
            std::cerr << "[UnifiedGraph] WARNING: Low GPU memory! Free: " << free_gb << " GB" << std::endl;
        }
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
    auto& ctx = AppContext::getInstance();
    const int width = ctx.config.detection_resolution;   // Use actual detection resolution
    const int height = ctx.config.detection_resolution;  // Square capture region
    const int yoloSize = 640;
    const int maxDetections = 100;
    
    try {
        // Initialize Triple Buffer System for async pipeline
        if (m_config.enableCapture && m_tripleBuffer) {
            for (int i = 0; i < 3; i++) {
                m_tripleBuffer->buffers[i].create(height, width, 4);  // BGRA
                CUDA_CHECK(cudaEventCreateWithFlags(&m_tripleBuffer->events[i], cudaEventDisableTiming));
                m_tripleBuffer->isReady[i] = false;
            }
            std::cout << "[UnifiedGraph] Triple buffer system initialized" << std::endl;
        }
        
        // Allocate GPU buffers
        m_captureBuffer.create(height, width, 4);  // BGRA
        m_preprocessBuffer.create(height, width, 3);  // BGR
        
        // YOLO input buffer (640x640x3 in CHW format)
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
    std::cout << "[UnifiedPipeline] Starting main loop (" << ctx.config.target_fps << " FPS target)" << std::endl;
    
    const auto targetFrameTime = std::chrono::microseconds(static_cast<int64_t>(1000000.0f / ctx.config.target_fps));
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
    
    while (!m_shouldStop && !ctx.should_exit) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        
        // 전체 파이프라인 실행 (캡처→추론→마우스)
        bool success = false;
        try {
            success = executeGraph(m_primaryStream);
        } catch (const std::exception& e) {
            std::cerr << "[UnifiedPipeline] Exception in pipeline: " << e.what() << std::endl;
            success = false;
        }
        
        if (!success) {
            // 에러 시 짧은 대기 후 재시도
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // 목표 FPS 유지를 위한 정밀 타이밍
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto frameTime = frameEnd - frameStart;
        
        if (frameTime < targetFrameTime) {
            std::this_thread::sleep_for(targetFrameTime - frameTime);
        }
        
        // 성능 통계 업데이트 (매 100프레임마다)
        if (m_state.frameCount % 100 == 0) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(frameTime);
            float latencyMs = duration.count() / 1000.0f;
            updateStatistics(latencyMs);
            
            if (m_state.frameCount % 1000 == 0) {
                std::cout << "[UnifiedPipeline] Frame " << m_state.frameCount 
                          << " - Avg latency: " << m_state.avgLatency << "ms" << std::endl;
            }
        }
        
        m_lastFrameTime = frameEnd;
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
    
    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[Pipeline] Failed to create execution context" << std::endl;
        return false;
    }
    
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
    m_imgScale = 1.0f / 255.0f;  // Standard normalization for YOLO models
    m_numClasses = 80;  // Default COCO classes, should be configured properly
    
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
                std::cout << "[Pipeline] Allocated input '" << name << "': " << size << " bytes" << std::endl;
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
                std::cout << "[Pipeline] Allocated output '" << name << "': " << size << " bytes" << std::endl;
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
    
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // Optimized tensor address setting with validation
    // Pre-check input bindings before setting addresses
    for (const auto& inputName : m_inputNames) {
        auto bindingIt = m_inputBindings.find(inputName);
        if (bindingIt == m_inputBindings.end() || bindingIt->second == nullptr) {
            std::cerr << "[Pipeline] Input binding not found or null for: " << inputName << std::endl;
            return false;
        }
        
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
        return false;
    }
    
    // Record inference completion event for pipeline synchronization
    if (m_detectionEvent) {
        cudaEventRecord(m_detectionEvent, stream);
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
    static float cached_confidence_threshold = 0.25f;
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
            max_candidates,
            maxDecodedTargets,
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
            max_candidates,
            maxDecodedTargets,
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

    // Step 2: Get decoded count for further processing
    int decodedCount = 0;
    cudaError_t countCopyErr = cudaMemcpyAsync(&decodedCount, m_d_decodedCount, sizeof(int), 
                                               cudaMemcpyDeviceToHost, stream);
    if (countCopyErr != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to copy decoded count: " << cudaGetErrorString(countCopyErr) << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }
    
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        std::cerr << "[Pipeline] Stream sync failed: " << cudaGetErrorString(syncErr) << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    // Early exit if no detections were decoded
    if (decodedCount == 0) {
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    // Step 3: Class ID filtering
    if (m_d_classFilteredTargets && m_d_classFilteredCount && m_d_allowFlags) {
        cudaError_t filterErr = filterTargetsByClassIdGpu(
            m_d_decodedTargets,
            decodedCount,
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
        // No class filtering - copy decoded targets directly
        if (m_d_classFilteredTargets && m_d_classFilteredCount) {
            cudaMemcpyAsync(m_d_classFilteredTargets, m_d_decodedTargets, 
                          decodedCount * sizeof(Target), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(m_d_classFilteredCount, m_d_decodedCount, 
                          sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Get class filtered count
    int classFilteredCount = 0;
    if (m_d_classFilteredCount) {
        cudaMemcpyAsync(&classFilteredCount, m_d_classFilteredCount, sizeof(int), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        classFilteredCount = decodedCount;
    }

    // Early exit if no detections after class filtering
    if (classFilteredCount == 0) {
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
        return;
    }

    // Step 4: Color filtering (if enabled and available)
    Target* nmsInputTargets = m_d_classFilteredTargets ? m_d_classFilteredTargets : m_d_decodedTargets;
    int effectiveFilteredCount = classFilteredCount;

    // TODO: Color filtering will be implemented when color mask integration is complete
    // For now, we skip color filtering step

    // Step 5: NMS (Non-Maximum Suppression)
    if (m_d_finalTargets && m_d_finalTargetsCount && 
        m_d_x1 && m_d_y1 && m_d_x2 && m_d_y2 && m_d_areas && 
        m_d_scores_nms && m_d_classIds_nms && m_d_iou_matrix && 
        m_d_keep && m_d_indices) {
        
        // Use cached frame dimensions for CUDA Graph compatibility
        static int cached_frame_width = 640;
        static int cached_frame_height = 640;
        if (!m_graphCaptured) {
            cached_frame_width = ctx.config.detection_resolution;
            cached_frame_height = ctx.config.detection_resolution;
        }
        
        try {
            NMSGpu(
                nmsInputTargets,
                effectiveFilteredCount,
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
            
            // Validate detections after NMS
            int finalCount = 0;
            cudaMemcpyAsync(&finalCount, m_d_finalTargetsCount, sizeof(int), 
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            if (finalCount > 0 && finalCount <= cached_max_detections) {
                validateTargetsGpu(m_d_finalTargets, finalCount, stream);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline] Exception during NMSGpu: " << e.what() << std::endl;
            if (m_d_finalTargetsCount) {
                cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
            }
        }
    } else {
        std::cerr << "[Pipeline] NMS buffers not properly allocated!" << std::endl;
        if (m_d_finalTargetsCount) {
            cudaMemsetAsync(m_d_finalTargetsCount, 0, sizeof(int), stream);
        }
    }
    
    // Log first detection result for debugging
    static int detection_log_count = 0;
    if (detection_log_count < 5 && m_d_finalTargets && m_d_finalTargetsCount) {  // Log first 5 frames only
        int finalCount = 0;
        cudaMemcpy(&finalCount, m_d_finalTargetsCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (finalCount > 0) {
            Target first_detection;
            cudaMemcpy(&first_detection, m_d_finalTargets, sizeof(Target), cudaMemcpyDeviceToHost);
            std::cout << "[MODEL OUTPUT] First detection - x:" << first_detection.x 
                      << " y:" << first_detection.y << " w:" << first_detection.width 
                      << " h:" << first_detection.height << " conf:" << first_detection.confidence 
                      << " class:" << first_detection.classId << " Frame: " << detection_log_count << std::endl;
        }
        detection_log_count++;
    }
}

void UnifiedGraphPipeline::performTargetSelection(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Check if we have final targets to select from
    if (!m_d_finalTargets || !m_d_finalTargetsCount) {
        std::cerr << "[Pipeline] No final targets available for selection" << std::endl;
        return;
    }
    
    // Get the count of final targets
    int finalCount = 0;
    cudaError_t countErr = cudaMemcpyAsync(&finalCount, m_d_finalTargetsCount, sizeof(int), 
                                          cudaMemcpyDeviceToHost, stream);
    if (countErr != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to get final target count: " << cudaGetErrorString(countErr) << std::endl;
        return;
    }
    
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        std::cerr << "[Pipeline] Stream sync failed in target selection: " << cudaGetErrorString(syncErr) << std::endl;
        return;
    }
    
    // Early exit if no targets available
    if (finalCount <= 0) {
        // Clear best target data
        if (m_d_bestTargetIndex) {
            cudaMemsetAsync(m_d_bestTargetIndex, -1, sizeof(int), stream);
        }
        return;
    }
    
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
    
    // Call GPU target selection
    cudaError_t selectErr = findClosestTargetGpu(
        m_d_finalTargets,
        finalCount,
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
    } else {
        // Log final target data for debugging  
        static int final_target_log_count = 0;
        if (final_target_log_count < 5) {  // Log first 5 frames only
            cudaStreamSynchronize(stream);  // Ensure target selection is done
            Target final_target;
            int target_index;
            cudaMemcpy(&final_target, m_d_bestTarget, sizeof(Target), cudaMemcpyDeviceToHost);
            cudaMemcpy(&target_index, m_d_bestTargetIndex, sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << "[FINAL TARGET] Selected target - x:" << final_target.x 
                      << " y:" << final_target.y << " w:" << final_target.width 
                      << " h:" << final_target.height << " conf:" << final_target.confidence 
                      << " class:" << final_target.classId << " index:" << target_index 
                      << " Frame: " << final_target_log_count << std::endl;
            final_target_log_count++;
        }
    }
}

} // namespace needaimbot