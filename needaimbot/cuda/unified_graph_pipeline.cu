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

namespace needaimbot {

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
}

UnifiedGraphPipeline::~UnifiedGraphPipeline() {
    shutdown();
    
    if (m_state.startEvent) cudaEventDestroy(m_state.startEvent);
    if (m_state.endEvent) cudaEventDestroy(m_state.endEvent);
}

bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    // Create primary stream
    cudaError_t err = cudaStreamCreate(&m_primaryStream);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Failed to create stream: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
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
    
    std::cout << "[UnifiedGraph] Pipeline initialized successfully" << std::endl;
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
    
    // 2. Preprocessing (color conversion + resize)
    if (m_config.enableDetection) {
        // BGRA to BGR conversion using existing kernel
        CudaImageProcessing::bgra2bgr(m_captureBuffer, m_preprocessBuffer, stream);
        
        // TODO: Add resize kernel for YOLO input
        // resizeImage(m_preprocessBuffer, m_d_yoloInput, 640, 640, stream);
    }
    
    // 3. TensorRT Inference
    if (m_config.enableDetection && m_detector) {
        // TODO: Detector must expose async inference method
        // m_detector->runInferenceAsync(m_d_yoloInput, m_d_inferenceOutput, stream);
    }
    
    // 4. Postprocessing (NMS, filtering, target selection)
    if (m_config.enableDetection) {
        // TODO: Integrate postprocessing kernels
        // launchNMSKernel(m_d_inferenceOutput, m_d_nmsOutput, stream);
        // launchFilterKernel(m_d_nmsOutput, m_d_filteredOutput, stream);
        // launchTargetSelectionKernel(m_d_filteredOutput, m_d_selectedTarget, stream);
    }
    
    // 5. Tracking (Kalman filter)
    if (m_config.enableTracking && m_tracker) {
        // TODO: Integrate GPU Kalman filter
        // m_tracker->predictAsync(stream);
        // m_tracker->updateAsync(m_d_selectedTarget, stream);
        // m_tracker->getOutputAsync(m_d_trackedTarget, stream);
    }
    
    // 6. PID Control
    if (m_config.enablePIDControl && m_pidController) {
        // TODO: Integrate GPU PID controller
        // m_pidController->computeAsync(m_d_trackedTarget, m_d_pidOutput, stream);
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
    if (!m_state.graphReady) {
        // Fall back to direct execution or trigger graph capture
        if (m_state.needsRebuild) {
            if (!captureGraph(stream)) {
                return executeDirect(stream);
            }
        } else {
            return executeDirect(stream);
        }
    }
    
    if (!stream) stream = m_primaryStream;
    
    // Record start time if profiling
    if (m_config.enableProfiling) {
        cudaEventRecord(m_state.startEvent, stream);
    }
    
    // Execute the graph
    cudaError_t err = cudaGraphLaunch(m_graphExec, stream);
    if (err != cudaSuccess) {
        std::cerr << "[UnifiedGraph] Graph execution failed: " 
                  << cudaGetErrorString(err) << std::endl;
        m_state.needsRebuild = true;
        return false;
    }
    
    // Record end time if profiling
    if (m_config.enableProfiling) {
        cudaEventRecord(m_state.endEvent, stream);
        cudaEventSynchronize(m_state.endEvent);
        
        float latency;
        cudaEventElapsedTime(&latency, m_state.startEvent, m_state.endEvent);
        updateStatistics(latency);
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
    if (!stream) stream = m_primaryStream;
    
    // Direct execution without graph
    // This is the fallback path when graph capture fails
    
    // 1. Capture stage (if resource is mapped)
    if (m_config.enableCapture && m_cudaResource) {
        // Note: D3D11 interop operations can't be captured in graphs
        // We need a workaround for this
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0);
        
        // Copy to our buffer
        cudaMemcpy2DFromArrayAsync(
            m_captureBuffer.data(), m_captureBuffer.step(),
            array, 0, 0,
            m_captureBuffer.cols() * 4, m_captureBuffer.rows(),
            cudaMemcpyDeviceToDevice, stream
        );
    }
    
    // 2. Detection pipeline
    if (m_config.enableDetection && m_detector) {
        // Preprocessing
        // preprocessImage(m_captureBuffer, m_preprocessBuffer, stream); // TODO: implement
        
        // Inference (TensorRT)
        // Note: We need detector to expose a method that works with our buffers
        // For now, this is a placeholder
        
        // Postprocessing
        int numDetections = 0;
        // postprocessDetections(m_d_inferenceOutput, m_d_detections, 
        //                     &numDetections, stream); // TODO: implement
    }
    
    // 3. Tracking
    if (m_config.enableTracking && m_tracker) {
        int numTracked = 0;
        // processKalmanFilter(m_tracker, m_d_detections, 100,
        //                   m_d_trackedTargets, &numTracked, stream, false); // TODO: fix namespace
    }
    
    // 4. PID Control
    if (m_config.enablePIDControl && m_pidController) {
        // m_pidController->compute(m_d_trackedTargets, m_d_pidOutput, stream); // TODO: implement
    }
    
    // 5. Copy results
    if (m_d_outputBuffer && m_d_pidOutput) {
        cudaMemcpyAsync(m_d_outputBuffer, m_d_pidOutput, 
                       2 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    
    return true;
}

bool UnifiedGraphPipeline::addCaptureNode(cudaStream_t stream) {
    // Capture operations are tricky with D3D11 interop
    // We'll use a memcpy node as placeholder
    cudaGraphNode_t node;
    cudaMemcpy3DParms copyParams = {0};
    
    // Setup copy parameters
    copyParams.srcArray = m_cudaArray;
    copyParams.dstPtr = make_cudaPitchedPtr(
        m_captureBuffer.data(), 
        m_captureBuffer.step(),
        m_captureBuffer.cols() * 4,
        m_captureBuffer.rows()
    );
    copyParams.extent = make_cudaExtent(
        m_captureBuffer.cols() * 4,
        m_captureBuffer.rows(),
        1
    );
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    cudaError_t err = cudaGraphAddMemcpyNode(&node, m_graph, nullptr, 0, &copyParams);
    if (err == cudaSuccess) {
        m_captureNodes.push_back(node);
        return true;
    }
    
    return false;
}

bool UnifiedGraphPipeline::addPreprocessNode(cudaStream_t stream) {
    // Add color conversion and preprocessing nodes
    std::vector<cudaGraphNode_t> preprocessNodes;
    
    // 1. BGRA to BGR conversion
    {
        cudaGraphNode_t colorNode;
        // Use external bgra2bgr function from cuda_image_processing
        // We'll call it directly in the stream during capture
        CudaImageProcessing::bgra2bgr(m_captureBuffer, m_preprocessBuffer, stream);
        
        // For graph capture, we need to get the actual kernel function pointer
        // This is handled during stream capture mode
        preprocessNodes.push_back(colorNode);
    }
    
    // 2. Resize and normalize for YOLO (640x640)
    {
        cudaGraphNode_t resizeNode;
        // Resize kernel would go here
        // For now, handled during stream capture
        preprocessNodes.push_back(resizeNode);
    }
    
    // Store nodes for dependency tracking
    m_inferenceNodes.insert(m_inferenceNodes.end(), 
                           preprocessNodes.begin(), 
                           preprocessNodes.end());
    
    return true;
}

bool UnifiedGraphPipeline::addInferenceNode(cudaStream_t stream) {
    if (!m_detector) return false;
    
    // TensorRT 8.5+ supports graph capture for inference
    // The detector must call enqueueV3 during the capture phase
    // This will automatically add the inference operations to the graph
    
    // During graph capture mode, TensorRT will add its own nodes
    // We just need to ensure the detector's inference is called
    // with the correct buffers during capture
    
    // The actual inference call happens during stream capture
    // m_detector->runInference(m_preprocessBuffer.data(), 
    //                          m_d_inferenceOutput, stream);
    
    // For tracking purposes, add a placeholder node
    cudaGraphNode_t node;
    m_inferenceNodes.push_back(node);
    
    return true;
}

bool UnifiedGraphPipeline::addPostprocessNode(cudaStream_t stream) {
    // Add NMS and filtering kernels
    std::vector<cudaGraphNode_t> postNodes;
    
    // These kernels will be captured during stream recording
    // 1. NMS kernel
    // 2. Filtering kernel  
    // 3. Target selection kernel
    
    // The actual kernel launches happen during capture
    // We track nodes for dependency management
    cudaGraphNode_t nmsNode, filterNode, targetNode;
    postNodes.push_back(nmsNode);
    postNodes.push_back(filterNode);
    postNodes.push_back(targetNode);
    
    m_postprocessNodes.insert(m_postprocessNodes.end(),
                             postNodes.begin(), postNodes.end());
    
    return true;
}

bool UnifiedGraphPipeline::addTrackingNode(cudaStream_t stream) {
    if (!m_tracker) return true;
    
    // Kalman filter predict and update kernels
    std::vector<cudaGraphNode_t> trackNodes;
    
    // During capture, the tracker kernels will be recorded
    // m_tracker->predict(stream);
    // m_tracker->update(m_d_detections, stream);
    
    cudaGraphNode_t predictNode, updateNode;
    trackNodes.push_back(predictNode);
    trackNodes.push_back(updateNode);
    
    m_trackingNodes.insert(m_trackingNodes.end(),
                          trackNodes.begin(), trackNodes.end());
    
    return true;
}

bool UnifiedGraphPipeline::addPIDNode(cudaStream_t stream) {
    if (!m_pidController) return true;
    
    // PID control computation kernel
    cudaGraphNode_t pidNode;
    
    // During capture:
    // m_pidController->compute(m_d_trackedTargets, m_d_pidOutput, stream);
    
    m_pidNodes.push_back(pidNode);
    return true;
}

bool UnifiedGraphPipeline::addResultCopyNode(cudaStream_t stream) {
    if (!m_d_outputBuffer || !m_d_pidOutput) return true;
    
    cudaGraphNode_t node;
    cudaMemcpy3DParms copyParams = {0};
    
    // Setup simple 1D copy as 3D parameters
    copyParams.srcPtr = make_cudaPitchedPtr(m_d_pidOutput, 2 * sizeof(float), 2 * sizeof(float), 1);
    copyParams.dstPtr = make_cudaPitchedPtr(m_d_outputBuffer, 2 * sizeof(float), 2 * sizeof(float), 1);
    copyParams.extent = make_cudaExtent(2 * sizeof(float), 1, 1);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    // Collect all previous nodes as dependencies
    std::vector<cudaGraphNode_t> allDependencies;
    allDependencies.insert(allDependencies.end(), m_pidNodes.begin(), m_pidNodes.end());
    
    cudaError_t err = cudaGraphAddMemcpyNode(&node, m_graph,
                                            allDependencies.data(), 
                                            allDependencies.size(),
                                            &copyParams);
    
    return err == cudaSuccess;
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
    m_h_inputBuffer = nullptr;
    m_h_outputBuffer = nullptr;
}

} // namespace needaimbot