#include "unified_graph_pipeline.h"
#include "cuda_error_check.h"
#include "detection/postProcess.h"
#include <cuda_runtime.h>
#include <iostream>

// Constructor - Initialize multi-stream architecture
UnifiedGraphPipeline::UnifiedGraphPipeline() {
    // Create multiple CUDA streams for pipeline parallelism
    m_primaryStream = std::make_unique<CudaStream>();
    m_inferenceStream = std::make_unique<CudaStream>();
    m_postprocessStream = std::make_unique<CudaStream>();
    m_copyStream = std::make_unique<CudaStream>();
    
    // Create synchronization events
    m_inferenceStartEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    m_inferenceEndEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    m_postprocessEndEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    m_nmsCompleteEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    m_previewReadyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    
    // Initialize other events
    m_lastFrameEnd = std::make_unique<CudaEvent>();
    m_copyEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    
    std::cout << "[CUDA Graph] Multi-stream pipeline initialized with 4 streams" << std::endl;
}

// Enhanced initialize with multi-stream support
bool UnifiedGraphPipeline::initialize(const UnifiedPipelineConfig& config) {
    m_config = config;
    
    // Allocate buffers
    if (!allocateBuffers()) {
        std::cerr << "[CUDA Graph] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    // Initialize triple buffer
    m_tripleBuffer = std::make_unique<TripleBuffer>();
    
    // Set stream priorities for optimal scheduling
    int minPriority, maxPriority;
    cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);
    
    // Inference gets highest priority
    cudaStreamCreateWithPriority(&m_inferenceStream->stream, cudaStreamNonBlocking, maxPriority);
    // Post-processing gets second priority  
    cudaStreamCreateWithPriority(&m_postprocessStream->stream, cudaStreamNonBlocking, (maxPriority + minPriority) / 2);
    
    std::cout << "[CUDA Graph] Stream priorities set - Inference: " << maxPriority 
              << ", PostProcess: " << (maxPriority + minPriority) / 2 << std::endl;
    
    return true;
}

// Enhanced graph capture with NMS integration
bool UnifiedGraphPipeline::captureGraph(cudaStream_t stream) {
    if (m_graphCaptured) {
        return true;
    }
    
    cudaStream_t captureStream = stream ? stream : m_primaryStream->get();
    
    // Begin graph capture
    CUDA_CHECK(cudaStreamBeginCapture(captureStream, 
        static_cast<cudaStreamCaptureMode>(m_config.graphCaptureMode)));
    
    // Capture preprocessing on primary stream
    if (!capturePreprocessGraph(captureStream)) {
        cudaStreamEndCapture(captureStream, &m_graph);
        return false;
    }
    
    // Fork to inference stream with event synchronization
    CUDA_CHECK(cudaEventRecord(*m_inferenceStartEvent, captureStream));
    CUDA_CHECK(cudaStreamWaitEvent(m_inferenceStream->get(), *m_inferenceStartEvent, 0));
    
    // Capture inference on dedicated stream
    if (!captureInferenceGraph(m_inferenceStream->get())) {
        cudaStreamEndCapture(captureStream, &m_graph);
        return false;
    }
    
    // Signal inference completion
    CUDA_CHECK(cudaEventRecord(*m_inferenceEndEvent, m_inferenceStream->get()));
    
    // Fork to post-processing stream
    CUDA_CHECK(cudaStreamWaitEvent(m_postprocessStream->get(), *m_inferenceEndEvent, 0));
    
    // Capture post-processing including NMS on dedicated stream
    if (!capturePostprocessGraph(m_postprocessStream->get())) {
        cudaStreamEndCapture(captureStream, &m_graph);
        return false;
    }
    
    // Capture NMS as part of the graph
    if (!captureNMSGraph(m_postprocessStream->get())) {
        cudaStreamEndCapture(captureStream, &m_graph);
        return false;
    }
    
    // Signal NMS completion
    CUDA_CHECK(cudaEventRecord(*m_nmsCompleteEvent, m_postprocessStream->get()));
    
    // Synchronize back to primary stream
    CUDA_CHECK(cudaStreamWaitEvent(captureStream, *m_nmsCompleteEvent, 0));
    
    // Capture tracking if enabled
    if (m_config.enableDetection) {
        captureTrackingGraph(captureStream);
    }
    
    // End graph capture
    CUDA_CHECK(cudaStreamEndCapture(captureStream, &m_graph));
    
    // Instantiate the graph
    CUDA_CHECK(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 
        m_config.graphInstantiateFlags));
    
    m_graphCaptured = true;
    m_state.graphReady = true;
    
    std::cout << "[CUDA Graph] Graph captured with multi-stream support and NMS integration" << std::endl;
    return true;
}

// New: Capture NMS operations in the graph
bool UnifiedGraphPipeline::captureNMSGraph(cudaStream_t stream) {
    if (!m_d_decodedTargets || !m_d_finalTargets) {
        return false;
    }
    
    // Create kernel node parameters for NMS
    cudaKernelNodeParams nmsParams = {0};
    
    // Get NMS kernel function pointer
    void* nmsKernel = reinterpret_cast<void*>(&NMSGpu);
    
    // Set up NMS kernel arguments
    void* nmsArgs[] = {
        &m_d_decodedTargets->data(),      // Input detections
        &m_d_decodedCount->data(),         // Input count
        &m_d_finalTargets->data(),         // Output detections
        &m_d_finalTargetsCount->data(),    // Output count
        &m_config.nmsThreshold,            // NMS threshold
        &m_config.detectionWidth,          // Frame width
        &m_config.detectionHeight,         // Frame height
        // NMS temporary buffers
        &m_d_x1->data(),
        &m_d_y1->data(),
        &m_d_x2->data(),
        &m_d_y2->data(),
        &m_d_areas->data(),
        &m_d_scores_nms->data(),
        &m_d_classIds_nms->data(),
        &m_d_iou_matrix->data(),
        &m_d_keep->data(),
        &m_d_indices->data()
    };
    
    // Configure kernel launch parameters
    dim3 blockDim(256);
    dim3 gridDim((1000 + blockDim.x - 1) / blockDim.x);  // Max 1000 detections
    
    nmsParams.func = nmsKernel;
    nmsParams.gridDim = gridDim;
    nmsParams.blockDim = blockDim;
    nmsParams.sharedMemBytes = 0;
    nmsParams.kernelParams = nmsArgs;
    nmsParams.extra = nullptr;
    
    // Add NMS kernel node to the graph
    cudaGraphNode_t nmsNode;
    CUDA_CHECK(cudaGraphAddKernelNode(&nmsNode, m_graph, nullptr, 0, &nmsParams));
    
    // Store the node for potential updates
    m_nmsNodes.push_back(nmsNode);
    m_namedNodes["nms_kernel"] = nmsNode;
    
    std::cout << "[CUDA Graph] NMS operations captured in graph" << std::endl;
    return true;
}

// Execute graph with multi-stream parallelism
bool UnifiedGraphPipeline::executeGraphNonBlocking(cudaStream_t stream) {
    if (!m_state.graphReady || !m_graphExec) {
        return false;
    }
    
    // Execute the graph on primary stream (graph handles multi-stream internally)
    cudaStream_t execStream = stream ? stream : m_primaryStream->get();
    CUDA_CHECK(cudaGraphLaunch(m_graphExec, execStream));
    
    // Record events for async operations on copy stream
    CUDA_CHECK(cudaEventRecord(*m_copyEvent, m_copyStream->get()));
    
    // Async copy results to pinned memory on copy stream
    int bufferIdx = m_currentPipelineIdx.load();
    if (m_d_finalTargetsCount && m_d_finalTargets) {
        // Copy target count
        CUDA_CHECK(cudaMemcpyAsync(
            &m_tripleBuffer->target_count[bufferIdx],
            m_d_finalTargetsCount->data(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            m_copyStream->get()
        ));
        
        // Copy target data if any
        CUDA_CHECK(cudaMemcpyAsync(
            m_tripleBuffer->h_target_coords_pinned[bufferIdx].data(),
            m_d_finalTargets->data(),
            sizeof(Target),
            cudaMemcpyDeviceToHost,
            m_copyStream->get()
        ));
        
        // Signal target data ready
        CUDA_CHECK(cudaEventRecord(m_tripleBuffer->target_ready_events[bufferIdx], 
            m_copyStream->get()));
        m_tripleBuffer->target_data_valid[bufferIdx] = true;
    }
    
    // Update profiling asynchronously
    if (m_config.enableProfiling) {
        updateProfilingAsync(execStream);
    }
    
    return true;
}

// Placeholder implementations for other methods
bool UnifiedGraphPipeline::capturePreprocessGraph(cudaStream_t stream) {
    // Preprocessing kernels would be captured here
    return true;
}

bool UnifiedGraphPipeline::captureInferenceGraph(cudaStream_t stream) {
    // TensorRT inference would be captured here (but not in graph as per requirement)
    return true;
}

bool UnifiedGraphPipeline::capturePostprocessGraph(cudaStream_t stream) {
    // Post-processing kernels would be captured here
    return true;
}

bool UnifiedGraphPipeline::captureTrackingGraph(cudaStream_t stream) {
    // Tracking kernels would be captured here
    return true;
}

bool UnifiedGraphPipeline::allocateBuffers() {
    // Allocate NMS buffers
    const int maxDetections = 1000;
    
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
    
    m_d_decodedTargets = std::make_unique<CudaMemory<Target>>(maxDetections);
    m_d_decodedCount = std::make_unique<CudaMemory<int>>(1);
    m_d_finalTargets = std::make_unique<CudaMemory<Target>>(maxDetections);
    m_d_finalTargetsCount = std::make_unique<CudaMemory<int>>(1);
    
    return true;
}

void UnifiedGraphPipeline::updateProfilingAsync(cudaStream_t stream) {
    if (m_state.startEvent && m_state.endEvent) {
        float ms = 0;
        cudaEventElapsedTime(&ms, *m_state.startEvent, *m_state.endEvent);
        m_state.lastLatency = ms;
        m_state.avgLatency = (m_state.avgLatency * m_state.frameCount + ms) / (m_state.frameCount + 1);
        m_state.frameCount++;
    }
}

// Destructor
UnifiedGraphPipeline::~UnifiedGraphPipeline() {
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
    }
    if (m_graph) {
        cudaGraphDestroy(m_graph);
    }
}