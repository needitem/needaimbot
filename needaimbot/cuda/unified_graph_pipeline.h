#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include "simple_cuda_mat.h"
#include "../core/Target.h"

// TensorRT includes for Phase 1 integration
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>

// Forward declarations outside namespace
class GPUKalmanTracker;

namespace needaimbot {

// Forward declarations for internal implementation classes
class DynamicCudaGraph;
class PipelineCoordinator;

// Graph node types for tracking
enum class GraphNodeType {
    CAPTURE_MAP,
    CAPTURE_COPY,
    CAPTURE_UNMAP,
    PREPROCESSING,
    INFERENCE,
    POSTPROCESSING,
    FILTERING,
    TRACKING_PREDICT,
    TRACKING_UPDATE,
    RESULT_COPY
};

// Graph execution state
struct GraphExecutionState {
    bool graphReady = false;
    bool needsRebuild = false;
    int frameCount = 0;
    float avgLatency = 0.0f;
    float lastLatency = 0.0f;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
};

// Unified pipeline configuration
struct UnifiedPipelineConfig {
    // Pipeline stages enable flags
    bool enableCapture = true;
    bool enableDetection = true;
    bool enableTracking = true;
    
    // Model configuration (Phase 1 integration)
    std::string modelPath;
    
    // Detection parameters
    float confThreshold = 0.4f;
    float nmsThreshold = 0.45f;
    int detectionWidth = 256;   // Detection/capture width
    int detectionHeight = 256;  // Detection/capture height
    
    // Graph optimization flags
    bool useGraphOptimization = true;
    bool allowGraphUpdate = true;
    bool enableProfiling = false;
    
    // Performance tuning
    int maxBatchSize = 1;
    int graphCaptureMode = cudaStreamCaptureModeGlobal;
    int graphInstantiateFlags = 0;
};



class UnifiedGraphPipeline {
public:
    UnifiedGraphPipeline();
    ~UnifiedGraphPipeline();
    
    // Initialize pipeline with components
    bool initialize(const UnifiedPipelineConfig& config);
    void shutdown();
    
    // Set component references
    void setTracker(::GPUKalmanTracker* tracker) { m_tracker = tracker; }
    
    // Main execution methods
    bool captureGraph(cudaStream_t stream = nullptr);
    bool executeGraph(cudaStream_t stream = nullptr);
    
        
    // Pipeline data management
    void setInputTexture(cudaGraphicsResource_t resource) { m_cudaResource = resource; }
    void setDesktopDuplication(void* duplication, void* device, void* context, void* texture) {
        m_desktopDuplication = duplication;
        m_d3dDevice = device;
        m_d3dContext = context;
        m_captureTextureD3D = texture;
    }
    void setInputFrame(const SimpleCudaMat& frame);
    void setOutputBuffer(float* d_output) { m_d_outputBuffer = d_output; }
    
    // TensorRT integration methods (Phase 1)
    bool initializeTensorRT(const std::string& modelFile);
    bool loadEngine(const std::string& modelFile);
    nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath);
    void getInputNames();
    void getOutputNames();
    void getBindings();
    bool runInferenceAsync(cudaStream_t stream);
    void performIntegratedPostProcessing(cudaStream_t stream);

    void performTargetSelection(cudaStream_t stream);
    
    // Main loop methods
    void runMainLoop();
    void stopMainLoop();
    
    // State and statistics
    GraphExecutionState getState() const { return m_state; }
    float getAverageLatency() const { return m_state.avgLatency; }
    bool isGraphReady() const { return m_state.graphReady; }
    
    // Frame access for preview
    const SimpleCudaMat& getCaptureBuffer() const { return m_captureBuffer; }
    
    
private:
    // Advanced graph and stream management
    DynamicCudaGraph* m_dynamicGraph = nullptr;
    PipelineCoordinator* m_coordinator = nullptr;
    
    // Two-stage graph pipeline for conditional execution
    cudaGraph_t m_graph = nullptr;                    // Legacy monolithic graph
    cudaGraphExec_t m_graphExec = nullptr;            // Legacy graph exec
    cudaGraph_t m_detectionGraph = nullptr;           // Stage 1: Detection only
    cudaGraphExec_t m_detectionGraphExec = nullptr;
    cudaGraph_t m_trackingGraph = nullptr;            // Stage 2: Tracking + PID
    cudaGraphExec_t m_trackingGraphExec = nullptr;
    cudaStream_t m_primaryStream = nullptr;
    
    // Pipeline synchronization
    cudaEvent_t m_detectionEvent = nullptr;
    cudaEvent_t m_trackingEvent = nullptr;
    bool m_prevFrameHasTarget = false;
    
    // Graph nodes for dynamic updates
    std::vector<cudaGraphNode_t> m_captureNodes;
    std::vector<cudaGraphNode_t> m_inferenceNodes;
    std::vector<cudaGraphNode_t> m_postprocessNodes;
    std::vector<cudaGraphNode_t> m_trackingNodes;
    std::vector<cudaGraphNode_t> m_pidNodes;
    
    // Node name mapping for dynamic updates
    std::unordered_map<std::string, cudaGraphNode_t> m_namedNodes;
    
    // Triple buffering for async pipeline
    struct TripleBuffer {
        SimpleCudaMat buffers[3];
        std::atomic<int> captureIdx{0};
        std::atomic<int> inferenceIdx{1};
        std::atomic<int> displayIdx{2};
        cudaEvent_t events[3];
        bool isReady[3] = {false, false, false};
    };
    std::unique_ptr<TripleBuffer> m_tripleBuffer;
    
    // Component pointers
    ::GPUKalmanTracker* m_tracker = nullptr;
    
    // TensorRT engine management (Phase 1 integration)
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    // Input/Output binding management
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::unordered_map<std::string, size_t> m_inputSizes;
    std::unordered_map<std::string, size_t> m_outputSizes;
    std::unordered_map<std::string, void*> m_inputBindings;
    std::unordered_map<std::string, void*> m_outputBindings;
    
    // Inference related information
    std::string m_inputName;
    nvinfer1::Dims m_inputDims;
    float m_imgScale;
    int m_numClasses;
    // Pipeline buffers (GPU memory)
    SimpleCudaMat m_captureBuffer;
    SimpleCudaMat m_preprocessBuffer;
    float* m_d_preprocessBuffer = nullptr;  // Raw preprocess buffer pointer
    Target* m_d_tracks = nullptr;           // GPU tracking data
    
    // NMS temporary buffers (allocated once, reused)
    int* m_d_numDetections = nullptr;
    int* m_d_x1 = nullptr;
    int* m_d_y1 = nullptr;
    int* m_d_x2 = nullptr;
    int* m_d_y2 = nullptr;
    float* m_d_areas = nullptr;
    float* m_d_scores_nms = nullptr;
    int* m_d_classIds_nms = nullptr;
    float* m_d_iou_matrix = nullptr;
    bool* m_d_keep = nullptr;
    int* m_d_indices = nullptr;
    int* m_d_outputCount = nullptr;
    float* m_d_yoloInput = nullptr;        // YOLO model input (640x640x3)
    float* m_d_inferenceOutput = nullptr;  // Raw inference output
    float* m_d_nmsOutput = nullptr;        // After NMS
    float* m_d_filteredOutput = nullptr;   // After filtering
    Target* m_d_detections = nullptr;      // Detection results
    Target* m_d_selectedTarget = nullptr;  // Selected target
    Target* m_d_trackedTarget = nullptr;   // After Kalman tracking
    Target* m_d_trackedTargets = nullptr;  // Multiple tracked targets
    
    // Post-processing buffers (Phase 3 integration)
    Target* m_d_decodedTargets = nullptr;      // Decoded detections from inference
    int* m_d_decodedCount = nullptr;           // Count of decoded detections
    Target* m_d_finalTargets = nullptr;        // Final NMS output
    int* m_d_finalTargetsCount = nullptr;      // Final count after NMS
    Target* m_d_classFilteredTargets = nullptr; // After class filtering
    int* m_d_classFilteredCount = nullptr;     // Count after class filtering
    Target* m_d_colorFilteredTargets = nullptr; // After color filtering
    int* m_d_colorFilteredCount = nullptr;     // Count after color filtering
    
    // Class filtering control buffer
    unsigned char* m_d_allowFlags = nullptr;   // Class filtering flags
    
    // Additional post-processing metadata
    std::unordered_map<std::string, std::vector<int64_t>> m_outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> m_outputTypes;
    
    // Target selection buffers
    int* m_d_bestTargetIndex = nullptr;        // Selected target index
    Target* m_d_bestTarget = nullptr;          // Selected target data
    float* m_d_outputBuffer = nullptr;     // Final output buffer
    
    // Pinned host memory for zero-copy access
    unsigned char* m_h_inputBuffer = nullptr;  // Pinned input buffer
    float* m_h_outputBuffer = nullptr;         // Pinned output buffer (x,y)
    
    // Resource management
    cudaGraphicsResource_t m_cudaResource = nullptr;
    cudaArray_t m_cudaArray = nullptr;
    
    // Desktop Duplication for screen capture
    void* m_desktopDuplication = nullptr;  // IDXGIOutputDuplication*
    void* m_d3dDevice = nullptr;           // ID3D11Device*
    void* m_d3dContext = nullptr;          // ID3D11DeviceContext*
    void* m_captureTextureD3D = nullptr;   // ID3D11Texture2D*
    
    // Configuration and state
    UnifiedPipelineConfig m_config;
    GraphExecutionState m_state;
    std::mutex m_graphMutex;
    bool m_hasFrameData = false;
    
    // Main loop control
    std::atomic<bool> m_shouldStop{false};
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;

    bool validateGraph();
    void cleanupGraph();
    void updateStatistics(float latency);
    
    // Buffer allocation
    bool allocateBuffers();
    void deallocateBuffers();
    
    // Two-stage pipeline helpers
    void checkTargetsAsync(cudaStream_t stream);
    void updateProfilingAsync(cudaStream_t stream);
    
    // Graph capture methods
    bool capturePreprocessGraph(cudaStream_t stream);
    bool captureInferenceGraph(cudaStream_t stream);
    bool capturePostprocessGraph(cudaStream_t stream);
    bool captureTrackingGraph(cudaStream_t stream);
    
    // Graph state
    bool m_graphCaptured = false;
};

// Global pipeline instance manager
class PipelineManager {
public:
    static PipelineManager& getInstance() {
        static PipelineManager instance;
        return instance;
    }
    
    UnifiedGraphPipeline* getPipeline() { return m_pipeline.get(); }
    
    bool initializePipeline(const UnifiedPipelineConfig& config) {
        m_pipeline = std::make_unique<UnifiedGraphPipeline>();
        return m_pipeline->initialize(config);
    }
    
    void shutdownPipeline() {
        if (m_pipeline) {
            m_pipeline->shutdown();
            m_pipeline.reset();
        }
    }
    
    // Main loop delegation
    void runMainLoop() {
        if (m_pipeline) {
            m_pipeline->runMainLoop();
        }
    }
    
    void stopMainLoop() {
        if (m_pipeline) {
            m_pipeline->stopMainLoop();
        }
    }
    
private:
    PipelineManager() = default;
    ~PipelineManager() = default;
    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;
    
    std::unique_ptr<UnifiedGraphPipeline> m_pipeline;
};

} // namespace needaimbot