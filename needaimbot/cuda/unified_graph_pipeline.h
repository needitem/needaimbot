#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include "simple_cuda_mat.h"
#include "../core/Target.h"

// Forward declarations outside namespace
class Detector;
class GPUKalmanTracker;
class GpuPIDController;

namespace needaimbot {

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
    PID_CONTROL,
    RESULT_COPY
};

// Graph execution state
struct GraphExecutionState {
    bool graphReady = false;
    bool needsRebuild = false;
    int frameCount = 0;
    float avgLatency = 0.0f;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
};

// Unified pipeline configuration
struct UnifiedPipelineConfig {
    // Pipeline stages enable flags
    bool enableCapture = true;
    bool enableDetection = true;
    bool enableTracking = true;
    bool enablePIDControl = true;
    
    // Graph optimization flags
    bool useGraphOptimization = true;
    bool allowGraphUpdate = true;
    bool enableProfiling = false;
    
    // Performance tuning
    int maxBatchSize = 1;
    int graphCaptureMode = cudaStreamCaptureModeGlobal;
    int graphInstantiateFlags = 0;
};

// Forward declaration for internal classes
class DynamicCudaGraph;
class PipelineCoordinator;

class UnifiedGraphPipeline {
public:
    UnifiedGraphPipeline();
    ~UnifiedGraphPipeline();
    
    // Initialize pipeline with components
    bool initialize(const UnifiedPipelineConfig& config);
    void shutdown();
    
    // Set component references
    void setDetector(::Detector* detector) { m_detector = detector; }
    void setTracker(::GPUKalmanTracker* tracker) { m_tracker = tracker; }
    void setPIDController(::GpuPIDController* pidController) { m_pidController = pidController; }
    
    // Main execution methods
    bool captureGraph(cudaStream_t stream = nullptr);
    bool executeGraph(cudaStream_t stream = nullptr);
    bool updateGraph(cudaStream_t stream = nullptr);
    
    // Direct execution (non-graph fallback)
    bool executeDirect(cudaStream_t stream = nullptr);
    
    // Dynamic parameter updates (no graph recapture needed)
    bool updateConfidenceThreshold(float threshold);
    bool updateNMSThreshold(float threshold);
    bool updateTargetSelectionParams(float centerWeight, float sizeWeight);
    
    // Pipeline data management
    void setInputTexture(cudaGraphicsResource_t resource) { m_cudaResource = resource; }
    void setOutputBuffer(float* d_output) { m_d_outputBuffer = d_output; }
    
    // State and statistics
    GraphExecutionState getState() const { return m_state; }
    float getAverageLatency() const { return m_state.avgLatency; }
    bool isGraphReady() const { return m_state.graphReady; }
    
private:
    // Advanced graph and stream management
    std::unique_ptr<DynamicCudaGraph> m_dynamicGraph;
    std::unique_ptr<PipelineCoordinator> m_coordinator;
    
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
    ::Detector* m_detector = nullptr;
    ::GPUKalmanTracker* m_tracker = nullptr;
    ::GpuPIDController* m_pidController = nullptr;
    
    // Pipeline buffers (GPU memory)
    SimpleCudaMat m_captureBuffer;
    SimpleCudaMat m_preprocessBuffer;
    
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
    float* m_d_pidOutput = nullptr;        // PID controller output (x,y)
    float* m_d_outputBuffer = nullptr;     // Final output buffer
    
    // Pinned host memory for zero-copy access
    unsigned char* m_h_inputBuffer = nullptr;  // Pinned input buffer
    float* m_h_outputBuffer = nullptr;         // Pinned output buffer (x,y)
    
    // Resource management
    cudaGraphicsResource_t m_cudaResource = nullptr;
    cudaArray_t m_cudaArray = nullptr;
    
    // Configuration and state
    UnifiedPipelineConfig m_config;
    GraphExecutionState m_state;
    std::mutex m_graphMutex;
    
    // Internal methods
    bool createGraphNodes(cudaStream_t stream);
    bool addCaptureNode(cudaStream_t stream);
    bool addPreprocessNode(cudaStream_t stream);
    bool addInferenceNode(cudaStream_t stream);
    bool addPostprocessNode(cudaStream_t stream);
    bool addTrackingNode(cudaStream_t stream);
    bool addPIDNode(cudaStream_t stream);
    bool addResultCopyNode(cudaStream_t stream);
    
    bool validateGraph();
    void cleanupGraph();
    void updateStatistics(float latency);
    
    // Buffer allocation
    bool allocateBuffers();
    void deallocateBuffers();
    
    // Two-stage pipeline helpers
    void checkTargetsAsync(cudaStream_t stream);
    void updateProfilingAsync(cudaStream_t stream);
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
    
private:
    PipelineManager() = default;
    ~PipelineManager() = default;
    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;
    
    std::unique_ptr<UnifiedGraphPipeline> m_pipeline;
};

} // namespace needaimbot