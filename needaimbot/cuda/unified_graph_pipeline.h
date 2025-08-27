#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include <memory>
#include <atomic>
#include <array>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include "simple_cuda_mat.h"
#include "cuda_resource_manager.h"
#include "../core/Target.h"
#include "../utils/cuda_utils.h"  // For CudaEvent and CudaStream

// TensorRT includes for Phase 1 integration
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>


namespace needaimbot {

// Forward declarations for internal implementation classes

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
    std::unique_ptr<CudaEvent> startEvent;  // Using RAII wrapper
    std::unique_ptr<CudaEvent> endEvent;    // Using RAII wrapper
    
    // Default constructor
    GraphExecutionState() = default;
    
    // Move constructor
    GraphExecutionState(GraphExecutionState&& other) noexcept
        : graphReady(other.graphReady)
        , needsRebuild(other.needsRebuild)
        , frameCount(other.frameCount)
        , avgLatency(other.avgLatency)
        , lastLatency(other.lastLatency)
        , startEvent(std::move(other.startEvent))
        , endEvent(std::move(other.endEvent))
    {}
    
    // Move assignment operator
    GraphExecutionState& operator=(GraphExecutionState&& other) noexcept {
        if (this != &other) {
            graphReady = other.graphReady;
            needsRebuild = other.needsRebuild;
            frameCount = other.frameCount;
            avgLatency = other.avgLatency;
            lastLatency = other.lastLatency;
            startEvent = std::move(other.startEvent);
            endEvent = std::move(other.endEvent);
        }
        return *this;
    }
    
    // Delete copy constructor and copy assignment
    GraphExecutionState(const GraphExecutionState&) = delete;
    GraphExecutionState& operator=(const GraphExecutionState&) = delete;
};

// Unified pipeline configuration
struct UnifiedPipelineConfig {
    // Pipeline stages enable flags
    bool enableCapture = true;
    bool enableDetection = true;
    
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
};;;



class UnifiedGraphPipeline {
public:
    UnifiedGraphPipeline();
    ~UnifiedGraphPipeline();
    
    // Initialize pipeline with components
    bool initialize(const UnifiedPipelineConfig& config);
    void shutdown();
    
    
    // Main execution methods
    bool captureGraph(cudaStream_t stream = nullptr);
    
    // Non-blocking pipeline execution methods
    bool executeGraphNonBlocking(cudaStream_t stream = nullptr);
    void processMouseMovementAsync();
    
        
    // Pipeline data management
    void setInputTexture(cudaGraphicsResource_t resource) { m_cudaResource = resource; }
    void setDesktopDuplication(void* duplication, void* device, void* context, void* texture) {
        m_desktopDuplication = duplication;
        m_d3dDevice = device;
        m_d3dContext = context;
        m_captureTextureD3D = texture;
    }
    void setInputFrame(const SimpleCudaMat& frame);
    void setOutputBuffer(float* d_output) { 
        // Note: This function is for compatibility with external code
        // The actual output buffer is managed internally with RAII
        // If needed, we can copy data to the external buffer after processing
        m_externalOutputBuffer = d_output;
    }
    
    // TensorRT integration methods (Phase 1)
    bool initializeTensorRT(const std::string& modelFile);
    bool loadEngine(const std::string& modelFile);
    int getModelInputResolution() const;
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
    const GraphExecutionState& getState() const { return m_state; }
    float getAverageLatency() const { return m_state.avgLatency; }
    bool isGraphReady() const { return m_state.graphReady; }
    
    // Frame access for preview
    const SimpleCudaMat& getCaptureBuffer() const { return m_captureBuffer; }
    
    
private:
    // Advanced graph and stream management
    // Simple graph management
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    std::unique_ptr<CudaStream> m_primaryStream;  // Using RAII wrapper
    
    // Dedicated streams for pipeline stages (ordered execution within each stream)
    std::unique_ptr<CudaStream> m_captureStream;    // Frame capture only
    std::unique_ptr<CudaStream> m_inferenceStream;  // Preprocessing + inference + postprocessing
    std::unique_ptr<CudaStream> m_copyStream;       // Host memory transfers
    
    // Simple event for preview (using RAII)
    std::unique_ptr<CudaEvent> m_previewReadyEvent;
    
    // Graph nodes for dynamic updates
    std::vector<cudaGraphNode_t> m_captureNodes;
    std::vector<cudaGraphNode_t> m_inferenceNodes;
    std::vector<cudaGraphNode_t> m_postprocessNodes;
    std::vector<cudaGraphNode_t> m_trackingNodes;
    std::vector<cudaGraphNode_t> m_pidNodes;
    
    // Node name mapping for dynamic updates
    std::unordered_map<std::string, cudaGraphNode_t> m_namedNodes;
    
    // Class filter caching to avoid redundant uploads
    bool m_classFilterDirty = true;
    std::vector<unsigned char> m_cachedClassFilter;
    
    // Simple and efficient stream-based pipeline with ordered execution
    struct TripleBuffer {
        // Shared frame buffers for memory efficiency (8MB total vs 24MB)
        SimpleCudaMat buffers[3];
        std::atomic<int> writeIdx{0};  // Single atomic for write index
        
        // Stage completion events for non-blocking dependency management
        std::array<CudaEvent, 3> captureComplete;
        std::array<CudaEvent, 3> inferenceComplete; 
        std::array<CudaEvent, 3> copyComplete;
        
        // Target data with simple validity flags
        std::array<CudaPinnedMemory<Target>, 3> h_target_coords_pinned;
        bool target_data_valid[3] = {false, false, false};
        int target_count[3] = {0, 0, 0};
        
        // Constructor with proper initialization
        TripleBuffer() {
            // Initialize events with optimal flags
            for (int i = 0; i < 3; i++) {
                captureComplete[i] = CudaEvent(cudaEventDisableTiming);
                inferenceComplete[i] = CudaEvent(cudaEventDisableTiming);
                copyComplete[i] = CudaEvent(cudaEventDisableTiming);
                
                // Initialize pinned memory for zero-copy transfers
                h_target_coords_pinned[i] = CudaPinnedMemory<Target>(1, cudaHostAllocDefault);
                target_data_valid[i] = false;
                target_count[i] = 0;
            }
        }
        
        // Get next write index (atomic increment, no complex state management)
        int getNextWriteIndex() {
            return writeIdx.fetch_add(1, std::memory_order_relaxed) % 3;
        }
        
        // Find ready data for consumption (non-blocking check)
        int findReadyData() {
            for (int i = 0; i < 3; ++i) {
                if (target_data_valid[i] && copyComplete[i].query() == cudaSuccess) {
                    return i;
                }
            }
            return -1; // No ready data
        }
        
        // Mark data as consumed
        void markConsumed(int idx) {
            if (idx >= 0 && idx < 3) {
                target_data_valid[idx] = false;
                target_count[idx] = 0;
            }
        }
        
        // Initialize frame buffers (called once during setup)
        void initializeFrameBuffers(int height, int width, int channels) {
            for (int i = 0; i < 3; i++) {
                if (buffers[i].empty()) {
                    buffers[i].create(height, width, channels);
                    
                    // Clear pinned memory to prevent garbage values
                    if (h_target_coords_pinned[i].get()) {
                        memset(h_target_coords_pinned[i].get(), 0, sizeof(Target));
                    }
                }
            }
        }
        
        // Clear all pending data (for cleanup/reset)
        void clearAllData() {
            for (int i = 0; i < 3; i++) {
                target_data_valid[i] = false;
                target_count[i] = 0;
            }
        }
        
        // Destructor - RAII handles cleanup
        ~TripleBuffer() = default;
    };
    std::unique_ptr<TripleBuffer> m_tripleBuffer;
    
    
    // TensorRT engine management (Phase 1 integration)
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    // Input/Output binding management
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::unordered_map<std::string, size_t> m_inputSizes;
    std::unordered_map<std::string, size_t> m_outputSizes;
    std::unordered_map<std::string, std::unique_ptr<CudaMemory<uint8_t>>> m_inputBindings;
    std::unordered_map<std::string, std::unique_ptr<CudaMemory<uint8_t>>> m_outputBindings;
    
    // Inference related information
    std::string m_inputName;
    nvinfer1::Dims m_inputDims;
    int m_modelInputResolution = 320; // 캐시된 모델 입력 해상도
    float m_imgScale;
    int m_numClasses;
    // Pipeline buffers (GPU memory)
    SimpleCudaMat m_captureBuffer;
    SimpleCudaMat m_preprocessBuffer;
    std::unique_ptr<CudaMemory<float>> m_d_preprocessBuffer;  // Preprocess buffer 
    std::unique_ptr<CudaMemory<Target>> m_d_tracks;           // GPU tracking data
    
    // NMS temporary buffers (allocated once, reused) - Using RAII
    std::unique_ptr<CudaMemory<int>> m_d_numDetections;
    std::unique_ptr<CudaMemory<int>> m_d_x1;
    std::unique_ptr<CudaMemory<int>> m_d_y1;
    std::unique_ptr<CudaMemory<int>> m_d_x2;
    std::unique_ptr<CudaMemory<int>> m_d_y2;
    std::unique_ptr<CudaMemory<float>> m_d_areas;
    std::unique_ptr<CudaMemory<float>> m_d_scores_nms;
    std::unique_ptr<CudaMemory<int>> m_d_classIds_nms;
    std::unique_ptr<CudaMemory<float>> m_d_iou_matrix;
    std::unique_ptr<CudaMemory<bool>> m_d_keep;
    std::unique_ptr<CudaMemory<int>> m_d_indices;
    std::unique_ptr<CudaMemory<int>> m_d_outputCount;
    std::unique_ptr<CudaMemory<float>> m_d_yoloInput;        // YOLO model input (640x640x3)
    std::unique_ptr<CudaMemory<float>> m_d_inferenceOutput;  // Inference output (managed by RAII)
    std::unique_ptr<CudaMemory<float>> m_d_nmsOutput;        // After NMS
    std::unique_ptr<CudaMemory<float>> m_d_filteredOutput;   // After filtering
    std::unique_ptr<CudaMemory<Target>> m_d_detections;      // Detection results
    std::unique_ptr<CudaMemory<Target>> m_d_selectedTarget;  // Selected target
    
    // Post-processing buffers (Phase 3 integration) - Using RAII
    std::unique_ptr<CudaMemory<Target>> m_d_decodedTargets;      // Decoded detections from inference
    std::unique_ptr<CudaMemory<int>> m_d_decodedCount;           // Count of decoded detections
    std::unique_ptr<CudaMemory<Target>> m_d_finalTargets;        // Final NMS output
    std::unique_ptr<CudaMemory<int>> m_d_finalTargetsCount;      // Final count after NMS
    std::unique_ptr<CudaMemory<Target>> m_d_classFilteredTargets; // After class filtering
    std::unique_ptr<CudaMemory<int>> m_d_classFilteredCount;     // Count after class filtering
    std::unique_ptr<CudaMemory<Target>> m_d_colorFilteredTargets; // After color filtering
    std::unique_ptr<CudaMemory<int>> m_d_colorFilteredCount;     // Count after color filtering
    
    // Class filtering control buffer - Using RAII
    std::unique_ptr<CudaMemory<unsigned char>> m_d_allowFlags;   // Class filtering flags
    
    // Additional post-processing metadata
    std::unordered_map<std::string, std::vector<int64_t>> m_outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> m_outputTypes;
    
    // Target selection buffers - Using RAII
    std::unique_ptr<CudaMemory<int>> m_d_bestTargetIndex;        // Selected target index
    std::unique_ptr<CudaMemory<Target>> m_d_bestTarget;          // Selected target data
    std::unique_ptr<CudaMemory<float>> m_d_outputBuffer;     // Final output buffer
    
    // Pinned host memory for zero-copy access
    std::unique_ptr<CudaPinnedMemory<unsigned char>> m_h_inputBuffer;  // Pinned input buffer
    std::unique_ptr<CudaPinnedMemory<float>> m_h_outputBuffer;         // Pinned output buffer (x,y)
    
    // External output buffer (for compatibility)
    float* m_externalOutputBuffer = nullptr;
    
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
    
    // Pipeline state for non-blocking execution
    std::atomic<int> m_currentPipelineIdx{0};    // Current pipeline buffer index
    std::atomic<int> m_prevPipelineIdx{2};       // Previous pipeline buffer index
    
    // Main loop control
    std::atomic<bool> m_shouldStop{false};
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    
    // Event management for profiling and synchronization (using RAII)
    std::unique_ptr<CudaEvent> m_lastFrameEnd;      // For frame timing
    std::unique_ptr<CudaEvent> m_copyEvent;         // For copy synchronization
    
    // Copy state management for async preview updates
    bool m_copyInProgress = false;
    int m_h_finalCount = 0;
    std::vector<Target> m_h_finalTargets;

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