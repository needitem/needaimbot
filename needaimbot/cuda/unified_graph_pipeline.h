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

// Forward declarations
class AppContext;
struct D3D11_TEXTURE2D_DESC;
struct D3D11_BOX;

namespace needaimbot {

// Mouse movement result structure for unified GPU-CPU transfer
struct MouseMovement {
    int dx;
    int dy;
};

// Enhanced memory arena structure for small frequently used buffers
struct SmallBufferArena {
    // Integer counters (7 total)
    int* numDetections;           // Detection count
    int* outputCount;            // Output count
    int* decodedCount;           // Decoded detections count  
    int* finalTargetsCount;      // Final NMS count
    int* classFilteredCount;     // Class filtered count
    int* colorFilteredCount;     // Color filtered count  
    int* bestTargetIndex;        // Selected target index
    
    // Structure data (3 total)
    Target* selectedTarget;      // Selected target (1)
    Target* bestTarget;          // Best target (1) 
    MouseMovement* mouseMovement; // Mouse movement result (1)
    
    // Additional small buffers to reduce allocation overhead (NEW)
    unsigned char* allowFlags;    // Class filtering flags (64 bytes)
    bool* keepFlags;             // NMS keep flags (small buffer)
    int* tempIndices;            // Temporary indices for sorting (small buffer)
    float* tempScores;           // Temporary scores buffer (small buffer)
    
    // Raw arena memory
    std::unique_ptr<CudaMemory<uint8_t>> arenaBuffer;
    
    // Initialize pointers to arena offsets
    void initializePointers(uint8_t* basePtr) {
        size_t offset = 0;
        
        // Integer counters (aligned to int boundary)
        numDetections = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        outputCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        decodedCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        finalTargetsCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        classFilteredCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        colorFilteredCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        bestTargetIndex = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        
        // Align to Target boundary (typically 8-byte aligned)
        offset = (offset + alignof(Target) - 1) & ~(alignof(Target) - 1);
        
        selectedTarget = reinterpret_cast<Target*>(basePtr + offset);
        offset += sizeof(Target);
        bestTarget = reinterpret_cast<Target*>(basePtr + offset);
        offset += sizeof(Target);
        
        // Align to MouseMovement boundary
        offset = (offset + alignof(MouseMovement) - 1) & ~(alignof(MouseMovement) - 1);
        mouseMovement = reinterpret_cast<MouseMovement*>(basePtr + offset);
        offset += sizeof(MouseMovement);
        
        // Additional small buffers (NEW)
        allowFlags = reinterpret_cast<unsigned char*>(basePtr + offset);
        offset += 64; // Constants::MAX_CLASSES_FOR_FILTERING
        
        // Align to bool boundary
        offset = (offset + alignof(bool) - 1) & ~(alignof(bool) - 1);
        keepFlags = reinterpret_cast<bool*>(basePtr + offset);
        offset += sizeof(bool) * 128; // Small buffer for NMS keep flags
        
        // Align to int boundary
        offset = (offset + alignof(int) - 1) & ~(alignof(int) - 1);
        tempIndices = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int) * 64; // Temporary indices buffer
        
        // Align to float boundary  
        offset = (offset + alignof(float) - 1) & ~(alignof(float) - 1);
        tempScores = reinterpret_cast<float*>(basePtr + offset);
    }
    
    // Calculate total arena size needed (ENHANCED)
    static size_t calculateArenaSize() {
        size_t size = sizeof(int) * 7;  // 7 integer counters
        
        // Align for Target structures
        size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
        size += sizeof(Target) * 2;  // selectedTarget, bestTarget
        
        // Align for MouseMovement
        size = (size + alignof(MouseMovement) - 1) & ~(alignof(MouseMovement) - 1);
        size += sizeof(MouseMovement);
        
        // Additional small buffers (NEW - reduces individual allocations)
        size += 64;                    // allowFlags (class filtering)
        size = (size + alignof(bool) - 1) & ~(alignof(bool) - 1);
        size += sizeof(bool) * 128;    // keepFlags (NMS)
        size = (size + alignof(int) - 1) & ~(alignof(int) - 1);
        size += sizeof(int) * 64;      // tempIndices
        size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
        size += sizeof(float) * 64;    // tempScores
        
        return size;
    }
};

// OPTIMIZATION: Unified GPU Memory Arena with dynamic IOU allocation
struct UnifiedGPUArena {
    std::unique_ptr<CudaMemory<uint8_t>> megaArena;
    
    // Pointers to different buffer types within the arena
    float* yoloInput;              // YOLO model input buffer
    float* nmsOutput;              // NMS algorithm output
    float* filteredOutput;         // Post-filtering output
    Target* decodedTargets;        // Decoded detection targets
    Target* finalTargets;          // Final processed targets
    Target* classFilteredTargets;  // Class-filtered targets
    Target* colorFilteredTargets;  // Color-filtered targets  
    Target* detections;            // Raw detections
    
    // NMS working buffers
    int* x1; int* y1; int* x2; int* y2;  // Bounding box coordinates
    float* areas;                         // Target areas
    float* scores_nms;                   // NMS scores
    int* classIds_nms;                   // NMS class IDs
    bool* keep;                          // NMS keep flags
    int* indices;                        // Target indices
    
    // OPTIMIZATION: Dynamic IOU matrix allocation (saves 4-64MB)
    std::unique_ptr<CudaMemory<float>> iou_matrix_dynamic;  // Allocated on-demand based on actual detections
    float* iou_matrix = nullptr;                           // Points to dynamic allocation
    size_t current_iou_size = 0;                          // Track current allocation size
    
    void initializePointers(uint8_t* basePtr, int maxDetections, int yoloSize);
    static size_t calculateArenaSize(int maxDetections, int yoloSize);
    
    // Dynamic IOU matrix management
    bool allocateIOUMatrix(int actualDetections, cudaStream_t stream = nullptr);
    void releaseIOUMatrix();
};

// OPTIMIZATION: Double Buffer (replaces Triple Buffer for 33% memory savings)
class DoubleBuffer {
public:
    SimpleCudaMat buffers[2];  // Only 2 buffers instead of 3
    std::array<CudaEvent, 2> captureComplete;
    std::array<CudaEvent, 2> preprocessComplete;
    std::array<CudaEvent, 2> inferenceComplete;
    std::array<CudaEvent, 2> copyComplete;
    
    // Pinned host memory for results
    std::array<std::unique_ptr<CudaPinnedMemory<MouseMovement>>, 2> h_movement_pinned;
    std::array<bool, 2> movement_data_ready{false, false};
    
    std::atomic<int> writeIndex{0};
    std::atomic<int> readIndex{1};
    
    void initializeFrameBuffers(int height, int width, int channels);
    int getNextWriteIndex();
    int findReadyMovementData();
    void markMovementConsumed(int index);
    void clearAllData();
};

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

// Forward declaration of PostProcessingConfig struct
struct PostProcessingConfig {
    int max_detections;
    float nms_threshold;
    float confidence_threshold;
    std::string postprocess;
    
    void updateFromContext(const AppContext& ctx, bool graphCaptured);
};



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
    const SimpleCudaMat& getCaptureBuffer() const { 
        return m_preview.enabled && !m_preview.previewBuffer.empty() ? m_preview.previewBuffer : m_unifiedCaptureBuffer; 
    }
    
    
private:
    // Advanced graph and stream management
    // Simple graph management
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    // OPTIMIZATION: Unified CUDA streams (4→2 reduction for better performance)
    std::unique_ptr<CudaStream> m_pipelineStream;   // capture + inference + preprocessing
    std::unique_ptr<CudaStream> m_outputStream;     // postprocessing + copy + output
    
    // Simple event for preview (using RAII)
    std::unique_ptr<CudaEvent> m_previewReadyEvent;
    
    // Graph nodes for dynamic updates
    std::vector<cudaGraphNode_t> m_captureNodes;
    std::vector<cudaGraphNode_t> m_inferenceNodes;
    std::vector<cudaGraphNode_t> m_postprocessNodes;
    
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
        std::array<CudaEvent, 3> preprocessComplete;  // Added for preprocessing completion
        std::array<CudaEvent, 3> inferenceComplete; 
        std::array<CudaEvent, 3> copyComplete;
        
        // Mouse movement data unified structure (8 bytes per frame)
        std::array<CudaPinnedMemory<MouseMovement>, 3> h_movement_pinned;  // Combined dx, dy
        bool movement_data_ready[3] = {false, false, false};              // Validity managed here
        
        // Constructor with proper initialization
        TripleBuffer() {
            // Initialize events with optimal flags
            for (int i = 0; i < 3; i++) {
                captureComplete[i] = CudaEvent(cudaEventDisableTiming);
                preprocessComplete[i] = CudaEvent(cudaEventDisableTiming);
                inferenceComplete[i] = CudaEvent(cudaEventDisableTiming);
                copyComplete[i] = CudaEvent(cudaEventDisableTiming);
                
                // Initialize pinned memory for unified mouse movement data
                h_movement_pinned[i] = CudaPinnedMemory<MouseMovement>(1, cudaHostAllocDefault);
                movement_data_ready[i] = false;
            }
        }
        
        // Get next write index (atomic increment, no complex state management)
        int getNextWriteIndex() {
            return writeIdx.fetch_add(1, std::memory_order_relaxed) % 3;
        }
        
        // Find ready mouse movement data (non-blocking check)
        int findReadyMovementData() {
            for (int i = 0; i < 3; ++i) {
                if (movement_data_ready[i] && copyComplete[i].query() == cudaSuccess) {
                    return i;
                }
            }
            return -1; // No ready data
        }
        
        // Mark movement data as consumed
        void markMovementConsumed(int idx) {
            if (idx >= 0 && idx < 3) {
                movement_data_ready[idx] = false;
            }
        }
        
        // Initialize frame buffers (called once during setup)
        void initializeFrameBuffers(int height, int width, int channels) {
            for (int i = 0; i < 3; i++) {
                if (buffers[i].empty()) {
                    buffers[i].create(height, width, channels);
                    
                    // Clear pinned memory to prevent garbage values
                    if (h_movement_pinned[i].get()) {
                        h_movement_pinned[i].get()->dx = 0;
                        h_movement_pinned[i].get()->dy = 0;
                    }
                }
            }
        }
        
        // Clear all pending movement data (for cleanup/reset)
        void clearAllData() {
            for (int i = 0; i < 3; i++) {
                movement_data_ready[i] = false;
            }
        }
        
        // Check if any GPU work is still active (non-blocking)
        bool hasActiveWork() {
            for (int i = 0; i < 3; i++) {
                if (captureComplete[i].query() != cudaSuccess ||
                    preprocessComplete[i].query() != cudaSuccess ||
                    inferenceComplete[i].query() != cudaSuccess ||
                    copyComplete[i].query() != cudaSuccess) {
                    return true;  // Some work is still pending
                }
            }
            return false;  // All work completed
        }
        
        // Destructor - RAII handles cleanup
        ~TripleBuffer() = default;
    };
    // OPTIMIZATION: Double buffer instead of triple buffer (33% memory savings)
    std::unique_ptr<DoubleBuffer> m_doubleBuffer;
    
    
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
    // OPTIMIZATION: Unified buffer for capture and preprocessing (in-place processing)
    // Saves 25% memory by eliminating redundant buffer
    SimpleCudaMat m_unifiedCaptureBuffer;  // Single buffer for capture + preprocessing
    std::unique_ptr<CudaMemory<float>> m_d_preprocessBuffer;  // Preprocess buffer 
    
    // OPTIMIZATION: All buffers moved to UnifiedGPUArena (20+ allocations → 1)
    std::unique_ptr<CudaMemory<float>> m_d_inferenceOutput;  // TensorRT inference output (separate from arena)
    
    // Class filtering control buffer - Using RAII
    // m_d_allowFlags moved to SmallBufferArena for memory efficiency
    
    // Additional post-processing metadata
    std::unordered_map<std::string, std::vector<int64_t>> m_outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> m_outputTypes;
    
    // Memory arena for small frequently allocated buffers (OPTIMIZATION)
    SmallBufferArena m_smallBufferArena;
    
    // OPTIMIZATION: Unified GPU memory arena (replaces 15+ individual allocations)
    UnifiedGPUArena m_unifiedArena;
    
    std::unique_ptr<CudaMemory<float>> m_d_outputBuffer;         // Final output buffer
    
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
    
    // OPTIMIZATION: Event Pool for reusing CUDA events
    struct EventPool {
        std::vector<std::unique_ptr<CudaEvent>> available;
        std::vector<std::unique_ptr<CudaEvent>> inUse;
        
        CudaEvent* acquire() {
            if (available.empty()) {
                inUse.push_back(std::make_unique<CudaEvent>(cudaEventDisableTiming));
                return inUse.back().get();
            }
            inUse.push_back(std::move(available.back()));
            available.pop_back();
            return inUse.back().get();
        }
        
        void release(CudaEvent* event) {
            auto it = std::find_if(inUse.begin(), inUse.end(),
                [event](const std::unique_ptr<CudaEvent>& e) { return e.get() == event; });
            if (it != inUse.end()) {
                available.push_back(std::move(*it));
                inUse.erase(it);
            }
        }
        
        void clear() {
            available.clear();
            inUse.clear();
        }
    } m_eventPool;
    
    // Event management using pool
    CudaEvent* m_lastFrameEnd = nullptr;      // From event pool
    CudaEvent* m_copyEvent = nullptr;         // From event pool
    
    // OPTIMIZATION: Preview-conditional state (only allocated when show_window=true)
    struct PreviewState {
        bool enabled = false;
        bool copyInProgress = false;
        int finalCount = 0;
        std::vector<Target> finalTargets;
        SimpleCudaMat previewBuffer;  // Only allocated when preview enabled
    } m_preview;

    bool validateGraph();
    void cleanupGraph();
    void updateStatistics(float latency);
    
    // Buffer allocation
    bool allocateBuffers();
    void deallocateBuffers();
    
    // Two-stage pipeline helpers
    void updateProfilingAsync(cudaStream_t stream);
    
    // Graph capture methods
    bool capturePreprocessGraph(cudaStream_t stream);
    bool captureInferenceGraph(cudaStream_t stream);
    bool capturePostprocessGraph(cudaStream_t stream);
    bool captureTrackingGraph(cudaStream_t stream);

    // Main loop helper methods
    void handleAimbotDeactivation();
    void clearCountBuffers();
    void clearDoubleBufferData();
    void clearHostPreviewData(AppContext& ctx);
    void handleAimbotActivation();
    bool executePipelineWithErrorHandling();

    // Pipeline execution helper methods
    std::pair<int, int> calculateCaptureCenter(const AppContext& ctx, const D3D11_TEXTURE2D_DESC& desktopDesc);
    D3D11_BOX createCaptureBox(int centerX, int centerY, int captureSize, const D3D11_TEXTURE2D_DESC& desktopDesc);
    bool performDesktopCapture(int writeIdx, const AppContext& ctx);
    bool performFrameCapture(int writeIdx);
    bool performPreprocessing(int writeIdx);
    void updatePreviewBuffer(const SimpleCudaMat& currentBuffer);
    bool performInference(int writeIdx);
    int findHeadClassId(const AppContext& ctx);
    bool performResultCopy(int writeIdx);

    // Post-processing helper methods
    void clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream);
    cudaError_t decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType, 
                                const std::vector<int64_t>& shape, 
                                const PostProcessingConfig& config, cudaStream_t stream);
    bool validateYoloDecodeBuffers(int maxDecodedTargets, int max_candidates);
    void updateClassFilterIfNeeded(cudaStream_t stream);

    // NMS processing methods
    void performNMSProcessing(const PostProcessingConfig& config, cudaStream_t stream);
    void copyDecodedToFinalTargets(const PostProcessingConfig& config, cudaStream_t stream);
    void performStandardNMS(const PostProcessingConfig& config, cudaStream_t stream);
    bool validateNMSBuffers();
    void executeNMSKernel(const PostProcessingConfig& config, cudaStream_t stream);
    void handleNMSResults(const PostProcessingConfig& config, cudaStream_t stream);
    void handlePreviewUpdate(const PostProcessingConfig& config, cudaStream_t stream);
    void updatePreviewTargets(const PostProcessingConfig& config);
    void startPreviewCopy(const PostProcessingConfig& config, cudaStream_t stream);
    
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