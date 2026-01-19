#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <atomic>
#include <array>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include "simple_cuda_mat.h"
#include "cuda_resource_manager.h"
#include "../core/Target.h"
#include "../core/constants.h"
#include "../utils/cuda_utils.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>

struct AppContext;
class ICaptureProvider;

namespace gpa {

struct MouseMovement {
    int dx;
    int dy;
};

struct PIDState {
    float prev_error_x;
    float prev_error_y;
    float integral_x;
    float integral_y;
};

// Frame metadata for synchronous capture
struct FrameMetadata {
    uint64_t frameId = 0;          // Monotonic frame counter
    uint64_t presentQpc = 0;       // DXGI LastPresentQpc (0 if unsupported)
    uint32_t width = 0;
    uint32_t height = 0;
};

// Performance metrics for monitoring
struct PerformanceMetrics {
    uint64_t totalFrames = 0;
    double totalLatencyMs = 0.0;
    uint64_t droppedFrames = 0;
    uint64_t duplicateFrames = 0;
    uint64_t busySpinCount = 0;
    uint64_t yieldCount = 0;
    uint64_t sleepCount = 0;
    uint64_t captureWaitCount = 0;
    uint64_t inputPendingCount = 0;
    uint64_t frameSkipCount = 0;
    uint64_t memcpySkipCount = 0;
    std::chrono::steady_clock::time_point lastReportTime;
    
    void reset() {
        totalFrames = 0;
        totalLatencyMs = 0.0;
        droppedFrames = 0;
        duplicateFrames = 0;
        busySpinCount = 0;
        yieldCount = 0;
        sleepCount = 0;
        captureWaitCount = 0;
        inputPendingCount = 0;
        frameSkipCount = 0;
        memcpySkipCount = 0;
        lastReportTime = std::chrono::steady_clock::now();
    }
    
    void logIfNeeded(const char* prefix = "[Perf]") {
        if (totalFrames == 0) return;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastReportTime).count();
        if (elapsed >= 10) {  // Log every 10 seconds
            double avgMs = totalLatencyMs / totalFrames;
            printf("%s Last 10s: %llu frames (dropped=%llu, dup=%llu), %.2fms avg (%.1f FPS)\n",
                   prefix,
                   (unsigned long long)totalFrames,
                   (unsigned long long)droppedFrames,
                   (unsigned long long)duplicateFrames,
                   avgMs,
                   1000.0 / avgMs);
            printf("%s   Waits: busySpin=%llu, yield=%llu, sleep=%llu\n",
                   prefix,
                   (unsigned long long)busySpinCount,
                   (unsigned long long)yieldCount,
                   (unsigned long long)sleepCount);
            printf("%s   Optimizations: captureWait=%llu, inputPending=%llu, frameSkip=%llu, memcpySkip=%llu\n",
                   prefix,
                   (unsigned long long)captureWaitCount,
                   (unsigned long long)inputPendingCount,
                   (unsigned long long)frameSkipCount,
                   (unsigned long long)memcpySkipCount);
            reset();
        }
    }
};

struct SmallBufferArena {
    int* numDetections;
    int* outputCount;
    int* decodedCount;
    int* finalTargetsCount;
    int* classFilteredCount;
    int* bestTargetIndex;
    
    Target* selectedTarget;
    Target* bestTarget;
    MouseMovement* mouseMovement;
    PIDState* pidState;

    unsigned char* allowFlags;
    bool* keepFlags;
    int* tempIndices;
    float* tempScores;
    
    std::unique_ptr<CudaMemory<uint8_t>> arenaBuffer;
    
    void initializePointers(uint8_t* basePtr) {
        size_t offset = 0;
        
        numDetections = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        outputCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        decodedCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        finalTargetsCount = decodedCount;  // Share count storage with decoded results
        classFilteredCount = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        bestTargetIndex = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int);
        
        offset = (offset + alignof(Target) - 1) & ~(alignof(Target) - 1);
        
        selectedTarget = reinterpret_cast<Target*>(basePtr + offset);
        offset += sizeof(Target);
        bestTarget = reinterpret_cast<Target*>(basePtr + offset);
        offset += sizeof(Target);
        
        offset = (offset + alignof(MouseMovement) - 1) & ~(alignof(MouseMovement) - 1);
        mouseMovement = reinterpret_cast<MouseMovement*>(basePtr + offset);
        offset += sizeof(MouseMovement);

        offset = (offset + alignof(PIDState) - 1) & ~(alignof(PIDState) - 1);
        pidState = reinterpret_cast<PIDState*>(basePtr + offset);
        offset += sizeof(PIDState);

        allowFlags = reinterpret_cast<unsigned char*>(basePtr + offset);
        offset += 64;

        offset = (offset + alignof(bool) - 1) & ~(alignof(bool) - 1);
        keepFlags = reinterpret_cast<bool*>(basePtr + offset);
        offset += sizeof(bool) * 128;
        
        offset = (offset + alignof(int) - 1) & ~(alignof(int) - 1);
        tempIndices = reinterpret_cast<int*>(basePtr + offset);
        offset += sizeof(int) * 64;
        
        offset = (offset + alignof(float) - 1) & ~(alignof(float) - 1);
        tempScores = reinterpret_cast<float*>(basePtr + offset);
    }
    
    static size_t calculateArenaSize() {
        size_t size = sizeof(int) * 5;

        size = (size + alignof(Target) - 1) & ~(alignof(Target) - 1);
        size += sizeof(Target) * 2;

        size = (size + alignof(MouseMovement) - 1) & ~(alignof(MouseMovement) - 1);
        size += sizeof(MouseMovement);

        size = (size + alignof(PIDState) - 1) & ~(alignof(PIDState) - 1);
        size += sizeof(PIDState);

        size += 64;
        size = (size + alignof(bool) - 1) & ~(alignof(bool) - 1);
        size += sizeof(bool) * 128;
        size = (size + alignof(int) - 1) & ~(alignof(int) - 1);
        size += sizeof(int) * 64;
        size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
        size += sizeof(float) * 64;

        return size;
    }
};

struct UnifiedGPUArena {
    std::unique_ptr<CudaMemory<uint8_t>> megaArena;
    
    float* yoloInput;
    Target* decodedTargets;
    Target* finalTargets;
    Target* nmsTemp;  // Temporary buffer for NMS compaction
    
    
    void initializePointers(uint8_t* basePtr, int maxDetections, int yoloSize);
    static size_t calculateArenaSize(int maxDetections, int yoloSize);
};



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

struct GraphExecutionState {
    bool graphReady = false;
    bool needsRebuild = false;
    int frameCount = 0;
    float avgLatency = 0.0f;
    float lastLatency = 0.0f;
    std::unique_ptr<CudaEvent> startEvent;
    std::unique_ptr<CudaEvent> endEvent;
    
    GraphExecutionState() = default;
    
    GraphExecutionState(GraphExecutionState&& other) noexcept
        : graphReady(other.graphReady)
        , needsRebuild(other.needsRebuild)
        , frameCount(other.frameCount)
        , avgLatency(other.avgLatency)
        , lastLatency(other.lastLatency)
        , startEvent(std::move(other.startEvent))
        , endEvent(std::move(other.endEvent))
    {}
    
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
    
    GraphExecutionState(const GraphExecutionState&) = delete;
    GraphExecutionState& operator=(const GraphExecutionState&) = delete;
};

struct UnifiedPipelineConfig {
    bool enableCapture = true;
    bool enableDetection = true;
    
    std::string modelPath;
    
    float confThreshold = 0.4f;
    int detectionWidth = 256;
    int detectionHeight = 256;
    
    bool useGraphOptimization = true;
    bool allowGraphUpdate = true;
    bool enableProfiling = false;
    
    int maxBatchSize = 1;
    int graphCaptureMode = cudaStreamCaptureModeGlobal;
    int graphInstantiateFlags = 0;
};;;

struct PostProcessingConfig {
    int max_detections;
    float confidence_threshold;
    std::string postprocess;
    
    void updateFromContext(const AppContext& ctx, bool graphCaptured);
};



class UnifiedGraphPipeline {
public:
    UnifiedGraphPipeline();
    ~UnifiedGraphPipeline();
    
    bool initialize(const UnifiedPipelineConfig& config);
    void shutdown();
    

    bool captureGraph(cudaStream_t stream = nullptr);
    bool updateGraphExec();  // Update existing graph without recapture

    // v2: single unified execution entry point (no failure enums)
    bool executeFrame(cudaStream_t stream = nullptr);
    
        
    void setCapture(ICaptureProvider* capture) { m_capture = capture; }

    struct CaptureStats {
        int lastWidth = 0;
        int lastHeight = 0;
        int roiLeft = 0;
        int roiTop = 0;
        int roiSize = 0;
        bool gpuDirect = false;
        bool hasFrame = false;
        bool previewEnabled = false;
        bool previewHasHost = false;
        const char* backend = nullptr;
    };

    void getCaptureStats(CaptureStats& out) const;
    void setOutputBuffer(float* d_output) { 
        m_externalOutputBuffer = d_output;
    }
    
    bool initializeTensorRT(const std::string& modelFile);
    bool loadEngine(const std::string& modelFile);
    int getModelInputResolution() const;
    void getInputNames();
    void getOutputNames();
    void getBindings();
    bool runInferenceAsync(cudaStream_t stream);
    void performIntegratedPostProcessing(cudaStream_t stream);

    void performTargetSelection(cudaStream_t stream);
    
    void runMainLoop();
    void stopMainLoop();
    
    const GraphExecutionState& getState() const { return m_state; }
    float getAverageLatency() const { return m_state.avgLatency; }
    bool isGraphReady() const { return m_state.graphReady; }
    void setGraphRebuildNeeded() { m_state.needsRebuild = true; }
    
    const SimpleCudaMat& getPreviewBuffer() const { 
        return m_preview.previewBuffer; 
    }
    
    bool isPreviewAvailable() const {
        return m_preview.enabled && !m_preview.previewBuffer.empty();
    }

    bool getPreviewSnapshot(SimpleMat& outFrame);

    struct UIGPUBuffers {
        Target* finalTargets;
        int* finalTargetsCount;
        Target* bestTarget;
        int* bestTargetIndex;
        cudaStream_t uiStream;
    };
    
    UIGPUBuffers getUIGPUBuffers() {
        UIGPUBuffers buffers;
        buffers.finalTargets = m_unifiedArena.finalTargets;
        buffers.finalTargetsCount = m_smallBufferArena.finalTargetsCount;
        buffers.bestTarget = m_smallBufferArena.bestTarget;
        buffers.bestTargetIndex = m_smallBufferArena.bestTargetIndex;
        buffers.uiStream = nullptr;
        return buffers;
    }
    
    bool hasNewFrameData() const { return m_state.frameCount > m_lastUIReadFrame; }
    void markUIFrameRead() { m_lastUIReadFrame = m_state.frameCount; }

    void markFrameCompleted();

private:
    mutable std::atomic<uint64_t> m_lastUIReadFrame{0};
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    std::unique_ptr<CudaStream> m_pipelineStream;
    
    std::unique_ptr<CudaEvent> m_previewReadyEvent;
    
    std::vector<cudaGraphNode_t> m_captureNodes;
    std::vector<cudaGraphNode_t> m_inferenceNodes;
    std::vector<cudaGraphNode_t> m_postprocessNodes;
    
    std::unordered_map<std::string, cudaGraphNode_t> m_namedNodes;

    // Graph update tracking
    cudaGraph_t m_updateGraph = nullptr;  // Temporary graph for update comparison
    cudaGraphNode_t m_preprocessNode = nullptr;
    cudaGraphNode_t m_inferenceNode = nullptr;
    std::vector<cudaGraphNode_t> m_postprocessKernelNodes;

    // v2 Lock-free config cache - updated explicitly or from background, read without locks in hot path
    struct CachedConfig {
        // PID parameters
        struct {
            float kp_x, kp_y;
            float ki_x, ki_y;
            float kd_x, kd_y;
            float integral_max;
            float derivative_max;
        } pid;

        // Target selection
        struct {
            float head_y_offset;
            float body_y_offset;
            float iou_stickiness_threshold;
            int head_class_id;
        } targeting;

        // Detection
        struct {
            int max_detections;
            int detection_resolution;
            float confidence_threshold;
            bool enable_nms;
            float nms_iou_threshold;
            std::array<unsigned char, 80> class_filter;  // Fixed size for cache-friendly access
            std::array<unsigned char, 80> prev_class_filter;  // Previous filter for change detection
        } detection;

        // Movement filtering
        struct {
            int deadband_enter_x = 0;
            int deadband_exit_x = 0;
            int deadband_enter_y = 0;
            int deadband_exit_y = 0;
            bool disable_upward_aim = false;
        } filtering;

        // Color filter for target selection
        struct {
            bool enabled = false;
            int color_mode = 0;  // 0=RGB, 1=HSV
            int target_mode = 0; // 0=ratio, 1=absolute count
            int comparison = 0;  // 0=above (>=), 1=below (<=), 2=between (min-max)
            int r_min = 0, r_max = 255;
            int g_min = 0, g_max = 255;
            int b_min = 0, b_max = 255;
            int h_min = 0, h_max = 179;
            int s_min = 0, s_max = 255;
            int v_min = 0, v_max = 255;
            float min_ratio = 0.1f;
            float max_ratio = 1.0f;
            int min_count = 10;
            int max_count = 10000;
        } color_filter;

        // Generation counter - incremented when config changes
        std::atomic<uint32_t> generation{0};
    } m_cachedConfig;

    // Capture buffer for synchronous frame acquisition
    SimpleCudaMat m_captureBuffer;

    std::unique_ptr<CudaStream> m_previewStream;

    // Frame tracking for duplicate detection
    std::atomic<uint64_t> m_nextFrameId{0};
    std::atomic<uint64_t> m_lastProcessedPresentQpc{0};

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::unordered_map<std::string, size_t> m_inputSizes;
    std::unordered_map<std::string, size_t> m_outputSizes;
    std::unordered_map<std::string, std::unique_ptr<CudaMemory<uint8_t>>> m_inputBindings;
    std::unordered_map<std::string, std::unique_ptr<CudaMemory<uint8_t>>> m_outputBindings;
    std::vector<void*> m_inputAddressCache;
    std::vector<void*> m_outputAddressCache;
    bool m_bindingsNeedUpdate = true;
    int m_primaryInputIndex = -1;
    
    std::string m_inputName;
    nvinfer1::Dims m_inputDims;
    nvinfer1::DataType m_inputDataType = nvinfer1::DataType::kFLOAT;
    int m_modelInputResolution = 320;
    float m_imgScale;
    int m_numClasses;

    std::unique_ptr<CudaMemory<float>> m_d_inferenceOutput;
    
    
    std::unordered_map<std::string, std::vector<int64_t>> m_outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> m_outputTypes;
    
    SmallBufferArena m_smallBufferArena;
    
    UnifiedGPUArena m_unifiedArena;
    
    std::unique_ptr<CudaMemory<float>> m_d_outputBuffer;
    
    std::unique_ptr<CudaPinnedMemory<MouseMovement>> m_h_movement;
    std::unique_ptr<CudaPinnedMemory<unsigned char>> m_h_allowFlags;
    
    // Host pinned buffers for target data (for debug overlay)
    std::unique_ptr<CudaPinnedMemory<Target>> m_h_targets;
    std::unique_ptr<CudaPinnedMemory<int>> m_h_targetCount;
    static constexpr int MAX_HOST_TARGETS = 64;

    bool m_mouseMovementUsesMappedMemory = false;

    bool configureMouseMovementBuffer();

    float* m_externalOutputBuffer = nullptr;

    ICaptureProvider* m_capture = nullptr;
    int m_lastCaptureW = 0;
    int m_lastCaptureH = 0;
    bool m_lastGpuDirect = false;

    UnifiedPipelineConfig m_config;
    GraphExecutionState m_state;
    std::mutex m_graphMutex;
    
    // Performance tracking
    PerformanceMetrics m_perfMetrics;

    std::atomic<bool> m_allowMovement{false};
    std::atomic<bool> m_shouldStop{false};
    std::atomic<bool> m_frameInFlight{false};

    // Movement filter state - per-thread, no lock needed in callback
    struct MovementFilterState {
        bool skipNext = true;
        bool inSettleX = false;
        bool inSettleY = false;
        int lastEmitX = 0;
        int lastEmitY = 0;
    } m_filterState;

    std::chrono::steady_clock::time_point m_lastFrameTime{};
    mutable std::mutex m_previewMutex;
    
    
    struct PreviewState {
        bool enabled = false;
        bool copyInProgress = false;
        bool hasValidHostPreview = false;
        int finalCount = 0;
        std::vector<Target> finalTargets;
        SimpleCudaMat previewBuffer;
        SimpleMat hostPreview;
        // Host preview pinning for faster D2H copies
        bool hostPreviewPinned = false;
        size_t hostPreviewPinnedSize = 0;
        // Throttle preview copies to reduce overhead
        std::chrono::steady_clock::time_point lastCopyTime{};
    } m_preview;

    struct CaptureRegionCache {
        int detectionRes = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        bool usingAimShootOffset = false;
        int left = 0;
        int top = 0;
        int size = 0;
    } m_captureRegionCache;

    // Timestamp (QPC) of the last mouse input injection we issued.
    // Next capture waits until a frame with LastPresentTime >= this value.
    std::atomic<uint64_t> m_pendingInputQpc{0};

    // Whether DXGI LastPresentQpc is supported by the capture backend.
    bool m_qpcSupported = false;

    bool validateGraph();
    void cleanupGraph();
    bool allocateBuffers();
    void deallocateBuffers();

    bool ensurePrimaryInputBindingAliased();
    void ensureFinalTargetAliases();

    bool capturePreprocessGraph(cudaStream_t stream);
    bool captureInferenceGraph(cudaStream_t stream);
    bool capturePostprocessGraph(cudaStream_t stream);
    bool captureTrackingGraph(cudaStream_t stream);

    void handleAimbotDeactivation();
    void clearCountBuffers();
    void clearMovementData();
    void resetMovementFilter();
    void invalidateSelectedTarget(cudaStream_t stream = nullptr);
    MouseMovement filterMouseMovement(const MouseMovement& raw, bool enabled);
    void clearHostPreviewData(AppContext& ctx);
    void handleAimbotActivation();

    // Frame completion callback
    bool enqueueFrameCompletionCallback(cudaStream_t stream, const FrameMetadata& metadata);

    bool updateDDACaptureRegion(const AppContext& ctx);

    // Pipeline stage helpers
    bool acquireFrameSync(FrameMetadata& outMetadata);  // Synchronous capture
    bool performPreprocessing(cudaStream_t stream);
    void updatePreviewBuffer(const SimpleCudaMat& currentBuffer);
    void updatePreviewBufferAllocation();
    bool performInference(cudaStream_t stream);

    void clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream);
    cudaError_t decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType,
                                const std::vector<int64_t>& shape,
                                const PostProcessingConfig& config, cudaStream_t stream);
    bool validateYoloDecodeBuffers(int maxDecodedTargets, int max_candidates);

    void handlePreviewUpdate(const PostProcessingConfig& config, cudaStream_t stream);
    void updatePreviewTargets(const PostProcessingConfig& config);
    void startPreviewCopy(const PostProcessingConfig& config, cudaStream_t stream);

    bool m_graphCaptured = false;

    void refreshCachedBindings();
    bool bindStaticTensorAddresses();

    // Config cache helpers
    void refreshConfigCache(const AppContext& ctx);
    void updateConfig(const AppContext& ctx);

    // Post-processing config (member instead of static for thread safety)
    PostProcessingConfig m_postProcessConfig{Constants::MAX_DETECTIONS, 0.001f, "yolo12"};
    
    // Config generation tracking for change-detection based updates
    uint32_t m_lastConfigGeneration = 0;

public:
    // Mark runtime-controlled parameters dirty; in v2 this triggers a config cache refresh.
    void markPidConfigDirty();
};

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

} // namespace gpa
