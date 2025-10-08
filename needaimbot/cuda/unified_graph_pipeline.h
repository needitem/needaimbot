#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <functional>
#include "simple_cuda_mat.h"
#include "cuda_resource_manager.h"
#include "../core/Target.h"
#include "../utils/cuda_utils.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>

struct AppContext;
class ICaptureProvider;

namespace needaimbot {

struct MouseMovement {
    int dx;
    int dy;
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
    float nmsThreshold = 0.45f;
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
    
    bool executeFrame(cudaStream_t stream = nullptr);
    bool executeNormalPipeline(cudaStream_t stream = nullptr);
    
        
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
    void setInputFrame(const SimpleCudaMat& frame);
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
    struct RegisteredHostBuffer {
        size_t size = 0;
        bool registered = false;
        bool permanentFailure = false;
    };

    mutable std::atomic<uint64_t> m_lastUIReadFrame{0};
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    std::unique_ptr<CudaStream> m_pipelineStream;
    
    std::unique_ptr<CudaEvent> m_previewReadyEvent;
    
    std::vector<cudaGraphNode_t> m_captureNodes;
    std::vector<cudaGraphNode_t> m_inferenceNodes;
    std::vector<cudaGraphNode_t> m_postprocessNodes;
    
    std::unordered_map<std::string, cudaGraphNode_t> m_namedNodes;
    
    std::atomic<bool> m_classFilterDirty{true};
    std::vector<unsigned char> m_cachedClassFilter;
    std::atomic<int> m_cachedHeadClassId{-1};
    std::atomic<size_t> m_cachedHeadClassNameHash{0};
    std::atomic<size_t> m_cachedClassSettingsSize{0};
    
    // Unified capture buffer - removed duplicate m_unifiedCaptureBuffer
    SimpleCudaMat m_captureBuffer;
    SimpleCudaMat m_nextCaptureBuffer;

    std::unique_ptr<CudaStream> m_captureStream;
    std::unique_ptr<CudaEvent> m_captureReadyEvent;
    bool m_captureInFlight = false;

    bool ensureFrameReady();
    bool scheduleNextFrameCapture(bool forceSync);
    bool waitForCaptureCompletion();
    bool copyFrameToBuffer(void* frameData, unsigned int width, unsigned int height,
                           SimpleCudaMat& targetBuffer, cudaStream_t stream);
    
    
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
    int m_modelInputResolution = 320;
    float m_imgScale;
    int m_numClasses;
    // m_unifiedCaptureBuffer removed - using m_captureBuffer instead
    std::unique_ptr<CudaMemory<float>> m_d_preprocessBuffer;
    
    std::unique_ptr<CudaMemory<float>> m_d_inferenceOutput;
    
    
    std::unordered_map<std::string, std::vector<int64_t>> m_outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> m_outputTypes;
    
    SmallBufferArena m_smallBufferArena;
    
    UnifiedGPUArena m_unifiedArena;
    
    std::unique_ptr<CudaMemory<float>> m_d_outputBuffer;
    
    std::unique_ptr<CudaPinnedMemory<MouseMovement>> m_h_movement;
    std::unique_ptr<CudaPinnedMemory<unsigned char>> m_h_allowFlags;

    bool m_mouseMovementUsesMappedMemory = false;

    bool configureMouseMovementBuffer();

    float* m_externalOutputBuffer = nullptr;

    ICaptureProvider* m_capture = nullptr;
    int m_lastCaptureW = 0;
    int m_lastCaptureH = 0;
    bool m_lastGpuDirect = false;

    std::unordered_map<void*, RegisteredHostBuffer> m_registeredCaptureBuffers;
    
    UnifiedPipelineConfig m_config;
    GraphExecutionState m_state;
    std::mutex m_graphMutex;
    bool m_hasFrameData = false;

    std::atomic<bool> m_allowMovement{false};
    std::atomic<bool> m_shouldStop{false};
    std::atomic<bool> m_frameInFlight{false};
    mutable std::mutex m_movementFilterMutex;
    bool m_skipNextMovement{true};
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
    MouseMovement filterMouseMovement(const MouseMovement& rawMovement, bool movementEnabled);
    void clearHostPreviewData(AppContext& ctx);
    void handleAimbotActivation();

    bool enqueueFrameCompletionCallback(cudaStream_t stream);
    bool enqueueMovementResetCallback(cudaStream_t stream);

    bool updateDDACaptureRegion(const AppContext& ctx);
    bool performFrameCapture();
    bool performFrameCaptureDirectToUnified();
    bool copyDDAFrameToGPU(void* frameData, unsigned int width, unsigned int height);
    bool ensureCaptureBufferRegistered(void* frameData, size_t size);
    void unregisterStaleCaptureBuffers(void* keepPtr);
    void releaseRegisteredCaptureBuffers();
    bool performPreprocessing();
    void updatePreviewBuffer(const SimpleCudaMat& currentBuffer);
    void updatePreviewBufferAllocation();  // Dynamic allocation based on show_window state
    bool performInference();
    int findHeadClassId(AppContext& ctx);

    void clearDetectionBuffers(const PostProcessingConfig& config, cudaStream_t stream);
    cudaError_t decodeYoloOutput(void* d_rawOutputPtr, nvinfer1::DataType outputType, 
                                const std::vector<int64_t>& shape, 
                                const PostProcessingConfig& config, cudaStream_t stream);
    bool validateYoloDecodeBuffers(int maxDecodedTargets, int max_candidates);
    void updateClassFilterIfNeeded(cudaStream_t stream);

    // Removed redundant NMS and copy functions - integrated into main processing
    void handlePreviewUpdate(const PostProcessingConfig& config, cudaStream_t stream);
    void updatePreviewTargets(const PostProcessingConfig& config);
    void startPreviewCopy(const PostProcessingConfig& config, cudaStream_t stream);

    bool m_graphCaptured = false;

    void refreshCachedBindings();
    bool bindStaticTensorAddresses();
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

}
