// Simple TensorRT inference with optimizations
// - GPU preprocessing (RGB input with bilinear resize)
// - GPU postprocessing (decode + fused target selection + nonlinear P)
// - IoU-based target stickiness (hysteresis)
// - Full CUDA Graph capture (preprocess + inference + postprocess)
// - Pinned host transfers
// - Single D2H transfer (InferenceResult struct, 40 bytes)
// - Event-backed callback worker for low-latency completion handling
// - FP16 input/output support (native, no conversion)
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include "simple_postprocess.h"

namespace gpa {

struct Detection {
    float x1, y1, x2, y2;  // bbox
    float confidence;
    int classId;
};

class SimpleInference {
public:
    // Raw function pointer - zero overhead (no std::function heap allocation)
    using InferenceCallback = void(*)(const InferenceResult&, void*);

    // Pre-allocated callback data (eliminates per-frame heap allocation)
    struct CallbackData {
        InferenceCallback callback = nullptr;
        void* userData = nullptr;
        InferenceResult* resultPtr = nullptr;
    };

    struct LaunchStats {
        uint64_t graph = 0;
        uint64_t standard = 0;
        uint64_t graphFallback = 0;
    };

    SimpleInference();
    ~SimpleInference();

    bool loadEngine(const std::string& enginePath);

    // Must be set before loadEngine(). Values are clamped to a safe range.
    void setMaxDetections(int maxDetections) {
        if (m_loaded) return;
        if (maxDetections < 1) maxDetections = 1;
        if (maxDetections > 4096) maxDetections = 4096;
        m_maxDetections = maxDetections;
    }

    bool captureFullGraphForShape(int sourceWidth, int sourceHeight,
                                  int graphSlotCount,
                                  float confThreshold, int headClassId, float headBonus,
                                  uint32_t allowedClassMask, const AimConfig& aimConfig,
                                  float iouStickinessThreshold, float headYOffset, float bodyYOffset);
    bool isFullGraphReadyForShape(int sourceWidth, int sourceHeight,
                                  int graphSlotCount,
                                  float confThreshold, int headClassId, float headBonus,
                                  uint32_t allowedClassMask, const AimConfig& aimConfig,
                                  float iouStickinessThreshold, float headYOffset, float bodyYOffset) const;

    // Run inference with GPU callback - lowest latency option
    bool runInferenceWithCallback(void* pinnedData, int width, int height,
                                  float confThreshold, int headClassId, float headBonus,
                                  uint32_t allowedClassMask,
                                  const AimConfig& aimConfig,
                                  float iouStickinessThreshold,
                                  float headYOffset, float bodyYOffset,
                                  InferenceCallback callback, void* userData = nullptr);

    int getCallbacksInFlight() const {
        return m_callbacksInFlight.load(std::memory_order_acquire);
    }

    LaunchStats takeLaunchStats();

    // Per-stage timing (opt-in). Must be enabled before loadEngine().
    void setStageTimingEnabled(bool enabled) {
        if (m_loaded) return;
        m_stageTimingEnabled = enabled;
    }

    // Override the completion-callback worker's CPU core. >=0 pins to that core,
    // <0 leaves it unpinned. Must be set before loadEngine() (which starts the
    // worker). When unset, the worker keeps its default of pinning to last-1.
    static constexpr int kCallbackAffinityUnset = -1000;
    void setCallbackAffinity(int core) {
        if (m_loaded) return;
        m_callbackAffinityCore = core;
    }

    struct StageTimingStats {
        uint64_t samples = 0;
        // Microsecond totals (host-side conversion of cudaEventElapsedTime millis)
        uint64_t h2dUsTotal = 0;
        uint64_t preprocessUsTotal = 0;
        uint64_t inferenceUsTotal = 0;
        uint64_t postprocessUsTotal = 0;
        uint64_t d2hUsTotal = 0;
        uint64_t h2dUsMax = 0;
        uint64_t preprocessUsMax = 0;
        uint64_t inferenceUsMax = 0;
        uint64_t postprocessUsMax = 0;
        uint64_t d2hUsMax = 0;
    };
    StageTimingStats takeStageTimingStats();

    cudaStream_t getStream() const { return m_stream; }

private:
    static constexpr int kMaxCallbacksInFlight = 4;
    static constexpr int kMaxGraphShapes = 3;  // LRU shape cache size

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger m_logger;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;

    // GPU buffers
    void* m_d_rawInput = nullptr;    // Raw RGB HWC uint8 input
    void* m_d_chwInput = nullptr;    // CHW float32 or float16 (preprocessed)
    void* m_d_output = nullptr;      // Model output (float32 or float16)
    cudaStream_t m_stream = nullptr;

    int m_maxDetections = 100;

    // GPU fused pipeline buffers
    Detection* m_d_selectedTarget = nullptr;  // Persistent selected target for IoU stickiness
    AimState* m_d_aimState = nullptr;         // Persistent movement state on GPU
    AimConfig* m_d_runtimeAimConfig = nullptr; // Runtime movement parameters read by graph kernels
    Detection* m_d_stage1BestDist = nullptr;  // Stage-1 per-block best-by-distance
    float* m_d_stage1DistScore = nullptr;     // Stage-1 per-block distance score
    Detection* m_d_stage1BestIou = nullptr;   // Stage-1 per-block best-by-IoU
    float* m_d_stage1IouScore = nullptr;      // Stage-1 per-block IoU score

    // Result buffers (per callback slot for both standard and graph paths)
    std::array<InferenceResult*, kMaxCallbacksInFlight> m_d_inferenceResult{};
    std::array<InferenceResult*, kMaxCallbacksInFlight> m_h_inferenceResultPinned{};

    // Pinned host memory for fast transfers
    uint8_t* m_h_rawPinned = nullptr;

    // Multi-shape CUDA Graph cache. One bucket per (sourceW, sourceH) pair.
    // Each bucket holds one graph per result slot so multiple frames can be
    // in flight on the stream without aliasing result buffers. The graphs run
    // shared device buffers; only H2D length + kernel-arg-baked shape differ.
    struct GraphShapeBucket {
        int sourceW = 0;
        int sourceH = 0;
        bool used = false;
        uint64_t lastUseTick = 0;
        int slotCount = 0;
        std::array<cudaGraph_t, kMaxCallbacksInFlight> graphs{};
        std::array<cudaGraphExec_t, kMaxCallbacksInFlight> graphExecs{};
        std::array<cudaGraphNode_t, kMaxCallbacksInFlight> h2dNodes{};
        std::array<size_t, kMaxCallbacksInFlight> rawSizes{};
    };
    std::array<GraphShapeBucket, kMaxGraphShapes> m_shapeBuckets{};
    uint64_t m_graphUseTickCounter = 0;

    // Per-stage CUDA events for timing diagnostics. One event quintuple per
    // result slot - graphs record into these as graph-event nodes. Events
    // are timing-enabled only when m_stageTimingEnabled is true at load time.
    bool m_stageTimingEnabled = false;
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_stageEvtStart{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_stageEvtPostH2D{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_stageEvtPostPreprocess{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_stageEvtPostInference{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_stageEvtPostPostprocess{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_stageEvtEnd{};
    std::array<std::atomic<bool>, kMaxCallbacksInFlight> m_stageEvtValid{};
    std::atomic<uint64_t> m_stageSamples{0};
    std::atomic<uint64_t> m_stageH2DUsTotal{0};
    std::atomic<uint64_t> m_stagePreprocessUsTotal{0};
    std::atomic<uint64_t> m_stageInferenceUsTotal{0};
    std::atomic<uint64_t> m_stagePostprocessUsTotal{0};
    std::atomic<uint64_t> m_stageD2HUsTotal{0};
    std::atomic<uint64_t> m_stageH2DUsMax{0};
    std::atomic<uint64_t> m_stagePreprocessUsMax{0};
    std::atomic<uint64_t> m_stageInferenceUsMax{0};
    std::atomic<uint64_t> m_stagePostprocessUsMax{0};
    std::atomic<uint64_t> m_stageD2HUsMax{0};

    int m_inputH = 320;         // Model input height (target)
    int m_inputW = 320;         // Model input width (target)
    size_t m_rawInputCapacityBytes = 0; // Allocated raw-input capacity on device
    float m_crosshairX = 160.0f;
    float m_crosshairY = 160.0f;
    int m_numBoxes = 2100;
    int m_numClasses = 2;
    bool m_loaded = false;
    bool m_inputFP16 = false;   // Input tensor is FP16
    bool m_outputFP16 = false;  // Output tensor is FP16
    bool m_tensorAddressesBound = false;  // TRT10 static tensor addresses bound once

    // Cached graph parameters
    float m_cachedConfThreshold = 0.35f;
    int m_cachedHeadClassId = 1;
    float m_cachedHeadBonus = 0.15f;
    uint32_t m_cachedAllowedClassMask = 0xFFFFFFFF;
    AimConfig m_enqueuedRuntimeAimConfig;
    bool m_hasEnqueuedRuntimeAimConfig = false;
    float m_cachedIouThreshold = 0.3f;
    float m_cachedHeadYOffset = 1.0f;
    float m_cachedBodyYOffset = 0.15f;

    std::array<CallbackData, kMaxCallbacksInFlight> m_callbackDataSlots{};
    std::array<std::atomic<bool>, kMaxCallbacksInFlight> m_callbackSlotBusy{};
    std::array<std::atomic<bool>, kMaxCallbacksInFlight> m_callbackSlotPending{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_callbackEvents{};
    std::atomic<int> m_callbacksInFlight{0};
    std::atomic<uint64_t> m_graphLaunchCount{0};
    std::atomic<uint64_t> m_standardLaunchCount{0};
    std::atomic<uint64_t> m_graphFallbackCount{0};
    uint32_t m_callbackSlotCursor = 0;
    std::atomic<bool> m_callbackWorkerRunning{false};
    int m_callbackAffinityCore = kCallbackAffinityUnset;  // see setCallbackAffinity()
    std::thread m_callbackWorkerThread;
    std::condition_variable m_callbackWorkerCv;
    std::mutex m_callbackWorkerMutex;

    void callbackWorkerLoop();
    void destroyFullGraphs();              // Destroys every shape bucket
    void destroyBucket(GraphShapeBucket& bucket);
    bool uploadRuntimeAimConfig(const AimConfig& aimConfig, bool force = false);
    // True only when non-shape params match the cached set.
    bool nonShapeParamsMatch(float confThreshold, int headClassId, float headBonus,
                             uint32_t allowedClassMask, const AimConfig& aimConfig,
                             float iouStickinessThreshold, float headYOffset,
                             float bodyYOffset) const;
    int findBucketIndex(int sourceWidth, int sourceHeight) const;
    int pickBucketForCapture(int sourceWidth, int sourceHeight);
    void touchBucket(int bucketIndex);
    bool ensureRawInputCapacity(size_t requiredBytes);
    void recordStageTimings(int slotIndex);

    // Execute full fused pipeline (H2D + preprocess + inference + postprocess + D2H)
    bool executeFusedPipeline(void* rawInput, int width, int height,
                              float confThreshold, int headClassId, float headBonus,
                              uint32_t allowedClassMask, const AimConfig& aimConfig,
                              float iouThreshold, float headYOffset, float bodyYOffset,
                              int resultSlot);

    // Execute pipeline without H2D transfer (for CUDA Graph - H2D is done separately)
    bool executeFusedPipelinePostH2D(int width, int height,
                                     float confThreshold, int headClassId, float headBonus,
                                     uint32_t allowedClassMask, const AimConfig& aimConfig,
                                     float iouThreshold, float headYOffset, float bodyYOffset,
                                     int resultSlot);

    // Input size helper
    static constexpr int inputBytesPerPixel() { return 3; }
};

} // namespace gpa
