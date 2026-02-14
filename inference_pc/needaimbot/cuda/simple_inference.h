// Simple TensorRT inference with optimizations
// - GPU preprocessing (BGRA/RGB input with bilinear resize)
// - GPU postprocessing (decode + fused target selection + PID)
// - IoU-based target stickiness (hysteresis)
// - Full CUDA Graph capture (preprocess + inference + postprocess)
// - Zero-copy pinned memory transfers
// - Single D2H transfer (InferenceResult struct, 40 bytes)
// - GPU Callback API (cudaLaunchHostFunc) for lowest latency
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
        SimpleInference* owner = nullptr;
        int slotIndex = -1;
    };

    SimpleInference();
    ~SimpleInference();

    bool loadEngine(const std::string& enginePath);

    // Set BGRA input mode (call before captureFullGraph)
    // When true, preprocessing expects BGRA HWC input (4 bytes/pixel)
    // When false, preprocessing expects RGB HWC input (3 bytes/pixel)
    void setBgraInput(bool bgra) { m_bgraInput = bgra; }
    // Must be set before loadEngine(). Values are clamped to a safe range.
    void setMaxDetections(int maxDetections) {
        if (m_loaded) return;
        if (maxDetections < 1) maxDetections = 1;
        if (maxDetections > 4096) maxDetections = 4096;
        m_maxDetections = maxDetections;
    }

    // Capture full CUDA graph (preprocess + inference + decode + fused)
    bool captureFullGraph(float confThreshold, int headClassId, float headBonus,
                          uint32_t allowedClassMask, const PIDConfig& pidConfig,
                          float iouStickinessThreshold, float headYOffset, float bodyYOffset);

    // Run inference with GPU callback - lowest latency option
    bool runInferenceWithCallback(void* pinnedData, int width, int height,
                                  float confThreshold, int headClassId, float headBonus,
                                  uint32_t allowedClassMask,
                                  const PIDConfig& pidConfig,
                                  float iouStickinessThreshold,
                                  float headYOffset, float bodyYOffset,
                                  InferenceCallback callback, void* userData = nullptr);

    bool isCallbackInFlight() const {
        return m_callbacksInFlight.load(std::memory_order_acquire) > 0;
    }

    bool hasSubmissionCapacity() const {
        return m_callbacksInFlight.load(std::memory_order_acquire) < kMaxCallbacksInFlight;
    }

    cudaStream_t getStream() const { return m_stream; }

private:
    static constexpr int kMaxCallbacksInFlight = 4;

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger m_logger;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;

    // GPU buffers
    void* m_d_rawInput = nullptr;    // Raw input (RGB or BGRA HWC uint8)
    void* m_d_chwInput = nullptr;    // CHW float32 or float16 (preprocessed)
    void* m_d_output = nullptr;      // Model output (float32 or float16)
    cudaStream_t m_stream = nullptr;

    int m_maxDetections = 100;

    // GPU fused pipeline buffers
    Detection* m_d_selectedTarget = nullptr;  // Persistent selected target for IoU stickiness
    PIDState* m_d_pidState = nullptr;         // Persistent PID state on GPU
    Detection* m_d_stage1BestDist = nullptr;  // Stage-1 per-block best-by-distance
    float* m_d_stage1DistScore = nullptr;     // Stage-1 per-block distance score
    Detection* m_d_stage1BestIou = nullptr;   // Stage-1 per-block best-by-IoU
    float* m_d_stage1IouScore = nullptr;      // Stage-1 per-block IoU score

    // Result buffers (per callback slot for standard path, slot 0 for graph path)
    std::array<InferenceResult*, kMaxCallbacksInFlight> m_d_inferenceResult{};
    std::array<InferenceResult*, kMaxCallbacksInFlight> m_h_inferenceResultPinned{};

    // Pinned host memory for fast transfers
    uint8_t* m_h_rawPinned = nullptr;

    // CUDA Graph for full pipeline
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCaptured = false;

    int m_inputH = 320;         // Model input height (target)
    int m_inputW = 320;         // Model input width (target)
    size_t m_rawInputBytes = 0; // Bytes per input frame at model resolution
    size_t m_rawInputCapacityBytes = 0; // Allocated raw-input capacity on device
    float m_crosshairX = 160.0f;
    float m_crosshairY = 160.0f;
    int m_numBoxes = 2100;
    int m_numClasses = 2;
    bool m_loaded = false;
    bool m_inputFP16 = false;   // Input tensor is FP16
    bool m_outputFP16 = false;  // Output tensor is FP16
    bool m_bgraInput = false;   // True for BGRA input, false for RGB
    bool m_tensorAddressesBound = false;  // TRT10 static tensor addresses bound once

    // Cached graph parameters
    float m_cachedConfThreshold = 0.35f;
    int m_cachedHeadClassId = 1;
    float m_cachedHeadBonus = 0.15f;
    uint32_t m_cachedAllowedClassMask = 0xFFFFFFFF;
    PIDConfig m_cachedPidConfig;
    float m_cachedIouThreshold = 0.3f;
    float m_cachedHeadYOffset = 1.0f;
    float m_cachedBodyYOffset = 0.15f;

    std::array<CallbackData, kMaxCallbacksInFlight> m_callbackDataSlots{};
    std::array<std::atomic<bool>, kMaxCallbacksInFlight> m_callbackSlotBusy{};
    std::array<std::atomic<bool>, kMaxCallbacksInFlight> m_callbackSlotPending{};
    std::array<cudaEvent_t, kMaxCallbacksInFlight> m_callbackEvents{};
    std::atomic<int> m_callbacksInFlight{0};
    uint32_t m_callbackSlotCursor = 0;
    std::atomic<bool> m_callbackWorkerRunning{false};
    std::thread m_callbackWorkerThread;
    std::condition_variable m_callbackWorkerCv;
    std::mutex m_callbackWorkerMutex;

    void callbackWorkerLoop();

    // Execute full fused pipeline (H2D + preprocess + inference + postprocess + D2H)
    bool executeFusedPipeline(void* rawInput, int width, int height,
                              float confThreshold, int headClassId, float headBonus,
                              uint32_t allowedClassMask, const PIDConfig& pidConfig,
                              float iouThreshold, float headYOffset, float bodyYOffset,
                              int resultSlot);

    // Execute pipeline without H2D transfer (for CUDA Graph - H2D is done separately)
    bool executeFusedPipelinePostH2D(int width, int height,
                                     float confThreshold, int headClassId, float headBonus,
                                     uint32_t allowedClassMask, const PIDConfig& pidConfig,
                                     float iouThreshold, float headYOffset, float bodyYOffset,
                                     int resultSlot);

    // Input size helper
    int inputBytesPerPixel() const { return m_bgraInput ? 4 : 3; }
};

} // namespace gpa
