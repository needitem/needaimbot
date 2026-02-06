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

#include <string>
#include <cstdint>
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
    // Pre-allocated callback data (public for static callback function access)
    struct CallbackData {
        void(*callback)(const InferenceResult&, void*);
        void* userData;
        InferenceResult* resultPtr;
    };

    SimpleInference();
    ~SimpleInference();

    bool loadEngine(const std::string& enginePath);
    bool isLoaded() const { return m_loaded; }

    // Set BGRA input mode (call before captureFullGraph)
    // When true, preprocessing expects BGRA HWC input (4 bytes/pixel)
    // When false, preprocessing expects RGB HWC input (3 bytes/pixel)
    void setBgraInput(bool bgra) { m_bgraInput = bgra; }
    bool isBgraInput() const { return m_bgraInput; }

    // Capture full CUDA graph (preprocess + inference + decode + fused)
    bool captureFullGraph(float confThreshold, int headClassId, float headBonus,
                          uint32_t allowedClassMask, const PIDConfig& pidConfig,
                          float iouStickinessThreshold, float headYOffset, float bodyYOffset);
    bool isGraphCaptured() const { return m_graphCaptured; }

    // =========================================================================
    // GPU CALLBACK API - RECOMMENDED (lowest latency)
    // =========================================================================
    // Uses cudaLaunchHostFunc to execute callback immediately when GPU finishes.
    // No CPU waiting - callback runs on CUDA's internal thread.
    //
    // When CUDA graph is captured, uses graph launch for faster execution.
    //
    // Callback signature: void callback(const InferenceResult& result, void* userData)
    //
    // THREAD SAFETY: Callback runs on CUDA thread, NOT main thread!
    // - Keep callback fast (just send mouse command)
    // - Don't access non-thread-safe resources
    // =========================================================================

    // Raw function pointer - zero overhead (no std::function heap allocation)
    using InferenceCallback = void(*)(const InferenceResult&, void*);

    bool runInferenceWithCallback(void* pinnedData, int width, int height,
                                  float confThreshold, int headClassId, float headBonus,
                                  uint32_t allowedClassMask,
                                  const PIDConfig& pidConfig,
                                  float iouStickinessThreshold,
                                  float headYOffset, float bodyYOffset,
                                  InferenceCallback callback, void* userData = nullptr);

    int getModelResolution() const { return m_inputH; }
    int getNumClasses() const { return m_numClasses; }
    bool isInputFP16() const { return m_inputFP16; }
    bool isOutputFP16() const { return m_outputFP16; }
    cudaStream_t getStream() const { return m_stream; }

private:
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

    // GPU postprocessing buffers
    Detection* m_d_decoded = nullptr;     // Decoded detections on GPU
    int* m_d_decodedCount = nullptr;      // Detection count on GPU
    Detection* m_d_bestTarget = nullptr;  // Best target on GPU
    int* m_d_hasTarget = nullptr;         // Whether target found (GPU)
    static constexpr int kMaxDetections = 100;

    // GPU fused pipeline buffers
    Detection* m_d_selectedTarget = nullptr;  // Persistent selected target for IoU stickiness
    PIDState* m_d_pidState = nullptr;         // Persistent PID state on GPU
    MouseMovement* m_d_mouseMovement = nullptr; // Mouse movement output on GPU

    // Combined result buffer for single D2H transfer
    InferenceResult* m_d_inferenceResult = nullptr;  // GPU
    InferenceResult* m_h_inferenceResultPinned = nullptr;  // Pinned host

    // Pinned host memory for fast transfers
    uint8_t* m_h_rawPinned = nullptr;

    // CUDA Graph for full pipeline
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCaptured = false;

    int m_inputH = 320;         // Model input height (target)
    int m_inputW = 320;         // Model input width (target)
    int m_numBoxes = 2100;
    int m_numClasses = 2;
    bool m_loaded = false;
    bool m_inputFP16 = false;   // Input tensor is FP16
    bool m_outputFP16 = false;  // Output tensor is FP16
    bool m_bgraInput = false;   // True for BGRA input, false for RGB

    // Cached graph parameters
    float m_cachedConfThreshold = 0.35f;
    int m_cachedHeadClassId = 1;
    float m_cachedHeadBonus = 0.15f;
    uint32_t m_cachedAllowedClassMask = 0xFFFFFFFF;
    PIDConfig m_cachedPidConfig;
    float m_cachedIouThreshold = 0.3f;
    float m_cachedHeadYOffset = 1.0f;
    float m_cachedBodyYOffset = 0.15f;

    // Pre-allocated callback data (eliminates per-frame heap allocation)
    CallbackData m_callbackData{};

    // Execute full fused pipeline (H2D + preprocess + inference + postprocess + D2H)
    void executeFusedPipeline(void* rawInput, int width, int height,
                              float confThreshold, int headClassId, float headBonus,
                              uint32_t allowedClassMask, const PIDConfig& pidConfig,
                              float iouThreshold, float headYOffset, float bodyYOffset);

    // Execute pipeline without H2D transfer (for CUDA Graph - H2D is done separately)
    void executeFusedPipelinePostH2D(int width, int height,
                                      float confThreshold, int headClassId, float headBonus,
                                      uint32_t allowedClassMask, const PIDConfig& pidConfig,
                                      float iouThreshold, float headYOffset, float bodyYOffset);

    // Input size helper
    int inputBytesPerPixel() const { return m_bgraInput ? 4 : 3; }
};

} // namespace gpa
