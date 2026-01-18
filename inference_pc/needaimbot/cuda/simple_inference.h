// Simple TensorRT inference with optimizations
// - GPU preprocessing (with resolution-aware skip)
// - GPU postprocessing (decode + target selection on GPU)
// - Fused target selection + PID movement on GPU
// - IoU-based target stickiness (hysteresis)
// - CUDA Graph capture for minimal kernel launch overhead
// - Pinned memory for fast host-device transfers
// - FP16 input/output support (native, no conversion)
#pragma once

#include <string>
#include <vector>
#include <memory>
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
    SimpleInference();
    ~SimpleInference();

    bool loadEngine(const std::string& enginePath);
    bool isLoaded() const { return m_loaded; }

    // Capture CUDA graph for optimized execution
    bool captureGraph();
    bool isGraphCaptured() const { return m_graphCaptured; }

    // Run inference on host RGB buffer (will upload to GPU)
    // Uses CUDA graph if captured, otherwise standard execution
    // CPU decode version (legacy)
    int runInference(const uint8_t* h_rgbData, int width, int height,
                     float confThreshold, std::vector<Detection>& outDetections);

    // GPU decode version - returns best target directly
    // Avoids D2H copy of all detections, only copies single best target
    // allowedClassMask: bitmask of allowed classes (bit N = class N allowed)
    // Returns true if a target was found
    bool runInferenceGpu(const uint8_t* h_rgbData, int width, int height,
                         float confThreshold, int headClassId, float headBonus,
                         uint32_t allowedClassMask,
                         Detection& outBestTarget);

    // Full GPU pipeline: inference + decode + target selection + PID movement
    // Returns mouse movement directly, minimal CPU involvement
    // allowedClassMask: bitmask of allowed classes (bit N = class N allowed)
    // Returns true if a target was found
    bool runInferenceFused(const uint8_t* h_rgbData, int width, int height,
                           float confThreshold, int headClassId, float headBonus,
                           uint32_t allowedClassMask,
                           const PIDConfig& pidConfig,
                           float iouStickinessThreshold,
                           float headYOffset, float bodyYOffset,
                           MouseMovement& outMovement,
                           Detection* outBestTarget = nullptr);

    int getModelResolution() const { return m_inputH; }
    int getNumClasses() const { return m_numClasses; }
    bool isInputFP16() const { return m_inputFP16; }
    bool isOutputFP16() const { return m_outputFP16; }
    cudaStream_t getStream() const { return m_stream; }

    // Access GPU output buffer for external GPU postprocessing
    void* getOutputBuffer() const { return m_d_output; }
    int getNumBoxes() const { return m_numBoxes; }

private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger m_logger;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;

    // GPU buffers
    void* m_d_rgbInput = nullptr;    // RGB HWC uint8 input
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

    // Pinned host memory for fast transfers
    uint8_t* m_h_rgbPinned = nullptr;
    void* m_h_outputPinned = nullptr;  // FP16 or FP32 depending on model
    Detection* m_h_bestTargetPinned = nullptr;  // Pinned memory for best target
    int* m_h_hasTargetPinned = nullptr;         // Pinned memory for has_target flag
    MouseMovement* m_h_mouseMovementPinned = nullptr;  // Pinned memory for mouse movement
    size_t m_outputPinnedSize = 0;

    // CUDA Graph
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCaptured = false;

    int m_inputH = 320;         // Model input height (target)
    int m_inputW = 320;         // Model input width (target)
    int m_srcH = 320;           // Current source height (for resize)
    int m_srcW = 320;           // Current source width (for resize)
    int m_numBoxes = 2100;
    int m_numClasses = 2;
    bool m_loaded = false;
    bool m_inputFP16 = false;   // Input tensor is FP16
    bool m_outputFP16 = false;  // Output tensor is FP16

    // Internal methods
    void executeStandard();
    void executeGraph();
    void decodeOutput(float confThreshold, std::vector<Detection>& outDetections);
};

} // namespace gpa
