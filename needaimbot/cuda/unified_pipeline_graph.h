#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>

// Forward declarations
class GPUPIDController;
class GPUKalmanFilter;
class MemoryPool;
struct Detection;
struct TargetInfo;

// Pipeline parameters for dynamic updates
struct PipelineParams {
    float confThreshold = 0.45f;
    float nmsThreshold = 0.45f;
    float targetSelectionFov = 100.0f;
    float pidKp = 0.35f;
    float pidKi = 0.02f;
    float pidKd = 0.15f;
    float smoothingFactor = 0.3f;
    float predictionFrames = 2.0f;
};

class UnifiedPipelineGraph {
public:
    UnifiedPipelineGraph();
    ~UnifiedPipelineGraph();
    
    // Initialize the pipeline with dimensions and TensorRT context
    bool initialize(int captureWidth, int captureHeight,
                   int modelWidth, int modelHeight,
                   nvinfer1::IExecutionContext* trtContext);
    
    // Create the CUDA graph (called automatically on first execute if needed)
    bool createGraph();
    
    // Execute the entire pipeline with a single graph launch
    // Input: captureData - raw BGRA image from capture
    // Output: dx, dy - mouse movement calculated by PID
    //         targetCount - number of targets detected
    bool execute(void* captureData, float& dx, float& dy, int& targetCount);
    
    // Update pipeline parameters (may require graph recreation)
    bool updateParameters(const PipelineParams& params);
    
    // Get pipeline statistics
    void getStatistics(float& fps, float& latency, int& graphNodes);
    
    // Check if graph is created and ready
    bool isReady() const;
    
private:
    // Implementation details hidden in .cu file
    class Impl;
    Impl* pImpl;
};

// C Interface for external integration
extern "C" {
    UnifiedPipelineGraph* createUnifiedPipeline();
    void destroyUnifiedPipeline(UnifiedPipelineGraph* pipeline);
    
    bool initializeUnifiedPipeline(UnifiedPipelineGraph* pipeline,
                                  int captureWidth, int captureHeight,
                                  int modelWidth, int modelHeight,
                                  void* trtContext);
    
    bool executeUnifiedPipeline(UnifiedPipelineGraph* pipeline,
                               void* captureData,
                               float* dx, float* dy,
                               int* targetCount);
    
    bool updatePipelineParams(UnifiedPipelineGraph* pipeline,
                             float confThreshold,
                             float nmsThreshold,
                             float pidKp, float pidKi, float pidKd);
}