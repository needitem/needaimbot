#ifndef DETECTOR_H
#define DETECTOR_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <NvInfer.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>
#include <cuda_fp16.h>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/cudaimgproc.hpp>
#include <filesystem>

#include "postProcess.h"
#include "CudaBuffer.h"
#include "DetectionExchange.h"

struct TrackedTarget {
    int id;
    Detection detection;
    int frames_since_last_seen;
    // Add other tracking-specific data here, like velocity
};

// TensorRT utility functions
nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath);
nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& enginePath);

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;



struct Detection; 
struct Config; 



constexpr int MAX_CLASSES_FOR_FILTERING = 64; 

class Detector
{
public:
    Detector();
    ~Detector();
    void initialize(const std::string &modelFile);
    bool initializeCudaContext();
    void processFrame(const cv::cuda::GpuMat &frame);
    void processFrame(const cv::Mat &frame);
    void inferenceThread();
    void setCaptureEvent(HANDLE event) { captureEvent = event; }
    
    
    

    // Lock-free detection exchange
    DetectionExchange detectionExchange;
    
    // Legacy mutex-based system (for backward compatibility)
    std::mutex detectionMutex;
    int detectionVersion;
    std::condition_variable detectionCV;
    
    
    

    float img_scale;

    int getInputHeight() const { return inputDims.d[2]; }
    int getInputWidth() const { return inputDims.d[3]; }
    
    // Batched results structure for reduced memory transfers
    struct BatchedResults {
        int finalCount;
        int bestIndex;
        float bestScore;
        Detection bestTarget;
        int matchingIndex;
        float matchingScore;
    };
    
    // GPU memory for batched results
    BatchedResults* m_batchedResultsGpu;
    BatchedResults m_batchedResultsHost;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, size_t> outputSizes;

    std::atomic<bool> shouldExit;
    std::condition_variable inferenceCV;

    cv::cuda::GpuMat currentFrame;
    cv::Mat currentFrameCpu;
    bool frameReady;
    bool frameIsGpu;
    
    HANDLE captureEvent = nullptr;

    // CUDA Graph support removed for optimization
    
    // Multiple CUDA streams for pipeline optimization
    cudaStream_t m_preprocessStream = nullptr;
    cudaStream_t m_inferenceStream = nullptr;
    cudaStream_t m_postprocessStream = nullptr;
    
    // Events for stream synchronization
    cudaEvent_t m_preprocessDone = nullptr;
    cudaEvent_t m_inferenceDone2 = nullptr;
    cudaEvent_t m_postprocessDone = nullptr;

    cv::cuda::GpuMat resizedBuffer;
    
    cv::cuda::GpuMat m_hsvMaskGpu;

    
    CudaBuffer<Detection> m_decodedDetectionsGpu;
    CudaBuffer<int> m_decodedCountGpu;
    int m_decodedCountHost = 0;

    CudaBuffer<Detection> m_finalDetectionsGpu;
    CudaBuffer<int> m_finalDetectionsCountGpu;
    int m_finalDetectionsCountHost = 0;
    CudaBuffer<Detection> m_classFilteredDetectionsGpu;
    CudaBuffer<int> m_classFilteredCountGpu;
    int m_classFilteredCountHost = 0;

    CudaBuffer<float> m_scoresGpu;
    CudaBuffer<int> m_bestTargetIndexGpu;
    int m_bestTargetIndexHost = -1;
    Detection m_bestTargetHost;
    bool m_hasBestTarget = false;
    int m_headClassId = -1;
    
    // For matching previous target
    CudaBuffer<int> m_matchingIndexGpu;
    CudaBuffer<float> m_matchingScoreGpu;
    
    // Target persistence to reduce flickering
    int m_targetLostFrameCount = 0;
    static constexpr int TARGET_LOST_THRESHOLD = 1; // Wait 1 frame before clearing target
    
    // Sticky target to prevent jumping between targets
    float m_lastTargetScore = 0.0f;
    static constexpr float STICKY_TARGET_THRESHOLD = 0.8f; // New target must be 20% better to switch

    // Target Tracking System
    std::vector<TrackedTarget> m_tracked_targets;
    int m_next_target_id = 0;

    bool m_isTargetLocked;
    Detection m_lockedTargetInfo;

    CudaBuffer<unsigned char> m_d_ignore_flags_gpu;

    CudaBuffer<int> m_nms_d_x1;
    CudaBuffer<int> m_nms_d_y1;
    CudaBuffer<int> m_nms_d_x2;
    CudaBuffer<int> m_nms_d_y2;
    CudaBuffer<float> m_nms_d_areas;
    CudaBuffer<float> m_nms_d_scores;
    CudaBuffer<int> m_nms_d_classIds;
    CudaBuffer<float> m_nms_d_iou_matrix;
    CudaBuffer<bool> m_nms_d_keep;
    CudaBuffer<int> m_nms_d_indices;     

    
    std::vector<unsigned char> m_host_ignore_flags_uchar;
    bool m_ignore_flags_need_update;
    
    // CUDA streams and events for pipelining
    static constexpr int NUM_STREAMS = 3;
    cudaStream_t m_streams[NUM_STREAMS];
    cudaEvent_t m_events[NUM_STREAMS];
    int m_currentStreamIdx = 0;
    
    // Pipeline buffers for each stream
    struct PipelineBuffer {
        cv::cuda::GpuMat resizedFrame;
        CudaBuffer<float> inputBuffer;
        CudaBuffer<float> outputBuffer;
        bool inUse = false;
    };
    PipelineBuffer m_pipelineBuffers[NUM_STREAMS];
    
    // Legacy stream variables (for compatibility)
    cudaStream_t& m_computeStream = m_streams[0];
    cudaStream_t& m_memoryStream = m_streams[1];
    cudaEvent_t m_inferenceDone;
    cudaEvent_t processingDone;
    cudaEvent_t postprocessCopyDone;
    HANDLE m_captureDoneEvent;

    std::mutex hsvMaskMutex; 

    bool isCudaContextInitialized() const { return m_cudaContextInitialized; } 
    cv::cuda::GpuMat getHsvMaskGpu() const; 

    void start();
    void stop();

private:
    std::thread m_captureThread;
    std::thread m_inferenceThread;

    static float calculate_host_iou(const cv::Rect& box1, const cv::Rect& box2); 
    bool m_cudaContextInitialized = false; 
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    nvinfer1::Dims inputDims;

    cv::cuda::Stream cvStream;
    cv::cuda::Stream preprocessCvStream;
    cv::cuda::Stream postprocessCvStream;

    cudaStream_t stream;
    cudaStream_t preprocessStream;
    cudaStream_t postprocessStream;

    bool usePinnedMemory; 

    

    std::mutex inferenceMutex;

    std::unordered_map<std::string, size_t> inputSizes;
    std::unordered_map<std::string, void *> inputBindings;
    std::unordered_map<std::string, void *> outputBindings;
    std::unordered_map<std::string, std::vector<int64_t>> outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> outputTypes; 
    int numClasses;

    size_t getSizeByDim(const nvinfer1::Dims &dims);
    size_t getElementSize(nvinfer1::DataType dtype);

    std::string inputName;
    void *inputBufferDevice; 

    cv::cuda::GpuMat floatBuffer;
    std::vector<cv::cuda::GpuMat> channelBuffers;

    bool checkCudaError(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] Error " << message << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    }

    
    void performGpuPostProcessing(cudaStream_t stream);


    void synchronizeStreams(cv::cuda::Stream &cvStream, cudaStream_t cudaStream)
    {
        cudaEvent_t event;
        cudaEventCreate(&event);

        cvStream.enqueueHostCallback([](int, void *userData)
                                     { cudaEventRecord(static_cast<cudaEvent_t>(userData)); }, &event);

        cudaStreamWaitEvent(cudaStream, event, 0);
        cudaEventDestroy(event);
    }

    void loadEngine(const std::string &engineFile);
    void preProcess(const cv::cuda::GpuMat &frame, cudaStream_t stream);
    void getInputNames();
    void getOutputNames();
    void getBindings();
    void initializeBuffers();
    
};

#endif 
