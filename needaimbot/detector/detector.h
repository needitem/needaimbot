#ifndef NEEDAIMBOT_DETECTOR_DETECTOR_H
#define NEEDAIMBOT_DETECTOR_DETECTOR_H

// Only include Windows headers when not compiling CUDA
#ifndef __CUDACC__
#include "../core/windows_headers.h"
#endif

#include "../cuda/simple_cuda_mat.h"
#include "../cuda/detection/cuda_image_processing.h"
#include "../cuda/detection/cuda_float_processing.h"
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
#include <cuda_runtime_api.h>
#include <filesystem>

#include "../cuda/detection/postProcess.h"
#include "CudaBuffer.h"



// Forward declarations for tracking
// ByteTracker removed - tracking handled by GPU pipeline
class GPUTracker;
struct GPUTrackingContext;

// TensorRT utility functions
nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath);
nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& enginePath);

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

// LEGACY: GPU-calculated mouse movement structure - no longer used
// Mouse movement is now calculated in unified_graph_pipeline.cu
// struct MouseMovement {
//     float dx;
//     float dy;
//     bool hasTarget;
//     float targetDistance;
// };

#include "../core/Target.h"

// Use only Target type for all detections
using TrackedObject = Target;

class Config; 



constexpr int MAX_CLASSES_FOR_FILTERING = 64; 

/**
 * @brief High-performance AI-based object detector with CUDA acceleration
 * 
 * Provides real-time object detection using TensorRT inference engine
 * with GPU-accelerated preprocessing and postprocessing pipelines.
 * Includes target tracking, filtering, and advanced memory management.
 */
class Detector
{
public:
    Detector();
    ~Detector();
    void initialize(const std::string &modelFile);
    bool initializeCudaContext();
    void processFrame(const SimpleCudaMat &frame);
    void processFrame(const SimpleMat &frame);
    
    // CUDA Graph integration methods
    bool runInferenceAsync(float* d_input, float* d_output, cudaStream_t stream);
    void processFrameWithGraph(const unsigned char* h_frameData, cudaStream_t stream);
    float2 getMouseCoordsAsync(cudaStream_t stream);  // Get final mouse coordinates
    
    void inferenceThread();
    void setCaptureEvent(HANDLE event) { captureEvent = event; }
    
    
    

    // Mutex-based detection system
    std::mutex detectionMutex;
    int detectionVersion;
    std::condition_variable detectionCV;
    
    
    

    float img_scale;

    int getInputHeight() const { return static_cast<int>(inputDims.d[2]); }
    int getInputWidth() const { return static_cast<int>(inputDims.d[3]); }
    


    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, size_t> outputSizes;

    std::atomic<bool> should_exit;
    std::condition_variable inferenceCV;

    SimpleCudaMat currentFrame;
    SimpleMat currentFrameCpu;
    std::atomic<bool> frameReady;
    std::atomic<bool> frameIsGpu;
    
    HANDLE captureEvent = nullptr;


    SimpleCudaMat resizedBuffer;
    
    SimpleCudaMat m_colorMaskGpu;  // For RGB or HSV color filtering

    
    CudaBuffer<Target> m_decodedTargetsGpu;
    CudaBuffer<int> m_decodedCountGpu;

    CudaBuffer<Target> m_finalTargetsGpu;
    CudaBuffer<int> m_finalTargetsCountGpu;
    int m_finalTargetsCountHost = 0;
    std::unique_ptr<Target[]> m_finalTargetsHost;
    CudaBuffer<Target> m_classFilteredTargetsGpu;
    CudaBuffer<int> m_classFilteredCountGpu;
    
    // RGB filtered detections buffer (for optimized pipeline)
    CudaBuffer<Target> m_colorFilteredTargetsGpu;
    CudaBuffer<int> m_colorFilteredCountGpu;

    CudaBuffer<float> m_scoresGpu;

    int m_bestTargetIndexHost = -1;
    Target m_bestTargetHost;
    bool m_hasBestTarget = false;
    int m_headClassId = -1;
    
    // GPU buffers for target selection
    CudaBuffer<int> m_bestTargetIndexGpu;
    CudaBuffer<Target> m_bestTargetGpu;
    
    // Temporary buffers for multi-block reduction
    CudaBuffer<float> m_tempBlockScores;
    CudaBuffer<int> m_tempBlockIndices;
    
    // For matching previous target
    CudaBuffer<int> m_matchingIndexGpu;
    CudaBuffer<float> m_matchingScoreGpu;
    

    // GPU Tracking System
    GPUTrackingContext* m_gpuTrackerContext = nullptr;
    CudaBuffer<Target> m_trackedTargetsGpu;  // GPU buffer for tracked targets
    int m_trackedTargetsCount = 0;
    
    std::vector<TrackedObject> m_trackedObjects;
    std::mutex m_trackingMutex;  // Mutex to protect m_trackedObjects
    
    // GPU Kalman Filter
    class GPUKalmanTracker* m_gpuKalmanTracker = nullptr;
    CudaBuffer<Target> m_kalmanPredictionsGpu;
    CudaBuffer<int> m_kalmanPredictionsCountGpu;
    bool m_kalmanGraphInitialized = false;
    
    // Tracking handled by GPU pipeline
    std::unique_ptr<GPUTracker> m_gpuTracker;

    bool m_isTargetLocked;
    Target m_lockedTargetInfo;
    int m_lockedTrackId = -1;  // Track ID of locked target

    CudaBuffer<unsigned char> m_d_allow_flags_gpu;

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

    
    std::vector<unsigned char> m_host_allow_flags_uchar;
    bool m_allow_flags_need_update;
    
    
    // Double buffering for ultra-fast processing
    struct DoubleBuffer {
        SimpleCudaMat frameBuffers[2];
        cudaEvent_t readyEvents[2];
        int currentReadIdx = 0;
        int currentWriteIdx = 1;
        void swap() {
            currentReadIdx = (currentReadIdx + 1) % 2;
            currentWriteIdx = (currentWriteIdx + 1) % 2;
        }
    };
    DoubleBuffer m_doubleBuffer;
    
    // Event-based synchronization for pipeline stages
    cudaEvent_t m_inferenceDone;
    cudaEvent_t m_preprocessDone; 
    cudaEvent_t processingDone;
    HANDLE m_captureDoneEvent;
    
    // Additional events for fine-grained control
    cudaEvent_t m_postprocessEvent;
    cudaEvent_t m_colorFilterEvent;
    cudaEvent_t m_trackingEvent;
    cudaEvent_t m_finalCopyEvent;
    
    // Triple buffering for results
    static constexpr int kNumResultBuffers = 3;
    struct ResultBuffer {
        CudaBuffer<Target> targets;
        CudaBuffer<int> count;
        cudaEvent_t readyEvent;
        std::atomic<bool> isReady{false};
        int frameId = -1;
    };
    ResultBuffer m_resultBuffers[kNumResultBuffers];
    std::atomic<int> m_currentWriteBuffer{0};
    std::atomic<int> m_currentReadBuffer{1};
    std::atomic<int> m_processingBuffer{2};

    std::mutex colorMaskMutex; 

    bool isCudaContextInitialized() const { return m_cudaContextInitialized; } 
    SimpleCudaMat getColorMaskGpu() const; 

    void start();
    void stop();
    
    // Kalman filter management
    void initializeKalmanFilter();
    void destroyKalmanFilter();

private:
    std::thread m_captureThread;
    std::thread m_inferenceThread;
    
    // GPU 최적화 관련
    void warmupGPU();
    
    // CUDA Graph 최적화
    cudaGraph_t m_inferenceGraph = nullptr;
    cudaGraphExec_t m_inferenceGraphExec = nullptr;
    bool m_graphCaptured = false;
    void captureInferenceGraph(const SimpleCudaMat& frameGpu);

    static float calculate_host_iou(const Target& det1, const Target& det2); 
    bool m_cudaContextInitialized = false; 
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    nvinfer1::Dims inputDims;

    // CUDA streams removed - using native CUDA streams only

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

    SimpleCudaMatFloat floatBuffer;
    std::vector<SimpleCudaMatFloat> channelBuffers;

    bool checkCudaError(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] Error " << message << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    }

    
    void performGpuPostProcessing(cudaStream_t stream);
    
public:
    // GPU 감지 결과 직접 반환 (CPU 복사 없이)
    std::pair<void*, int> getLatestDetectionsGPU() const {
        if (m_finalTargetsCountHost > 0) {
            return std::make_pair(m_finalTargetsGpu.get(), m_finalTargetsCountHost);
        }
        return std::make_pair(nullptr, 0);
    }

    void synchronizeStreams(cudaStream_t stream1, cudaStream_t stream2)
    {
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event, stream1);
        cudaStreamWaitEvent(stream2, event, 0);
        cudaEventDestroy(event);
    }

    void loadEngine(const std::string &engineFile);
    void preProcess(const SimpleCudaMat &frame, cudaStream_t stream);
    void getInputNames();
    void getOutputNames();
    void getBindings();
    void initializeBuffers();
    
};

#endif // NEEDAIMBOT_DETECTOR_DETECTOR_H
