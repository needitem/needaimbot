#ifndef DETECTOR_H
#define DETECTOR_H

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

#include "postProcess.h"

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

// Forward declaration for the Detection struct
// struct Detection; // Already in postProcess.h
struct Detection; // Forward declaration if needed, or include postProcess.h
struct Config; // Forward declaration

// Define using constexpr to allow definition in header (implies inline)
// const int MAX_CLASSES_FOR_FILTERING = 64; // Old
constexpr int MAX_CLASSES_FOR_FILTERING = 64; // Example value - Adjust if needed

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
    // void releaseDetections(); // No longer needed as public API?
    // bool getLatestDetections(std::vector<cv::Rect> &boxes, std::vector<int> &classes); // Use best target info instead
    void setCaptureEvent(cudaEvent_t event);

    std::mutex detectionMutex;

    int detectionVersion;
    std::condition_variable detectionCV;
    // --- Removed members related to CPU post-processing results ---
    // std::vector<cv::Rect> detectedBoxes; // No longer populated directly
    // std::vector<int> detectedClasses;   // No longer populated directly

    float img_scale;

    int getInputHeight() const { return inputDims.d[2]; }
    int getInputWidth() const { return inputDims.d[3]; }

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, size_t> outputSizes;

    std::atomic<bool> shouldExit;
    std::condition_variable inferenceCV;

    cv::cuda::GpuMat currentFrame;
    cv::Mat currentFrameCpu;
    bool frameReady;
    bool frameIsGpu;

    cudaEvent_t processingDone;
    cudaEvent_t postprocessCopyDone;
    cudaEvent_t m_captureDoneEvent = nullptr;

    cv::cuda::GpuMat resizedBuffer;

    // --- Members for GPU Post-processing Pipeline ---
    Detection* m_decodedDetectionsGpu = nullptr;   // GPU buffer for decoded detections (before NMS)
    int* m_decodedCountGpu = nullptr;        // GPU buffer for count of decoded detections (atomic)
    int m_decodedCountHost = 0;          // Host copy of decoded count (for debugging/logging)

    // --- Members for Final NMS Results (GPU & Host) ---
    Detection* m_finalDetectionsGpu = nullptr;        // GPU buffer for final detections (after NMS)
    int* m_finalDetectionsCountGpu = nullptr;     // GPU buffer for count of final detections (after NMS)
    int m_finalDetectionsCountHost = 0;        // Host copy of final NMS count
    Detection* m_classFilteredDetectionsGpu = nullptr; // Detections after class ID filter
    int* m_classFilteredCountGpu = nullptr;      // Count after class ID filter
    int m_classFilteredCountHost = 0;          // Host count after class filter

    // --- Members for Scoring & Best Target (GPU & Host) ---
    float* m_scoresGpu = nullptr;             // GPU buffer for scores
    int* m_bestTargetIndexGpu = nullptr;  // GPU buffer for best target index
    int m_bestTargetIndexHost = -1;       // Host copy of best target index
    Detection m_bestTargetHost;           // Host copy of best target data
    bool m_hasBestTarget = false;          // Flag indicating if a valid target was found

    // New member for GPU ignore flags (using unsigned char for CUDA compatibility)
    // bool* m_d_ignore_flags_gpu = nullptr; // Old type
    unsigned char* m_d_ignore_flags_gpu = nullptr;

    bool isCudaContextInitialized() const { return m_cudaContextInitialized; } // Getter for the flag

private:
    bool m_cudaContextInitialized = false; // Add this flag
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

    // --- Removed members related to CPU output buffers ---
    // std::unordered_map<std::string, std::vector<float>> outputDataBuffers;
    // std::unordered_map<std::string, std::vector<__half>> outputDataBuffersHalf;
    // std::unordered_map<std::string, void *> pinnedOutputBuffers; // Keep if used for other things, or remove
    bool usePinnedMemory; // Keep if used elsewhere

    std::mutex inferenceMutex;

    std::unordered_map<std::string, size_t> inputSizes;
    std::unordered_map<std::string, void *> inputBindings;
    std::unordered_map<std::string, void *> outputBindings;
    std::unordered_map<std::string, std::vector<int64_t>> outputShapes;
    std::unordered_map<std::string, nvinfer1::DataType> outputTypes; // Keep this
    int numClasses;

    // New member for size of ignore flags buffer (matches MAX_CLASSES_FOR_FILTERING)
    // size_t m_d_ignore_flags_size = 0; // Not strictly needed if using a const for size

    size_t getSizeByDim(const nvinfer1::Dims &dims);
    size_t getElementSize(nvinfer1::DataType dtype);

    std::string inputName;
    void *inputBufferDevice; // Keep for preprocess
    // std::unordered_map<std::string, std::vector<float>> outputDataBuffers; // Removed
    // std::unordered_map<std::string, std::vector<__half>> outputDataBuffersHalf; // Removed


    cv::cuda::GpuMat floatBuffer;
    std::vector<cv::cuda::GpuMat> channelBuffers;

    bool checkCudaError(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] Error " << message << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    }

    // --- Memory allocation/deallocation helper (Definitions moved inside class) --- 
    template<typename T>
    cudaError_t allocateGpuBuffer(T*& buffer, size_t count, const char* name) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
        cudaError_t err = cudaMalloc(&buffer, count * sizeof(T));
        if (err != cudaSuccess) {
             std::cerr << "[CUDA] Failed to allocate GPU buffer '" << name << "': " << cudaGetErrorString(err) << std::endl;
        }
        return err;
    }
    template<typename T>
    void freeGpuBuffer(T*& buffer) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }

    // --- Post-processing function (modified) ---
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
    void preProcess(const cv::cuda::GpuMat &frame);
    void getInputNames();
    void getOutputNames();
    void getBindings();
    void initializeBuffers();
    // void performGpuPostProcessing(cudaStream_t stream); // Moved up
};

#endif // DETECTOR_H