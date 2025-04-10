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

#include "postProcess.h"

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

// Forward declaration for the Detection struct
struct Detection;

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
    void releaseDetections();
    bool getLatestDetections(std::vector<cv::Rect> &boxes, std::vector<int> &classes);
    void setCaptureEvent(cudaEvent_t event);

    std::mutex detectionMutex;

    int detectionVersion;
    std::condition_variable detectionCV;
    std::vector<cv::Rect> detectedBoxes;
    std::vector<int> detectedClasses;
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

    // GPU Post-processing results
    void* m_finalDetectionsGpu = nullptr;        // Buffer for final detections on GPU
    int* m_finalDetectionsCountGpu = nullptr;     // Buffer for count of final detections on GPU
    int m_finalDetectionsCountHost = 0;        // Count of final detections on Host
    std::vector<Detection> m_finalDetectionsHost; // Buffer for final detections on Host (temp)

    // GPU Scoring & Best Target Selection results
    float* m_scoresGpu = nullptr;             // Buffer for scores on GPU
    int* m_bestTargetIndexGpu = nullptr;  // Buffer for best target index on GPU (size 1)
    int m_bestTargetIndexHost = -1;       // Best target index on Host
    Detection m_bestTargetHost;           // Best target details on Host
    bool m_hasBestTarget = false;         // Flag if a valid best target exists

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

    std::unordered_map<std::string, void *> pinnedOutputBuffers;
    bool usePinnedMemory;

    std::mutex inferenceMutex;

    std::unordered_map<std::string, size_t> inputSizes;
    std::unordered_map<std::string, void *> inputBindings;
    std::unordered_map<std::string, void *> outputBindings;
    std::unordered_map<std::string, std::vector<int64_t>> outputShapes;
    int numClasses;

    size_t getSizeByDim(const nvinfer1::Dims &dims);
    size_t getElementSize(nvinfer1::DataType dtype);

    std::string inputName;
    void *inputBufferDevice;
    std::unordered_map<std::string, std::vector<float>> outputDataBuffers;
    std::unordered_map<std::string, std::vector<__half>> outputDataBuffersHalf;
    std::unordered_map<std::string, nvinfer1::DataType> outputTypes;

    cv::cuda::GpuMat floatBuffer;
    std::vector<cv::cuda::GpuMat> channelBuffers;

    bool checkCudaError(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] Error " << message << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    }

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
    void performGpuPostProcessing(cudaStream_t stream);
};

#endif // DETECTOR_H