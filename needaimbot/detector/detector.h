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
    
    
    void setCaptureEvent(cudaEvent_t event);

    std::mutex detectionMutex;

    int detectionVersion;
    std::condition_variable detectionCV;
    
    
    

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
    
    cudaStream_t m_computeStream = nullptr;
    cudaStream_t m_memoryStream = nullptr;
    cudaEvent_t m_preprocessDone = nullptr;
    cudaEvent_t m_inferenceDone = nullptr;

    cv::cuda::GpuMat resizedBuffer;
    
    cv::cuda::GpuMat m_hsvMaskGpu;

    
    Detection* m_decodedDetectionsGpu = nullptr;   
    int* m_decodedCountGpu = nullptr;        
    int m_decodedCountHost = 0;          

    
    Detection* m_finalDetectionsGpu = nullptr;        
    int* m_finalDetectionsCountGpu = nullptr;     
    int m_finalDetectionsCountHost = 0;        
    Detection* m_classFilteredDetectionsGpu = nullptr; 
    int* m_classFilteredCountGpu = nullptr;      
    int m_classFilteredCountHost = 0;          

    
    float* m_scoresGpu = nullptr;             
    int* m_bestTargetIndexGpu = nullptr;  
    int m_bestTargetIndexHost = -1;       
    Detection m_bestTargetHost;           
    bool m_hasBestTarget = false;          
    int m_headClassId = -1;                

    
    bool m_isTargetLocked;
    Detection m_lockedTargetInfo;       
    int m_lockedTargetLostFrames;       

    
    
    unsigned char* m_d_ignore_flags_gpu = nullptr;

    
    int* m_nms_d_x1 = nullptr;
    int* m_nms_d_y1 = nullptr;
    int* m_nms_d_x2 = nullptr;
    int* m_nms_d_y2 = nullptr;
    float* m_nms_d_areas = nullptr;
    float* m_nms_d_scores = nullptr;    
    int* m_nms_d_classIds = nullptr;  
    float* m_nms_d_iou_matrix = nullptr;
    bool* m_nms_d_keep = nullptr;
    int* m_nms_d_indices = nullptr;     

    
    std::vector<unsigned char> m_host_ignore_flags_uchar;
    bool m_ignore_flags_need_update;

    std::mutex hsvMaskMutex; 

    bool isCudaContextInitialized() const { return m_cudaContextInitialized; } 
    cv::cuda::GpuMat getHsvMaskGpu() const; 

private:
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
    
};

#endif 
