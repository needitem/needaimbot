#ifndef NEEDAIMBOT_DETECTOR_DETECTOR_H
#define NEEDAIMBOT_DETECTOR_DETECTOR_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../cuda/simple_cuda_mat.h"
#include "../cuda/cuda_image_processing.h"
#include "../cuda/cuda_float_processing.h"
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

#include "../postprocess/postProcess.h"
#include "CudaBuffer.h"

// GPU 메모리 풀 최적화 클래스
class CudaMemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
        Block* next;
    };
    
    Block* free_blocks;
    void* pool_base;
    size_t pool_size;
    size_t used_bytes;
    size_t alignment = 256;
    std::mutex pool_mutex;
    
    size_t align_up(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
public:
    CudaMemoryPool(size_t total_size) : free_blocks(nullptr), used_bytes(0) {
        cudaMalloc(&pool_base, total_size);
        pool_size = total_size;
        
        free_blocks = new Block;
        free_blocks->ptr = pool_base;
        free_blocks->size = total_size;
        free_blocks->in_use = false;
        free_blocks->next = nullptr;
    }
    
    ~CudaMemoryPool() {
        if (pool_base) cudaFree(pool_base);
        while (free_blocks) {
            Block* next = free_blocks->next;
            delete free_blocks;
            free_blocks = next;
        }
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        size = align_up(size, alignment);
        
        Block* current = free_blocks;
        while (current) {
            if (!current->in_use && current->size >= size) {
                current->in_use = true;
                used_bytes += size;
                
                if (current->size > size + alignment) {
                    Block* new_block = new Block;
                    new_block->ptr = static_cast<char*>(current->ptr) + size;
                    new_block->size = current->size - size;
                    new_block->in_use = false;
                    new_block->next = current->next;
                    
                    current->size = size;
                    current->next = new_block;
                }
                
                return current->ptr;
            }
            current = current->next;
        }
        
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        Block* current = free_blocks;
        while (current) {
            if (current->ptr == ptr && current->in_use) {
                current->in_use = false;
                used_bytes -= current->size;
                return;
            }
            current = current->next;
        }
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        Block* current = free_blocks;
        while (current) {
            current->in_use = false;
            current = current->next;
        }
        used_bytes = 0;
    }
    
    size_t getUsedBytes() const { return used_bytes; }
    size_t getTotalBytes() const { return pool_size; }
};

/**
 * @brief Represents a target being tracked across frames
 * 
 * Contains detection information and temporal tracking data
 * for maintaining target continuity in video sequences.
 */
struct TrackedTarget {
    int id;                       ///< Unique identifier for this target
    Detection detection;          ///< Latest detection information
    int frames_since_last_seen;  ///< Number of frames since last detection
    // TODO: Add velocity, acceleration, and prediction data
};

// TensorRT utility functions
nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath);
nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& enginePath);

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;



struct Detection; 
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
    void inferenceThread();
    void setCaptureEvent(HANDLE event) { captureEvent = event; }
    
    
    

    // Mutex-based detection system
    std::mutex detectionMutex;
    int detectionVersion;
    std::condition_variable detectionCV;
    
    
    

    float img_scale;

    int getInputHeight() const { return static_cast<int>(inputDims.d[2]); }
    int getInputWidth() const { return static_cast<int>(inputDims.d[3]); }
    
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
    
    // Pinned memory for ultra-fast transfers
    BatchedResults* m_batchedResultsPinned;
    int* m_pinnedBestIndex;
    int* m_pinnedMatchingIndex;
    cudaEvent_t m_finalCopyEvent;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, size_t> outputSizes;

    std::atomic<bool> should_exit;
    std::condition_variable inferenceCV;

    SimpleCudaMat currentFrame;
    SimpleMat currentFrameCpu;
    bool frameReady;
    bool frameIsGpu;
    
    HANDLE captureEvent = nullptr;


    SimpleCudaMat resizedBuffer;
    
    SimpleCudaMat m_hsvMaskGpu;

    
    CudaBuffer<Detection> m_decodedDetectionsGpu;
    CudaBuffer<int> m_decodedCountGpu;

    CudaBuffer<Detection> m_finalDetectionsGpu;
    CudaBuffer<int> m_finalDetectionsCountGpu;
    int m_finalDetectionsCountHost = 0;
    std::chrono::steady_clock::time_point m_lastDetectionTime;
    CudaBuffer<Detection> m_classFilteredDetectionsGpu;
    CudaBuffer<int> m_classFilteredCountGpu;

    CudaBuffer<float> m_scoresGpu;
    CudaBuffer<int> m_bestTargetIndexGpu;
    int m_bestTargetIndexHost = -1;
    Detection m_bestTargetHost;
    bool m_hasBestTarget = false;
    int m_headClassId = -1;
    
    // For matching previous target
    CudaBuffer<int> m_matchingIndexGpu;
    CudaBuffer<float> m_matchingScoreGpu;
    

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
    
    
    // Double buffering for ultra-fast processing
    struct DoubleBuffer {
        SimpleCudaMat frameBuffers[2];
        BatchedResults resultBuffers[2];
        cudaEvent_t readyEvents[2];
        int currentReadIdx = 0;
        int currentWriteIdx = 1;
        void swap() {
            currentReadIdx = (currentReadIdx + 1) % 2;
            currentWriteIdx = (currentWriteIdx + 1) % 2;
        }
    };
    DoubleBuffer m_doubleBuffer;
    
    cudaEvent_t m_inferenceDone;
    cudaEvent_t m_preprocessDone; 
    cudaEvent_t processingDone;
    HANDLE m_captureDoneEvent;

    std::mutex hsvMaskMutex; 

    bool isCudaContextInitialized() const { return m_cudaContextInitialized; } 
    SimpleCudaMat getHsvMaskGpu() const; 

    void start();
    void stop();

private:
    std::thread m_captureThread;
    std::thread m_inferenceThread;

    static float calculate_host_iou(const Detection& det1, const Detection& det2); 
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
