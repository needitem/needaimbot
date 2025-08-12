#include "AppContext.h"

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>

#include "../cuda/cuda_image_processing.h"
#include "../cuda/cuda_float_processing.h"
#include "../cuda/color_filter.h"
#include "../cuda/gpu_kalman_filter.h"
#include "../cuda/gpu_tracker.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>
#include <vector>
#include <mutex>
#include <limits>
#include <cfloat>

#include "detector.h"
#include "../needaimbot.h"
#include "../include/other_tools.h"
#include "../core/constants.h"

#include "../postprocess/postProcess.h"
#include "../postprocess/filterGpu.h"
#include "../config/config.h"
#include "../tracking/ByteTracker.h"
#include "../cuda/gpu_tracker.h"

#if defined(__has_include)
#  if __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>  
#    define NVTX_PUSH(p) nvtxRangePushA(p)
#    define NVTX_POP() nvtxRangePop()
#  else
#    define NVTX_PUSH(p)
#    define NVTX_POP()
#  endif
#else
#  define NVTX_PUSH(p)
#  define NVTX_POP()
#endif



extern std::atomic<bool> detectionPaused;

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;




extern std::mutex configMutex; 




// External filter functions (separated for better performance)
extern cudaError_t filterTargetsByClassIdGpu(
    const Target* decodedTargets,
    int numDecodedTargets,
    Target* filteredTargets,
    int* filteredCount,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    int max_output_detections,
    cudaStream_t stream
);

extern cudaError_t filterTargetsByColorGpu(
    const Target* d_input_detections,
    int num_input_detections,
    Target* d_output_detections,
    int* d_output_count,
    const unsigned char* d_color_mask,
    int mask_pitch,
    int min_color_pixels,
    bool remove_color_matches,
    int max_output_detections,
    cudaStream_t stream
);

// Simple CPU function to find closest target based on distance

Detector::Detector()
    : frameReady(false),
    detectionVersion(0),
    inputBufferDevice(nullptr),
    img_scale(1.0f),
    numClasses(0),
    m_captureDoneEvent(nullptr),
    m_cudaContextInitialized(false),
    m_hasBestTarget(false),
    m_bestTargetIndexHost(-1),
    m_finalTargetsCountHost(0),
    m_host_allow_flags_uchar(MAX_CLASSES_FOR_FILTERING, 0), 
    m_allow_flags_need_update(true) 
    , m_isTargetLocked(false)
 
{
    // Initialize CUDA events for async pipeline
    cudaEventCreateWithFlags(&m_preprocessDone, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_inferenceDone, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&processingDone, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_postprocessEvent, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_colorFilterEvent, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_trackingEvent, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_finalCopyEvent, cudaEventDisableTiming);
    
    // Initialize result buffer events
    for (int i = 0; i < kNumResultBuffers; ++i) {
        cudaEventCreateWithFlags(&m_resultBuffers[i].readyEvent, cudaEventDisableTiming);
    }
    
    // Initialize double buffer events
    cudaEventCreateWithFlags(&m_doubleBuffer.readyEvents[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_doubleBuffer.readyEvents[1], cudaEventDisableTiming);
    
    // Initialize GPU tracker context
    auto& ctx = AppContext::getInstance();
    if (ctx.config.enable_tracking) {
        try {
            m_gpuTrackerContext = initGPUTracker(
                ctx.config.tracker_max_age,
                ctx.config.tracker_min_hits,
                ctx.config.tracker_iou_threshold
            );
        } catch (const std::exception& e) {
            std::cerr << "[Detector] Failed to initialize GPU tracker: " << e.what() << std::endl;
            m_gpuTrackerContext = nullptr;
        }
    }
    
    // Initialize GPU Kalman filter if enabled
    initializeKalmanFilter();
}

Detector::~Detector()
{
    
    try {
        // 1. 먼저 모든 CUDA 스트림 동기화 (가장 중요)
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        if (preprocessStream) {
            cudaStreamSynchronize(preprocessStream);
        }
        if (postprocessStream) {
            cudaStreamSynchronize(postprocessStream);
        }
        
        // 2. 이벤트 정리 (스트림보다 먼저)
        if (m_preprocessDone) {
            cudaEventDestroy(m_preprocessDone);
            m_preprocessDone = nullptr;
        }
        if (m_inferenceDone) {
            cudaEventDestroy(m_inferenceDone);
            m_inferenceDone = nullptr;
        }
        if (processingDone) {
            cudaEventDestroy(processingDone);
            processingDone = nullptr;
        }
        
        // Destroy additional events
        if (m_postprocessEvent) cudaEventDestroy(m_postprocessEvent);
        if (m_colorFilterEvent) cudaEventDestroy(m_colorFilterEvent);
        if (m_trackingEvent) cudaEventDestroy(m_trackingEvent);
        if (m_finalCopyEvent) cudaEventDestroy(m_finalCopyEvent);
        
        // Destroy result buffer events
        for (int i = 0; i < kNumResultBuffers; ++i) {
            if (m_resultBuffers[i].readyEvent) {
                cudaEventDestroy(m_resultBuffers[i].readyEvent);
            }
        }
        
        // Destroy double buffer events
        if (m_doubleBuffer.readyEvents[0]) cudaEventDestroy(m_doubleBuffer.readyEvents[0]);
        if (m_doubleBuffer.readyEvents[1]) cudaEventDestroy(m_doubleBuffer.readyEvents[1]);
        
        // 3. 메모리 버퍼 정리
        
        // 5. Input/Output 바인딩 정리
        for (auto& binding : inputBindings) {
            if (binding.second) {
                cudaFree(binding.second);
                binding.second = nullptr;
            }
        }
        inputBindings.clear();

        for (auto& binding : outputBindings) {
            if (binding.second) {
                cudaFree(binding.second);
                binding.second = nullptr;
            }
        }
        outputBindings.clear();

        if (inputBufferDevice) {
            cudaFree(inputBufferDevice);
            inputBufferDevice = nullptr;
        }
        
        // 6. CUDA Graph 정리
        if (m_inferenceGraphExec) {
            cudaGraphExecDestroy(m_inferenceGraphExec);
            m_inferenceGraphExec = nullptr;
        }
        if (m_inferenceGraph) {
            cudaGraphDestroy(m_inferenceGraph);
            m_inferenceGraph = nullptr;
        }
        
        // 7. GPU Kalman filter cleanup
        if (m_gpuKalmanTracker) {
            destroyGPUKalmanTracker(m_gpuKalmanTracker);
            m_gpuKalmanTracker = nullptr;
        }
        
        // 8. 스트림 정리 (마지막에)
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (preprocessStream) {
            cudaStreamDestroy(preprocessStream);
            preprocessStream = nullptr;
        }
        if (postprocessStream) {
            cudaStreamDestroy(postprocessStream);
            postprocessStream = nullptr;
        }
        
        std::cout << "[Detector] Resource cleanup completed successfully." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[Detector] Error during cleanup: " << e.what() << std::endl;
    }
    
    // Note: Removed cudaDeviceReset() as it affects all CUDA contexts,
    // not just this instance. Proper cleanup is handled by destructors.
}
// CUDA Graph integration: Async inference without CPU synchronization
bool Detector::runInferenceAsync(float* d_input, float* d_output, cudaStream_t stream) {
    if (!context || !engine) {
        return false;
    }
    
    // Set input/output bindings
    void* bindings[] = { d_input, d_output };
    
    // Run inference asynchronously on the provided stream
    // TensorRT 8.5+ supports graph capture for enqueueV3
    bool success = context->enqueueV3(stream);
    
    if (!success) {
        std::cerr << "[Detector] Async inference failed" << std::endl;
        return false;
    }
    
    return true;
}

// Process frame using CUDA Graph pipeline
void Detector::processFrameWithGraph(const unsigned char* h_frameData, cudaStream_t stream) {
    // This method is called by UnifiedGraphPipeline
    // All operations must be async and use the provided stream
    
    // The actual processing is handled by UnifiedGraphPipeline
    // which orchestrates the entire pipeline including:
    // 1. H2D copy
    // 2. Preprocessing 
    // 3. Inference (via runInferenceAsync)
    // 4. Postprocessing
    // 5. Tracking
    // 6. PID control
    // 7. D2H copy of results
}

// Get final mouse coordinates after graph execution
float2 Detector::getMouseCoordsAsync(cudaStream_t stream) {
    // Synchronize to ensure results are ready
    cudaStreamSynchronize(stream);
    
    // Return the final mouse coordinates
    // These should be in pinned memory for fast access
    float2 coords;
    coords.x = m_bestTargetHost.center_x;  // Use correct member name
    coords.y = m_bestTargetHost.center_y;
    
    return coords;
}

void Detector::getInputNames()
{
    auto& ctx = AppContext::getInstance();
    inputNames.clear();
    inputSizes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputNames.emplace_back(name);
        }
    }



    // Allocate NMS buffers - balanced between capacity and memory usage
    const int nms_buffer_size = 300; // Reasonable buffer for detections (was causing illegal memory access with 1000)
    m_nms_d_x1.allocate(nms_buffer_size);
    m_nms_d_y1.allocate(nms_buffer_size);
    m_nms_d_x2.allocate(nms_buffer_size);
    m_nms_d_y2.allocate(nms_buffer_size);
    m_nms_d_areas.allocate(nms_buffer_size);
    m_nms_d_scores.allocate(nms_buffer_size);
    m_nms_d_classIds.allocate(nms_buffer_size);
    m_nms_d_iou_matrix.allocate(nms_buffer_size * nms_buffer_size);
    m_nms_d_keep.allocate(nms_buffer_size);
    m_nms_d_indices.allocate(nms_buffer_size);
}

void Detector::getOutputNames()
{
    auto& ctx = AppContext::getInstance();
    outputNames.clear();
    outputSizes.clear();
    outputTypes.clear();
    outputShapes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputNames.emplace_back(name);
            outputTypes[name] = engine->getTensorDataType(name);
            
        }
    }
}

void Detector::getBindings()
{
    auto& ctx = AppContext::getInstance();
    for (auto& binding : inputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    inputBindings.clear();

    for (auto& binding : outputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    outputBindings.clear();

    for (const auto& name : inputNames)
    {
        size_t size = inputSizes[name];
        if (size > 0)
        {
            void* ptr = nullptr;

            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess)
            {
                inputBindings[name] = ptr;
            }
            else
            {
                std::cerr << "[Detector] Failed to allocate input memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        if (size > 0) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess)
            {
                outputBindings[name] = ptr;
            } else
            {
                std::cerr << "[Detector] Failed to allocate output memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
}

bool Detector::initializeCudaContext()
{
    auto& ctx = AppContext::getInstance();
    
    cudaError_t cuda_err = cudaSetDevice(ctx.config.cuda_device_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "[Detector] ERROR: Failed to set CUDA device " << ctx.config.cuda_device_id 
                  << ": " << cudaGetErrorString(cuda_err) << std::endl;
        m_cudaContextInitialized = false; 
        return false; 
    }
    // Silent - device set successfully

    // Create multiple streams for pipeline optimization with priorities
    // Get stream priority range
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    // Create streams with appropriate priorities (inference gets highest priority)
    if (!checkCudaError(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority), "creating main stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreateWithPriority(&preprocessStream, cudaStreamNonBlocking, (greatestPriority + leastPriority) / 2), "creating preprocess stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreateWithPriority(&postprocessStream, cudaStreamNonBlocking, (greatestPriority + leastPriority) / 2), "creating postprocess stream")) { m_cudaContextInitialized = false; return false; }

    // OpenCV streams removed - using native CUDA streams only
    
    if (!checkCudaError(cudaEventCreateWithFlags(&m_preprocessDone, cudaEventDisableTiming), "creating preprocessDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&m_inferenceDone, cudaEventDisableTiming), "creating inferenceDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&processingDone, cudaEventDisableTiming), "creating processingDone event")) { m_cudaContextInitialized = false; return false; }

    // GPU 워밍업 - 첫 추론 시간 단축
    warmupGPU();
    
    // GPU 메모리 대역폭 최적화 설정
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    // 메모리 풀 사전 할당으로 동적 할당 오버헤드 제거
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // GPU 메모리의 10%를 메모리 풀로 예약
    size_t pool_size = total_mem / 10;
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, ctx.config.cuda_device_id);
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &pool_size);
    
    // Memory pool initialized
    
    // GPU 메모리 병합 최적화
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 2);
    
    // L2 캐시 설정
    size_t l2CacheSize = 0;
    cudaDeviceGetLimit(&l2CacheSize, cudaLimitPersistingL2CacheSize);
    
    if (l2CacheSize < 16 * 1024 * 1024) {
        size_t newL2CacheSize = std::min(l2CacheSize * 2, size_t(8 * 1024 * 1024));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, newL2CacheSize);
    }

    m_cudaContextInitialized = true; 
    return true; 
}

void Detector::initialize(const std::string& modelFile)
{
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) {
        std::cerr << "[Detector] CUDA context not initialized. Skipping TensorRT engine load and GPU memory allocation." << std::endl;
        return; 
    }

    // Create a simple logger for TensorRT
    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Suppress TensorRT internal errors
            if (severity <= Severity::kERROR && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    static SimpleLogger logger;
    
    // TensorRT initialized
    
    runtime.reset(nvinfer1::createInferRuntime(logger));
    loadEngine(modelFile);

    if (!engine)
    {
        std::cerr << "[Detector] Engine loading failed" << std::endl;
        return;
    }

    context.reset(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "[Detector] Context creation failed" << std::endl;
        return;
    }
    
    // TensorRT 10.x 최적화 설정
    context->setOptimizationProfileAsync(0, stream);
    
    // CUDA 그래프 호환성 설정
    context->setEnqueueEmitsProfile(false);
    
    // 영구 캐시 활성화 - 시스템 제한에 맞게 조정
    size_t systemL2CacheLimit = 0;
    cudaDeviceGetLimit(&systemL2CacheLimit, cudaLimitPersistingL2CacheSize);
    
    size_t requestedCacheLimit = static_cast<size_t>(ctx.config.persistent_cache_limit_mb) * 1024 * 1024;
    size_t actualCacheLimit = std::min(requestedCacheLimit, systemL2CacheLimit);
    
    context->setPersistentCacheLimit(actualCacheLimit);

    getInputNames();
    getOutputNames();

    if (inputNames.empty())
    {
        std::cerr << "[Detector] No input tensors found" << std::endl;
        return;
    }

    inputName = inputNames[0];
    // use dynamic ONNX input resolution
    int r = ctx.config.onnx_input_resolution;
    inputDims = nvinfer1::Dims4{1, 3, r, r};
    context->setInputShape(inputName.c_str(), inputDims);

    if (!context->allInputDimensionsSpecified())
    {
        std::cerr << "[Detector] Failed to set input dimensions" << std::endl;
        return;
    }

    for (const auto& inName : inputNames)
    {
        nvinfer1::Dims dims = context->getTensorShape(inName.c_str());
        nvinfer1::DataType dtype = engine->getTensorDataType(inName.c_str());
        inputSizes[inName] = getSizeByDim(dims) * getElementSize(dtype);
    }

    for (const auto& outName : outputNames)
    {
        nvinfer1::Dims dims = context->getTensorShape(outName.c_str());
        nvinfer1::DataType dtype = engine->getTensorDataType(outName.c_str());
        outputSizes[outName] = getSizeByDim(dims) * getElementSize(dtype);
        outputTypes[outName] = dtype;
        
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; j++)
        {
            shape.push_back(dims.d[j]);
        }
        outputShapes[outName] = shape;
    }

    getBindings();
    
    initializeBuffers();
 

    
    m_allow_flags_need_update = true;

    
    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());

        if (ctx.config.postprocess == "yolo10")
        {
            numClasses = 11;
        } else if (outDims.nbDims == 3) {
            // For yolo8/9/11/12, output is typically [1, num_features, num_boxes]
            // num_features = 4 (bbox) + 1 (confidence) + num_classes
            // So, num_classes = num_features - 5
            numClasses = static_cast<int>(outDims.d[1] - 5); // Corrected: 4 for bbox, 1 for confidence
        } else {
            std::cerr << "[Detector] Warning: Unknown output dimensions for class calculation. Assuming 0."
                      << std::endl;
            numClasses = 0;
        }
    }

    img_scale = static_cast<float>(ctx.config.detection_resolution) / 640;
    
    
    m_headClassId = -1; 
    if (!ctx.config.head_class_name.empty()) {
        for (const auto& class_setting : ctx.config.class_settings) {
            if (class_setting.name == ctx.config.head_class_name) {
                m_headClassId = class_setting.id;
                break;
            }
        }
    }
    
    nvinfer1::Dims actualInputDims = context->getTensorShape(inputName.c_str());
    int c = static_cast<int>(actualInputDims.d[1]);
    int h = static_cast<int>(actualInputDims.d[2]);
    int w = static_cast<int>(actualInputDims.d[3]);
    
    if (resizedBuffer.empty() || resizedBuffer.rows() != h || resizedBuffer.cols() != w) {
        resizedBuffer.create(h, w, 3);
    }
    
    if (floatBuffer.empty() || floatBuffer.rows() != h || floatBuffer.cols() != w) {
        floatBuffer.create(h, w, 4); // Support up to 4 channels (BGRA)
    }
    
    channelBuffers.resize(c);
    for (int i = 0; i < c; ++i)
    {
        if (channelBuffers[i].empty() || channelBuffers[i].rows() != h || channelBuffers[i].cols() != w) {
            channelBuffers[i].create(h, w, 1);  // 1 channel for float data
        }
    }
    
    
    for (const auto& name : inputNames)
    {
        context->setTensorAddress(name.c_str(), inputBindings[name]);
    }
    for (const auto& name : outputNames)
    {
        context->setTensorAddress(name.c_str(), outputBindings[name]);
    }
}

size_t Detector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0) return 0;
        size *= dims.d[i];
    }
    return size;
}

size_t Detector::getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        default: return 0;
    }
}

void Detector::loadEngine(const std::string& modelFile)
{
    auto& ctx = AppContext::getInstance();
    std::string engineFilePath;
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();

    if (extension == ".engine")
    {
        engineFilePath = modelFile;
    }
    else if (extension == ".onnx")
    {
        // generate engine filename with resolution and precision suffixes
        std::string baseName = modelPath.stem().string();
        baseName += "_" + std::to_string(ctx.config.onnx_input_resolution);
        if (ctx.config.export_enable_fp16) baseName += "_fp16";
        if (ctx.config.export_enable_fp8)  baseName += "_fp8";
        std::string engineFilename = baseName + ".engine";
        engineFilePath = (modelPath.parent_path() / engineFilename).string();

        if (!fileExists(engineFilePath))
        {
            std::cout << "[Detector] Building engine from ONNX model" << std::endl;

            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile);
            if (builtEngine)
            {
                nvinfer1::IHostMemory* serializedEngine = builtEngine->serialize();

                if (serializedEngine)
                {
                    std::ofstream engineFile(engineFilePath, std::ios::binary);
                    if (engineFile)
                    {
                        engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
                        engineFile.close();
                        
                        std::cout << "[Detector] Engine saved to: " << engineFilePath << std::endl;
                    }
                    delete serializedEngine;
                }
                delete builtEngine;
            }
        }
    }
    else
    {
        std::cerr << "[Detector] Unsupported model format: " << extension << std::endl;
        return;
    }

    // Loading engine
    engine.reset(loadEngineFromFile(engineFilePath));
}

void Detector::processFrame(const SimpleCudaMat& frame)
{
    auto& ctx = AppContext::getInstance();
    
    static int process_call_count = 0;
    process_call_count++;
    
    if (!isCudaContextInitialized()) {
        std::cerr << "[Detector] CUDA context not initialized!" << std::endl;
        return;
    }

    if (ctx.detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
        // Don't use memset on non-POD types, use assignment instead
        m_bestTargetHost = Target();  // Use default constructor
        m_bestTargetIndexHost = -1;
        m_finalTargetsCountHost = 0;
        detectionVersion++;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Lock-free frame transfer
    // Skip if previous frame still processing (frame drop for low latency)
    if (frameReady.load(std::memory_order_acquire)) {
        return; // Skip frame if inference slower than capture
    }
    
    currentFrame = frame.clone();
    frameIsGpu.store(true, std::memory_order_release);
    frameReady.store(true, std::memory_order_release);
    
    // No need to notify - using busy wait

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    ctx.g_current_process_frame_time_ms.store(duration.count());
    ctx.add_to_history(ctx.g_process_frame_time_history, duration.count(), ctx.g_process_frame_history_mutex);
}

void Detector::processFrame(const SimpleMat& frame)
{
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) return; 

    if (ctx.detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
        // Don't use memset on non-POD types, use assignment instead
        m_bestTargetHost = Target();  // Use default constructor
        m_bestTargetIndexHost = -1;
        m_finalTargetsCountHost = 0;
        detectionVersion++;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Lock-free frame transfer
    if (frameReady.load(std::memory_order_acquire)) {
        return; // Skip frame if inference slower than capture
    }
    
    currentFrameCpu = frame;
    frameIsGpu.store(false, std::memory_order_release);
    frameReady.store(true, std::memory_order_release);
    // No need to notify - using busy wait

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    ctx.g_current_process_frame_time_ms.store(duration.count());
    ctx.add_to_history(ctx.g_process_frame_time_history, duration.count(), ctx.g_process_frame_history_mutex);
}

void Detector::inferenceThread()
{
    std::cout << "[InferenceThread] Starting inference thread" << std::endl;
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) {
        std::cerr << "[InferenceThread] CUDA context not initialized. Inference thread exiting." << std::endl;
        return;
    }
    std::cout << "[InferenceThread] CUDA context initialized" << std::endl;
    
    // CPU 스레드 친화성 설정 (성능 코어에 고정)
    {
        HANDLE thread = GetCurrentThread();
        DWORD_PTR mask = 0x0F; // 첫 4개 코어 사용 (보통 성능 코어)
        SetThreadAffinityMask(thread, mask);
        SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
        // Thread optimization set
    }
    
    // Cache frequently accessed config values to reduce mutex contention
    struct CachedConfig {
        int detection_resolution;
        int max_detections;
        float confidence_threshold;
        float nms_threshold;
        float distance_weight;
        float confidence_weight;
        float sticky_target_threshold;
        bool enable_color_filter;
        int min_color_pixels_required;
        bool color_filter_remove_on_match;
        std::string postprocess;
        
        void update(const Config& config) {
            detection_resolution = config.detection_resolution;
            max_detections = config.max_detections;
            confidence_threshold = config.confidence_threshold;
            nms_threshold = config.nms_threshold;
            distance_weight = config.distance_weight;
            confidence_weight = config.confidence_weight;
            sticky_target_threshold = config.sticky_target_threshold;
            enable_color_filter = config.enable_color_filter;
            min_color_pixels_required = config.min_color_pixels;
            color_filter_remove_on_match = config.remove_color_matches;
            postprocess = config.postprocess;
        }
    } cached_config;
    
    // Initial config cache
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_config.update(ctx.config);
    }
    
    // Update cache periodically (every 100 frames)
    static int frame_counter = 0;

    SimpleCudaMat frameGpu;
    static auto last_cycle_start_time = std::chrono::high_resolution_clock::time_point{};

    std::cout << "[InferenceThread] Entering main loop" << std::endl;
    static int inference_loop_count = 0;
    
    while (!ctx.should_exit)
    {
        inference_loop_count++;
        auto cycle_start_time = std::chrono::high_resolution_clock::now();
        
        // Calculate detector cycle time (time between cycle starts)
        if (last_cycle_start_time.time_since_epoch().count() != 0) {
            std::chrono::duration<float, std::milli> cycle_duration_ms = cycle_start_time - last_cycle_start_time;
            ctx.g_current_detector_cycle_time_ms.store(cycle_duration_ms.count());
            ctx.add_to_history(ctx.g_detector_cycle_time_history, cycle_duration_ms.count(), ctx.g_detector_cycle_history_mutex);
        }
        last_cycle_start_time = cycle_start_time;
        
        NVTX_PUSH("Detector Inference Loop");

        if (ctx.should_exit) {
            std::cout << "[InferenceThread] Exit signal detected in main loop" << std::endl;
            break;
        }

        // Update allow flags if needed
        if (m_allow_flags_need_update) {
            std::lock_guard<std::mutex> lock(ctx.configMutex);
            // Reset all flags to 0 (don't allow)
            std::fill(m_host_allow_flags_uchar.begin(), m_host_allow_flags_uchar.end(), 0);
            
            // Set allow flags based on class settings
            for (const auto& class_setting : ctx.config.class_settings) {
                if (class_setting.id >= 0 && class_setting.id < MAX_CLASSES_FOR_FILTERING) {
                    m_host_allow_flags_uchar[class_setting.id] = class_setting.allow ? 1 : 0;
                }
            }
            
            // Copy to GPU
            cudaMemcpyAsync(m_d_allow_flags_gpu.get(), m_host_allow_flags_uchar.data(), 
                           MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), 
                           cudaMemcpyHostToDevice, stream);
            
            m_allow_flags_need_update = false;
            
        }

        if (ctx.detector_model_changed.load()) {
            // CUDA Graph removed for optimization

            // Re-initialize detector components
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                context.reset();
                engine.reset();
                // Free and reallocate bindings and buffers
                for (auto& binding : inputBindings) { if (binding.second) cudaFree(binding.second); }
                inputBindings.clear();
                for (auto& binding : outputBindings) { if (binding.second) cudaFree(binding.second); }
                outputBindings.clear();
                // Free other buffers...
            }
            initialize("models/" + ctx.config.ai_model);
            img_scale = static_cast<float>(ctx.config.detection_resolution) / 640;
            ctx.detector_model_changed.store(false);
        }

        if (AppContext::getInstance().should_exit) {
            std::cout << "[InferenceThread] Exit signal received" << std::endl;
            break;
        }
        
        // Try to get frame without lock - using atomic operations
        if (!frameReady.load(std::memory_order_acquire)) {
            // Sleep briefly to reduce CPU usage when no frame is available
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;  // No frame available, skip this iteration
        }
        
        // Process the frame
        if (frameIsGpu.load(std::memory_order_acquire)) {
            frameGpu = std::move(currentFrame);
        } else {
            frameGpu.create(currentFrameCpu.rows(), currentFrameCpu.cols(), currentFrameCpu.channels());
            frameGpu.upload(currentFrameCpu.data(), currentFrameCpu.step());
        }
        
        // Mark frame as processed
        frameReady.store(false, std::memory_order_release);

        if (!context) {
            std::cerr << "[InferenceThread] TensorRT context not initialized!" << std::endl;
            continue;
        }

        if (!frameGpu.empty())
        {
            try
            {
                auto inference_start_time = std::chrono::high_resolution_clock::now();
                
                // Update config cache every 100 frames to reduce mutex contention
                if (++frame_counter % 100 == 0) {
                    std::lock_guard<std::mutex> lock(ctx.configMutex);
                    cached_config.update(ctx.config);
                }

                // Pipeline optimization: parallelize previous frame post-processing with current preprocessing
                static bool firstFrame = true;
                
                if (!firstFrame) {
                    // Wait for previous frame post-processing to complete
                    cudaEventSynchronize(processingDone);
                }
                
                // Execute preprocessing
                preProcess(frameGpu, preprocessStream);
                cudaEventRecord(m_preprocessDone, preprocessStream);
                
                // CUDA Graph based inference (capture after sufficient frames)
                // Increase delay to ensure WindowsGraphicsCapture is fully initialized
                if (!m_graphCaptured && frame_counter > 60 && frame_counter % 10 == 0) {
                    // Clear any pending CUDA errors before attempting graph capture
                    cudaError_t pendingError = cudaGetLastError();
                    if (pendingError != cudaSuccess) {
                        std::cerr << "[Graph] Clearing pending CUDA error: " << cudaGetErrorString(pendingError) << std::endl;
                        // Skip this capture attempt if there was an error
                        continue;
                    }
                    captureInferenceGraph(frameGpu);
                }
                
                bool enqueueSuccess;
                // 전처리 완료 대기
                cudaStreamWaitEvent(stream, m_preprocessDone, cudaEventWaitExternal);
                
                if (m_graphCaptured && m_inferenceGraphExec) {
                    // Graph 실행 (추론만)
                    cudaError_t graphLaunchResult = cudaGraphLaunch(m_inferenceGraphExec, stream);
                    if (graphLaunchResult != cudaSuccess) {
                        std::cerr << "[InferenceThread] CUDA Graph launch failed: " << cudaGetErrorString(graphLaunchResult) << std::endl;
                        // Fallback to regular inference
                        m_graphCaptured = false;
                        cudaGraphExecDestroy(m_inferenceGraphExec);
                        cudaGraphDestroy(m_inferenceGraph);
                        m_inferenceGraphExec = nullptr;
                        m_inferenceGraph = nullptr;
                        enqueueSuccess = context->enqueueV3(stream);
                    } else {
                        enqueueSuccess = true;
                    }
                } else {
                    // 일반 추론 (Graph 캡처 전)
                    enqueueSuccess = context->enqueueV3(stream);
                }
                
                cudaEventRecord(m_inferenceDone, stream);
                
                // Post-processing always runs separately (due to dynamic size)
                cudaStreamWaitEvent(postprocessStream, m_inferenceDone, cudaEventWaitExternal);
                performGpuPostProcessing(postprocessStream);
                
                if (!enqueueSuccess) {
                    std::cerr << "[Detector] TensorRT inference failed" << std::endl;
                    continue;
                }
                
                // 후처리 완료 이벤트 기록
                cudaEventRecord(processingDone, postprocessStream);
                
                firstFrame = false;
                
                // GPU에서 detection 결과 복사 (간단한 방식)
                cudaError_t copyErr = cudaMemcpyAsync(&m_finalTargetsCountHost, m_finalTargetsCountGpu.get(), 
                               sizeof(int), cudaMemcpyDeviceToHost, postprocessStream);
                if (copyErr != cudaSuccess) {
                    std::cerr << "[InferenceThread] Failed to copy detection count: " << cudaGetErrorString(copyErr) << std::endl;
                    continue;
                }
                
                // Record event instead of sync - we'll check it later when needed
                cudaEventRecord(m_postprocessEvent, postprocessStream);
                
                // 검출된 객체가 있으면 detection 데이터도 복사
                static int frame_counter = 0;
                int current_frame = ++frame_counter;
                
                // Wait for count to be ready before checking it
                cudaEventSynchronize(m_postprocessEvent);
                
                if (m_finalTargetsCountHost > 0) {
                    
                    // Validate buffer size
                    if (m_finalTargetsCountHost > Constants::MAX_DETECTIONS) {
                        std::cerr << "[InferenceThread] Target count exceeds buffer size: " 
                                  << m_finalTargetsCountHost << " > " << Constants::MAX_DETECTIONS << std::endl;
                        m_finalTargetsCountHost = Constants::MAX_DETECTIONS;
                    }
                    
                    // Ensure GPU buffer is valid before copying
                    if (!m_finalTargetsGpu.get()) {
                        std::cerr << "[InferenceThread] GPU detection buffer is null!" << std::endl;
                        continue;
                    }
                    
                    if (!m_finalTargetsHost) {
                        std::cerr << "[InferenceThread] Host detection buffer is null!" << std::endl;
                        continue;
                    }
                    
                    copyErr = cudaMemcpyAsync(m_finalTargetsHost.get(), m_finalTargetsGpu.get(), 
                                   m_finalTargetsCountHost * sizeof(Target), 
                                   cudaMemcpyDeviceToHost, postprocessStream);
                    if (copyErr != cudaSuccess) {
                        std::cerr << "[InferenceThread] Failed to copy detections: " << cudaGetErrorString(copyErr) << std::endl;
                        continue;
                    }
                    
                    // Record final copy event for later synchronization
                    cudaEventRecord(m_finalCopyEvent, postprocessStream);
                    
                    // Wait for copy to complete before processing on CPU
                    cudaEventSynchronize(m_finalCopyEvent);
                    
                    // Filter out detections touching screen edges (only for very edge cases)
                    int valid_count = 0;
                    const int edge_margin = 2; // Reduced margin - only filter if really at edge
                    const int screen_width = ctx.config.detection_resolution;
                    const int screen_height = ctx.config.detection_resolution;
                    
                    // Safety check
                    if (m_finalTargetsCountHost > Constants::MAX_DETECTIONS) {
                        std::cerr << "[ERROR] Target count exceeds maximum: " << m_finalTargetsCountHost << std::endl;
                        m_finalTargetsCountHost = Constants::MAX_DETECTIONS;
                    }
                    
                    for (int i = 0; i < m_finalTargetsCountHost; i++) {
                        Target& det = m_finalTargetsHost[i];
                        
                        // Only filter if BOTH position is at edge AND size is unreasonably large
                        bool at_corner = (det.x <= edge_margin && det.y <= edge_margin);
                        bool too_large = (det.width > screen_width * 0.8f || det.height > screen_height * 0.8f);
                        
                        if (at_corner && too_large) {
                            // This is likely a false detection at screen corner
                            continue; // Skip this detection
                        }
                        
                        // Fix invalid dimensions if needed
                        if (det.width <= 0) det.width = 1;
                        if (det.height <= 0) det.height = 1;
                        
                        // Copy valid detection (might overwrite itself, which is fine)
                        if (valid_count < Constants::MAX_DETECTIONS) {
                            m_finalTargetsHost[valid_count] = det;
                            valid_count++;
                        }
                    }
                    
                    // Update count to only valid detections
                    m_finalTargetsCountHost = valid_count;
                }

                auto inference_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float, std::milli> inference_duration_ms = inference_end_time - inference_start_time;
                ctx.g_current_inference_time_ms.store(inference_duration_ms.count());
                ctx.add_to_history(ctx.g_inference_time_history, inference_duration_ms.count(), ctx.g_inference_history_mutex);
                
                // Apply SORT tracking if enabled
                std::vector<TrackedObject> tracked_targets;
                
                // Initialize or destroy tracker based on config
                if (ctx.config.enable_tracking && !m_byteTracker) {
                    try {
                        m_byteTracker = std::make_unique<ByteTracker>();
                        m_byteTracker->setTrackThresh(ctx.config.byte_track_thresh);
                        m_byteTracker->setHighThresh(ctx.config.byte_high_thresh);
                        m_byteTracker->setMatchThresh(ctx.config.byte_match_thresh);
                        m_byteTracker->setMaxTimeLost(ctx.config.byte_max_time_lost);
                    } catch (const std::exception& e) {
                        std::cerr << "[Tracker] Failed to initialize ByteTracker: " << e.what() << std::endl;
                    }
                } else if (!ctx.config.enable_tracking && m_byteTracker) {
                    // Clean up tracker if tracking is disabled
                    m_byteTracker.reset();
                    {
                        std::lock_guard<std::mutex> lock(m_trackingMutex);
                        m_trackedObjects.clear();
                    }
                    std::cout << "[Tracker] ByteTracker disabled and cleaned up" << std::endl;
                }
                
                // GPU Tracking System  
                if (ctx.config.enable_tracking && m_finalTargetsCountHost > 0 && m_gpuTrackerContext) {
                    
                    // Allocate tracked targets buffer if needed
                    if (!m_trackedTargetsGpu.get()) {
                        m_trackedTargetsGpu.allocate(Constants::MAX_DETECTIONS);
                    }
                    
                    // Allocate count buffer
                    CudaBuffer<int> trackedCountGpu(1);
                    
                    // Run GPU tracking directly on GPU memory
                    updateGPUTrackerDirect(
                        m_gpuTrackerContext,
                        m_finalTargetsGpu.get(),        // Input: detections already on GPU
                        m_finalTargetsCountHost,         // Number of detections
                        m_trackedTargetsGpu.get(),          // Output: tracked targets on GPU
                        trackedCountGpu.get(),               // Output: number of tracked targets
                        postprocessStream,                   // CUDA stream
                        0.033f                               // dt: ~30 FPS
                    );
                    
                    // Replace final detections with tracked targets
                    cudaMemcpyAsync(m_finalTargetsGpu.get(), 
                                   m_trackedTargetsGpu.get(),
                                   Constants::MAX_DETECTIONS * sizeof(Target),
                                   cudaMemcpyDeviceToDevice, 
                                   postprocessStream);
                    
                    // Update count
                    int tracked_count = 0;
                    cudaMemcpyAsync(&tracked_count, 
                                   trackedCountGpu.get(),
                                   sizeof(int), 
                                   cudaMemcpyDeviceToHost, 
                                   postprocessStream);
                    cudaEventRecord(m_trackingEvent, postprocessStream);
                    
                    // Update final count
                    if (tracked_count > 0 && tracked_count <= Constants::MAX_DETECTIONS) {
                        m_finalTargetsCountHost = tracked_count;
                        cudaMemcpyAsync(m_finalTargetsCountGpu.get(), 
                                       &m_finalTargetsCountHost,
                                       sizeof(int), 
                                       cudaMemcpyHostToDevice, 
                                       postprocessStream);
                        
                        // Copy to host for CPU-based operations if needed
                        cudaMemcpyAsync(m_finalTargetsHost.get(),
                                       m_finalTargetsGpu.get(),
                                       m_finalTargetsCountHost * sizeof(Target),
                                       cudaMemcpyDeviceToHost,
                                       postprocessStream);
                        
                        // Update CPU tracking cache for other uses
                        {
                            std::lock_guard<std::mutex> lock(m_trackingMutex);
                            m_trackedObjects.clear();
                            m_trackedObjects.reserve(tracked_count);
                            
                            // Wait for tracking data to be ready
                            cudaEventSynchronize(m_trackingEvent);
                            for (int i = 0; i < tracked_count; i++) {
                                m_trackedObjects.push_back(m_finalTargetsHost[i]);
                            }
                            
                            // Handle target lock
                            if (ctx.config.enable_target_lock) {
                                // Check if locked target still exists
                                if (m_isTargetLocked) {
                                    bool lockStillValid = false;
                                    for (int i = 0; i < tracked_count; i++) {
                                        if (m_finalTargetsHost[i].id == m_lockedTrackId) {
                                            lockStillValid = true;
                                            break;
                                        }
                                    }
                                    
                                    if (!lockStillValid) {
                                        // Lost lock
                                        m_isTargetLocked = false;
                                        m_lockedTrackId = -1;
                                    }
                                }
                                
                                // If not locked and targets available, lock onto closest target
                                if (!m_isTargetLocked && tracked_count > 0) {
                                    // Find closest target
                                    float minDist = FLT_MAX;
                                    int closestIdx = -1;
                                    float centerX = cached_config.detection_resolution / 2.0f;
                                    float centerY = cached_config.detection_resolution / 2.0f;
                                    
                                    for (int i = 0; i < tracked_count; i++) {
                                        float dx = m_finalTargetsHost[i].x - centerX;
                                        float dy = m_finalTargetsHost[i].y - centerY;
                                        float dist = dx * dx + dy * dy;
                                        
                                        if (dist < minDist) {
                                            minDist = dist;
                                            closestIdx = i;
                                        }
                                    }
                                    
                                    if (closestIdx >= 0) {
                                        m_isTargetLocked = true;
                                        m_lockedTrackId = m_finalTargetsHost[closestIdx].id;
                                    }
                                }
                            } else {
                                // Target lock disabled
                                m_isTargetLocked = false;
                                m_lockedTrackId = -1;
                            }
                        }
                        
                        // GPU Tracker debug log removed
                    }
                } else if (ctx.config.enable_tracking && m_finalTargetsCountHost > 0) {
                    std::cout << "[GPU Tracker] Tracking enabled but GPU tracker not initialized" << std::endl;
                }
                
                // Apply GPU Kalman filter if enabled
                if (ctx.config.enable_kalman_filter && m_gpuKalmanTracker && m_finalTargetsCountHost > 0) {
                    
                    // Use fixed frame-based delta time (1 frame = 1.0)
                    const float frameDelta = 1.0f;
                    
                    // Update Kalman filter settings with frame-based delta
                    updateKalmanFilterSettings(
                        frameDelta,
                        ctx.config.kalman_process_noise,
                        ctx.config.kalman_measurement_noise,
                        postprocessStream
                    );
                    
                    // Initialize CUDA graph on first use
                    if (ctx.config.kalman_use_cuda_graph && !m_kalmanGraphInitialized) {
                        initializeKalmanGraph(m_gpuKalmanTracker, postprocessStream);
                        m_kalmanGraphInitialized = true;
                        std::cout << "[Kalman] CUDA graph initialized for Kalman filter" << std::endl;
                    }
                    
                    // Process with Kalman filter
                    // Convert lookahead time to frames (assuming 60fps base)
                    float lookaheadFrames = ctx.config.kalman_lookahead_time * 60.0f;
                    
                    processKalmanFilter(
                        m_gpuKalmanTracker,
                        m_finalTargetsGpu.get(),
                        m_finalTargetsCountHost,
                        m_kalmanPredictionsGpu.get(),
                        m_kalmanPredictionsCountGpu.get(),
                        postprocessStream,
                        ctx.config.kalman_use_cuda_graph,
                        lookaheadFrames
                    );
                    
                    // Update final targets with Kalman predictions
                    int kalmanCount = 0;
                    cudaMemcpyAsync(&kalmanCount, m_kalmanPredictionsCountGpu.get(), 
                                   sizeof(int), cudaMemcpyDeviceToHost, postprocessStream);
                    cudaStreamSynchronize(postprocessStream);
                    
                    if (kalmanCount > 0) {
                        // Replace targets with Kalman predictions
                        cudaMemcpyAsync(m_finalTargetsGpu.get(), m_kalmanPredictionsGpu.get(),
                                       kalmanCount * sizeof(Target), 
                                       cudaMemcpyDeviceToDevice, postprocessStream);
                        m_finalTargetsCountHost = kalmanCount;
                        cudaMemcpyAsync(m_finalTargetsCountGpu.get(), &m_finalTargetsCountHost,
                                       sizeof(int), cudaMemcpyHostToDevice, postprocessStream);
                    }
                }
                
                // GPU에서 거리 기반 타겟 선택
                if (m_finalTargetsCountHost > 0 && !ctx.should_exit) {
                    
                    // Crosshair is now always at center since capture region already includes offset
                    float crosshairX = cached_config.detection_resolution / 2.0f;
                    float crosshairY = cached_config.detection_resolution / 2.0f;
                    
                    // Validate buffers before GPU operation
                    if (!m_finalTargetsGpu.get()) {
                        std::cerr << "[Aimbot] ERROR: m_finalTargetsGpu is null!" << std::endl;
                        continue;
                    }
                    
                    if (!m_bestTargetIndexGpu.get() || !m_bestTargetGpu.get()) {
                        std::cerr << "[Aimbot] ERROR: Target buffers not allocated!" << std::endl;
                        continue;
                    }
                    
                    // Check if we have a locked target first
                    bool useLockedTarget = false;
                    int lockedTargetIndex = -1;
                    
                    if (ctx.config.enable_target_lock && m_isTargetLocked) {
                        // Find the locked target in current frame
                        for (int i = 0; i < m_finalTargetsCountHost; i++) {
                            if (m_finalTargetsHost[i].id == m_lockedTrackId) {
                                lockedTargetIndex = i;
                                useLockedTarget = true;
                                break;
                            }
                        }
                    }
                    
                    cudaError_t target_err;
                    
                    if (useLockedTarget) {
                        // Use the locked target directly
                        cudaMemcpyAsync(m_bestTargetIndexGpu.get(), &lockedTargetIndex, 
                                       sizeof(int), cudaMemcpyHostToDevice, postprocessStream);
                        cudaMemcpyAsync(m_bestTargetGpu.get(), &m_finalTargetsHost[lockedTargetIndex], 
                                       sizeof(Target), cudaMemcpyHostToDevice, postprocessStream);
                        target_err = cudaSuccess;
                    } else {
                        // Find closest target on GPU
                        target_err = findClosestTargetGpu(
                            m_finalTargetsGpu.get(),
                            m_finalTargetsCountHost,
                            crosshairX,
                            crosshairY,
                            m_bestTargetIndexGpu.get(),
                            m_bestTargetGpu.get(),
                            postprocessStream
                        );
                    }
                    
                    if (target_err == cudaSuccess) {
                        
                        // Copy results to host
                        
                        // Validate GPU buffer before copying
                        if (!m_bestTargetIndexGpu.get()) {
                            std::cerr << "[Aimbot] ERROR: m_bestTargetIndexGpu is null!" << std::endl;
                            continue;
                        }
                        
                        if (!m_bestTargetGpu.get()) {
                            std::cerr << "[Aimbot] ERROR: m_bestTargetGpu is null!" << std::endl;
                            continue;
                        }
                        
                        cudaError_t indexCopyErr = cudaMemcpyAsync(&m_bestTargetIndexHost, m_bestTargetIndexGpu.get(), 
                                       sizeof(int), cudaMemcpyDeviceToHost, postprocessStream);
                        if (indexCopyErr != cudaSuccess) {
                            std::cerr << "[InferenceThread] Failed to copy target index: " << cudaGetErrorString(indexCopyErr) << std::endl;
                            continue;
                        }
                        
                        
                        // Copy Target structure directly (it's now POD)
                        Target temp_detection;
                        
                        cudaError_t targetCopyErr = cudaMemcpyAsync(&temp_detection, m_bestTargetGpu.get(), 
                                       sizeof(Target), cudaMemcpyDeviceToHost, postprocessStream);
                        if (targetCopyErr != cudaSuccess) {
                            std::cerr << "[InferenceThread] Failed to copy target data: " << cudaGetErrorString(targetCopyErr) << std::endl;
                            continue;
                        }
                        
                        cudaError_t syncErr = cudaStreamSynchronize(postprocessStream);
                        if (syncErr != cudaSuccess) {
                            std::cerr << "[InferenceThread] Stream sync after target copy failed: " << cudaGetErrorString(syncErr) << std::endl;
                            continue;
                        }
                        
                        
                        // Safely copy POD fields to m_bestTargetHost
                        try {
                            // Don't use memset on non-POD types! Just assign fields directly.
                            
                            // Copy POD fields
                            m_bestTargetHost.confidence = temp_detection.confidence;
                            m_bestTargetHost.classId = temp_detection.classId;
                            m_bestTargetHost.x = temp_detection.x;
                            m_bestTargetHost.y = temp_detection.y;
                            m_bestTargetHost.width = temp_detection.width;
                            m_bestTargetHost.height = temp_detection.height;
                            m_bestTargetHost.id = temp_detection.id;
                            m_bestTargetHost.center_x = temp_detection.center_x;
                            m_bestTargetHost.center_y = temp_detection.center_y;
                            m_bestTargetHost.velocity_x = temp_detection.velocity_x;
                            m_bestTargetHost.velocity_y = temp_detection.velocity_y;
                            // age and time_since_update removed from Target (now in SORTTracker metadata)
                            
                        } catch (const std::exception& e) {
                            std::cerr << "[Aimbot] Exception updating best target: " << e.what() << std::endl;
                            continue;
                        } catch (...) {
                            std::cerr << "[Aimbot] Unknown exception updating best target" << std::endl;
                            continue;
                        }
                        
                        
                        // Update detection state
                        {
                            std::lock_guard<std::mutex> lock(detectionMutex);
                            
                            if (m_bestTargetIndexHost >= 0) {
                                m_hasBestTarget = true;
                                
                                // Calculate movement deltas and push to event queue
                                if (ctx.aiming) {
                                    float centerX = cached_config.detection_resolution / 2.0f;
                                    float centerY = cached_config.detection_resolution / 2.0f;
                                    
                                    MouseEvent event;
                                    event.dx = m_bestTargetHost.center_x - centerX;
                                    event.dy = m_bestTargetHost.center_y - centerY;
                                    event.has_target = true;
                                    event.target = m_bestTargetHost;
                                    event.timestamp = std::chrono::steady_clock::now();
                                    
                                    {
                                        std::lock_guard<std::mutex> event_lock(ctx.mouse_event_mutex);
                                        ctx.mouse_event_queue.push(event);
                                    }
                                    ctx.mouse_events_available = true;
                                    ctx.mouse_event_cv.notify_one();
                                }
                            } else {
                                m_hasBestTarget = false;
                                m_bestTargetHost = Target();  // Use default constructor
                            }
                            
                        }
                        detectionVersion++;
                        detectionCV.notify_one();
                    } else {
                        std::cerr << "[Detector] Error in GPU target selection: " << cudaGetErrorString(target_err) << std::endl;
                        
                        std::lock_guard<std::mutex> lock(detectionMutex);
                        m_hasBestTarget = false;
                        m_bestTargetHost = Target();  // Use default constructor
                        m_bestTargetIndexHost = -1;
                        detectionVersion++;
                        detectionCV.notify_one();
                    }
                } else {
                    // No detections or should exit
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    m_hasBestTarget = false;
                    m_bestTargetHost = Target();  // Use default constructor
                    m_bestTargetIndexHost = -1;
                    
                    // Send no-target event if aiming
                    if (ctx.aiming) {
                        MouseEvent event;
                        event.dx = 0;
                        event.dy = 0;
                        event.has_target = false;
                        event.target = Target();
                        event.timestamp = std::chrono::steady_clock::now();
                        
                        {
                            std::lock_guard<std::mutex> event_lock(ctx.mouse_event_mutex);
                            ctx.mouse_event_queue.push(event);
                        }
                        ctx.mouse_events_available = true;
                        ctx.mouse_event_cv.notify_one();
                    }
                    
                    detectionVersion++;
                    detectionCV.notify_one();
                }

            }
            catch (const std::exception& e)
            {
                std::cerr << "[InferenceThread] Exception caught in inference loop: " << e.what() << std::endl;
                m_hasBestTarget = false;
                m_bestTargetHost = Target();  // Use default constructor
                m_bestTargetIndexHost = -1;
                m_finalTargetsCountHost = 0;
            }
            catch (...)
            {
                std::cerr << "[InferenceThread] Unknown exception caught in inference loop" << std::endl;
                m_hasBestTarget = false;
                m_bestTargetHost = Target();  // Use default constructor
                m_bestTargetIndexHost = -1;
                m_finalTargetsCountHost = 0;
            }
        }
        NVTX_POP();
    }
}

void Detector::performGpuPostProcessing(cudaStream_t stream) {
    
    auto& ctx = AppContext::getInstance();
    if (outputNames.empty()) {
        std::cerr << "[PostProcess] No output names found for post-processing." << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    static int postProcessCount = 0;
    postProcessCount++;

    const std::string& primaryOutputName = outputNames[0];
    void* d_rawOutputPtr = outputBindings[primaryOutputName];
    nvinfer1::DataType outputType = outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = outputShapes[primaryOutputName];


    if (!d_rawOutputPtr) {
        std::cerr << "[PostProcess] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    

    // Clear all detection buffers at the start of processing
    cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);

    cudaError_t decodeErr = cudaSuccess;

    // Use cached config values for CUDA Graph compatibility
    // These should be set once before graph capture
    static int cached_max_detections = Constants::MAX_DETECTIONS; // Use max from constants (100)
    static float cached_nms_threshold = 0.45f;
    static float cached_confidence_threshold = 0.25f;  // Default confidence threshold
    static std::string cached_postprocess = "yolo12";
    static bool config_cached = false;
    
    // Always update cache from config when not in graph capture mode
    if (!m_graphCaptured) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_max_detections = ctx.config.max_detections;
        cached_nms_threshold = ctx.config.nms_threshold;
        cached_confidence_threshold = ctx.config.confidence_threshold;
        cached_postprocess = ctx.config.postprocess;
        config_cached = true;
    }

    // Use a reasonable buffer for decoding - balance between capacity and memory
    // Too large buffers cause illegal memory access in NMS
    int maxDecodedTargets = 300;  // Reasonable buffer for detections
    
    
    if (cached_postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        // Pass large buffer size to decode ALL detections
        decodeErr = decodeYolo10Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            cached_confidence_threshold,
            this->img_scale,
            m_decodedTargetsGpu.get(),
            m_decodedCountGpu.get(),
            max_candidates,
            maxDecodedTargets,  // Large buffer for all detections
            stream);
    } else if (cached_postprocess == "yolo8" || cached_postprocess == "yolo9" || cached_postprocess == "yolo11" || cached_postprocess == "yolo12") {
        int max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        
        // Validate parameters before calling
        if (!m_decodedTargetsGpu.get() || !m_decodedCountGpu.get()) {
            std::cerr << "[PostProcess] Target buffers not allocated!" << std::endl;
            cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        if (max_candidates <= 0 || maxDecodedTargets <= 0) {
            std::cerr << "[PostProcess] Invalid buffer sizes: max_candidates=" << max_candidates 
                      << ", maxDecodedTargets=" << maxDecodedTargets << std::endl;
            cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        decodeErr = decodeYolo11Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            cached_confidence_threshold,
            this->img_scale,
            m_decodedTargetsGpu.get(),
            m_decodedCountGpu.get(),
            max_candidates,
            maxDecodedTargets,  // Large buffer for all detections
            stream);
    } else {
        std::cerr << "[Detector] Unsupported post-processing type for GPU decoding: " << cached_postprocess << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    if (decodeErr != cudaSuccess) {
        std::cerr << "[PostProcess] GPU decoding kernel launch/execution failed: " << cudaGetErrorString(decodeErr) << std::endl;
        std::cerr << "[PostProcess] CUDA Error Code: " << decodeErr << std::endl;
        std::cerr << "[PostProcess] Postprocess type: " << cached_postprocess << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    

    // Always get the actual decoded count for correct processing
    int decodedCount = 0;
    int classFilteredCount = 0;
    
    // Get actual decoded count (needed for correct filtering)
    cudaError_t countCopyErr = cudaMemcpyAsync(&decodedCount, m_decodedCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (countCopyErr != cudaSuccess) {
        std::cerr << "[PostProcess] Failed to copy decoded count: " << cudaGetErrorString(countCopyErr) << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        std::cerr << "[PostProcess] Stream sync failed: " << cudaGetErrorString(syncErr) << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    
    
    
    // If no detections were decoded, clear final count and return early
    if (decodedCount == 0) {
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    
    // Only sync and get detailed debug info when not in graph mode
    if (!m_graphCaptured) {
        
    }
    
    // Step 1: Class ID filtering (after confidence filtering in decode)
    cudaError_t filterErr = filterTargetsByClassIdGpu(
        m_decodedTargetsGpu.get(),
        decodedCount,  // Use actual count or max for graph mode
        m_classFilteredTargetsGpu.get(),
        m_classFilteredCountGpu.get(),
        m_d_allow_flags_gpu.get(),
        MAX_CLASSES_FOR_FILTERING,
        300,  // Reasonable buffer for filtered detections
        stream
    );
    if (!checkCudaError(filterErr, "filtering detections by class ID GPU")) {
        std::cerr << "[PostProcess] Class ID filtering failed" << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    
    
    // Always get class filtered count for correct processing
    cudaMemcpyAsync(&classFilteredCount, m_classFilteredCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Check class filtered count (only debug when not in graph mode)
    if (!m_graphCaptured) {
        
    }
    
    // Early exit if no detections after class filtering
    if (classFilteredCount == 0) {
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    
    // Step 2: RGB color filtering (before NMS for better efficiency)
    // Get color mask if available
    SimpleCudaMat colorMask = getColorMaskGpu();
    const unsigned char* colorMaskPtr = nullptr;
    int maskPitch = 0;
    
    // For non-graph mode, apply color filtering if mask exists
    Target* nmsInputTargets = nullptr;
    int effectiveFilteredCount = classFilteredCount;  // Use actual filtered count
    
    if (!m_graphCaptured && !colorMask.empty()) {
        colorMaskPtr = colorMask.data();
        maskPitch = static_cast<int>(colorMask.step());
        
        // Apply RGB filtering
        cudaError_t colorFilterErr = filterTargetsByColorGpu(
            m_classFilteredTargetsGpu.get(),
            classFilteredCount,  // Process actual class-filtered count
            m_colorFilteredTargetsGpu.get(),
            m_colorFilteredCountGpu.get(),
            colorMaskPtr,
            maskPitch,
            10,  // min_color_pixels threshold
            false,  // keep detections WITH color matches
            300,  // max output
            stream
        );
        
        if (!checkCudaError(colorFilterErr, "filtering detections by color GPU")) {
            cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        // Use color-filtered detections for NMS
        nmsInputTargets = m_colorFilteredTargetsGpu.get();
        
        // Check color filtered count
        int colorFilteredCount = 0;
        cudaMemcpyAsync(&colorFilteredCount, m_colorFilteredCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        effectiveFilteredCount = colorFilteredCount;  // Update count for NMS
    } else {
        // No color filtering, use class-filtered detections directly
        nmsInputTargets = m_classFilteredTargetsGpu.get();
    }
    
    // Step 3: NMS (after all filtering for maximum efficiency)
    try {
        // Use cached frame dimensions for CUDA Graph compatibility
        static int cached_frame_width = 640;
        static int cached_frame_height = 640;
        if (!m_graphCaptured) {
            cached_frame_width = ctx.config.detection_resolution;
            cached_frame_height = ctx.config.detection_resolution;
        }
        
        // Validate NMS buffers before calling
        if (!m_nms_d_x1.get() || !m_nms_d_y1.get() || !m_nms_d_x2.get() || !m_nms_d_y2.get() ||
            !m_nms_d_areas.get() || !m_nms_d_scores.get() || !m_nms_d_classIds.get() ||
            !m_nms_d_iou_matrix.get() || !m_nms_d_keep.get() || !m_nms_d_indices.get()) {
            std::cerr << "[Detector] ERROR: NMS buffers not properly allocated!" << std::endl;
            cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        
        // NMS will process filtered detections and output only max_detections
        NMSGpu(
            nmsInputTargets,
            effectiveFilteredCount, // Process all filtered detections
            m_finalTargetsGpu.get(),       
            m_finalTargetsCountGpu.get(),  
            cached_max_detections,  // Apply max_detections limit AFTER NMS 
            cached_nms_threshold,
            cached_frame_width,
            cached_frame_height,
            m_nms_d_x1.get(),
            m_nms_d_y1.get(),
            m_nms_d_x2.get(),
            m_nms_d_y2.get(),
            m_nms_d_areas.get(),
            m_nms_d_scores.get(),     
            m_nms_d_classIds.get(),   
            m_nms_d_iou_matrix.get(),
            m_nms_d_keep.get(),
            m_nms_d_indices.get(),
            stream
        );
        
        // Validate detections after NMS to ensure no invalid dimensions
        extern void validateTargetsGpu(Target* d_detections, int n, cudaStream_t stream);
        
        // Get final count to validate only actual detections
        int finalCount = 0;
        cudaMemcpyAsync(&finalCount, m_finalTargetsCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (finalCount > 0 && finalCount <= cached_max_detections) {
            validateTargetsGpu(m_finalTargetsGpu.get(), finalCount, stream);
        }
        
        // Target selection will be done on CPU after copying results
        
    } catch (const std::exception& e) {
         std::cerr << "[Detector] Exception during NMSGpu call: " << e.what() << std::endl;
         cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
    }
}

void Detector::preProcess(const SimpleCudaMat& frame, cudaStream_t stream)
{
    auto& ctx = AppContext::getInstance();
    if (frame.empty()) return;

    // This function will now use the stream passed for graph capture

    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = static_cast<int>(dims.d[1]);
    int h = static_cast<int>(dims.d[2]);
    int w = static_cast<int>(dims.d[3]);
    

    try
    {
        static SimpleCudaMat maskGpu_static;
        static SimpleCudaMat maskedImageGpu_static;

        if (ctx.config.circle_mask) {
            if (maskGpu_static.empty() || maskGpu_static.rows() != frame.rows() || maskGpu_static.cols() != frame.cols()) {
                SimpleMat mask(frame.rows(), frame.cols(), 1);
                mask.setZero();
                int centerX = mask.cols() / 2;
                int centerY = mask.rows() / 2;
                int radius = std::min(mask.cols(), mask.rows()) / 2;
                // Create circular mask manually
                for (int y = 0; y < mask.rows(); ++y) {
                    for (int x = 0; x < mask.cols(); ++x) {
                        int dx = x - centerX;
                        int dy = y - centerY;
                        if (dx * dx + dy * dy <= radius * radius) {
                            mask.set(y, x, 255);
                        }
                    }
                }
                maskGpu_static.create(mask.rows(), mask.cols(), 1);
                maskGpu_static.upload(mask.data(), mask.step());
            }
            maskedImageGpu_static.create(frame.rows(), frame.cols(), frame.channels());
            maskedImageGpu_static.setZero();
            CudaImageProcessing::applyMask(frame, maskedImageGpu_static, maskGpu_static, stream);
            CudaImageProcessing::resize(maskedImageGpu_static, resizedBuffer, w, h, stream);
        } else {
            CudaImageProcessing::resize(frame, resizedBuffer, w, h, stream);
        }

        // Process RGB color filter if enabled
        bool current_enable_color_filter = false;
        bool current_remove_color_matches = false;
        {
            std::lock_guard<std::mutex> lock(ctx.configMutex);
            current_enable_color_filter = ctx.config.enable_color_filter;
            current_remove_color_matches = ctx.config.remove_color_matches;
        }

        // Skip color processing entirely if not needed
        if (current_enable_color_filter && (current_remove_color_matches || ctx.config.min_color_pixels > 0)) {
            static SimpleCudaMat maskGpu_rgb_static;
            
            // Create color mask using RGB filter
            maskGpu_rgb_static.create(resizedBuffer.rows(), resizedBuffer.cols(), 1);
            
            // Use the RGB range filter (much faster than HSV)
            launchRGBRangeFilter(
                resizedBuffer.data(),
                maskGpu_rgb_static.data(),
                resizedBuffer.cols(),
                resizedBuffer.rows(),
                static_cast<int>(resizedBuffer.step()),
                ctx.config.rgb_min_r, ctx.config.rgb_max_r,
                ctx.config.rgb_min_g, ctx.config.rgb_max_g,
                ctx.config.rgb_min_b, ctx.config.rgb_max_b,
                stream
            );
            
            // Only resize if resolution changed
            static int last_resolution = 0;
            if (last_resolution != ctx.config.detection_resolution) {
                last_resolution = ctx.config.detection_resolution;
                SimpleCudaMat maskResized;
                CudaImageProcessing::resize(maskGpu_rgb_static, maskResized, last_resolution, last_resolution, stream);
                {
                    std::lock_guard<std::mutex> lock(colorMaskMutex);
                    m_colorMaskGpu = std::move(maskResized);
                }
            }
        } else {
            // Clear color mask if not in use
            std::lock_guard<std::mutex> lock(colorMaskMutex);
            if (!m_colorMaskGpu.empty()) {
                m_colorMaskGpu.release();
            }
        }

        // Handle both BGR (3 channel) and BGRA (4 channel) inputs
        int inputChannels = resizedBuffer.channels();
        
        // Convert to float first
        CudaFloatProcessing::convertToFloat(resizedBuffer, floatBuffer, 1.0f / 255.0f, 0.0f, stream);
        
        if (inputChannels == 4 && c == 3) {
            // Input is BGRA but model expects BGR - extract first 3 channels only
            // Create 4 channel buffers but only copy first 3 to model
            channelBuffers.resize(4);
            for (int i = 0; i < 4; ++i) {
                channelBuffers[i].create(h, w, 1);
            }
            CudaFloatProcessing::splitFloat(floatBuffer, channelBuffers.data(), stream);
            
            // Copy only BGR channels (0,1,2), skip Alpha (3)
            size_t channelSize = h * w * sizeof(float);
            for (int i = 0; i < 3; ++i)
            {
                cudaMemcpyAsync(
                    static_cast<float*>(inputBuffer) + i * h * w,
                    channelBuffers[i].data(),
                    channelSize,
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            }
        } else {
            // Standard path for matching channel counts
            channelBuffers.resize(c);
            for (int i = 0; i < c; ++i) {
                channelBuffers[i].create(h, w, 1);
            }
            CudaFloatProcessing::splitFloat(floatBuffer, channelBuffers.data(), stream);

            size_t channelSize = h * w * sizeof(float);
            for (int i = 0; i < c; ++i)
            {
                cudaMemcpyAsync(
                    static_cast<float*>(inputBuffer) + i * h * w,
                    channelBuffers[i].data(),
                    channelSize,
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            }
        }
    } catch (const std::exception& e)
    {
        std::cerr << "[Detector] Error in preProcess: " << e.what() << std::endl;
    }
}



void Detector::initializeBuffers() {
    auto& ctx = AppContext::getInstance();
    if (!context || !stream) {
        std::cerr << "[Detector] Error: Cannot initialize buffers without valid context and stream." << std::endl;
        return; 
    }

    // Determine max_candidates based on the model's output shape
    int max_candidates = 0;
    if (!outputNames.empty()) {
        const std::string& primaryOutputName = outputNames[0];
        const std::vector<int64_t>& shape = outputShapes[primaryOutputName];
        if (ctx.config.postprocess == "yolo10") {
            // YOLO10: shape is [1, num_detections, 6] where 6 = bbox(4) + conf(1) + class_id(1)
            // num_detections (shape[1]) is the actual number of candidates (e.g., 300)
            max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        } else if (ctx.config.postprocess == "yolo8" || ctx.config.postprocess == "yolo9" || ctx.config.postprocess == "yolo11" || ctx.config.postprocess == "yolo12") {
            // YOLO8/9/11/12: shape is [1, features, num_boxes] where features = bbox(4) + conf(1) + num_classes
            // num_boxes (shape[2]) is the number of candidates (e.g., 8400)
            max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        }
        
        // Buffer allocation completed for model output shape
    }

    if (max_candidates == 0) {
        std::cerr << "[Detector] Warning: Could not determine max_candidates from model output shape. Using default max_detections * 2." << std::endl;
        max_candidates = ctx.config.max_detections * 2; // Fallback
    }
    
    // Allocate extra buffer space for safety
    int buffer_size = max_candidates * 2;
    m_decodedTargetsGpu.allocate(buffer_size);
    m_decodedCountGpu.allocate(1);
    
    // Allocate buffers - balanced to avoid illegal memory access
    const int graph_buffer_size = Constants::MAX_DETECTIONS; // Final output size
    const int intermediate_buffer_size = 300; // Reasonable buffer for intermediate processing
    m_finalTargetsGpu.allocate(graph_buffer_size);
    m_finalTargetsCountGpu.allocate(1);
    m_finalTargetsHost = std::unique_ptr<Target[]>(new Target[graph_buffer_size]);
    m_classFilteredTargetsGpu.allocate(intermediate_buffer_size);  // Buffer for class filtered detections
    m_classFilteredCountGpu.allocate(1);
    m_colorFilteredTargetsGpu.allocate(intermediate_buffer_size);  // Buffer for color filtered detections
    m_colorFilteredCountGpu.allocate(1);
    m_scoresGpu.allocate(graph_buffer_size);

    m_matchingIndexGpu.allocate(1);
    m_matchingScoreGpu.allocate(1);
    
    // Allocate temporary buffers for multi-block reduction
    // Maximum number of blocks we might need
    const int max_blocks = (ctx.config.max_detections + 255) / 256;
    m_tempBlockScores.allocate(max_blocks);
    m_tempBlockIndices.allocate(max_blocks);

    m_d_allow_flags_gpu.allocate(MAX_CLASSES_FOR_FILTERING);
    
    // Allocate GPU buffers for target selection
    m_bestTargetIndexGpu.allocate(1);
    m_bestTargetGpu.allocate(1);
    
    // Initialize target selection buffers
    if (m_bestTargetIndexGpu.get()) {
        int neg_one = -1;
        cudaMemcpyAsync(m_bestTargetIndexGpu.get(), &neg_one, sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    if (m_bestTargetGpu.get()) {
        cudaMemsetAsync(m_bestTargetGpu.get(), 0, sizeof(Target), stream);
    }

    if (m_decodedCountGpu.get()) cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    if (m_finalTargetsCountGpu.get()) cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
    if (m_classFilteredCountGpu.get()) cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    
    // Initialize Target arrays to zero to prevent garbage values
    // Do NOT use 0xFF as it creates invalid float values (NaN)
    if (m_decodedTargetsGpu.get()) {
        cudaMemsetAsync(m_decodedTargetsGpu.get(), 0, buffer_size * sizeof(Target), stream);
    }
    if (m_finalTargetsGpu.get()) {
        cudaMemsetAsync(m_finalTargetsGpu.get(), 0, graph_buffer_size * sizeof(Target), stream);
    }
    if (m_classFilteredTargetsGpu.get()) {
        cudaMemsetAsync(m_classFilteredTargetsGpu.get(), 0, intermediate_buffer_size * sizeof(Target), stream);
    }
    if (m_colorFilteredTargetsGpu.get()) {
        cudaMemsetAsync(m_colorFilteredTargetsGpu.get(), 0, intermediate_buffer_size * sizeof(Target), stream);
    }

}


float Detector::calculate_host_iou(const Target& det1, const Target& det2) {
    int xA = (std::max)(det1.x, det2.x);
    int yA = (std::max)(det1.y, det2.y);
    int xB = (std::min)(det1.x + det1.width, det2.x + det2.width);
    int yB = (std::min)(det1.y + det1.height, det2.y + det2.height);

    
    int interArea = (std::max)(0, xB - xA) * (std::max)(0, yB - yA);

    
    int box1Area = det1.width * det1.height;
    int box2Area = det2.width * det2.height;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    
    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}











SimpleCudaMat Detector::getColorMaskGpu() const { 
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(colorMaskMutex)); 
    if (m_colorMaskGpu.empty()) {
        return SimpleCudaMat(); 
    }
    return m_colorMaskGpu.clone(); 
}

void Detector::warmupGPU()
{
    auto& ctx = AppContext::getInstance();
    
    // 더미 데이터로 GPU 워밍업
    if (inputBufferDevice && inputDims.nbDims >= 4) {
        size_t dummySize = static_cast<size_t>(inputDims.d[1]) * static_cast<size_t>(inputDims.d[2]) * static_cast<size_t>(inputDims.d[3]) * sizeof(float);
        
        float* dummyData = nullptr;
        cudaMalloc(&dummyData, dummySize);
        cudaMemset(dummyData, 0, dummySize);
        
        // 5번 정도 더미 추론 실행
        for (int i = 0; i < 5; i++) {
            if (context) {
                void* bindings[] = { dummyData };
                context->enqueueV3(stream);
                cudaStreamSynchronize(stream);
            }
        }
        
        cudaFree(dummyData);
    }
}


void Detector::captureInferenceGraph(const SimpleCudaMat& frameGpu)
{
    
    if (m_graphCaptured || !context || !frameGpu.data()) {
        return;
    }
    
    auto& ctx = AppContext::getInstance();
    
    // YOLO10 has dynamic output shape due to end-to-end NMS, skip graph capture
    if (ctx.config.postprocess == "yolo10") {
        return;
    }
    
    static int captureAttempts = 0;
    const int maxAttempts = 5;
    
    if (captureAttempts >= maxAttempts) {
        return;
    }
    
    captureAttempts++;
    
    // Clear any previous CUDA errors and check
    cudaError_t prevError = cudaGetLastError();
    if (prevError != cudaSuccess) {
        std::cerr << "[Graph] Previous CUDA error detected, skipping capture: " << cudaGetErrorString(prevError) << std::endl;
        return;
    }
    
    // Ensure all streams are idle
    cudaStreamSynchronize(preprocessStream);
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(postprocessStream);
    
    // Multiple warmup runs to ensure all allocations are done
    for (int i = 0; i < 3; i++) {
        bool warmupResult = context->enqueueV3(stream);
        if (!warmupResult) {
            std::cerr << "[Graph] Warmup inference #" << (i+1) << " failed" << std::endl;
            return;
        }
        cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            std::cerr << "[Graph] Warmup sync failed: " << cudaGetErrorString(syncErr) << std::endl;
            return;
        }
    }
    
    // Try to capture with relaxed mode to avoid conflicts with other CUDA operations
    cudaError_t captureResult = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (captureResult != cudaSuccess) {
        std::cerr << "[Graph] Failed to begin capture: " << cudaGetErrorString(captureResult) << std::endl;
        return;
    }
    
    // Capture inference only
    bool enqueueResult = context->enqueueV3(stream);
    if (!enqueueResult) {
        std::cerr << "[Graph] Failed to enqueue during capture" << std::endl;
        cudaStreamEndCapture(stream, nullptr);
        return;
    }
    
    // End capture
    cudaGraph_t tempGraph = nullptr;
    captureResult = cudaStreamEndCapture(stream, &tempGraph);
    
    if (!enqueueResult || captureResult != cudaSuccess) {
        if (tempGraph) cudaGraphDestroy(tempGraph);
        return;
    }
    
    // Validate and instantiate
    if (tempGraph != nullptr) {
        cudaGraphNode_t* nodes = nullptr;
        size_t numNodes = 0;
        cudaError_t getNodesResult = cudaGraphGetNodes(tempGraph, nullptr, &numNodes);
        
        if (getNodesResult == cudaSuccess && numNodes > 0) {
            cudaError_t instantiateResult = cudaGraphInstantiate(&m_inferenceGraphExec, tempGraph, nullptr, nullptr, 0);
            
            if (instantiateResult == cudaSuccess && m_inferenceGraphExec != nullptr) {
                m_inferenceGraph = tempGraph;
                m_graphCaptured = true;
                
                // Test launch
                cudaError_t launchResult = cudaGraphLaunch(m_inferenceGraphExec, stream);
                if (launchResult == cudaSuccess) {
                    cudaError_t syncErr = cudaStreamSynchronize(stream);
                    if (syncErr != cudaSuccess) {
                        std::cerr << "[Graph] Test launch sync failed: " << cudaGetErrorString(syncErr) << std::endl;
                        // Clean up
                        cudaGraphExecDestroy(m_inferenceGraphExec);
                        cudaGraphDestroy(m_inferenceGraph);
                        m_inferenceGraphExec = nullptr;
                        m_inferenceGraph = nullptr;
                        m_graphCaptured = false;
                        return;
                    }
                } else {
                    std::cerr << "[Graph] Test launch failed: " << cudaGetErrorString(launchResult) << std::endl;
                    // Clean up
                    cudaGraphExecDestroy(m_inferenceGraphExec);
                    cudaGraphDestroy(m_inferenceGraph);
                    m_inferenceGraphExec = nullptr;
                    m_inferenceGraph = nullptr;
                    m_graphCaptured = false;
                }
            } else {
                std::cerr << "[Graph] Failed to instantiate graph: " << cudaGetErrorString(instantiateResult) << std::endl;
                if (tempGraph) cudaGraphDestroy(tempGraph);
            }
        } else {
            if (tempGraph) cudaGraphDestroy(tempGraph);
        }
    }
}

void Detector::start()
{
    auto& ctx = AppContext::getInstance();
    ctx.should_exit = false;
    
    // Note: capture thread is handled separately in capture.cpp
    m_inferenceThread = std::thread(&Detector::inferenceThread, this);
}

void Detector::initializeKalmanFilter()
{
    auto& ctx = AppContext::getInstance();
    
    // Destroy existing Kalman filter if any
    destroyKalmanFilter();
    
    if (ctx.config.enable_kalman_filter) {
        
        try {
            m_gpuKalmanTracker = createGPUKalmanTracker(100, Constants::MAX_DETECTIONS);
            m_kalmanPredictionsGpu.allocate(Constants::MAX_DETECTIONS);
            m_kalmanPredictionsCountGpu.allocate(1);
            
            // Initialize filter constants with frame-based delta
            updateKalmanFilterSettings(
                1.0f,  // Frame-based: 1 frame = 1.0
                ctx.config.kalman_process_noise,
                ctx.config.kalman_measurement_noise,
                0  // Use default stream
            );
            
        } catch (const std::exception& e) {
            std::cerr << "[Detector] Failed to initialize GPU Kalman filter: " << e.what() << std::endl;
            m_gpuKalmanTracker = nullptr;
        }
    }
}

void Detector::destroyKalmanFilter()
{
    if (m_gpuKalmanTracker) {
        destroyGPUKalmanTracker(m_gpuKalmanTracker);
        m_gpuKalmanTracker = nullptr;
        m_kalmanGraphInitialized = false;
    }
}

void Detector::stop()
{
    auto& ctx = AppContext::getInstance();
    ctx.should_exit = true;
    
    // Notify detection CV only (inference uses busy wait now)
    detectionCV.notify_all();
    
    if (m_captureThread.joinable())
    {
        m_captureThread.join();
    }
    if (m_inferenceThread.joinable())
    {
        m_inferenceThread.join();
    }
}

// TensorRT utility functions implementation
nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath)
{
    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Suppress TensorRT internal errors
            if (severity <= Severity::kERROR && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    static SimpleLogger logger;

    auto& ctx = AppContext::getInstance();
    
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) return nullptr;

    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network) return nullptr;

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) return nullptr;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TensorRT] Failed to parse ONNX file: " << onnxPath << std::endl;
        return nullptr;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return nullptr;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB

    // 더 공격적인 TensorRT 최적화
    if (ctx.config.tensorrt_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TensorRT] FP16 optimization enabled" << std::endl;
    }
    
    
    // Additional optimization flags
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    
    // Enable tactics sources for better kernel selection
    config->setTacticSources(
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS) |
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT) |
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN)
    );
    
    // Profiling for optimal kernel selection
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // Create optimization profile for dynamic inputs
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "[TensorRT] Failed to create optimization profile" << std::endl;
        return nullptr;
    }

    // Set optimization profile for the input (assuming batch size = 1, channels = 3, and dynamic height/width)
    const char* inputName = network->getInput(0)->getName();
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    
    // For YOLO models, typically the input is [1, 3, height, width]
    // Set min, opt, and max dimensions - using the config's input resolution
    int resolution = ctx.config.onnx_input_resolution;
    nvinfer1::Dims minDims = nvinfer1::Dims4{1, 3, resolution, resolution};
    nvinfer1::Dims optDims = nvinfer1::Dims4{1, 3, resolution, resolution};
    nvinfer1::Dims maxDims = nvinfer1::Dims4{1, 3, resolution, resolution};
    
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& enginePath)
{
    class SimpleLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Suppress TensorRT internal errors
            if (severity <= Severity::kERROR && 
                (strstr(msg, "defaultAllocator.cpp") == nullptr) &&
                (strstr(msg, "enqueueV3") == nullptr)) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    static SimpleLogger logger;

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[TensorRT] Failed to open engine file: " << enginePath << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "[TensorRT] Failed to create runtime" << std::endl;
        return nullptr;
    }

    return runtime->deserializeCudaEngine(buffer.data(), size);
}
