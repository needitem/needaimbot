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
extern cudaError_t filterDetectionsByClassIdGpu(
    const Detection* decodedDetections,
    int numDecodedDetections,
    Detection* filteredDetections,
    int* filteredCount,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    int max_output_detections,
    cudaStream_t stream
);

extern cudaError_t filterDetectionsByColorGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
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
    m_finalDetectionsCountHost(0),
    m_host_allow_flags_uchar(MAX_CLASSES_FOR_FILTERING, 0), 
    m_allow_flags_need_update(true) 
    , m_isTargetLocked(false)
 
{

}

Detector::~Detector()
{
    std::cout << "[Detector] Starting resource cleanup..." << std::endl;
    
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
        
        // 7. 스트림 정리 (마지막에)
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
    std::cout << "[Detector] Successfully set CUDA device to " << ctx.config.cuda_device_id << "." << std::endl;

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
    
    std::cout << "[Detector] Memory pool initialized with " << (pool_size / (1024.0 * 1024.0)) << "MB" << std::endl;
    
    // GPU 메모리 병합 최적화
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 2);
    
    // L2 캐시 크기 제한 확인 및 설정
    size_t l2CacheSize = 0;
    cudaDeviceGetLimit(&l2CacheSize, cudaLimitPersistingL2CacheSize);
    std::cout << "[Detector] Current L2 cache limit: " << (l2CacheSize / (1024.0 * 1024.0)) << "MB" << std::endl;
    
    // 시스템이 지원하는 최대값으로 L2 캐시 설정
    if (l2CacheSize < 16 * 1024 * 1024) {
        // 시스템이 16MB를 지원하지 않으면 현재 값을 유지하거나 약간 증가
        size_t newL2CacheSize = std::min(l2CacheSize * 2, size_t(8 * 1024 * 1024)); // 최대 8MB까지만
        cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, newL2CacheSize);
        if (err == cudaSuccess) {
            std::cout << "[Detector] L2 cache limit set to: " << (newL2CacheSize / (1024.0 * 1024.0)) << "MB" << std::endl;
        } else {
            std::cout << "[Detector] Failed to set L2 cache limit: " << cudaGetErrorString(err) << std::endl;
        }
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
            if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
        }
    };
    static SimpleLogger logger;
    
    // TensorRT 버전 출력
    std::cout << "[Detector] TensorRT version: " << NV_TENSORRT_MAJOR << "." 
              << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
    
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
    // 메모리 할당 전략 최적화
    if (context->setOptimizationProfileAsync(0, stream)) {
        std::cout << "[Detector] Optimization profile set for better performance" << std::endl;
    }
    
    // CUDA 그래프 호환성 설정
    context->setEnqueueEmitsProfile(false);
    
    // 영구 캐시 활성화 - 시스템 제한에 맞게 조정
    size_t systemL2CacheLimit = 0;
    cudaDeviceGetLimit(&systemL2CacheLimit, cudaLimitPersistingL2CacheSize);
    
    size_t requestedCacheLimit = static_cast<size_t>(ctx.config.persistent_cache_limit_mb) * 1024 * 1024;
    size_t actualCacheLimit = std::min(requestedCacheLimit, systemL2CacheLimit);
    
    context->setPersistentCacheLimit(actualCacheLimit);
    std::cout << "[Detector] TensorRT L2 cache limit set to " << (actualCacheLimit / (1024.0 * 1024.0)) 
              << "MB (requested: " << ctx.config.persistent_cache_limit_mb 
              << "MB, system max: " << (systemL2CacheLimit / (1024.0 * 1024.0)) << "MB)" << std::endl;

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
    
    // BGRA input support flag - can be added to detector.h if needed later
    // bool m_supportBGRA = true; // RTX 40 series handles BGRA efficiently
    
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

    std::cout << "[Detector] Loading engine: " << engineFilePath << std::endl;
    engine.reset(loadEngineFromFile(engineFilePath));
}

void Detector::processFrame(const SimpleCudaMat& frame)
{
    auto& ctx = AppContext::getInstance();
    
    
    if (!isCudaContextInitialized()) {
        return;
    }

    if (ctx.detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
        memset(&m_bestTargetHost, 0, sizeof(Detection));
        m_bestTargetIndexHost = -1;
        m_finalDetectionsCountHost = 0;
        detectionVersion++;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 비동기 프레임 전달 - lock 시간 최소화
    {
        std::unique_lock<std::mutex> lock(inferenceMutex);
        
        // 이전 프레임이 아직 처리 중이면 스킵 (프레임 드롭으로 낮은 지연시간 유지)
        if (frameReady) {
            return; // 추론이 캡처보다 느리면 프레임 스킵
        }
        
        currentFrame = frame.clone();
        frameIsGpu = true;
        frameReady = true;
    }
    
    inferenceCV.notify_one();

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
        memset(&m_bestTargetHost, 0, sizeof(Detection));
        m_bestTargetIndexHost = -1;
        m_finalDetectionsCountHost = 0;
        detectionVersion++;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrameCpu = frame;
    frameIsGpu = false;
    frameReady = true;
    inferenceCV.notify_one();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    ctx.g_current_process_frame_time_ms.store(duration.count());
    ctx.add_to_history(ctx.g_process_frame_time_history, duration.count(), ctx.g_process_frame_history_mutex);
}

void Detector::inferenceThread()
{
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) {
        std::cerr << "[Detector Thread] CUDA context not initialized. Inference thread exiting." << std::endl;
        return;
    }
    
    // CPU 스레드 친화성 설정 (성능 코어에 고정)
    {
        HANDLE thread = GetCurrentThread();
        DWORD_PTR mask = 0x0F; // 첫 4개 코어 사용 (보통 성능 코어)
        SetThreadAffinityMask(thread, mask);
        SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
        std::cout << "[Detector] Thread affinity and priority set for optimal performance" << std::endl;
    }
    
    // Cache frequently accessed config values to reduce mutex contention
    struct CachedConfig {
        int detection_resolution;
        int max_detections;
        float confidence_threshold;
        float nms_threshold;
        float distance_weight;
        float confidence_weight;
        float crosshair_offset_x;
        float crosshair_offset_y;
        float aim_shoot_offset_x;
        float aim_shoot_offset_y;
        bool enable_aim_shoot_offset;
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
            crosshair_offset_x = config.crosshair_offset_x;
            crosshair_offset_y = config.crosshair_offset_y;
            aim_shoot_offset_x = config.aim_shoot_offset_x;
            aim_shoot_offset_y = config.aim_shoot_offset_y;
            enable_aim_shoot_offset = config.enable_aim_shoot_offset;
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

    while (!ctx.should_exit)
    {
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

        bool hasNewFrame = false;
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            // Wait indefinitely for a frame or exit signal - no timeout
            inferenceCV.wait(lock, [this] { return frameReady || AppContext::getInstance().should_exit; });
            
            if (AppContext::getInstance().should_exit) break;
            
            if (frameReady) {
                if (frameIsGpu) {
                    frameGpu = std::move(currentFrame);
                } else {
                    frameGpu.create(currentFrameCpu.rows(), currentFrameCpu.cols(), currentFrameCpu.channels());
                    frameGpu.upload(currentFrameCpu.data(), currentFrameCpu.step());
                }
                frameReady = false;
                hasNewFrame = true;
            }
        }

        if (!context) {
            std::cerr << "[Detector] Context not initialized" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (hasNewFrame && !frameGpu.empty())
        {
            try
            {
                auto inference_start_time = std::chrono::high_resolution_clock::now();
                
                // Update config cache every 100 frames to reduce mutex contention
                if (++frame_counter % 100 == 0) {
                    std::lock_guard<std::mutex> lock(ctx.configMutex);
                    cached_config.update(ctx.config);
                }

                // 파이프라인 최적화: 이전 프레임의 후처리와 현재 프레임의 전처리를 병렬화
                static bool firstFrame = true;
                
                if (!firstFrame) {
                    // 이전 프레임의 후처리가 완료될 때까지 대기
                    cudaEventSynchronize(processingDone);
                }
                
                // Execute preprocessing
                preProcess(frameGpu, preprocessStream);
                cudaEventRecord(m_preprocessDone, preprocessStream);
                
                // CUDA Graph 기반 추론 (충분한 프레임 후 캡처)
                if (!m_graphCaptured && frame_counter > 30 && frame_counter % 10 == 0) {
                    captureInferenceGraph(frameGpu);
                }
                
                bool enqueueSuccess;
                // 전처리 완료 대기
                cudaStreamWaitEvent(stream, m_preprocessDone, cudaEventWaitExternal);
                
                if (m_graphCaptured && m_inferenceGraphExec) {
                    // Graph 실행 (추론만)
                    cudaError_t graphLaunchResult = cudaGraphLaunch(m_inferenceGraphExec, stream);
                    if (graphLaunchResult != cudaSuccess) {
                        std::cerr << "[Detector] CUDA Graph launch failed: " << cudaGetErrorString(graphLaunchResult) << std::endl;
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
                
                // 후처리는 항상 별도로 실행 (동적 크기 때문에)
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
                cudaMemcpyAsync(&m_finalDetectionsCountHost, m_finalDetectionsCountGpu.get(), 
                               sizeof(int), cudaMemcpyDeviceToHost, postprocessStream);
                
                // 먼저 count를 동기화해서 가져온 후 detection 데이터 복사
                cudaStreamSynchronize(postprocessStream);
                
                // Debug: Check final detection count
                if (m_finalDetectionsCountHost > 0) {
                    std::cout << "[Detector] Final detections on CPU: " << m_finalDetectionsCountHost << std::endl;
                }
                
                // 검출된 객체가 있으면 detection 데이터도 복사
                if (m_finalDetectionsCountHost > 0) {
                    cudaMemcpyAsync(m_finalDetectionsHost.get(), m_finalDetectionsGpu.get(), 
                                   m_finalDetectionsCountHost * sizeof(Detection), 
                                   cudaMemcpyDeviceToHost, postprocessStream);
                    cudaStreamSynchronize(postprocessStream);
                    
                    // Debug: Check first detection
                    std::cout << "[Detector] First detection - x:" << m_finalDetectionsHost[0].x 
                              << " y:" << m_finalDetectionsHost[0].y 
                              << " w:" << m_finalDetectionsHost[0].width 
                              << " h:" << m_finalDetectionsHost[0].height 
                              << " conf:" << m_finalDetectionsHost[0].confidence 
                              << " class:" << m_finalDetectionsHost[0].classId << std::endl;
                }

                auto inference_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float, std::milli> inference_duration_ms = inference_end_time - inference_start_time;
                ctx.g_current_inference_time_ms.store(inference_duration_ms.count());
                ctx.add_to_history(ctx.g_inference_time_history, inference_duration_ms.count(), ctx.g_inference_history_mutex);
                
                
                // GPU에서 거리 기반 타겟 선택
                if (m_finalDetectionsCountHost > 0 && !ctx.should_exit) {
                    float crosshairX, crosshairY;
                    if (cached_config.enable_aim_shoot_offset && ctx.aiming && ctx.shooting) {
                        crosshairX = cached_config.detection_resolution / 2.0f + cached_config.aim_shoot_offset_x;
                        crosshairY = cached_config.detection_resolution / 2.0f + cached_config.aim_shoot_offset_y;
                    } else {
                        crosshairX = cached_config.detection_resolution / 2.0f + cached_config.crosshair_offset_x;
                        crosshairY = cached_config.detection_resolution / 2.0f + cached_config.crosshair_offset_y;
                    }
                    
                    // Find closest target on GPU
                    cudaError_t target_err = findClosestTargetGpu(
                        m_finalDetectionsGpu.get(),
                        m_finalDetectionsCountHost,
                        crosshairX,
                        crosshairY,
                        m_bestTargetIndexGpu.get(),
                        m_bestTargetGpu.get(),
                        postprocessStream
                    );
                    
                    if (target_err == cudaSuccess) {
                        // Copy results to host
                        cudaMemcpyAsync(&m_bestTargetIndexHost, m_bestTargetIndexGpu.get(), 
                                       sizeof(int), cudaMemcpyDeviceToHost, postprocessStream);
                        cudaMemcpyAsync(&m_bestTargetHost, m_bestTargetGpu.get(), 
                                       sizeof(Detection), cudaMemcpyDeviceToHost, postprocessStream);
                        cudaStreamSynchronize(postprocessStream);
                        
                        // Update detection state
                        {
                            std::lock_guard<std::mutex> lock(detectionMutex);
                            
                            if (m_bestTargetIndexHost >= 0) {
                                m_hasBestTarget = true;
                                std::cout << "[Detector] Best target selected - idx:" << m_bestTargetIndexHost 
                                          << " x:" << m_bestTargetHost.x 
                                          << " y:" << m_bestTargetHost.y << std::endl;
                            } else {
                                m_hasBestTarget = false;
                                memset(&m_bestTargetHost, 0, sizeof(Detection));
                                std::cout << "[Detector] No best target found" << std::endl;
                            }
                            
                            detectionVersion++;
                        }
                        detectionCV.notify_one();
                    } else {
                        std::cerr << "[Detector] Error in GPU target selection: " << cudaGetErrorString(target_err) << std::endl;
                        
                        std::lock_guard<std::mutex> lock(detectionMutex);
                        m_hasBestTarget = false;
                        memset(&m_bestTargetHost, 0, sizeof(Detection));
                        m_bestTargetIndexHost = -1;
                        detectionVersion++;
                        detectionCV.notify_one();
                    }
                } else {
                    // No detections or should exit
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    m_hasBestTarget = false;
                    memset(&m_bestTargetHost, 0, sizeof(Detection));
                    m_bestTargetIndexHost = -1;
                    detectionVersion++;
                    detectionCV.notify_one();
                }

            }
            catch (const std::exception& e)
            {
                std::cerr << "[Detector] Error during inference loop: " << e.what() << std::endl;
                m_hasBestTarget = false;
                memset(&m_bestTargetHost, 0, sizeof(Detection));
                m_bestTargetIndexHost = -1;
                m_finalDetectionsCountHost = 0;
            }
        } else if (hasNewFrame) {
             {
                std::lock_guard<std::mutex> lock(detectionMutex);
                m_hasBestTarget = false;
                memset(&m_bestTargetHost, 0, sizeof(Detection));
                m_bestTargetIndexHost = -1;
                m_finalDetectionsCountHost = 0;
                detectionVersion++;
             }
             detectionCV.notify_one();
        } else {
            // No new frame received - keep previous detection state
            // Don't clear target or increment version unnecessarily
        }
        NVTX_POP();
    }
}

void Detector::performGpuPostProcessing(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    if (outputNames.empty()) {
        std::cerr << "[Detector] No output names found for post-processing." << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    static int postProcessCount = 0;
    postProcessCount++;

    const std::string& primaryOutputName = outputNames[0];
    void* d_rawOutputPtr = outputBindings[primaryOutputName];
    nvinfer1::DataType outputType = outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = outputShapes[primaryOutputName];


    if (!d_rawOutputPtr) {
        std::cerr << "[Detector] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    // Clear all detection buffers at the start of processing
    cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);

    cudaError_t decodeErr = cudaSuccess;

    // Use cached config values for CUDA Graph compatibility
    // These should be set once before graph capture
    static int cached_max_detections = Constants::MAX_DETECTIONS; // Use max from constants (100)
    static float cached_nms_threshold = 0.45f;
    static float cached_confidence_threshold = 0.25f;
    static std::string cached_postprocess = "yolo12";
    static bool config_cached = false;
    
    // Only update cache when not in graph capture mode
    if (!m_graphCaptured && !config_cached) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_max_detections = ctx.config.max_detections;
        cached_nms_threshold = ctx.config.nms_threshold;
        cached_confidence_threshold = ctx.config.confidence_threshold;
        cached_postprocess = ctx.config.postprocess;
        config_cached = true;
    }

    // Use a reasonable buffer for decoding - balance between capacity and memory
    // Too large buffers cause illegal memory access in NMS
    int maxDecodedDetections = 300;  // Reasonable buffer for detections
    
    // GPU decoding debug info removed - enable for debugging if needed

    
    
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
            m_decodedDetectionsGpu.get(),
            m_decodedCountGpu.get(),
            max_candidates,
            maxDecodedDetections,  // Large buffer for all detections
            stream);
    } else if (cached_postprocess == "yolo8" || cached_postprocess == "yolo9" || cached_postprocess == "yolo11" || cached_postprocess == "yolo12") {
        int max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        // Pass large buffer size to decode ALL detections
         decodeErr = decodeYolo11Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            cached_confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu.get(),
            m_decodedCountGpu.get(),
            max_candidates,
            maxDecodedDetections,  // Large buffer for all detections
            stream);
    } else {
        std::cerr << "[Detector] Unsupported post-processing type for GPU decoding: " << cached_postprocess << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    if (decodeErr != cudaSuccess) {
        std::cerr << "[Detector] GPU decoding kernel launch/execution failed: " << cudaGetErrorString(decodeErr) << std::endl;
        std::cerr << "[Detector] CUDA Error Code: " << decodeErr << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    // For CUDA Graph compatibility, use fixed counts when in graph mode
    int decodedCount = maxDecodedDetections;  // Default to max for graph mode
    int classFilteredCount = maxDecodedDetections;
    
    // Only sync and get actual counts when not in graph mode
    if (!m_graphCaptured) {
        // Check decoded count
        cudaMemcpyAsync(&decodedCount, m_decodedCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        std::cout << "[DEBUG] Decoded count: " << decodedCount << std::endl;
        
        // If no detections were decoded, clear final count and return early
        if (decodedCount == 0) {
            cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        // Debug: Check first decoded detection
        if (decodedCount > 0) {
            Detection firstDecoded;
            cudaMemcpyAsync(&firstDecoded, m_decodedDetectionsGpu.get(), sizeof(Detection), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::cout << "[DEBUG] First decoded - x:" << firstDecoded.x 
                      << " y:" << firstDecoded.y 
                      << " w:" << firstDecoded.width 
                      << " h:" << firstDecoded.height 
                      << " conf:" << firstDecoded.confidence << std::endl;
        }
    }
    
    // Step 1: Class ID filtering (after confidence filtering in decode)
    cudaError_t filterErr = filterDetectionsByClassIdGpu(
        m_decodedDetectionsGpu.get(),
        decodedCount,  // Use actual count or max for graph mode
        m_classFilteredDetectionsGpu.get(),
        m_classFilteredCountGpu.get(),
        m_d_allow_flags_gpu.get(),
        MAX_CLASSES_FOR_FILTERING,
        300,  // Reasonable buffer for filtered detections
        stream
    );
    if (!checkCudaError(filterErr, "filtering detections by class ID GPU")) {
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    
    // Check class filtered count (only when not in graph mode)
    if (!m_graphCaptured) {
        cudaMemcpyAsync(&classFilteredCount, m_classFilteredCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        std::cout << "[DEBUG] Class filtered count: " << classFilteredCount << std::endl;
        
        // Debug: Check first class filtered detection
        if (classFilteredCount > 0) {
            Detection firstFiltered;
            cudaMemcpyAsync(&firstFiltered, m_classFilteredDetectionsGpu.get(), sizeof(Detection), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::cout << "[DEBUG] First class filtered - x:" << firstFiltered.x 
                      << " y:" << firstFiltered.y 
                      << " w:" << firstFiltered.width 
                      << " h:" << firstFiltered.height 
                      << " conf:" << firstFiltered.confidence << std::endl;
        }
    }
    
    // Step 2: RGB color filtering (before NMS for better efficiency)
    // Get color mask if available
    SimpleCudaMat colorMask = getColorMaskGpu();
    const unsigned char* colorMaskPtr = nullptr;
    int maskPitch = 0;
    
    // For non-graph mode, apply color filtering if mask exists
    Detection* nmsInputDetections = nullptr;
    int effectiveFilteredCount = classFilteredCount;  // Use actual filtered count
    
    if (!m_graphCaptured && !colorMask.empty()) {
        colorMaskPtr = colorMask.data();
        maskPitch = static_cast<int>(colorMask.step());
        
        // Apply RGB filtering
        cudaError_t colorFilterErr = filterDetectionsByColorGpu(
            m_classFilteredDetectionsGpu.get(),
            classFilteredCount,  // Process actual class-filtered count
            m_colorFilteredDetectionsGpu.get(),
            m_colorFilteredCountGpu.get(),
            colorMaskPtr,
            maskPitch,
            10,  // min_color_pixels threshold
            false,  // keep detections WITH color matches
            300,  // max output
            stream
        );
        
        if (!checkCudaError(colorFilterErr, "filtering detections by color GPU")) {
            cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        // Use color-filtered detections for NMS
        nmsInputDetections = m_colorFilteredDetectionsGpu.get();
        
        // Check color filtered count
        int colorFilteredCount = 0;
        cudaMemcpyAsync(&colorFilteredCount, m_colorFilteredCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        effectiveFilteredCount = colorFilteredCount;  // Update count for NMS
    } else {
        // No color filtering, use class-filtered detections directly
        nmsInputDetections = m_classFilteredDetectionsGpu.get();
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
            cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
            return;
        }
        
        // Debug: Check NMS input
        std::cout << "[DEBUG] NMS input count: " << effectiveFilteredCount << std::endl;
        if (effectiveFilteredCount > 0 && !m_graphCaptured) {
            Detection firstNmsInput;
            cudaMemcpyAsync(&firstNmsInput, nmsInputDetections, sizeof(Detection), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::cout << "[DEBUG] First NMS input - x:" << firstNmsInput.x 
                      << " y:" << firstNmsInput.y 
                      << " w:" << firstNmsInput.width 
                      << " h:" << firstNmsInput.height 
                      << " conf:" << firstNmsInput.confidence << std::endl;
        }
        
        // NMS will process filtered detections and output only max_detections
        NMSGpu(
            nmsInputDetections,
            effectiveFilteredCount, // Process all filtered detections
            m_finalDetectionsGpu.get(),       
            m_finalDetectionsCountGpu.get(),  
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
        
        // Target selection will be done on CPU after copying results
        
    } catch (const std::exception& e) {
         std::cerr << "[Detector] Exception during NMSGpu call: " << e.what() << std::endl;
         cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
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
            max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        } else if (ctx.config.postprocess == "yolo8" || ctx.config.postprocess == "yolo9" || ctx.config.postprocess == "yolo11" || ctx.config.postprocess == "yolo12") {
            max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        }
        
        std::cout << "[Detector] Buffer allocation info:" << std::endl;
        std::cout << "  - Primary output: " << primaryOutputName << std::endl;
        std::cout << "  - Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  - Max candidates: " << max_candidates << std::endl;
    }

    if (max_candidates == 0) {
        std::cerr << "[Detector] Warning: Could not determine max_candidates from model output shape. Using default max_detections * 2." << std::endl;
        max_candidates = ctx.config.max_detections * 2; // Fallback
    }
    
    // Allocate extra buffer space for safety
    int buffer_size = max_candidates * 2;
    std::cout << "[Detector] Allocating decoded detections buffer: " << buffer_size << " detections" << std::endl;
    m_decodedDetectionsGpu.allocate(buffer_size);
    m_decodedCountGpu.allocate(1);
    
    // Allocate buffers - balanced to avoid illegal memory access
    const int graph_buffer_size = Constants::MAX_DETECTIONS; // Final output size
    const int intermediate_buffer_size = 300; // Reasonable buffer for intermediate processing
    m_finalDetectionsGpu.allocate(graph_buffer_size);
    m_finalDetectionsCountGpu.allocate(1);
    m_finalDetectionsHost = std::make_unique<Detection[]>(graph_buffer_size);
    m_classFilteredDetectionsGpu.allocate(intermediate_buffer_size);  // Buffer for class filtered detections
    m_classFilteredCountGpu.allocate(1);
    m_colorFilteredDetectionsGpu.allocate(intermediate_buffer_size);  // Buffer for color filtered detections
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

    if (m_decodedCountGpu.get()) cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    if (m_finalDetectionsCountGpu.get()) cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
    if (m_classFilteredCountGpu.get()) cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    
    // Initialize Detection arrays to zero to prevent garbage values
    // Do NOT use 0xFF as it creates invalid float values (NaN)
    if (m_decodedDetectionsGpu.get()) {
        cudaMemsetAsync(m_decodedDetectionsGpu.get(), 0, buffer_size * sizeof(Detection), stream);
    }
    if (m_finalDetectionsGpu.get()) {
        cudaMemsetAsync(m_finalDetectionsGpu.get(), 0, graph_buffer_size * sizeof(Detection), stream);
    }
    if (m_classFilteredDetectionsGpu.get()) {
        cudaMemsetAsync(m_classFilteredDetectionsGpu.get(), 0, intermediate_buffer_size * sizeof(Detection), stream);
    }
    if (m_colorFilteredDetectionsGpu.get()) {
        cudaMemsetAsync(m_colorFilteredDetectionsGpu.get(), 0, intermediate_buffer_size * sizeof(Detection), stream);
    }

}


float Detector::calculate_host_iou(const Detection& det1, const Detection& det2) {
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
    std::cout << "[Detector] Warming up GPU..." << std::endl;
    
    // 더미 데이터로 GPU 워밍업
    if (inputBufferDevice && inputDims.nbDims >= 4) {
        size_t dummySize = static_cast<size_t>(inputDims.d[1]) * static_cast<size_t>(inputDims.d[2]) * static_cast<size_t>(inputDims.d[3]) * sizeof(float);
        
        // 작은 더미 데이터 생성
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
        std::cout << "[Detector] GPU warmup completed" << std::endl;
    }
}


void Detector::captureInferenceGraph(const SimpleCudaMat& frameGpu)
{
    if (m_graphCaptured || !context || !frameGpu.data()) return;
    
    static int captureAttempts = 0;
    const int maxAttempts = 5;
    
    if (captureAttempts >= maxAttempts) {
        std::cerr << "[Detector] CUDA Graph capture disabled after " << maxAttempts << " failed attempts" << std::endl;
        return;
    }
    
    captureAttempts++;
    std::cout << "[Detector] Capturing CUDA Graph for inference only (attempt " << captureAttempts << "/" << maxAttempts << ")..." << std::endl;
    
    // Clear any previous CUDA errors
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        std::cerr << "[Detector] Clearing previous CUDA error: " << cudaGetErrorString(lastError) << std::endl;
    }
    
    // Ensure all streams are idle before capture
    cudaStreamSynchronize(preprocessStream);
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(postprocessStream);
    
    // TensorRT 추론만 CUDA Graph로 캡처 (후처리는 제외)
    cudaError_t captureResult = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (captureResult != cudaSuccess) {
        std::cerr << "[Detector] Failed to begin CUDA Graph capture: " << cudaGetErrorString(captureResult) << std::endl;
        return;
    }
    
    // 추론 실행만 캡처
    bool enqueueResult = context->enqueueV3(stream);
    
    // Graph 캡처 종료
    cudaGraph_t tempGraph = nullptr;
    captureResult = cudaStreamEndCapture(stream, &tempGraph);
    
    if (!enqueueResult || captureResult != cudaSuccess) {
        std::cerr << "[Detector] TensorRT enqueue or capture failed during graph capture" << std::endl;
        if (tempGraph) cudaGraphDestroy(tempGraph);
        return;
    }
    
    // Check if capture was successful
    if (captureResult == cudaSuccess && tempGraph != nullptr) {
        // Validate graph before instantiation
        cudaGraphNode_t* nodes = nullptr;
        size_t numNodes = 0;
        cudaError_t getNodesResult = cudaGraphGetNodes(tempGraph, nullptr, &numNodes);
        
        if (getNodesResult == cudaSuccess && numNodes > 0) {
            std::cout << "[Detector] CUDA Graph captured with " << numNodes << " nodes" << std::endl;
            
            // Graph 인스턴스 생성
            cudaError_t instantiateResult = cudaGraphInstantiate(&m_inferenceGraphExec, tempGraph, nullptr, nullptr, 0);
            
            if (instantiateResult == cudaSuccess && m_inferenceGraphExec != nullptr) {
                m_inferenceGraph = tempGraph;
                m_graphCaptured = true;
                std::cout << "[Detector] CUDA Graph instantiated successfully" << std::endl;
                
                // Verify graph can be launched
                cudaError_t launchResult = cudaGraphLaunch(m_inferenceGraphExec, stream);
                if (launchResult == cudaSuccess) {
                    cudaStreamSynchronize(stream);
                    std::cout << "[Detector] CUDA Graph test launch successful" << std::endl;
                } else {
                    std::cerr << "[Detector] CUDA Graph test launch failed: " << cudaGetErrorString(launchResult) << std::endl;
                    // Clean up
                    cudaGraphExecDestroy(m_inferenceGraphExec);
                    cudaGraphDestroy(m_inferenceGraph);
                    m_inferenceGraphExec = nullptr;
                    m_inferenceGraph = nullptr;
                    m_graphCaptured = false;
                }
            } else {
                // Clean up on failure
                if (tempGraph) cudaGraphDestroy(tempGraph);
                std::cerr << "[Detector] Failed to instantiate CUDA Graph: " << cudaGetErrorString(instantiateResult) << std::endl;
            }
        } else {
            std::cerr << "[Detector] CUDA Graph has no nodes or failed to get nodes" << std::endl;
            if (tempGraph) cudaGraphDestroy(tempGraph);
        }
    } else {
        std::cerr << "[Detector] CUDA Graph capture failed: " << cudaGetErrorString(captureResult) << std::endl;
        // Clear any capture state
        cudaGetLastError();
    }
}

void Detector::start()
{
    auto& ctx = AppContext::getInstance();
    ctx.should_exit = false;
    
    // Note: capture thread is handled separately in capture.cpp
    m_inferenceThread = std::thread(&Detector::inferenceThread, this);
}

void Detector::stop()
{
    auto& ctx = AppContext::getInstance();
    ctx.should_exit = true;
    
    // Notify all condition variables to wake up waiting threads
    inferenceCV.notify_all(); 
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
            if (severity <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
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
    
    // INT8 양자화 활성화 (RTX 40 시리즈에 최적화)
    if (builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "[TensorRT] INT8 optimization enabled for maximum speed" << std::endl;
        
        // 더 강력한 양자화 설정
        // config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES); // TensorRT 10에서는 제거됨
        // INT8 캘리브레이션은 별도로 설정해야 함
    }
    
    // 추가 최적화 플래그
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK); // DLA 사용 시 GPU 폴백
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS); // 정밀도 제약 선호
    
    // 더 많은 커널 선택을 위한 tactics 소스 활성화
    config->setTacticSources(
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS) |
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT) |
        1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN)
    );
    
    // 프로파일링을 통한 최적 커널 선택
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
            if (severity <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
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
