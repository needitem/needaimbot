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




extern cudaError_t filterDetectionsByClassIdGpu(
    const Detection* decodedDetections,
    int numDecodedDetections,
    Detection* filteredDetections,
    int* filteredCount,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    const unsigned char* d_hsv_mask,
    int mask_pitch,
    int min_color_pixels,
    bool remove_hsv_matches,
    int max_output_detections,
    cudaStream_t stream
);

// Simple CPU function to find closest target based on distance
int findClosestTargetSimple(
    const Detection* detections,
    int count,
    float crosshairX,
    float crosshairY
) {
    if (count <= 0) return -1;
    
    float minDistance = FLT_MAX;
    int closestIdx = -1;
    
    for (int i = 0; i < count; i++) {
        float centerX = detections[i].x + detections[i].width * 0.5f;
        float centerY = detections[i].y + detections[i].height * 0.5f;
        
        float dx = centerX - crosshairX;
        float dy = centerY - crosshairY;
        float distance = dx * dx + dy * dy; // No need for sqrt since we're just comparing
        
        if (distance < minDistance) {
            minDistance = distance;
            closestIdx = i;
        }
    }
    
    return closestIdx;
}

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
    m_host_ignore_flags_uchar(MAX_CLASSES_FOR_FILTERING, 1), 
    m_ignore_flags_need_update(true) 
    , m_isTargetLocked(false)
    , m_lastDetectionTime(std::chrono::steady_clock::now()) 
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
            if (ctx.config.verbose)
            {
                std::cout << "[Detector] Detected input: " << name << std::endl;
            }
        }
    }



    
    m_nms_d_x1.allocate(ctx.config.max_detections);
    m_nms_d_y1.allocate(ctx.config.max_detections);
    m_nms_d_x2.allocate(ctx.config.max_detections);
    m_nms_d_y2.allocate(ctx.config.max_detections);
    m_nms_d_areas.allocate(ctx.config.max_detections);
    m_nms_d_scores.allocate(ctx.config.max_detections);
    m_nms_d_classIds.allocate(ctx.config.max_detections);
    m_nms_d_iou_matrix.allocate(ctx.config.max_detections * ctx.config.max_detections);
    m_nms_d_keep.allocate(ctx.config.max_detections);
    m_nms_d_indices.allocate(ctx.config.max_detections);
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
            
            if (ctx.config.verbose)
            {
                std::cout << "[Detector] Detected output: " << name << std::endl;
            }
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
                if (ctx.config.verbose)
                {
                    std::cout << "[Detector] Allocated " << size << " bytes for input " << name << std::endl;
                }
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
                if (ctx.config.verbose)
                {
                    std::cout << "[Detector] Allocated " << size << " bytes for output " << name << std::endl;
                }
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
    
    // GPU 메모리 병합 최적화
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 2);

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
    
    // 영구 캐시 활성화 - GPU의 실제 L2 캐시 크기에 맞춤
    // 대부분의 GPU는 16MB 이하의 L2 캐시를 가짐
    context->setPersistentCacheLimit(16 * 1024 * 1024); // 16MB (safe limit)

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

    
    m_ignore_flags_need_update = true;

    
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
                if (ctx.config.verbose) {
                    std::cout << "[Detector] Head class '" << ctx.config.head_class_name << "' identified with ID: " << m_headClassId << std::endl;
                }
                break;
            }
        }
        if (m_headClassId == -1 && ctx.config.verbose) {
            std::cout << "[Detector] Warning: Head class name '" << ctx.config.head_class_name << "' not found in class_settings. No specific head bonus will be applied." << std::endl;
        }
    } else if (ctx.config.verbose) {
        std::cout << "[Detector] head_class_name is empty in config. No specific head bonus will be applied based on name." << std::endl;
    }
    
    nvinfer1::Dims actualInputDims = context->getTensorShape(inputName.c_str());
    int c = static_cast<int>(actualInputDims.d[1]);
    int h = static_cast<int>(actualInputDims.d[2]);
    int w = static_cast<int>(actualInputDims.d[3]);
    
    if (resizedBuffer.empty() || resizedBuffer.rows() != h || resizedBuffer.cols() != w) {
        resizedBuffer.create(h, w, 3);
    }
    
    if (floatBuffer.empty() || floatBuffer.rows() != h || floatBuffer.cols() != w) {
        floatBuffer.create(h, w, 3); // We'll need to handle float types differently
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

        // Update ignore flags if needed
        if (m_ignore_flags_need_update) {
            std::lock_guard<std::mutex> lock(ctx.configMutex);
            // Reset all flags to 0 (don't ignore)
            std::fill(m_host_ignore_flags_uchar.begin(), m_host_ignore_flags_uchar.end(), 0);
            
            // Set ignore flags based on class settings
            for (const auto& class_setting : ctx.config.class_settings) {
                if (class_setting.id >= 0 && class_setting.id < MAX_CLASSES_FOR_FILTERING) {
                    m_host_ignore_flags_uchar[class_setting.id] = class_setting.ignore ? 1 : 0;
                }
            }
            
            // Copy to GPU
            cudaMemcpyAsync(m_d_ignore_flags_gpu.get(), m_host_ignore_flags_uchar.data(), 
                           MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), 
                           cudaMemcpyHostToDevice, stream);
            
            m_ignore_flags_need_update = false;
            
            if (ctx.config.verbose) {
                std::cout << "[Detector] Updated ignore flags - ";
                for (const auto& cs : ctx.config.class_settings) {
                    if (cs.ignore) {
                        std::cout << cs.name << "(ID:" << cs.id << ") ";
                    }
                }
                std::cout << "are ignored" << std::endl;
            }
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
            if (inferenceCV.wait_for(lock, std::chrono::milliseconds(10), [this] { return frameReady || AppContext::getInstance().should_exit; }))
            {
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

                // Execute preprocessing
                preProcess(frameGpu, preprocessStream);
                cudaEventRecord(m_preprocessDone, preprocessStream);
                
                // Wait for preprocessing to complete before inference
                cudaStreamWaitEvent(stream, m_preprocessDone, 0);
                
                // CUDA Graph 기반 추론 (첫 프레임은 Graph 캡처)
                if (!m_graphCaptured && frame_counter > 5) {
                    captureInferenceGraph();
                }
                
                bool enqueueSuccess;
                if (m_graphCaptured && m_inferenceGraphExec) {
                    // Graph 실행 (오버헤드 최소화)
                    cudaGraphLaunch(m_inferenceGraphExec, stream);
                    enqueueSuccess = true;
                } else {
                    // 일반 추론 (Graph 캡처 전)
                    enqueueSuccess = context->enqueueV3(stream);
                }
                
                if (!enqueueSuccess) {
                    std::cerr << "[Detector] TensorRT inference failed" << std::endl;
                    continue;
                }
                cudaEventRecord(m_inferenceDone, stream);
                
                // Start post-processing on a different stream
                cudaStreamWaitEvent(postprocessStream, m_inferenceDone, 0);
                performGpuPostProcessing(postprocessStream);
                cudaEventRecord(processingDone, postprocessStream);
                
                // GPU에서 detection 결과 복사 (간단한 방식)
                cudaMemcpyAsync(&m_finalDetectionsCountHost, m_finalDetectionsCountGpu.get(), 
                               sizeof(int), cudaMemcpyDeviceToHost, postprocessStream);
                
                // 먼저 count를 동기화해서 가져온 후 detection 데이터 복사
                cudaStreamSynchronize(postprocessStream);
                
                // 검출된 객체가 있으면 detection 데이터도 복사
                if (m_finalDetectionsCountHost > 0) {
                    cudaMemcpyAsync(m_finalDetectionsHost.get(), m_finalDetectionsGpu.get(), 
                                   m_finalDetectionsCountHost * sizeof(Detection), 
                                   cudaMemcpyDeviceToHost, postprocessStream);
                    cudaStreamSynchronize(postprocessStream);
                }

                auto inference_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float, std::milli> inference_duration_ms = inference_end_time - inference_start_time;
                ctx.g_current_inference_time_ms.store(inference_duration_ms.count());
                ctx.add_to_history(ctx.g_inference_time_history, inference_duration_ms.count(), ctx.g_inference_history_mutex);
                
                m_lastDetectionTime = std::chrono::steady_clock::now();
                
                // CPU에서 단순한 거리 기반 타겟 선택
                {
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    
                    // Always clear state first
                    m_hasBestTarget = false;
                    memset(&m_bestTargetHost, 0, sizeof(Detection));
                    m_bestTargetIndexHost = -1;
                    
                    // 검출된 객체가 있으면 가장 가까운 타겟 선택
                    if (m_finalDetectionsCountHost > 0 && !ctx.should_exit) {
                        float crosshairX = cached_config.detection_resolution / 2.0f + cached_config.crosshair_offset_x;
                        float crosshairY = cached_config.detection_resolution / 2.0f + cached_config.crosshair_offset_y;
                        
                        int closestIndex = findClosestTargetSimple(
                            m_finalDetectionsHost.get(),
                            m_finalDetectionsCountHost,
                            crosshairX,
                            crosshairY
                        );
                        
                        if (closestIndex >= 0) {
                            m_hasBestTarget = true;
                            m_bestTargetHost = m_finalDetectionsHost.get()[closestIndex];
                            m_bestTargetIndexHost = closestIndex;
                            
                            if (ctx.config.verbose) {
                                std::cout << "[Detector] Target selected - Index: " << m_bestTargetIndexHost 
                                          << ", Position: (" << m_bestTargetHost.x << ", " << m_bestTargetHost.y << ")"
                                          << ", Size: " << m_bestTargetHost.width << "x" << m_bestTargetHost.height
                                          << ", Confidence: " << m_bestTargetHost.confidence << std::endl;
                            }
                        }
                    }
                    
                    detectionVersion++;
                }
                detectionCV.notify_one();

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

    // Local config variables to reduce mutex locks
    int local_max_detections;
    float local_nms_threshold;
    float local_confidence_threshold;
    std::string local_postprocess;

    // Get color mask info
    const unsigned char* colorMaskPtr = nullptr;
    int maskPitch = 0;
    int current_min_color_pixels_val;
    bool current_remove_color_matches_val;
    int current_max_output_detections_val;

    { 
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        local_max_detections = ctx.config.max_detections;
        local_nms_threshold = ctx.config.nms_threshold;
        local_confidence_threshold = ctx.config.confidence_threshold;
        local_postprocess = ctx.config.postprocess;

        if (ctx.config.enable_color_filter) { 
            std::lock_guard<std::mutex> color_lock(colorMaskMutex); 
            if (!m_colorMaskGpu.empty()) {
                colorMaskPtr = m_colorMaskGpu.data();
                maskPitch = static_cast<int>(m_colorMaskGpu.step());
            }
        }
        current_min_color_pixels_val = ctx.config.min_color_pixels;
        current_remove_color_matches_val = ctx.config.remove_color_matches;
        current_max_output_detections_val = local_max_detections; 
    }

    int maxDecodedDetections = local_max_detections * 2;

    // GPU decoding debug info removed - enable for debugging if needed

    
    
    if (local_postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? static_cast<int>(shape[1]) : 0;
        
        
        decodeErr = decodeYolo10Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            local_confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu.get(),
            m_decodedCountGpu.get(),
            max_candidates,
            maxDecodedDetections,
            stream);
    } else if (local_postprocess == "yolo8" || local_postprocess == "yolo9" || local_postprocess == "yolo11" || local_postprocess == "yolo12") {
        int max_candidates = (shape.size() > 2) ? static_cast<int>(shape[2]) : 0;
        
        
         decodeErr = decodeYolo11Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            local_confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu.get(),
            m_decodedCountGpu.get(),
            max_candidates,
            maxDecodedDetections,
            stream);
    } else {
        std::cerr << "[Detector] Unsupported post-processing type for GPU decoding: " << local_postprocess << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    if (decodeErr != cudaSuccess) {
        std::cerr << "[Detector] GPU decoding kernel launch/execution failed: " << cudaGetErrorString(decodeErr) << std::endl;
        std::cerr << "[Detector] CUDA Error Code: " << decodeErr << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    // Check decoded count
    int decodedCount = 0;
    cudaMemcpyAsync(&decodedCount, m_decodedCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // If no detections were decoded, clear final count and return early
    if (decodedCount == 0) {
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }
    
    // Process all detections without syncing - use max possible count
    cudaError_t filterErr = filterDetectionsByClassIdGpu(
        m_decodedDetectionsGpu.get(),
        maxDecodedDetections, // Use max possible instead of syncing
        m_classFilteredDetectionsGpu.get(),
        m_classFilteredCountGpu.get(),
        m_d_ignore_flags_gpu.get(),
        MAX_CLASSES_FOR_FILTERING,
        colorMaskPtr,
        maskPitch,
        current_min_color_pixels_val,
        current_remove_color_matches_val,
        current_max_output_detections_val,
        stream
    );
    if (!checkCudaError(filterErr, "filtering detections by class ID GPU")) {
         cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
         return;
    }

    // Check filtered count for debugging
    int filteredCount = 0;
    cudaMemcpyAsync(&filteredCount, m_classFilteredCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Run NMS with max detections to avoid sync
    try {
        // Get frame dimensions from config
        int frame_width = ctx.config.detection_resolution;
        int frame_height = ctx.config.detection_resolution;
        
        NMSGpu(
            m_classFilteredDetectionsGpu.get(),
            local_max_detections, // Use max instead of actual count
            m_finalDetectionsGpu.get(),       
            m_finalDetectionsCountGpu.get(),  
            static_cast<int>(local_max_detections), 
            local_nms_threshold,
            frame_width,
            frame_height,
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

        // Convert to float and split channels
        CudaFloatProcessing::convertToFloat(resizedBuffer, floatBuffer, 1.0f / 255.0f, 0.0f, stream);
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
    m_finalDetectionsGpu.allocate(ctx.config.max_detections);
    m_finalDetectionsCountGpu.allocate(1);
    m_finalDetectionsHost = std::make_unique<Detection[]>(ctx.config.max_detections);
    m_classFilteredDetectionsGpu.allocate(ctx.config.max_detections);
    m_classFilteredCountGpu.allocate(1);
    m_scoresGpu.allocate(ctx.config.max_detections);

    m_matchingIndexGpu.allocate(1);
    m_matchingScoreGpu.allocate(1);
    
    // Allocate temporary buffers for multi-block reduction
    // Maximum number of blocks we might need
    const int max_blocks = (ctx.config.max_detections + 255) / 256;
    m_tempBlockScores.allocate(max_blocks);
    m_tempBlockIndices.allocate(max_blocks);

    m_d_ignore_flags_gpu.allocate(MAX_CLASSES_FOR_FILTERING);

    if (m_decodedCountGpu.get()) cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    if (m_finalDetectionsCountGpu.get()) cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
    if (m_classFilteredCountGpu.get()) cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);

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

void Detector::preallocateGPUResources()
{
    // GPU 메모리 풀 생성 (100MB)
    if (!m_gpuMemoryPool) {
        m_gpuMemoryPool = std::make_unique<CudaMemoryPool>(100 * 1024 * 1024);
        std::cout << "[Detector] GPU memory pool allocated: 100MB" << std::endl;
    }
    
    // Pinned 메모리 추가 할당 (빠른 CPU-GPU 전송용)
    void* pinnedBuffer = nullptr;
    size_t pinnedSize = 10 * 1024 * 1024; // 10MB
    cudaHostAlloc(&pinnedBuffer, pinnedSize, cudaHostAllocDefault);
    cudaFreeHost(pinnedBuffer); // 일단 할당 후 해제 (시스템이 재사용 가능한 상태로 유지)
}

void Detector::captureInferenceGraph()
{
    if (m_graphCaptured || !context) return;
    
    std::cout << "[Detector] Capturing CUDA Graph for inference pipeline..." << std::endl;
    
    // Graph 캡처 시작
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // 추론 파이프라인 기록
    context->enqueueV3(stream);
    
    // Graph 캡처 종료
    cudaStreamEndCapture(stream, &m_inferenceGraph);
    
    // Graph 인스턴스 생성
    cudaGraphInstantiate(&m_inferenceGraphExec, m_inferenceGraph, nullptr, nullptr, 0);
    
    m_graphCaptured = true;
    std::cout << "[Detector] CUDA Graph captured successfully" << std::endl;
}

void Detector::start()
{
    auto& ctx = AppContext::getInstance();
    ctx.should_exit = false;
    
    // GPU 리소스 사전 할당
    preallocateGPUResources();
    
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
