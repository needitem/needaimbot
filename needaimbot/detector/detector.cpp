#include "AppContext.h"

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>
#include <limits>
#include <cfloat>
#include <omp.h>
#include <float.h>

#include "detector.h"
#include "needaimbot.h"
#include "other_tools.h"

#include "postProcess.h"
#include "scoringGpu.h"
#include "filterGpu.h"
#include "config.h"

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




// extern Config config;  // Removed - use AppContext::getInstance().config instead
extern std::mutex configMutex; 

static bool error_logged = false;



extern cudaError_t filterDetectionsByClassIdGpu(
    const Detection* decodedDetections,
    int numDecodedDetections,
    Detection* filteredDetections,
    int* filteredCount,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    const unsigned char* d_hsv_mask,
    int mask_pitch,
    int min_hsv_pixels,
    bool remove_hsv_matches,
    int max_output_detections,
    cudaStream_t stream
);

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
    m_classFilteredCountHost(0),
    m_host_ignore_flags_uchar(MAX_CLASSES_FOR_FILTERING, 1), 
    m_ignore_flags_need_update(true) 
    , m_isTargetLocked(false) 
{
    // Initialize batched results
    cudaMalloc((void**)&m_batchedResultsGpu, sizeof(BatchedResults));
    memset(&m_batchedResultsHost, 0, sizeof(BatchedResults));
}

Detector::~Detector()
{
    if (m_batchedResultsGpu) {
        cudaFree(m_batchedResultsGpu);
        m_batchedResultsGpu = nullptr;
    }
    
    if (stream) cudaStreamDestroy(stream);
    if (preprocessStream) cudaStreamDestroy(preprocessStream);
    if (postprocessStream) cudaStreamDestroy(postprocessStream);
    if (m_preprocessStream) cudaStreamDestroy(m_preprocessStream);
    if (m_inferenceStream) cudaStreamDestroy(m_inferenceStream);
    if (m_postprocessStream) cudaStreamDestroy(m_postprocessStream);
    if (m_preprocessDone) cudaEventDestroy(m_preprocessDone);
    if (m_inferenceDone) cudaEventDestroy(m_inferenceDone);
    if (m_inferenceDone2) cudaEventDestroy(m_inferenceDone2);
    if (m_postprocessDone) cudaEventDestroy(m_postprocessDone);
    if (processingDone) cudaEventDestroy(processingDone);
    if (postprocessCopyDone) cudaEventDestroy(postprocessCopyDone);
    

    for (auto& binding : inputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }

    for (auto& binding : outputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }

    if (inputBufferDevice)
    {
        cudaFree(inputBufferDevice);
    }

    if (m_computeStream) {
        cudaStreamDestroy(m_computeStream);
        m_computeStream = nullptr;
    }
    if (m_memoryStream) {
        cudaStreamDestroy(m_memoryStream);
        m_memoryStream = nullptr;
    }
    if (m_preprocessDone) {
        cudaEventDestroy(m_preprocessDone);
        m_preprocessDone = nullptr;
    }
    if (m_inferenceDone) {
        cudaEventDestroy(m_inferenceDone);
        m_inferenceDone = nullptr;
    }
    
    // CUDA Graph removed for optimization
    
    // Synchronize to ensure all CUDA operations are complete
    cudaStreamSynchronize(stream);
    
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

    if (m_bestTargetIndexGpu.get()) cudaMemsetAsync(m_bestTargetIndexGpu.get(), 0xFF, sizeof(int), stream);

    
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

    // Create multiple streams for pipeline optimization
    if (!checkCudaError(cudaStreamCreate(&stream), "creating main stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&preprocessStream), "creating preprocess stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&postprocessStream), "creating postprocess stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&m_preprocessStream), "creating preprocess2 stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&m_inferenceStream), "creating inference2 stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&m_postprocessStream), "creating postprocess2 stream")) { m_cudaContextInitialized = false; return false; }

    
    cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    preprocessCvStream = cv::cuda::StreamAccessor::wrapStream(preprocessStream);
    postprocessCvStream = cv::cuda::StreamAccessor::wrapStream(postprocessStream);
    
    
    if (!checkCudaError(cudaEventCreateWithFlags(&m_preprocessDone, cudaEventDisableTiming), "creating preprocessDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&m_inferenceDone, cudaEventDisableTiming), "creating inferenceDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&processingDone, cudaEventDisableTiming), "creating processingDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&postprocessCopyDone, cudaEventDisableTiming), "creating postprocessCopyDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&m_inferenceDone2, cudaEventDisableTiming), "creating inferenceDone2 event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&m_postprocessDone, cudaEventDisableTiming), "creating postprocessDone event")) { m_cudaContextInitialized = false; return false; }
    
    // Pre-allocate common buffer sizes in memory pool
    std::vector<size_t> commonSizes = {
        1920 * 1080 * 3,           // Full HD image
        640 * 640 * 3,             // Common detection size
        512 * 512 * 3,             // Another common size
        sizeof(Detection) * 1000,   // Detection buffer
        sizeof(float) * 1000       // Score buffer
    };
    getGpuMemoryPool().preallocate(commonSizes, stream);

    

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
            numClasses = outDims.d[1] - 5; // Corrected: 4 for bbox, 1 for confidence
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
    int c = actualInputDims.d[1];
    int h = actualInputDims.d[2];
    int w = actualInputDims.d[3];
    
    if (resizedBuffer.empty() || resizedBuffer.size() != cv::Size(w, h)) {
        resizedBuffer.create(h, w, CV_8UC3);
    }
    
    if (floatBuffer.empty() || floatBuffer.size() != cv::Size(w, h)) {
        floatBuffer.create(h, w, CV_32FC3);
    }
    
    channelBuffers.resize(c);
    for (int i = 0; i < c; ++i)
    {
        if (channelBuffers[i].empty() || channelBuffers[i].size() != cv::Size(w, h)) {
            channelBuffers[i].create(h, w, CV_32F);
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

void Detector::processFrame(const cv::cuda::GpuMat& frame)
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
    currentFrame = frame;
    frameIsGpu = true;
    frameReady = true;
    inferenceCV.notify_one();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    ctx.g_current_process_frame_time_ms.store(duration.count());
    ctx.add_to_history(ctx.g_process_frame_time_history, duration.count(), ctx.g_process_frame_history_mutex);
}

void Detector::processFrame(const cv::Mat& frame)
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
    currentFrameCpu = frame.clone();
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
        bool enable_hsv_filter;
        int min_hsv_pixels_required;
        bool hsv_filter_remove_on_match;
        
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
            enable_hsv_filter = config.enable_hsv_filter;
            min_hsv_pixels_required = config.min_hsv_pixels;
            hsv_filter_remove_on_match = config.remove_hsv_matches;
        }
    } cached_config;
    
    // Initial config cache
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        cached_config.update(ctx.config);
    }

    cv::cuda::GpuMat frameGpu;
    static auto last_cycle_start_time = std::chrono::high_resolution_clock::time_point{};

    while (!ctx.shouldExit)
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

        if (ctx.shouldExit) {
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
            if (inferenceCV.wait_for(lock, std::chrono::milliseconds(10), [this] { return frameReady || AppContext::getInstance().shouldExit; }))
            {
                if (AppContext::getInstance().shouldExit) break;
                if (frameReady) {
                    if (frameIsGpu) {
                        frameGpu = std::move(currentFrame);
                    } else {
                        frameGpu.upload(currentFrameCpu);
                    }
                    frameReady = false;
                    hasNewFrame = true;
                }
            }
        }

        if (!context) {
            if (!error_logged) {
                std::cerr << "[Detector] Context not initialized" << std::endl;
                error_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        } else {
            error_logged = false;
        }

        if (hasNewFrame && !frameGpu.empty())
        {
            try
            {
                auto inference_start_time = std::chrono::high_resolution_clock::now();

                // Execute preprocessing
                preProcess(frameGpu, preprocessStream);
                cudaEventRecord(m_preprocessDone, preprocessStream);
                
                // Wait for preprocessing to complete before inference
                cudaStreamWaitEvent(stream, m_preprocessDone, 0);
                
                // Direct inference without CUDA Graph to avoid stream conflicts
                bool enqueueSuccess = context->enqueueV3(stream);
                if (!enqueueSuccess) {
                    std::cerr << "[Detector] TensorRT enqueueV3 failed" << std::endl;
                    continue;
                }
                cudaEventRecord(m_inferenceDone, stream);
                
                // Start post-processing on a different stream
                cudaStreamWaitEvent(postprocessStream, m_inferenceDone, 0);
                performGpuPostProcessing(postprocessStream);
                cudaEventRecord(processingDone, postprocessStream);

                auto inference_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float, std::milli> inference_duration_ms = inference_end_time - inference_start_time;
                ctx.g_current_inference_time_ms.store(inference_duration_ms.count());
                ctx.add_to_history(ctx.g_inference_time_history, inference_duration_ms.count(), ctx.g_inference_history_mutex);
                

                // Post-graph processing (memory copies and target selection)
                // Wait for postprocessing to complete before copying
                cudaStreamWaitEvent(stream, processingDone, 0);
                
                // Prepare batched results on GPU by copying individual elements
                // Since we can't launch a kernel from a .cpp file, we'll copy the data separately
                cudaMemcpyAsync(&m_batchedResultsHost.finalCount, m_finalDetectionsCountGpu.get(), 
                               sizeof(int), cudaMemcpyDeviceToHost, stream);
                
                // Copy best index and score if valid
                int bestIndexTemp = -1;
                cudaMemcpyAsync(&bestIndexTemp, m_bestTargetIndexGpu.get(), 
                               sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                m_batchedResultsHost.bestIndex = bestIndexTemp;
                if (bestIndexTemp >= 0 && bestIndexTemp < m_finalDetectionsCountHost) {
                    cudaMemcpyAsync(&m_batchedResultsHost.bestScore, &m_scoresGpu.get()[bestIndexTemp], 
                                   sizeof(float), cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(&m_batchedResultsHost.bestTarget, &m_finalDetectionsGpu.get()[bestIndexTemp], 
                                   sizeof(Detection), cudaMemcpyDeviceToHost, stream);
                } else {
                    m_batchedResultsHost.bestScore = -1.0f;
                }
                
                // Copy matching index and score if valid
                int matchingIndexTemp = -1;
                cudaMemcpyAsync(&matchingIndexTemp, m_matchingIndexGpu.get(), 
                               sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                m_batchedResultsHost.matchingIndex = matchingIndexTemp;
                if (matchingIndexTemp >= 0 && matchingIndexTemp < m_finalDetectionsCountHost) {
                    cudaMemcpyAsync(&m_batchedResultsHost.matchingScore, &m_scoresGpu.get()[matchingIndexTemp], 
                                   sizeof(float), cudaMemcpyDeviceToHost, stream);
                } else {
                    m_batchedResultsHost.matchingScore = -1.0f;
                }
                
                // Synchronize to ensure results are available
                cudaStreamSynchronize(stream);
                
                // Extract results from batched structure
                m_finalDetectionsCountHost = m_batchedResultsHost.finalCount;
                m_lastDetectionTime = std::chrono::steady_clock::now();
                
                // Target handling logic with proper reset
                {
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    
                    // Only proceed if we have detections
                    if (m_finalDetectionsCountHost > 0 && !ctx.shouldExit)
                    {
                        // First, calculate scores for all detections using cached config
                        calculateTargetScoresGpu(m_finalDetectionsGpu.get(), m_finalDetectionsCountHost, m_scoresGpu.get(), 
                            cached_config.detection_resolution, cached_config.detection_resolution, 
                            cached_config.distance_weight, cached_config.confidence_weight, m_headClassId, 
                            cached_config.crosshair_offset_x, cached_config.crosshair_offset_y, stream);
                        
                        // Find the overall best target
                        findBestTargetGpu(m_scoresGpu.get(), m_finalDetectionsCountHost, m_bestTargetIndexGpu.get(), stream);
                        
                        int newBestIndex = -1;
                        float newBestScore = FLT_MAX;
                        
                        // If we have a previous target, try to match it in the new detections
                        if (m_hasBestTarget) {
                            // Find the detection that best matches our previous target
                            findMatchingTargetGpu(m_finalDetectionsGpu.get(), m_finalDetectionsCountHost, 
                                m_bestTargetHost, m_matchingIndexGpu.get(), m_matchingScoreGpu.get(), stream);
                        }
                        
                        // Wait for all target processing to complete
                        cudaStreamSynchronize(stream);
                        
                        // Use batched results
                        newBestIndex = m_batchedResultsHost.bestIndex;
                        
                        if (m_hasBestTarget && newBestIndex >= 0) {
                            int matchingIndex = m_batchedResultsHost.matchingIndex;
                            
                            // If we found a match and it's valid
                            if (matchingIndex >= 0 && matchingIndex < m_finalDetectionsCountHost && 
                                newBestIndex >= 0 && newBestIndex < m_finalDetectionsCountHost) {
                                
                                float matchingScore = m_batchedResultsHost.matchingScore;
                                float bestScore = m_batchedResultsHost.bestScore;
                                
                                // Apply sticky target logic
                                // Only switch if the new target is significantly better
                                float threshold = 1.0f - cached_config.sticky_target_threshold; // Convert to "how much better" metric
                                if (bestScore < matchingScore * threshold) {
                                    // New target is significantly better, switch to it
                                    m_bestTargetIndexHost = newBestIndex;
                                    newBestScore = bestScore;
                                    m_bestTargetHost = m_batchedResultsHost.bestTarget;
                                } else {
                                    // Stick with the matched target
                                    m_bestTargetIndexHost = matchingIndex;
                                    newBestScore = matchingScore;
                                    // Need to copy the matching target separately
                                    cudaMemcpy(&m_bestTargetHost, &m_finalDetectionsGpu.get()[matchingIndex], 
                                             sizeof(Detection), cudaMemcpyDeviceToHost);
                                }
                            } else if (newBestIndex >= 0 && newBestIndex < m_finalDetectionsCountHost) {
                                // No match found, use the best target
                                m_bestTargetIndexHost = newBestIndex;
                                newBestScore = m_batchedResultsHost.bestScore;
                                m_bestTargetHost = m_batchedResultsHost.bestTarget;
                            } else {
                                // No valid targets at all
                                m_bestTargetIndexHost = -1;
                            }
                        } else if (newBestIndex >= 0 && newBestIndex < m_finalDetectionsCountHost) {
                            // No previous target, just use the best one
                            m_bestTargetIndexHost = newBestIndex;
                            newBestScore = m_batchedResultsHost.bestScore;
                            m_bestTargetHost = m_batchedResultsHost.bestTarget;
                        }
                        
                        // Update target tracking state
                        if (m_bestTargetIndexHost >= 0 && m_bestTargetIndexHost < m_finalDetectionsCountHost) {
                            m_hasBestTarget = true;
                        } else {
                            // No valid target found - clear immediately
                            m_hasBestTarget = false;
                            memset(&m_bestTargetHost, 0, sizeof(Detection));
                            m_bestTargetIndexHost = -1;
                        }
                    }
                    else if (m_finalDetectionsCountHost == 0) {
                        // No detections at all - clear immediately
                        m_hasBestTarget = false;
                        memset(&m_bestTargetHost, 0, sizeof(Detection));
                        m_bestTargetIndexHost = -1;
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

    const std::string& primaryOutputName = outputNames[0];
    void* d_rawOutputPtr = outputBindings[primaryOutputName];
    nvinfer1::DataType outputType = outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = outputShapes[primaryOutputName];

    if (!d_rawOutputPtr) {
        std::cerr << "[Detector] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    cudaError_t decodeErr = cudaSuccess;

    // Local config variables to reduce mutex locks
    int local_max_detections;
    float local_nms_threshold;
    float local_confidence_threshold;
    std::string local_postprocess;

    // Get HSV mask info
    const unsigned char* hsvMaskPtr = nullptr;
    int maskPitch = 0;
    int current_min_hsv_pixels_val;
    bool current_remove_hsv_matches_val;
    int current_max_output_detections_val;

    { 
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        local_max_detections = ctx.config.max_detections;
        local_nms_threshold = ctx.config.nms_threshold;
        local_confidence_threshold = ctx.config.confidence_threshold;
        local_postprocess = ctx.config.postprocess;

        if (ctx.config.enable_hsv_filter) { 
            std::lock_guard<std::mutex> hsv_lock(hsvMaskMutex); 
            if (!m_hsvMaskGpu.empty()) {
                hsvMaskPtr = m_hsvMaskGpu.ptr<unsigned char>();
                maskPitch = static_cast<int>(m_hsvMaskGpu.step);
            }
        }
        current_min_hsv_pixels_val = ctx.config.min_hsv_pixels;
        current_remove_hsv_matches_val = ctx.config.remove_hsv_matches;
        current_max_output_detections_val = local_max_detections; 
    }

    int maxDecodedDetections = local_max_detections * 2;

    // GPU decoding debug info removed - enable for debugging if needed

    if (local_postprocess == "yolo10") {
        int max_candidates = (shape.size() > 1) ? shape[1] : 0;
        
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
        int max_candidates = (shape.size() > 2) ? shape[2] : 0;
        
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
        hsvMaskPtr,
        maskPitch,
        current_min_hsv_pixels_val,
        current_remove_hsv_matches_val,
        current_max_output_detections_val,
        stream
    );
    if (!checkCudaError(filterErr, "filtering detections by class ID GPU")) {
         cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
         return;
    }

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
    } catch (const std::exception& e) {
         std::cerr << "[Detector] Exception during NMSGpu call: " << e.what() << std::endl;
         cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
    }
}

void Detector::preProcess(const cv::cuda::GpuMat& frame, cudaStream_t stream)
{
    auto& ctx = AppContext::getInstance();
    if (frame.empty()) return;

    // This function will now use the stream passed for graph capture
    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];

    try
    {
        static cv::cuda::GpuMat maskGpu_static;
        static cv::cuda::GpuMat maskedImageGpu_static;

        if (ctx.config.circle_mask) {
            if (maskGpu_static.empty() || maskGpu_static.size() != frame.size()) {
                cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::Point center(mask.cols / 2, mask.rows / 2);
                int radius = std::min(mask.cols, mask.rows) / 2;
                cv::circle(mask, center, radius, cv::Scalar(255), -1);
                maskGpu_static.upload(mask, cvStream);
            }
            maskedImageGpu_static.create(frame.size(), frame.type());
            maskedImageGpu_static.setTo(cv::Scalar::all(0), cvStream);
            frame.copyTo(maskedImageGpu_static, maskGpu_static, cvStream);
            cv::cuda::resize(maskedImageGpu_static, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, cvStream);
        } else {
            cv::cuda::resize(frame, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, cvStream);
        }

        // Only process HSV filter if enabled and actually being used
        bool current_enable_hsv_filter = false;
        bool current_remove_hsv_matches = false;
        {
            std::lock_guard<std::mutex> lock(ctx.configMutex);
            current_enable_hsv_filter = ctx.config.enable_hsv_filter;
            current_remove_hsv_matches = ctx.config.remove_hsv_matches;
        }

        // Skip HSV processing entirely if not needed
        if (current_enable_hsv_filter && (current_remove_hsv_matches || ctx.config.min_hsv_pixels > 0)) {
            static cv::cuda::GpuMat hsvGpu_static;
            static cv::cuda::GpuMat maskGpu_hsv_static;
            
            // Cache HSV bounds to avoid repeated access
            static cv::Scalar cached_lower, cached_upper;
            static bool bounds_cached = false;
            
            if (!bounds_cached || ctx.config.hsv_lower_h != cached_lower[0]) {
                cached_lower = cv::Scalar(ctx.config.hsv_lower_h, ctx.config.hsv_lower_s, ctx.config.hsv_lower_v);
                cached_upper = cv::Scalar(ctx.config.hsv_upper_h, ctx.config.hsv_upper_s, ctx.config.hsv_upper_v);
                bounds_cached = true;
            }
            
            cv::cuda::cvtColor(resizedBuffer, hsvGpu_static, cv::COLOR_BGR2HSV, 0, cvStream);
            cv::cuda::inRange(hsvGpu_static, cached_lower, cached_upper, maskGpu_hsv_static, cvStream);
            
            // Only resize if resolution changed
            static int last_resolution = 0;
            if (last_resolution != ctx.config.detection_resolution) {
                last_resolution = ctx.config.detection_resolution;
                cv::cuda::GpuMat maskResized;
                cv::cuda::resize(maskGpu_hsv_static, maskResized, cv::Size(last_resolution, last_resolution), 0, 0, cv::INTER_NEAREST, cvStream);
                {
                    std::lock_guard<std::mutex> lock(hsvMaskMutex);
                    m_hsvMaskGpu = maskResized;
                }
            }
        } else {
            // Clear HSV mask if not in use
            std::lock_guard<std::mutex> lock(hsvMaskMutex);
            if (!m_hsvMaskGpu.empty()) {
                m_hsvMaskGpu.release();
            }
        }

        resizedBuffer.convertTo(floatBuffer, CV_32F, 1.0f / 255.0f, 0, cvStream);
        cv::cuda::split(floatBuffer, channelBuffers, cvStream);

        size_t channelSize = h * w * sizeof(float);
        for (int i = 0; i < c; ++i)
        {
            cudaMemcpyAsync(
                static_cast<float*>(inputBuffer) + i * h * w,
                channelBuffers[i].ptr<float>(),
                channelSize,
                cudaMemcpyDeviceToDevice,
                stream
            );
        }
    } catch (const cv::Exception& e)
    {
        std::cerr << "[Detector] OpenCV error in preProcess: " << e.what() << std::endl;
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
    m_classFilteredDetectionsGpu.allocate(ctx.config.max_detections);
    m_classFilteredCountGpu.allocate(1);
    m_scoresGpu.allocate(ctx.config.max_detections);
    m_bestTargetIndexGpu.allocate(1);
    m_matchingIndexGpu.allocate(1);
    m_matchingScoreGpu.allocate(1);

    m_d_ignore_flags_gpu.allocate(MAX_CLASSES_FOR_FILTERING);

    if (m_decodedCountGpu.get()) cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    if (m_finalDetectionsCountGpu.get()) cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
    if (m_classFilteredCountGpu.get()) cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    if (m_bestTargetIndexGpu.get()) cudaMemsetAsync(m_bestTargetIndexGpu.get(), 0xFF, sizeof(int), stream);
}


float Detector::calculate_host_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = std::max(box1.x, box2.x);
    int yA = std::max(box1.y, box2.y);
    int xB = std::min(box1.x + box1.width, box2.x + box2.width);
    int yB = std::min(box1.y + box1.height, box2.y + box2.height);

    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);

    
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    
    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}











cv::cuda::GpuMat Detector::getHsvMaskGpu() const { 
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(hsvMaskMutex)); 
    if (m_hsvMaskGpu.empty()) {
        return cv::cuda::GpuMat(); 
    }
    return m_hsvMaskGpu.clone(); 
}

void Detector::start()
{
    auto& ctx = AppContext::getInstance();
    ctx.shouldExit = false;
    // Note: capture thread is handled separately in capture.cpp
    m_inferenceThread = std::thread(&Detector::inferenceThread, this);
}

void Detector::stop()
{
    auto& ctx = AppContext::getInstance();
    ctx.shouldExit = true;
    
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

    if (ctx.config.tensorrt_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TensorRT] FP16 optimization enabled" << std::endl;
    }

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
