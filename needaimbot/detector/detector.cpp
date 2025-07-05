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

#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>
#include <limits>
#include <omp.h>

#include "detector.h"
#include "nvinf.h"
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




extern Config config; 
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
    m_computeStream(nullptr),
    m_memoryStream(nullptr),
    m_bestTargetIndexHost(-1),
    m_finalDetectionsCountHost(0),
    m_classFilteredCountHost(0),
    m_host_ignore_flags_uchar(MAX_CLASSES_FOR_FILTERING, 1), 
    m_ignore_flags_need_update(true) 
    , m_isTargetLocked(false) 
    , m_lockedTargetLostFrames(0) 
{
    
}

Detector::~Detector()
{
    
    if (stream) cudaStreamDestroy(stream);
    if (preprocessStream) cudaStreamDestroy(preprocessStream);
    if (postprocessStream) cudaStreamDestroy(postprocessStream);
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

    
    if (!checkCudaError(cudaStreamCreate(&stream), "creating main stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&preprocessStream), "creating preprocess stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&postprocessStream), "creating postprocess stream")) { m_cudaContextInitialized = false; return false; }

    
    cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    preprocessCvStream = cv::cuda::StreamAccessor::wrapStream(preprocessStream);
    postprocessCvStream = cv::cuda::StreamAccessor::wrapStream(postprocessStream);
    
    
    if (!checkCudaError(cudaEventCreateWithFlags(&processingDone, cudaEventDisableTiming), "creating processingDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&postprocessCopyDone, cudaEventDisableTiming), "creating postprocessCopyDone event")) { m_cudaContextInitialized = false; return false; }

    

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

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
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
            numClasses = outDims.d[1] - 4;
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

            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile, gLogger);
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
    engine.reset(loadEngineFromFile(engineFilePath, runtime.get()));
}

void Detector::processFrame(const cv::cuda::GpuMat& frame)
{
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) return; 

    if (ctx.detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
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
    add_to_history(ctx.g_process_frame_time_history, duration.count(), ctx.g_process_frame_history_mutex);
}

void Detector::processFrame(const cv::Mat& frame)
{
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) return; 

    if (ctx.detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
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
    add_to_history(ctx.g_process_frame_time_history, duration.count(), ctx.g_process_frame_history_mutex);
}

void Detector::inferenceThread()
{
    auto& ctx = AppContext::getInstance();
    if (!isCudaContextInitialized()) {
        std::cerr << "[Detector Thread] CUDA context not initialized. Inference thread exiting." << std::endl;
        return;
    }

    cv::cuda::GpuMat frameGpu;
    static auto last_inference_loop_start_time = std::chrono::high_resolution_clock::time_point{};

    while (!ctx.shouldExit)
    {
        NVTX_PUSH("Detector Inference Loop");

        if (ctx.detector_model_changed.load()) {
            // Reset graph on model change
            if (m_graphExec) {
                cudaGraphExecDestroy(m_graphExec);
                m_graphExec = nullptr;
            }
            if (m_graph) {
                cudaGraphDestroy(m_graph);
                m_graph = nullptr;
            }
            m_isGraphInitialized = false;

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
            if (inferenceCV.wait_for(lock, std::chrono::milliseconds(100), [this] { return frameReady || AppContext::getInstance().shouldExit; }))
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
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        } else {
            error_logged = false;
        }

        if (hasNewFrame && !frameGpu.empty())
        {
            auto current_inference_loop_start_time = std::chrono::high_resolution_clock::now();
            if (last_inference_loop_start_time.time_since_epoch().count() != 0) {
                std::chrono::duration<float, std::milli> cycle_duration_ms = current_inference_loop_start_time - last_inference_loop_start_time;
                ctx.g_current_detector_cycle_time_ms.store(cycle_duration_ms.count());
                add_to_history(ctx.g_detector_cycle_time_history, cycle_duration_ms.count(), ctx.g_detector_cycle_history_mutex);
            }
            last_inference_loop_start_time = current_inference_loop_start_time;

            try
            {
                auto inference_start_time = std::chrono::high_resolution_clock::now();

                // Capture graph on first run or if not initialized
                if (!m_isGraphInitialized)
                {
                    // Use a dedicated stream for capture to avoid interference
                    cudaStream_t captureStream;
                    cudaStreamCreate(&captureStream);

                    cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

                    preProcess(frameGpu, captureStream);
                    context->enqueueV3(captureStream);
                    performGpuPostProcessing(captureStream);
                    
                    cudaStreamEndCapture(captureStream, &m_graph);
                    
                    cudaGraphInstantiate(&m_graphExec, m_graph, NULL, NULL, 0);
                    m_isGraphInitialized = true;

                    cudaStreamDestroy(captureStream);
                }

                // Launch the graph
                if (m_graphExec) {
                    cudaGraphLaunch(m_graphExec, stream);
                } else {
                    // Fallback or error
                    std::cerr << "[Detector] CUDA Graph execution failed: graph not initialized." << std::endl;
                }
                
                // Synchronize the main stream to wait for graph completion
                cudaStreamSynchronize(stream);

                auto inference_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float, std::milli> inference_duration_ms = inference_end_time - inference_start_time;
                ctx.g_current_inference_time_ms.store(inference_duration_ms.count());
                add_to_history(ctx.g_inference_time_history, inference_duration_ms.count(), ctx.g_inference_history_mutex);

                // Post-graph processing (memory copies and logic)
                int final_detections_count = 0;
                cudaMemcpy(&final_detections_count, m_finalDetectionsCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost);

                if (final_detections_count > 0)
                {
                    // These operations are fast and can run on the default stream after the graph
                    calculateTargetScoresGpu(m_finalDetectionsGpu.get(), final_detections_count, m_scoresGpu.get(), ctx.config.detection_resolution, ctx.config.detection_resolution, ctx.config.distance_weight, ctx.config.confidence_weight, m_headClassId, stream);
                    findBestTargetGpu(m_scoresGpu.get(), final_detections_count, m_bestTargetIndexGpu.get(), stream);
                    
                    cudaStreamSynchronize(stream); // Wait for scores and index

                    cudaMemcpy(&m_bestTargetIndexHost, m_bestTargetIndexGpu.get(), sizeof(int), cudaMemcpyDeviceToHost);
                    
                    if (m_bestTargetIndexHost >= 0 && m_bestTargetIndexHost < final_detections_count) {
                        cudaMemcpy(&m_bestTargetHost, &m_finalDetectionsGpu.get()[m_bestTargetIndexHost], sizeof(Detection), cudaMemcpyDeviceToHost);
                        m_hasBestTarget = true;
                    } else {
                        m_hasBestTarget = false;
                    }
                }
                else
                {
                    m_hasBestTarget = false;
                }

                {
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    detectionVersion++;
                }
                detectionCV.notify_one();
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Detector] Error during inference loop: " << e.what() << std::endl;
                m_hasBestTarget = false;
            }
        } else if (hasNewFrame) {
             {
                std::lock_guard<std::mutex> lock(detectionMutex);
                m_hasBestTarget = false;
                detectionVersion++;
             }
             detectionCV.notify_one();
        }
        NVTX_POP();
    }
}''

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
    { 
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        local_max_detections = ctx.config.max_detections;
        local_nms_threshold = ctx.config.nms_threshold;
        local_confidence_threshold = ctx.config.confidence_threshold;
        local_postprocess = ctx.config.postprocess;
    }

    int maxDecodedDetections = local_max_detections * 2;

    if (local_postprocess == "yolo10") {
        decodeErr = decodeYolo10Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            local_confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu.get(),
            m_decodedCountGpu.get(),
            maxDecodedDetections,
            stream);
    } else if (local_postprocess == "yolo8" || local_postprocess == "yolo9" || local_postprocess == "yolo11" || local_postprocess == "yolo12") {
         decodeErr = decodeYolo11Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            local_confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu.get(),
            m_decodedCountGpu.get(),
            maxDecodedDetections,
            stream);
    } else {
        std::cerr << "[Detector] Unsupported post-processing type for GPU decoding: " << local_postprocess << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    if (decodeErr != cudaSuccess) {
        std::cerr << "[Detector] GPU decoding kernel launch/execution failed: " << cudaGetErrorString(decodeErr) << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    
    int decodedCountHost = 0;
    cudaError_t decodeCountCopyErr = cudaMemcpyAsync(&decodedCountHost, m_decodedCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (decodeCountCopyErr != cudaSuccess) {
        std::cerr << "[Detector] Failed to copy decoded count DtoH: " << cudaGetErrorString(decodeCountCopyErr) << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    
    // Only sync when we need to read the count
    cudaStreamSynchronize(stream);
    
    int classFilteredCountHost = 0;
    if (decodedCountHost > 0) {
        int validDecodedDetections = std::min(decodedCountHost, static_cast<int>(local_max_detections * 2));
        if (validDecodedDetections > 0) {
            
            // Only update ignore flags when necessary
            if (m_ignore_flags_need_update) {
                { 
                    std::lock_guard<std::mutex> lock(ctx.configMutex);
                    
                    std::fill(m_host_ignore_flags_uchar.begin(), m_host_ignore_flags_uchar.end(), 1); 
                    for (const auto& class_setting : ctx.config.class_settings) {
                        if (class_setting.id >= 0 && class_setting.id < MAX_CLASSES_FOR_FILTERING) {
                            m_host_ignore_flags_uchar[class_setting.id] = static_cast<unsigned char>(class_setting.ignore);
                        }
                    }
                }

                if (m_d_ignore_flags_gpu.get()) {
                    cudaError_t copyErr = cudaMemcpyAsync(m_d_ignore_flags_gpu.get(), m_host_ignore_flags_uchar.data(), 
                                                          MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), 
                                                          cudaMemcpyHostToDevice, stream);
                    if (!checkCudaError(copyErr, "copying updated ignore flags to GPU")) {
                        cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
                        return;
                    }
                } else {
                    std::cerr << "[Detector] Ignore flags GPU buffer not allocated!" << std::endl;
                    cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
                    return;
                }
                m_ignore_flags_need_update = false; 
            }

            
            const unsigned char* hsvMaskPtr = nullptr;
            int maskPitch = 0;
            int current_min_hsv_pixels_val;
            bool current_remove_hsv_matches_val;
            int current_max_output_detections_val;

            { 
                std::lock_guard<std::mutex> lock(ctx.configMutex);
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
            
            
            cudaError_t filterErr = filterDetectionsByClassIdGpu(
                m_decodedDetectionsGpu.get(),
                validDecodedDetections,
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

            cudaError_t filteredCountCopyErr = cudaMemcpyAsync(&classFilteredCountHost, m_classFilteredCountGpu.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
            // Reduce sync points - only sync when necessary
            cudaStreamSynchronize(stream);
            if (filteredCountCopyErr != cudaSuccess) {
                std::cerr << "[Detector] Failed to copy filtered detection count DtoH: " << cudaGetErrorString(filteredCountCopyErr) << std::endl;
                classFilteredCountHost = 0;
            }
        } else {
            classFilteredCountHost = 0;
            cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
        }
    } else {
        classFilteredCountHost = 0;
        cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    }

    
    if (classFilteredCountHost > 0) {
        // Limit NMS input to reduce computation - only top 5 for performance
        int inputNmsCount = std::min(classFilteredCountHost, std::min(local_max_detections, 5)); 
        if (inputNmsCount > 0) {
            try {
                NMSGpu(
                    m_classFilteredDetectionsGpu.get(),
                    inputNmsCount,            
                    m_finalDetectionsGpu.get(),       
                    m_finalDetectionsCountGpu.get(),  
                    static_cast<int>(local_max_detections), 
                    local_nms_threshold,
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
        } else {
            cudaMemsetAsync(m_finalDetectionsCountGpu.get(), 0, sizeof(int), stream);
        }
    } else {
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
                cv::circle(mask, center, cv::Scalar(255), -1);
                maskGpu_static.upload(mask, cvStream);
            }
            maskedImageGpu_static.create(frame.size(), frame.type());
            maskedImageGpu_static.setTo(cv::Scalar::all(0), cvStream);
            frame.copyTo(maskedImageGpu_static, maskGpu_static, cvStream);
            cv::cuda::resize(maskedImageGpu_static, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, cvStream);
        } else {
            cv::cuda::resize(frame, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, cvStream);
        }

        bool current_enable_hsv_filter;
        // HSV config values read once, assuming they don't change during graph execution
        {
            std::lock_guard<std::mutex> lock(ctx.configMutex);
            current_enable_hsv_filter = ctx.config.enable_hsv_filter;
        }

        if (current_enable_hsv_filter) {
            static cv::cuda::GpuMat hsvGpu_static;
            static cv::cuda::GpuMat maskGpu_hsv_static;
            cv::cuda::cvtColor(resizedBuffer, hsvGpu_static, cv::COLOR_BGR2HSV, 0, cvStream);
            cv::Scalar lower(ctx.config.hsv_lower_h, ctx.config.hsv_lower_s, ctx.config.hsv_lower_v);
            cv::Scalar upper(ctx.config.hsv_upper_h, ctx.config.hsv_upper_s, ctx.config.hsv_upper_v);
            cv::cuda::inRange(hsvGpu_static, lower, upper, maskGpu_hsv_static, cvStream);
            
            cv::cuda::GpuMat maskResized;
            cv::cuda::resize(maskGpu_hsv_static, maskResized, cv::Size(ctx.config.detection_resolution, ctx.config.detection_resolution), 0, 0, cv::INTER_NEAREST, cvStream);
            {
                std::lock_guard<std::mutex> lock(hsvMaskMutex);
                m_hsvMaskGpu = maskResized;
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
    
    m_decodedDetectionsGpu.allocate(ctx.config.max_detections * 2);
    m_decodedCountGpu.allocate(1);
    m_finalDetectionsGpu.allocate(ctx.config.max_detections);
    m_finalDetectionsCountGpu.allocate(1);
    m_classFilteredDetectionsGpu.allocate(ctx.config.max_detections);
    m_classFilteredCountGpu.allocate(1);
    m_scoresGpu.allocate(ctx.config.max_detections);
    m_bestTargetIndexGpu.allocate(1);

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
    m_captureThread = std::thread(captureThread, ctx.config.detection_resolution, ctx.config.detection_resolution);
    m_inferenceThread = std::thread(&Detector::inferenceThread, this);
}

void Detector::stop()
{
    auto& ctx = AppContext::getInstance();
    ctx.shouldExit = true;
    if (m_captureThread.joinable())
    {
        m_captureThread.join();
    }
    if (m_inferenceThread.joinable())
    {
        m_inferenceThread.join();
    }
}
