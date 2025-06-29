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
    shouldExit(false),
    detectionVersion(0),
    inputBufferDevice(nullptr),
    img_scale(1.0f),
    numClasses(0),
    m_decodedDetectionsGpu(nullptr),
    m_decodedCountGpu(nullptr),
    m_finalDetectionsGpu(nullptr),
    m_finalDetectionsCountGpu(nullptr),
    m_classFilteredDetectionsGpu(nullptr),
    m_classFilteredCountGpu(nullptr),
    m_scoresGpu(nullptr),
    m_bestTargetIndexGpu(nullptr),
    m_captureDoneEvent(nullptr),
    m_cudaContextInitialized(false),
    m_hasBestTarget(false),
    m_computeStream(nullptr),
    m_memoryStream(nullptr),
    m_bestTargetIndexHost(-1),
    m_finalDetectionsCountHost(0),
    m_classFilteredCountHost(0),
    m_d_ignore_flags_gpu(nullptr),
    
    m_nms_d_x1(nullptr),
    m_nms_d_y1(nullptr),
    m_nms_d_x2(nullptr),
    m_nms_d_y2(nullptr),
    m_nms_d_areas(nullptr),
    m_nms_d_scores(nullptr),
    m_nms_d_classIds(nullptr),
    m_nms_d_iou_matrix(nullptr),
    m_nms_d_keep(nullptr),
    m_nms_d_indices(nullptr),
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

    freeGpuBuffer(m_decodedDetectionsGpu);
    freeGpuBuffer(m_decodedCountGpu);
    freeGpuBuffer(m_finalDetectionsGpu);
    freeGpuBuffer(m_finalDetectionsCountGpu);
    freeGpuBuffer(m_classFilteredDetectionsGpu);
    freeGpuBuffer(m_classFilteredCountGpu);
    freeGpuBuffer(m_scoresGpu);
    freeGpuBuffer(m_bestTargetIndexGpu);

    if (m_d_ignore_flags_gpu) {
        cudaFree(m_d_ignore_flags_gpu);
        m_d_ignore_flags_gpu = nullptr;
    }

    
    freeGpuBuffer(m_nms_d_x1);
    freeGpuBuffer(m_nms_d_y1);
    freeGpuBuffer(m_nms_d_x2);
    freeGpuBuffer(m_nms_d_y2);
    freeGpuBuffer(m_nms_d_areas);
    freeGpuBuffer(m_nms_d_scores);
    freeGpuBuffer(m_nms_d_classIds);
    freeGpuBuffer(m_nms_d_iou_matrix);
    freeGpuBuffer(m_nms_d_keep);
    freeGpuBuffer(m_nms_d_indices);

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
    inputNames.clear();
    inputSizes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputNames.emplace_back(name);
            if (config.verbose)
            {
                std::cout << "[Detector] Detected input: " << name << std::endl;
            }
        }
    }

    if (m_bestTargetIndexGpu) cudaMemsetAsync(m_bestTargetIndexGpu, 0xFF, sizeof(int), stream);

    
    allocateGpuBuffer(m_nms_d_x1, config.max_detections, "NMS d_x1");
    allocateGpuBuffer(m_nms_d_y1, config.max_detections, "NMS d_y1");
    allocateGpuBuffer(m_nms_d_x2, config.max_detections, "NMS d_x2");
    allocateGpuBuffer(m_nms_d_y2, config.max_detections, "NMS d_y2");
    allocateGpuBuffer(m_nms_d_areas, config.max_detections, "NMS d_areas");
    allocateGpuBuffer(m_nms_d_scores, config.max_detections, "NMS d_scores");
    allocateGpuBuffer(m_nms_d_classIds, config.max_detections, "NMS d_classIds");
    
    allocateGpuBuffer(m_nms_d_iou_matrix, config.max_detections * config.max_detections, "NMS d_iou_matrix"); 
    allocateGpuBuffer(m_nms_d_keep, config.max_detections, "NMS d_keep");
    allocateGpuBuffer(m_nms_d_indices, config.max_detections, "NMS d_indices");
}

void Detector::getOutputNames()
{
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
            
            if (config.verbose)
            {
                std::cout << "[Detector] Detected output: " << name << std::endl;
            }
        }
    }
}

void Detector::getBindings()
{
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
                if (config.verbose)
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
                if (config.verbose)
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
    
    cudaError_t cuda_err = cudaSetDevice(config.cuda_device_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "[Detector] ERROR: Failed to set CUDA device " << config.cuda_device_id 
                  << ": " << cudaGetErrorString(cuda_err) << std::endl;
        m_cudaContextInitialized = false; 
        return false; 
    }
    std::cout << "[Detector] Successfully set CUDA device to " << config.cuda_device_id << "." << std::endl;

    
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
    int r = config.onnx_input_resolution;
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

    cudaStreamCreateWithFlags(&m_computeStream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&m_memoryStream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&m_preprocessDone, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&m_inferenceDone, cudaEventDisableTiming);
    
    initializeBuffers(); 

    
    m_ignore_flags_need_update = true;

    
    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());

        if (config.postprocess == "yolo10")
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

    img_scale = static_cast<float>(config.detection_resolution) / 640;
    
    
    m_headClassId = -1; 
    if (!config.head_class_name.empty()) {
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == config.head_class_name) {
                m_headClassId = class_setting.id;
                if (config.verbose) {
                    std::cout << "[Detector] Head class '" << config.head_class_name << "' identified with ID: " << m_headClassId << std::endl;
                }
                break;
            }
        }
        if (m_headClassId == -1 && config.verbose) {
            std::cout << "[Detector] Warning: Head class name '" << config.head_class_name << "' not found in class_settings. No specific head bonus will be applied." << std::endl;
        }
    } else if (config.verbose) {
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
        baseName += "_" + std::to_string(config.onnx_input_resolution);
        if (config.export_enable_fp16) baseName += "_fp16";
        if (config.export_enable_fp8)  baseName += "_fp8";
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
    if (!isCudaContextInitialized()) return; 

    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
        return;
    }

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame;
    frameIsGpu = true;
    frameReady = true;
    inferenceCV.notify_one();
}

void Detector::processFrame(const cv::Mat& frame)
{
    if (!isCudaContextInitialized()) return; 

    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        m_hasBestTarget = false;
        return;
    }

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrameCpu = frame.clone();
    frameIsGpu = false;
    frameReady = true;
    inferenceCV.notify_one();
}

void Detector::inferenceThread()
{
    if (!isCudaContextInitialized()) { 
        std::cerr << "[Detector Thread] CUDA context not initialized. Inference thread exiting." << std::endl;
        return;
    }

    cv::cuda::GpuMat frameGpu;
    static auto last_inference_loop_start_time = std::chrono::high_resolution_clock::time_point{}; 

    // Preallocate host buffer for detections to avoid per-frame allocations
    thread_local std::vector<Detection> current_frame_detections;

    while (!shouldExit)
    {
        NVTX_PUSH("Detector Inference Loop");  

        if (detector_model_changed.load()) {
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                
                context.reset();
                engine.reset();
                
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

                freeGpuBuffer(m_decodedDetectionsGpu);
                freeGpuBuffer(m_decodedCountGpu);
                freeGpuBuffer(m_finalDetectionsGpu);
                freeGpuBuffer(m_finalDetectionsCountGpu);
                freeGpuBuffer(m_classFilteredDetectionsGpu);
                freeGpuBuffer(m_classFilteredCountGpu);
                freeGpuBuffer(m_scoresGpu);
                freeGpuBuffer(m_bestTargetIndexGpu);
            }
            
            initialize("models/" + config.ai_model);
            
            img_scale = static_cast<float>(config.detection_resolution) / 640;
            
            detector_model_changed.store(false);
        }
        
        cv::Mat frameCpu;
        bool isGpu = false;
        bool hasNewFrame = false;
        
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            
            if (!frameReady && !shouldExit)
            {
                inferenceCV.wait_for(lock, std::chrono::milliseconds(100), [this] { return frameReady || shouldExit; });
            }
            
            if (shouldExit) break;
            
            if (frameReady)
            {
                isGpu = frameIsGpu;
                if (isGpu) {
                    frameGpu = std::move(currentFrame);
                } else {
                    frameCpu = std::move(currentFrameCpu);
                }
                frameReady = false;
                hasNewFrame = true;
            }
        }
        
        if (!context)
        {
            if (!error_logged)
            {
                std::cerr << "[Detector] Context not initialized" << std::endl;
                error_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        else
        {
            error_logged = false;
        }
        
        if (hasNewFrame && ((isGpu && !frameGpu.empty()) || (!isGpu && !frameCpu.empty())))
        {
            auto current_inference_loop_start_time = std::chrono::high_resolution_clock::now();
            if (last_inference_loop_start_time.time_since_epoch().count() != 0) { 
                std::chrono::duration<float, std::milli> cycle_duration_ms = current_inference_loop_start_time - last_inference_loop_start_time;
                g_current_detector_cycle_time_ms.store(cycle_duration_ms.count());
                add_to_history(g_detector_cycle_time_history, cycle_duration_ms.count(), g_detector_cycle_history_mutex);
            }
            last_inference_loop_start_time = current_inference_loop_start_time;

            try
            {
                auto inference_start_time = std::chrono::high_resolution_clock::now(); 

                if (!isGpu && !frameCpu.empty()) {
                    frameGpu.upload(frameCpu, preprocessCvStream); 
                } else if (!isGpu) {
                    continue; 
                }

                if (frameGpu.empty()) {
                    continue;
                }

                preProcess(frameGpu);
                
                context->enqueueV3(stream);

                performGpuPostProcessing(stream);

                auto inference_end_time = std::chrono::high_resolution_clock::now(); 
                std::chrono::duration<float, std::milli> inference_duration_ms = inference_end_time - inference_start_time;
                g_current_inference_time_ms.store(inference_duration_ms.count());
                add_to_history(g_inference_time_history, inference_duration_ms.count(), g_inference_history_mutex);

                
                

                int final_detections_count = 0;
                cudaMemcpyAsync(&final_detections_count, m_finalDetectionsCountGpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                if (final_detections_count > 0)
                {
                    calculateTargetScoresGpu(
                        m_finalDetectionsGpu, 
                        final_detections_count, 
                        m_scoresGpu, 
                        config.detection_resolution, 
                        config.detection_resolution, 
                        config.distance_weight, 
                        config.confidence_weight, 
                        m_headClassId, 
                        stream
                    );

                    findBestTargetGpu(m_scoresGpu, final_detections_count, m_bestTargetIndexGpu, stream);

                    cudaMemcpyAsync(&m_bestTargetIndexHost, m_bestTargetIndexGpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(&m_bestTargetHost, &m_finalDetectionsGpu[m_bestTargetIndexHost], sizeof(Detection), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);

                    if (m_bestTargetIndexHost >= 0 && m_bestTargetIndexHost < final_detections_count)
                    {
                        m_hasBestTarget = true;
                    }
                    else
                    {
                        m_hasBestTarget = false;
                    }
                }
                else
                {
                    m_hasBestTarget = false;
                }
                // Notify update
                {
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    detectionVersion++;
                }
                detectionCV.notify_one();
            } catch (const std::exception& e)
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
        } else {
             
             
             
             
        }

        NVTX_POP();  
    }
}

void Detector::performGpuPostProcessing(cudaStream_t stream) {
    if (outputNames.empty()) {
        std::cerr << "[Detector] No output names found for post-processing." << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        return;
    }

    const std::string& primaryOutputName = outputNames[0];
    void* d_rawOutputPtr = outputBindings[primaryOutputName];
    nvinfer1::DataType outputType = outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = outputShapes[primaryOutputName];

    if (!d_rawOutputPtr) {
        std::cerr << "[Detector] Raw output GPU pointer is null for " << primaryOutputName << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        return;
    }

    cudaMemsetAsync(m_decodedCountGpu, 0, sizeof(int), stream);
    cudaError_t decodeErr = cudaSuccess;

    int maxDecodedDetections = config.max_detections * 2;

    if (config.postprocess == "yolo10") {
        decodeErr = decodeYolo10Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            config.confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu,
            m_decodedCountGpu,
            maxDecodedDetections,
            stream);
    } else if (config.postprocess == "yolo8" || config.postprocess == "yolo9" || config.postprocess == "yolo11" || config.postprocess == "yolo12") {
         decodeErr = decodeYolo11Gpu(
            d_rawOutputPtr,
            outputType,
            shape,
            numClasses,
            config.confidence_threshold,
            this->img_scale,
            m_decodedDetectionsGpu,
            m_decodedCountGpu,
            maxDecodedDetections,
            stream);
    } else {
        std::cerr << "[Detector] Unsupported post-processing type for GPU decoding: " << config.postprocess << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        return;
    }

    if (decodeErr != cudaSuccess) {
        std::cerr << "[Detector] GPU decoding kernel launch/execution failed: " << cudaGetErrorString(decodeErr) << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        return;
    }

    
    int decodedCountHost = 0;
    cudaError_t decodeCountCopyErr = cudaMemcpyAsync(&decodedCountHost, m_decodedCountGpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (decodeCountCopyErr != cudaSuccess) {
        std::cerr << "[Detector] Failed to copy decoded count DtoH: " << cudaGetErrorString(decodeCountCopyErr) << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        return;
    }

    
    // Synchronization needed to read decodedCountHost
    cudaStreamSynchronize(stream);
    
    int classFilteredCountHost = 0;
    if (decodedCountHost > 0) {
        int validDecodedDetections = std::min(decodedCountHost, static_cast<int>(config.max_detections * 2));
        if (validDecodedDetections > 0) {
            
            if (m_ignore_flags_need_update) {
                { 
                    std::lock_guard<std::mutex> lock(configMutex);
                    
                    std::fill(m_host_ignore_flags_uchar.begin(), m_host_ignore_flags_uchar.end(), 1); 
                    for (const auto& class_setting : config.class_settings) {
                        if (class_setting.id >= 0 && class_setting.id < MAX_CLASSES_FOR_FILTERING) {
                            m_host_ignore_flags_uchar[class_setting.id] = static_cast<unsigned char>(class_setting.ignore);
                        }
                    }
                }

                if (m_d_ignore_flags_gpu) {
                    cudaError_t copyErr = cudaMemcpyAsync(m_d_ignore_flags_gpu, m_host_ignore_flags_uchar.data(), 
                                                          MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), 
                                                          cudaMemcpyHostToDevice, stream);
                    if (!checkCudaError(copyErr, "copying updated ignore flags to GPU")) {
                        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
                        return;
                    }
                } else {
                    std::cerr << "[Detector] Ignore flags GPU buffer not allocated!" << std::endl;
                    cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
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
                std::lock_guard<std::mutex> lock(configMutex);
                if (config.enable_hsv_filter) { 
                    std::lock_guard<std::mutex> hsv_lock(hsvMaskMutex); 
                    if (!m_hsvMaskGpu.empty()) {
                        hsvMaskPtr = m_hsvMaskGpu.ptr<unsigned char>();
                        maskPitch = static_cast<int>(m_hsvMaskGpu.step);
                    }
                }
                current_min_hsv_pixels_val = config.min_hsv_pixels;
                current_remove_hsv_matches_val = config.remove_hsv_matches;
                current_max_output_detections_val = config.max_detections; 
            } 
            
            
            cudaError_t filterErr = filterDetectionsByClassIdGpu(
                m_decodedDetectionsGpu,
                validDecodedDetections,
                m_classFilteredDetectionsGpu,
                m_classFilteredCountGpu,
                m_d_ignore_flags_gpu,
                MAX_CLASSES_FOR_FILTERING,
                hsvMaskPtr,
                maskPitch,
                current_min_hsv_pixels_val,
                current_remove_hsv_matches_val,
                current_max_output_detections_val,
                stream
            );
            if (!checkCudaError(filterErr, "filtering detections by class ID GPU")) {
                 cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
                 return;
            }

            cudaError_t filteredCountCopyErr = cudaMemcpyAsync(&classFilteredCountHost, m_classFilteredCountGpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (filteredCountCopyErr != cudaSuccess) {
                std::cerr << "[Detector] Failed to copy filtered detection count DtoH: " << cudaGetErrorString(filteredCountCopyErr) << std::endl;
                classFilteredCountHost = 0;
            }
        } else {
            classFilteredCountHost = 0;
            cudaMemsetAsync(m_classFilteredCountGpu, 0, sizeof(int), stream);
        }
    } else {
        classFilteredCountHost = 0;
        cudaMemsetAsync(m_classFilteredCountGpu, 0, sizeof(int), stream);
    }

    
    if (classFilteredCountHost > 0) {
        // Limit NMS input to reduce computation - only top 10 for performance
        int inputNmsCount = std::min(classFilteredCountHost, std::min(static_cast<int>(config.max_detections), 10)); 
        if (inputNmsCount > 0) {
            try {
                NMSGpu(
                    m_classFilteredDetectionsGpu,
                    inputNmsCount,            
                    m_finalDetectionsGpu,       
                    m_finalDetectionsCountGpu,  
                    static_cast<int>(config.max_detections), 
                    config.nms_threshold,
                    
                    m_nms_d_x1,
                    m_nms_d_y1,
                    m_nms_d_x2,
                    m_nms_d_y2,
                    m_nms_d_areas,
                    m_nms_d_scores,     
                    m_nms_d_classIds,   
                    m_nms_d_iou_matrix,
                    m_nms_d_keep,
                    m_nms_d_indices,
                    stream
                );
            } catch (const std::exception& e) {
                 std::cerr << "[Detector] Exception during NMSGpu call: " << e.what() << std::endl;
                 cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
            }
        } else {
            cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        }
    } else {
         cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
    }
}

void Detector::preProcess(const cv::cuda::GpuMat& frame)
{
    if (frame.empty()) return;

    if (m_captureDoneEvent) {
        cudaStream_t underlyingPreprocessStream = cv::cuda::StreamAccessor::getStream(preprocessCvStream);
        cudaError_t waitErr = cudaStreamWaitEvent(underlyingPreprocessStream, m_captureDoneEvent, 0);
        if (waitErr != cudaSuccess) {
            std::cerr << "[Detector] cudaStreamWaitEvent failed in preProcess: " << cudaGetErrorString(waitErr) << std::endl;
        }
    }

    void* inputBuffer = inputBindings[inputName];

    if (!inputBuffer) return;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());

    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];

    try
    {
        cv::cuda::GpuMat preprocessedFrame;

        if (config.circle_mask) {
            cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
            cv::Point center(mask.cols / 2, mask.rows / 2);
            int radius = std::min(mask.cols, mask.rows) / 2;
            cv::circle(mask, center, radius, cv::Scalar(255), -1);
            cv::cuda::GpuMat maskGpu;
            maskGpu.upload(mask, preprocessCvStream);

            cv::cuda::GpuMat maskedImageGpu;
            maskedImageGpu.create(frame.size(), frame.type()); 
            maskedImageGpu.setTo(cv::Scalar::all(0), preprocessCvStream);
            frame.copyTo(maskedImageGpu, maskGpu, preprocessCvStream);
            
            cv::cuda::resize(maskedImageGpu, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, preprocessCvStream);
        } else {
            cv::cuda::resize(frame, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, preprocessCvStream);
        }

        
        bool current_enable_hsv_filter;
        int current_hsv_lower_h = 0, current_hsv_lower_s = 0, current_hsv_lower_v = 0;
        int current_hsv_upper_h = 0, current_hsv_upper_s = 0, current_hsv_upper_v = 0;

        { 
            std::lock_guard<std::mutex> lock(configMutex);
            current_enable_hsv_filter = config.enable_hsv_filter;
            if (current_enable_hsv_filter) { 
                current_hsv_lower_h = config.hsv_lower_h;
                current_hsv_lower_s = config.hsv_lower_s;
                current_hsv_lower_v = config.hsv_lower_v;
                current_hsv_upper_h = config.hsv_upper_h;
                current_hsv_upper_s = config.hsv_upper_s;
                current_hsv_upper_v = config.hsv_upper_v;
            }
        } 

        if (current_enable_hsv_filter) {
            cv::cuda::GpuMat hsvGpu;
            cv::cuda::cvtColor(resizedBuffer, hsvGpu, cv::COLOR_BGR2HSV, 0, preprocessCvStream);
            cv::Scalar lower(current_hsv_lower_h, current_hsv_lower_s, current_hsv_lower_v);
            cv::Scalar upper(current_hsv_upper_h, current_hsv_upper_s, current_hsv_upper_v);
            cv::cuda::GpuMat maskGpu;
            cv::cuda::inRange(hsvGpu, lower, upper, maskGpu, preprocessCvStream);
            
            int detRes = 0;
            {
                std::lock_guard<std::mutex> lock(configMutex);
                detRes = config.detection_resolution;
            }
            cv::cuda::GpuMat maskResized;
            cv::cuda::resize(maskGpu, maskResized, cv::Size(detRes, detRes), 0, 0, cv::INTER_NEAREST, preprocessCvStream);
            {
                std::lock_guard<std::mutex> lock(hsvMaskMutex);
                m_hsvMaskGpu = maskResized;
            }
        } else {
            std::lock_guard<std::mutex> lock(hsvMaskMutex); 
            m_hsvMaskGpu.release();
        }

        resizedBuffer.convertTo(floatBuffer, CV_32F, 1.0f / 255.0f, 0, preprocessCvStream);
        cv::cuda::split(floatBuffer, channelBuffers, preprocessCvStream);
        
        cudaEvent_t preprocessDoneEvent;
        cudaEventCreateWithFlags(&preprocessDoneEvent, cudaEventDisableTiming);
        
        cudaStream_t underlyingPreprocessStream = cv::cuda::StreamAccessor::getStream(preprocessCvStream);
        
        cudaEventRecord(preprocessDoneEvent, underlyingPreprocessStream);
        
        cudaStreamWaitEvent(stream, preprocessDoneEvent, 0);
        
        cudaEventDestroy(preprocessDoneEvent);

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

void Detector::setCaptureEvent(cudaEvent_t event) {
    m_captureDoneEvent = event;
}

void Detector::initializeBuffers() {
    
    if (!context || !stream) {
        std::cerr << "[Detector] Error: Cannot initialize buffers without valid context and stream." << std::endl;
        return; 
    }
    
    allocateGpuBuffer(m_decodedDetectionsGpu, config.max_detections * 2, "decoded detections");
    allocateGpuBuffer(m_decodedCountGpu, 1, "decoded count");
    allocateGpuBuffer(m_finalDetectionsGpu, config.max_detections, "final detections");
    allocateGpuBuffer(m_finalDetectionsCountGpu, 1, "final count");
    allocateGpuBuffer(m_classFilteredDetectionsGpu, config.max_detections, "class filtered detections");
    allocateGpuBuffer(m_classFilteredCountGpu, 1, "class filtered count");
    allocateGpuBuffer(m_scoresGpu, config.max_detections, "scores");
    allocateGpuBuffer(m_bestTargetIndexGpu, 1, "best index");

    
    cudaError_t err = cudaMalloc(&m_d_ignore_flags_gpu, MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to allocate GPU buffer for ignore flags: " << cudaGetErrorString(err) << std::endl;
        
    } 
    

    
    if (m_decodedCountGpu) cudaMemsetAsync(m_decodedCountGpu, 0, sizeof(int), stream);
    if (m_finalDetectionsCountGpu) cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
    if (m_classFilteredCountGpu) cudaMemsetAsync(m_classFilteredCountGpu, 0, sizeof(int), stream);
    if (m_bestTargetIndexGpu) cudaMemsetAsync(m_bestTargetIndexGpu, 0xFF, sizeof(int), stream);
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
