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

#include "detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "postProcess.h"
#include "scoringGpu.h"
#include "filterGpu.h"
#include "config.h"

// Assume a global pointer to the active capture object exists (Needs proper implementation)
// extern IScreenCapture* g_capture;
extern std::atomic<bool> detectionPaused;

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

// Assume detector is globally accessible or passed to captureThread
// extern IScreenCapture* g_capture; // Removed global capture dependency

extern Config config; // Declare external global config object
extern std::mutex configMutex; // Added extern declaration

static bool error_logged = false;

// Declaration of the CUDA kernel (must match the .cu file)
// This is a placeholder based on the new arguments. You MUST update your .cu file.
extern cudaError_t filterDetectionsByClassIdGpu(
    const Detection* decodedDetections,
    int numDecodedDetections,
    Detection* filteredDetections,
    int* filteredCount,
    const unsigned char* d_ignored_class_ids, // Changed to unsigned char*
    int max_check_id,                        // Size of the d_ignored_class_ids array
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
    m_bestTargetIndexHost(-1),
    m_finalDetectionsCountHost(0),
    m_classFilteredCountHost(0),
    m_d_ignore_flags_gpu(nullptr)
{
    // No CUDA initialization here anymore
}

Detector::~Detector()
{
    // Release CUDA resources if they were created
    if (stream) cudaStreamDestroy(stream);
    if (preprocessStream) cudaStreamDestroy(preprocessStream);
    if (postprocessStream) cudaStreamDestroy(postprocessStream);
    if (processingDone) cudaEventDestroy(processingDone);
    if (postprocessCopyDone) cudaEventDestroy(postprocessCopyDone);
    // m_captureDoneEvent is likely managed elsewhere (Capture class?)

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
    // Set the CUDA device using the value from config
    cudaError_t cuda_err = cudaSetDevice(config.cuda_device_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "[Detector] ERROR: Failed to set CUDA device " << config.cuda_device_id 
                  << ": " << cudaGetErrorString(cuda_err) << std::endl;
        m_cudaContextInitialized = false; // Ensure flag is false
        return false; // Return false on failure
    }
    std::cout << "[Detector] Successfully set CUDA device to " << config.cuda_device_id << "." << std::endl;

    // Create CUDA streams
    if (!checkCudaError(cudaStreamCreate(&stream), "creating main stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&preprocessStream), "creating preprocess stream")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaStreamCreate(&postprocessStream), "creating postprocess stream")) { m_cudaContextInitialized = false; return false; }

    // Create corresponding OpenCV CUDA streams (optional, but good practice if mixing OpenCV CUDA)
    cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    preprocessCvStream = cv::cuda::StreamAccessor::wrapStream(preprocessStream);
    postprocessCvStream = cv::cuda::StreamAccessor::wrapStream(postprocessStream);
    
    // Create CUDA events
    if (!checkCudaError(cudaEventCreateWithFlags(&processingDone, cudaEventDisableTiming), "creating processingDone event")) { m_cudaContextInitialized = false; return false; }
    if (!checkCudaError(cudaEventCreateWithFlags(&postprocessCopyDone, cudaEventDisableTiming), "creating postprocessCopyDone event")) { m_cudaContextInitialized = false; return false; }

    m_cudaContextInitialized = true; // Set flag to true on success
    return true; // Return true on success
}

void Detector::initialize(const std::string& modelFile)
{
    if (!isCudaContextInitialized()) {
        std::cerr << "[Detector] CUDA context not initialized. Skipping TensorRT engine load and GPU memory allocation." << std::endl;
        return; // Skip if CUDA context failed
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

    inputDims = nvinfer1::Dims4{1, 3, 640, 640};
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

    // --- Initialize GPU Buffers --- 
    initializeBuffers(); // Call the buffer initialization function

    // Determine numClasses based on output shape
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
        engineFilePath = modelPath.replace_extension(".engine").string();

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
                        
                        config.ai_model = std::filesystem::path(engineFilePath).filename().string();
                        config.saveConfig("config.ini");
                        
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
    if (!isCudaContextInitialized()) return; // Skip if CUDA context failed

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
    if (!isCudaContextInitialized()) return; // Skip if CUDA context failed

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
    if (!isCudaContextInitialized()) { // Check at the beginning of the thread
        std::cerr << "[Detector Thread] CUDA context not initialized. Inference thread exiting." << std::endl;
        return;
    }

    cv::cuda::GpuMat frameGpu;

    while (!shouldExit)
    {
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
            try
            {
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

                // --- Synchronization Point 1: Ensure PostProcessing (including NMS) is done --- 
                // cudaStreamSynchronize(stream); // Sync moved into performGpuPostProcessing and before DtoH copies

                m_hasBestTarget = false; // Reset target status
                m_finalDetectionsCountHost = 0; // Reset host count for final (NMS) results
                // m_classFilteredCountHost = 0; // No longer needed here

                // Get the count of detections after NMS (which are already class-filtered)
                cudaError_t finalCountCopyErr = cudaMemcpy(&m_finalDetectionsCountHost, m_finalDetectionsCountGpu, sizeof(int), cudaMemcpyDeviceToHost); // Use a blocking copy or ensure stream is synced before use
                // cudaStreamSynchronize(stream); // Ensure copy is complete if using async
                if (finalCountCopyErr != cudaSuccess) {
                     std::cerr << "[Detector] Failed to copy final NMS detection count DtoH: " << cudaGetErrorString(finalCountCopyErr) << std::endl;
                     m_finalDetectionsCountHost = 0;
                }

                // --- Scoring Stage (using final NMS detections, which are pre-filtered) ---
                if (m_finalDetectionsCountHost > 0) {
                    // Use the count after NMS for scoring
                    int validDetectionsForScoring = std::min((int)config.max_detections, m_finalDetectionsCountHost);
                    
                    if (validDetectionsForScoring > 0) {
                        // Score the final (class-filtered and NMS'd) detections
                        calculateTargetScoresGpu(
                            m_finalDetectionsGpu, // Use final detections
                            validDetectionsForScoring,    // Use final count
                            m_scoresGpu,
                            config.detection_resolution,
                            config.detection_resolution,
                            1.0f, // Use hardcoded distance weight
                            stream
                        );
                        cudaError_t scoreErr = cudaGetLastError();
                        if(scoreErr != cudaSuccess) std::cerr << "[Detector] Error after calculateTargetScoresGpu: " << cudaGetErrorString(scoreErr) << std::endl;

                        // Find best target among scored detections
                        cudaError_t findErr = findBestTargetGpu(
                            m_scoresGpu,
                            validDetectionsForScoring, // Use final count
                            m_bestTargetIndexGpu,
                            stream
                        );
                        if(findErr != cudaSuccess) std::cerr << "[Detector] Error after findBestTargetGpu: " << cudaGetErrorString(findErr) << std::endl;

                        // Copy best index from GPU to Host
                        cudaError_t indexCopyErr = cudaMemcpyAsync(
                            &m_bestTargetIndexHost,
                            m_bestTargetIndexGpu,
                            sizeof(int),
                            cudaMemcpyDeviceToHost,
                            stream
                        );
                        
                        // --- Synchronization Point: Ensure index copy is done --- 
                        cudaStreamSynchronize(stream); // Synchronize after index copy
                        if (indexCopyErr != cudaSuccess) {
                            std::cerr << "[Detector] Failed to copy best index DtoH: " << cudaGetErrorString(indexCopyErr) << std::endl;
                            m_bestTargetIndexHost = -1;
                        }

                        // Check if the best index is valid within the *final* detections
                        if (m_bestTargetIndexHost >= 0 && m_bestTargetIndexHost < validDetectionsForScoring) {
                            // Copy the best target *from the final NMS buffer*
                            cudaError_t targetCopyErr = cudaMemcpyAsync(
                                &m_bestTargetHost,                                      // Dest: Host struct
                                m_finalDetectionsGpu + m_bestTargetIndexHost, // Src: Element in *final* GPU array
                                sizeof(Detection),                                      // Size of one detection
                                cudaMemcpyDeviceToHost,
                                stream
                            );
                            // --- Synchronization Point: Ensure target copy is done --- 
                            cudaStreamSynchronize(stream); // Synchronize after target copy
                            if (targetCopyErr != cudaSuccess) {
                                std::cerr << "[Detector] Failed to copy best target DtoH: " << cudaGetErrorString(targetCopyErr) << std::endl;
                                m_hasBestTarget = false;
                            } else {
                                m_hasBestTarget = true;
                            }
                        } else {
                           // Best index out of bounds or negative
                           m_bestTargetIndexHost = -1;
                           m_hasBestTarget = false;
                        }
                    } else {
                         // No valid detections after NMS for scoring
                         m_bestTargetIndexHost = -1;
                         m_hasBestTarget = false;
                    }
                } else {
                     // No detections passed NMS (or class filter before NMS)
                     m_bestTargetIndexHost = -1;
                     m_hasBestTarget = false;
                }

                // Notify that detection is complete
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
             // Optional: Handle case where there's no new frame at all?
             // Maybe keep previous state or reset?
             // For now, we reset if hasNewFrame is false but the outer loop continues
             // (This branch might not be strictly necessary depending on overall flow)
        }
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

    // Copy decoded count from GPU to Host
    int decodedCountHost = 0;
    cudaError_t decodeCountCopyErr = cudaMemcpyAsync(&decodedCountHost, m_decodedCountGpu, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (decodeCountCopyErr != cudaSuccess) {
        std::cerr << "[Detector] Failed to copy decoded count DtoH: " << cudaGetErrorString(decodeCountCopyErr) << std::endl;
        cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
        return;
    }

    // --- Class ID Filtering Stage --- 
    int classFilteredCountHost = 0;
    if (decodedCountHost > 0) {
        int validDecodedDetections = std::min(decodedCountHost, static_cast<int>(config.max_detections * 2));
        if (validDecodedDetections > 0) {
            
            // 1. Prepare host-side ignore flags (vector<bool>)
            std::vector<bool> host_ignored_flags(MAX_CLASSES_FOR_FILTERING, true);
            {
                std::lock_guard<std::mutex> lock(configMutex);
                for (const auto& class_setting : config.class_settings) {
                    if (class_setting.id >= 0 && class_setting.id < MAX_CLASSES_FOR_FILTERING) {
                        host_ignored_flags[class_setting.id] = class_setting.ignore;
                    }
                }
            }

            // 2. Convert vector<bool> to vector<unsigned char>
            std::vector<unsigned char> host_ignored_uchar_flags(MAX_CLASSES_FOR_FILTERING);
            for (size_t i = 0; i < MAX_CLASSES_FOR_FILTERING; ++i) {
                host_ignored_uchar_flags[i] = static_cast<unsigned char>(host_ignored_flags[i]);
            }

            // 3. Copy host_ignored_uchar_flags to GPU buffer m_d_ignore_flags_gpu
            if (m_d_ignore_flags_gpu) {
                // Use host_ignored_uchar_flags.data() and correct size
                cudaError_t copyErr = cudaMemcpyAsync(m_d_ignore_flags_gpu, host_ignored_uchar_flags.data(), 
                                                      MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char), 
                                                      cudaMemcpyHostToDevice, stream);
                if (!checkCudaError(copyErr, "copying ignore flags (uchar) to GPU")) {
                    cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
                    return;
                }
            } else {
                std::cerr << "[Detector] Ignore flags GPU buffer not allocated!" << std::endl;
                cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
                return;
            }

            // 4. Call the GPU filter function (expecting unsigned char*)
            cudaError_t filterErr = filterDetectionsByClassIdGpu(
                m_decodedDetectionsGpu,         
                validDecodedDetections,           
                m_classFilteredDetectionsGpu,    
                m_classFilteredCountGpu,      
                m_d_ignore_flags_gpu,           // Pass unsigned char* buffer
                MAX_CLASSES_FOR_FILTERING,      
                config.max_detections,          
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

    // --- NMS Stage (Using class-filtered detections) ---
    if (classFilteredCountHost > 0) {
        int inputNmsCount = std::min(classFilteredCountHost, static_cast<int>(config.max_detections)); 
        if (inputNmsCount > 0) {
            try {
                NMSGpu(
                    m_classFilteredDetectionsGpu,
                    inputNmsCount,            
                    m_finalDetectionsGpu,       
                    m_finalDetectionsCountGpu,  
                    static_cast<int>(config.max_detections), 
                    config.nms_threshold,
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
    // Check if context and stream are valid before using them
    if (!context || !stream) {
        std::cerr << "[Detector] Error: Cannot initialize buffers without valid context and stream." << std::endl;
        return; 
    }
    // Use allocateGpuBuffer which is a member function
    allocateGpuBuffer(m_decodedDetectionsGpu, config.max_detections * 2, "decoded detections");
    allocateGpuBuffer(m_decodedCountGpu, 1, "decoded count");
    allocateGpuBuffer(m_finalDetectionsGpu, config.max_detections, "final detections");
    allocateGpuBuffer(m_finalDetectionsCountGpu, 1, "final count");
    allocateGpuBuffer(m_classFilteredDetectionsGpu, config.max_detections, "class filtered detections");
    allocateGpuBuffer(m_classFilteredCountGpu, 1, "class filtered count");
    allocateGpuBuffer(m_scoresGpu, config.max_detections, "scores");
    allocateGpuBuffer(m_bestTargetIndexGpu, 1, "best index");

    // Allocate GPU memory for ignore flags (as unsigned char)
    cudaError_t err = cudaMalloc(&m_d_ignore_flags_gpu, MAX_CLASSES_FOR_FILTERING * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to allocate GPU buffer for ignore flags: " << cudaGetErrorString(err) << std::endl;
        // Handle error
    } 
    // No initialization needed here, will be updated each frame

    // Initialize counts to zero on GPU
    if (m_decodedCountGpu) cudaMemsetAsync(m_decodedCountGpu, 0, sizeof(int), stream);
    if (m_finalDetectionsCountGpu) cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
    if (m_classFilteredCountGpu) cudaMemsetAsync(m_classFilteredCountGpu, 0, sizeof(int), stream);
    if (m_bestTargetIndexGpu) cudaMemsetAsync(m_bestTargetIndexGpu, 0xFF, sizeof(int), stream);
}