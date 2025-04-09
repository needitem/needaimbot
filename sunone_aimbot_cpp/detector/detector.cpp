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

// Assume a global pointer to the active capture object exists (Needs proper implementation)
// extern IScreenCapture* g_capture;
extern std::atomic<bool> detectionPaused;

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

// Assume detector is globally accessible or passed to captureThread
// extern IScreenCapture* g_capture; // Removed global capture dependency

static bool error_logged = false;

Detector::Detector()
    : frameReady(false),
    shouldExit(false),
    detectionVersion(0),
    inputBufferDevice(nullptr),
    img_scale(1.0f),
    numClasses(0),
    m_finalDetectionsGpu(nullptr),
    m_finalDetectionsCountGpu(nullptr),
    m_scoresGpu(nullptr),
    m_bestTargetIndexGpu(nullptr)
{
    cudaStreamCreate(&stream);
    cudaStreamCreate(&preprocessStream);
    cudaStreamCreate(&postprocessStream);

    cvStream = cv::cuda::Stream();
    preprocessCvStream = cv::cuda::Stream();
    postprocessCvStream = cv::cuda::Stream();
    
    cudaEventCreateWithFlags(&processingDone, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&postprocessCopyDone, cudaEventDisableTiming);
}

Detector::~Detector()
{
    cudaStreamDestroy(stream);
    cudaStreamDestroy(preprocessStream);
    cudaStreamDestroy(postprocessStream);
    cudaEventDestroy(processingDone);
    cudaEventDestroy(postprocessCopyDone);
    
    for (auto& buffer : pinnedOutputBuffers)
    {
        if (buffer.second) cudaFreeHost(buffer.second);
    }

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

    if (m_finalDetectionsGpu) {
        cudaFree(m_finalDetectionsGpu);
        m_finalDetectionsGpu = nullptr;
    }
    if (m_finalDetectionsCountGpu) {
        cudaFree(m_finalDetectionsCountGpu);
        m_finalDetectionsCountGpu = nullptr;
    }
    if (m_scoresGpu) {
        cudaFree(m_scoresGpu);
        m_scoresGpu = nullptr;
    }
    if (m_bestTargetIndexGpu) {
        cudaFree(m_bestTargetIndexGpu);
        m_bestTargetIndexGpu = nullptr;
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

void Detector::initialize(const std::string& modelFile)
{
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

    context->setInputShape(inputName.c_str(), nvinfer1::Dims4{1, 3, 640, 640});

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
        
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; j++)
        {
            shape.push_back(dims.d[j]);
        }

        outputShapes[outName] = shape;
        
        if (dtype == nvinfer1::DataType::kHALF) {
            size_t numElements = outputSizes[outName] / sizeof(__half);
            outputDataBuffersHalf[outName].reserve(numElements);
        } else if (dtype == nvinfer1::DataType::kFLOAT) {
            size_t numElements = outputSizes[outName] / sizeof(float);
            outputDataBuffers[outName].reserve(numElements);
        }
    }

    getBindings();

    size_t maxDetections = config.max_detections > 0 ? config.max_detections : 100;
    size_t finalDetectionsSize = sizeof(Detection) * maxDetections;
    if (m_finalDetectionsGpu) cudaFree(m_finalDetectionsGpu);
    cudaError_t err_det = cudaMalloc(&m_finalDetectionsGpu, finalDetectionsSize);
    if (err_det != cudaSuccess) {
        std::cerr << "[Detector] Failed to allocate GPU memory for final detections: " << cudaGetErrorString(err_det) << std::endl;
    }

    if (m_finalDetectionsCountGpu) cudaFree(m_finalDetectionsCountGpu);
    cudaError_t err_count = cudaMalloc(&m_finalDetectionsCountGpu, sizeof(int));
    if (err_count != cudaSuccess) {
        std::cerr << "[Detector] Failed to allocate GPU memory for detection count: " << cudaGetErrorString(err_count) << std::endl;
    }
    
    m_finalDetectionsHost.resize(maxDetections);

    // Allocate GPU memory for scoring results
    size_t scoresSize = maxDetections * sizeof(float);
    if (m_scoresGpu) cudaFree(m_scoresGpu);
    cudaError_t err_scores = cudaMalloc(&m_scoresGpu, scoresSize);
    if (err_scores != cudaSuccess) {
        std::cerr << "[Detector] Failed to allocate GPU memory for scores: " << cudaGetErrorString(err_scores) << std::endl;
    }

    // Allocate GPU memory for best target index
    if (m_bestTargetIndexGpu) cudaFree(m_bestTargetIndexGpu);
    cudaError_t err_idx = cudaMalloc(&m_bestTargetIndexGpu, sizeof(int));
    if (err_idx != cudaSuccess) {
        std::cerr << "[Detector] Failed to allocate GPU memory for best target index: " << cudaGetErrorString(err_idx) << std::endl;
    }

    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());

        if (config.postprocess == "yolo10")
        {
            numClasses = 11;
        } else
        {
            numClasses = outDims.d[1] - 4;
        }
    }

    img_scale = static_cast<float>(config.detection_resolution) / 640;
    
    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];
    
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

    if (config.use_pinned_memory)
    {
        for (const auto& outName : outputNames)
        {
            size_t size = outputSizes[outName];
            
            if (pinnedOutputBuffers.find(outName) != pinnedOutputBuffers.end() && 
                pinnedOutputBuffers[outName] != nullptr) {
                continue;
            }
            
            void* hostBuffer = nullptr;
            cudaError_t status = cudaMallocHost(&hostBuffer, size);
            if (status == cudaSuccess)
            {
                pinnedOutputBuffers[outName] = hostBuffer;
            }
        }
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
    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();
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
    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();
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
            }
            
            initialize("models/" + config.ai_model);
            
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
        }
        
        cv::Mat frameCpu;
        bool isGpu = false;
        bool hasNewFrame = false;
        
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            
            if (!frameReady && !shouldExit)
            {
                inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
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
                // Upload CPU frame to GPU if necessary before preprocessing
                if (!isGpu) {
                    frameGpu.upload(frameCpu, preprocessCvStream); 
                }

                preProcess(frameGpu); // Always preprocess the GpuMat
                
                context->enqueueV3(stream);

                // --- Copy raw output from GPU to CPU Host buffers --- 
                for (const auto& name : outputNames)
                {
                    size_t size = outputSizes[name];
                    nvinfer1::DataType dtype = outputTypes[name];
                    void* outputBindingPtr = outputBindings[name];
                    void* hostBufferPtr = nullptr;

                    if (dtype == nvinfer1::DataType::kHALF) {
                        // Ensure buffer exists and is correctly sized
                        size_t numElements = size / sizeof(__half);
                        if (outputDataBuffersHalf.find(name) == outputDataBuffersHalf.end() || outputDataBuffersHalf[name].size() != numElements) {
                            outputDataBuffersHalf[name].resize(numElements);
                        }
                        // Determine destination buffer (pinned or regular)
                        hostBufferPtr = outputDataBuffersHalf[name].data();
                        if (config.use_pinned_memory && pinnedOutputBuffers.count(name)) {
                            hostBufferPtr = pinnedOutputBuffers[name]; // Use pinned memory if available and configured
                        }
                    } else if (dtype == nvinfer1::DataType::kFLOAT) {
                        size_t numElements = size / sizeof(float);
                         if (outputDataBuffers.find(name) == outputDataBuffers.end() || outputDataBuffers[name].size() != numElements) {
                            outputDataBuffers[name].resize(numElements);
                        }
                        hostBufferPtr = outputDataBuffers[name].data();
                         if (config.use_pinned_memory && pinnedOutputBuffers.count(name)) {
                            hostBufferPtr = pinnedOutputBuffers[name]; // Use pinned memory
                        }
                    } else {
                        // Handle other types if necessary, or log error
                        std::cerr << "[Detector] Unsupported output data type for copy: " << static_cast<int>(dtype) << " for output " << name << std::endl;
                        continue; // Skip copy for unsupported type
                    }

                    if (hostBufferPtr && outputBindingPtr) {
                        cudaMemcpyAsync(
                            hostBufferPtr,          // Destination (Host: pinned or regular)
                            outputBindingPtr,       // Source (GPU)
                            size,                   // Size in bytes
                            cudaMemcpyDeviceToHost,
                            stream                 // Use the main inference stream
                        );
                    } else if (!outputBindingPtr) {
                         std::cerr << "[Detector] Output binding pointer is null for " << name << std::endl;
                    } else {
                         std::cerr << "[Detector] Host buffer pointer is null for " << name << " (Type: " << static_cast<int>(dtype) << ")" << std::endl;
                    }
                }
                // Synchronize after submitting all copies for this frame
                cudaStreamSynchronize(stream); 
                
                // --- Handle Pinned Memory Transfer (if used) --- 
                // If pinned memory was used, copy data from pinned host buffers to regular std::vectors
                if (config.use_pinned_memory) {
                    for (const auto& name : outputNames) {
                         nvinfer1::DataType dtype = outputTypes[name];
                         size_t size = outputSizes[name];
                         if (pinnedOutputBuffers.count(name)) {
                             void* pinnedHostPtr = pinnedOutputBuffers[name];
                             if (dtype == nvinfer1::DataType::kHALF) {
                                 memcpy(outputDataBuffersHalf[name].data(), pinnedHostPtr, size);
                             } else if (dtype == nvinfer1::DataType::kFLOAT) {
                                 memcpy(outputDataBuffers[name].data(), pinnedHostPtr, size);
                             } 
                         } 
                    }
                }

                // --- Perform Post-processing (CPU Decode + GPU NMS) --- 
                performGpuPostProcessing(stream); // This function will now use the CPU buffers populated above

                // --- Synchronize GPU NMS operations --- 
                cudaStreamSynchronize(stream); // Ensure NMSGpu operations & count copy are complete

                // Reset best target flag
                m_hasBestTarget = false;

                // --- Calculate Scores, Find Best Target, and Copy ONLY Best Target to Host --- 
                if (m_finalDetectionsCountHost > 0) {
                    // Ensure count doesn't exceed buffer sizes
                    int validDetections = std::min((int)m_finalDetectionsHost.size(), m_finalDetectionsCountHost);
                    
                    if (validDetections > 0) {
                        // Calculate scores on GPU
                        calculateTargetScoresGpu(
                            static_cast<Detection*>(m_finalDetectionsGpu),
                            validDetections,
                            m_scoresGpu,
                            config.detection_resolution, // Use config values for resolution
                            config.detection_resolution,
                            config.disable_headshot,
                            config.class_head, // Make sure class_head is defined in config
                            stream
                        );

                        // Find the index of the best target on GPU
                        findBestTargetGpu(
                            m_scoresGpu,
                            validDetections,
                            m_bestTargetIndexGpu,
                            stream
                        );

                        // Copy the best index from GPU to Host
                        cudaMemcpyAsync(
                            &m_bestTargetIndexHost,
                            m_bestTargetIndexGpu,
                            sizeof(int),
                            cudaMemcpyDeviceToHost,
                            stream
                        );
                        // Synchronize to get the best index on host
                        cudaStreamSynchronize(stream);

                        // If a valid best target index was found, copy that single detection to host
                        if (m_bestTargetIndexHost >= 0 && m_bestTargetIndexHost < validDetections) {
                            cudaMemcpyAsync(
                                &m_bestTargetHost,                                               // Dest: Host struct
                                static_cast<Detection*>(m_finalDetectionsGpu) + m_bestTargetIndexHost, // Src: Element in GPU array
                                sizeof(Detection),                                               // Size of one detection
                                cudaMemcpyDeviceToHost,
                                stream
                            );
                            // Synchronize to ensure the best target data is ready on host
                            cudaStreamSynchronize(stream);
                            m_hasBestTarget = true; // Mark that we have a valid target
                        } else {
                           m_bestTargetIndexHost = -1; // Ensure index is -1 if something went wrong
                        }
                    } else {
                         m_bestTargetIndexHost = -1; // No valid detections to score
                    }
                } else {
                     m_bestTargetIndexHost = -1; // No detections from NMS
                }

                // --- Update shared detection results (Only best target related info if needed) --- 
                {
                    std::lock_guard<std::mutex> lock(detectionMutex);
                    // Clear old full detection lists
                    detectedBoxes.clear();
                    detectedClasses.clear();
                    
                    // We no longer populate detectedBoxes/detectedClasses here.
                    // Mouse thread will directly use m_hasBestTarget and m_bestTargetHost.
                    
                    // Update version number regardless
                    detectionVersion++;
                }
                detectionCV.notify_one(); // Notify waiting threads (mouse thread)

            } catch (const std::exception& e)
            {
                std::cerr << "[Detector] Error during inference: " << e.what() << std::endl;
            }
        }
    }
}

void Detector::releaseDetections()
{
    std::lock_guard<std::mutex> lock(detectionMutex);
    detectedBoxes.clear();
    detectedClasses.clear();
}

bool Detector::getLatestDetections(std::vector<cv::Rect>& boxes, std::vector<int>& classes)
{
    std::lock_guard<std::mutex> lock(detectionMutex);

    if (!detectedBoxes.empty())
    {
        boxes = detectedBoxes;
        classes = detectedClasses;
        return true;
    }
    return false;
}

void Detector::preProcess(const cv::cuda::GpuMat& frame)
{
    if (frame.empty()) return;

    // --- Wait for Capture Event (using member variable) ---
    if (m_captureDoneEvent) {
        // Make the preprocess stream wait for the capture event
        cudaStream_t underlyingPreprocessStream = cv::cuda::StreamAccessor::getStream(preprocessCvStream);
        cudaError_t waitErr = cudaStreamWaitEvent(underlyingPreprocessStream, m_captureDoneEvent, 0);
        if (waitErr != cudaSuccess) {
            std::cerr << "[Detector] cudaStreamWaitEvent failed in preProcess: " << cudaGetErrorString(waitErr) << std::endl;
            // Handle error, maybe return or throw
        }
    }
    // --- End Wait for Capture Event ---

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
        
        // Synchronize preprocess operations before copying to TensorRT buffer
        cudaEvent_t preprocessDoneEvent;
        cudaEventCreateWithFlags(&preprocessDoneEvent, cudaEventDisableTiming);
        
        // Get the underlying CUDA stream from the OpenCV stream
        cudaStream_t underlyingPreprocessStream = cv::cuda::StreamAccessor::getStream(preprocessCvStream);
        
        // Record the event on the underlying CUDA stream
        cudaEventRecord(preprocessDoneEvent, underlyingPreprocessStream);
        
        // Make the main TensorRT stream wait for the preprocess event
        cudaStreamWaitEvent(stream, preprocessDoneEvent, 0);
        
        // Destroy the event (consider managing lifecycle more carefully if needed)
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

void Detector::performGpuPostProcessing(cudaStream_t stream) {
    if (outputNames.empty()) {
        std::cerr << "[Detector] No output names found for post-processing." << std::endl;
        m_finalDetectionsCountHost = 0; // Ensure count is 0 if no output
        return;
    }

    const std::string& primaryOutputName = outputNames[0]; // Assuming primary output is sufficient
    nvinfer1::DataType outputType = outputTypes[primaryOutputName];
    const std::vector<int64_t>& shape = outputShapes[primaryOutputName];

    std::vector<Detection> decoded_detections;
    const float* outputDataPtr = nullptr;
    std::vector<float> tempFloatBuffer; // Needed if input is half

    // --- Get pointer to CPU buffer and handle FP16 conversion --- 
    if (outputType == nvinfer1::DataType::kHALF) {
        if (outputDataBuffersHalf.find(primaryOutputName) == outputDataBuffersHalf.end()) {
             std::cerr << "[Detector] Half precision output buffer not found for " << primaryOutputName << std::endl;
             m_finalDetectionsCountHost = 0;
             return;
        }
        const std::vector<__half>& halfData = outputDataBuffersHalf[primaryOutputName];
        size_t numElements = halfData.size();
        tempFloatBuffer.resize(numElements); 
        // Perform conversion from __half to float on CPU
        for (size_t i = 0; i < numElements; ++i) {
            tempFloatBuffer[i] = __half2float(halfData[i]);
        }
        outputDataPtr = tempFloatBuffer.data();

    } else if (outputType == nvinfer1::DataType::kFLOAT) {
        if (outputDataBuffers.find(primaryOutputName) == outputDataBuffers.end()) {
             std::cerr << "[Detector] Float output buffer not found for " << primaryOutputName << std::endl;
             m_finalDetectionsCountHost = 0;
             return;
        }
        outputDataPtr = outputDataBuffers[primaryOutputName].data();
    } else {
         std::cerr << "[Detector] Unsupported output data type for CPU decoding: " << static_cast<int>(outputType) << std::endl;
         m_finalDetectionsCountHost = 0;
         return;
    }

    if (!outputDataPtr) {
         std::cerr << "[Detector] Failed to get valid CPU output data pointer." << std::endl;
         m_finalDetectionsCountHost = 0;
         return;
    }

    // --- Perform CPU Decoding --- 
    // Pass img_scale to the decode function
    // Note: Need to ensure img_scale is up-to-date if detection resolution changes
    if (config.postprocess == "yolo10") {
        decoded_detections = decodeYolo10(
            outputDataPtr,
            shape,
            numClasses,
            config.confidence_threshold,
            this->img_scale
        );
    } else if (config.postprocess == "yolo8" || config.postprocess == "yolo9" || config.postprocess == "yolo11" || config.postprocess == "yolo12") {
         decoded_detections = decodeYolo11(
            outputDataPtr,
            shape,
            numClasses,
            config.confidence_threshold,
            this->img_scale
        );
    } else {
        std::cerr << "[Detector] Unsupported post-processing type for CPU decoding: " << config.postprocess << std::endl;
        m_finalDetectionsCountHost = 0;
        return; // Or handle differently
    }

    // --- Perform GPU NMS --- 
    if (!decoded_detections.empty()) {
        // Allocate temporary GPU buffer for decoded detections
        Detection* tempDecodedGpu = nullptr;
        size_t decodedSize = decoded_detections.size() * sizeof(Detection);
        cudaError_t allocErr = cudaMallocAsync(&tempDecodedGpu, decodedSize, stream);

        if (allocErr == cudaSuccess) {
            // Copy decoded detections from CPU to temporary GPU buffer
            cudaMemcpyAsync(tempDecodedGpu, decoded_detections.data(), decodedSize, cudaMemcpyHostToDevice, stream);

            try {
                // Call the modified NMSGpu
                NMSGpu(
                    tempDecodedGpu,              // Input detections (GPU)
                    decoded_detections.size(),   // Number of input detections
                    static_cast<Detection*>(m_finalDetectionsGpu), // Cast output buffer to Detection*
                    m_finalDetectionsCountGpu,   // Output count of filtered detections (GPU)
                    (int)m_finalDetectionsHost.size(), // Max size of the output buffer (using host buffer size)
                    config.nms_threshold,
                    stream
                );
            } catch (const std::exception& e) {
                 std::cerr << "[Detector] NMSGpu call failed: " << e.what() << std::endl;
                 // If NMSGpu fails, set count to 0
                 cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
            }
            // Free the temporary GPU buffer
            cudaFreeAsync(tempDecodedGpu, stream);
        } else {
             std::cerr << "[Detector] Failed to allocate temp GPU buffer for NMS input: " << cudaGetErrorString(allocErr) << std::endl;
             cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream); // Set count to 0 on allocation failure
        }
    } else {
         // No detections after decoding, set GPU count to 0
         cudaMemsetAsync(m_finalDetectionsCountGpu, 0, sizeof(int), stream);
    }

    // --- Update Host Count --- 
    // Copy the final detection count from GPU to Host
    cudaMemcpyAsync(
        &m_finalDetectionsCountHost, 
        m_finalDetectionsCountGpu, 
        sizeof(int), 
        cudaMemcpyDeviceToHost, 
        stream
    );

    // Removed memcpy from decoded_detections to m_finalDetectionsHost
    /*
    m_finalDetectionsCountHost = static_cast<int>(decoded_detections.size());
    int countToCopy = std::min((int)m_finalDetectionsHost.size(), m_finalDetectionsCountHost);
    if (countToCopy > 0) {
         memcpy(m_finalDetectionsHost.data(), decoded_detections.data(), countToCopy * sizeof(Detection));
    }
    */

}

void Detector::setCaptureEvent(cudaEvent_t event) {
    m_captureDoneEvent = event;
}