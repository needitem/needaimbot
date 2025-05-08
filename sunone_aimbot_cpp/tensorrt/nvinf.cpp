#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "nvinf.h"
#include "sunone_aimbot_cpp.h"

Logger gLogger;

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
    {
        std::string devMsg = msg;

        std::string magicTag = "Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.";
        std::string old_deserialization = "Using old deserialization call on a weight-separated plan file.";
        if (devMsg.find(magicTag) != std::string::npos || devMsg.find(old_deserialization) != std::string::npos)
        {
            std::cout << "[TensorRT] ERROR: This engine model is not suitable for execution. Please delete this engine model and set the ONNX version of this model in the settings. The program will export the model automatically." << std::endl;
        }
        else
        {
            std::cout << "[TensorRT] " << severityLevelName(severity) << ": " << msg << std::endl;
        }
    }
}

const char* Logger::severityLevelName(nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case nvinfer1::ILogger::Severity::kERROR:          return "ERROR";
        case nvinfer1::ILogger::Severity::kWARNING:        return "WARNING";
        case nvinfer1::ILogger::Severity::kINFO:           return "INFO";
        case nvinfer1::ILogger::Severity::kVERBOSE:        return "VERBOSE";
        default:                                           return "UNKNOWN";
    }
}

nvinfer1::IBuilder* createInferBuilder()
{
    return nvinfer1::createInferBuilder(gLogger);
}

nvinfer1::INetworkDefinition* createNetwork(nvinfer1::IBuilder* builder)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    return builder->createNetworkV2(explicitBatch);
}

nvinfer1::IBuilderConfig* createBuilderConfig(nvinfer1::IBuilder* builder)
{
    return builder->createBuilderConfig();
}

nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFile, nvinfer1::IRuntime* runtime)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "[TensorRT] Error opening the engine file: " << engineFile << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine)
    {
        std::cerr << "[TensorRT] Engine deserialization error from file: " << engineFile << std::endl;
        return nullptr;
    }

    if (config.verbose)
    {
        std::cout << "[TensorRT] The engine was successfully loaded from the file: " << engineFile << std::endl;
    }
    return engine;
}

nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile, nvinfer1::ILogger& logger)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    nvinfer1::IBuilderConfig* cfg = builder->createBuilderConfig();

    // --- Timing Cache Load ---
    const char* cachePath = "tensorrt_timing_cache.bin";
    std::ifstream cacheFileRead(cachePath, std::ios::binary);
    std::vector<char> cacheData;
    if (cacheFileRead.good()) {
        cacheFileRead.seekg(0, cacheFileRead.end);
        size_t size = cacheFileRead.tellg();
        cacheFileRead.seekg(0, cacheFileRead.beg);
        cacheData.resize(size);
        cacheFileRead.read(cacheData.data(), size);
        cacheFileRead.close();
    }

    nvinfer1::ITimingCache* timingCache = nullptr;
    if (!cacheData.empty()) {
        timingCache = cfg->createTimingCache(cacheData.data(), cacheData.size());
    } else {
        timingCache = cfg->createTimingCache(nullptr, 0); // Create an empty cache
    }
    if (timingCache) {
        cfg->setTimingCache(*timingCache, false); // false: ignore discrepancies if cache is from a different TRT version or hardware
    } else {
        std::cerr << "[TensorRT] Warning: Could not create timing cache." << std::endl;
    }
    // --- End Timing Cache Load ---

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "[TensorRT] ERROR: Error parsing the ONNX file: " << onnxFile << std::endl;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    int input_res = config.onnx_input_resolution;

    profile->setDimensions(
        network->getInput(0)->getName(),
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4(1, 3, input_res, input_res)
    );
    profile->setDimensions(
        network->getInput(0)->getName(),
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4(1, 3, input_res, input_res)
    );
    profile->setDimensions(
        network->getInput(0)->getName(),
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4(1, 3, input_res, input_res)
    );

    cfg->addOptimizationProfile(profile);


    if (config.export_enable_fp16)
    {
        if (config.verbose)
            std::cout << "[TensorRT] Set FP16" << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (config.export_enable_fp8)
    {
        if (config.verbose)
            std::cout << "[TensorRT] Set FP8" << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP8);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[TensorRT] Building engine (this may take several minutes)..." << std::endl;

    auto plan = builder->buildSerializedNetwork(*network, *cfg);
    if (!plan)
    {
        std::cerr << "[TensorRT] ERROR: Could not build the engine" << std::endl;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }

    // --- Timing Cache Save ---
    if (timingCache) { // timingCache should exist if created above
        nvinfer1::IHostMemory* serializedCache = timingCache->serialize();
        if (serializedCache) {
            std::ofstream cacheFileWrite(cachePath, std::ios::binary);
            if (cacheFileWrite.good()) {
                cacheFileWrite.write(static_cast<const char*>(serializedCache->data()), serializedCache->size());
                cacheFileWrite.close();
                if (config.verbose) {
                    std::cout << "[TensorRT] Timing cache saved to: " << cachePath << std::endl;
                }
            } else {
                std::cerr << "[TensorRT] ERROR: Could not open timing cache file for writing: " << cachePath << std::endl;
            }
            delete serializedCache;
        }
    }
    // Note: TensorRT IBuilderConfig takes ownership of the ITimingCache object set via setTimingCache.
    // So, we don't need to explicitly delete timingCache here if it was successfully set.
    // If createTimingCache failed or was not set, and we allocated it, we would need to manage its lifecycle.
    // However, createTimingCache is called on cfg, and setTimingCache also on cfg.
    // If timingCache was created but not set (e.g. setTimingCache failed, though unlikely),
    // then builder->destroyTimingCache(timingCache) might be needed if it wasn't managed by cfg.
    // Given the current logic, cfg should manage it.
    // --- End Timing Cache Save ---

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!engine)
    {
        std::cerr << "[TensorRT] ERROR: Could not create engine" << std::endl;
        delete plan;
        delete runtime;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }

    nvinfer1::IHostMemory* serializedModel = engine->serialize();
    std::string engineFile = onnxFile.substr(0, onnxFile.find_last_of('.')) + ".engine";
    std::ofstream p(engineFile, std::ios::binary);
    if (!p)
    {
        std::cerr << "[TensorRT] ERROR: Could not open file to write: " << engineFile << std::endl;
        delete serializedModel;
        delete engine;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }
    p.write(static_cast<const char*>(plan->data()), plan->size());
    p.close();

    delete plan;
    delete runtime;
    delete parser;
    delete network;
    delete cfg;
    delete builder;

    std::cout << "[TensorRT] The engine was built and saved to the file: " << engineFile << std::endl;
    return engine;
}