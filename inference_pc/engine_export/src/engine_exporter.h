#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferVersion.h>
#include <memory>
#include <string>
#include "config.h"
#include "logger.h"

class EngineExporter {
public:
    explicit EngineExporter(const ExportConfig& config);
    ~EngineExporter();

    bool exportEngine();

private:
    bool loadOnnxModel();
    bool buildEngine();
    bool saveEngine();
    bool validateInputFile();
    bool validateOutputPath();

    void setupBuilderConfig();
    void setupOptimizationProfile();
    void printModelInfo();

    ExportConfig m_config;
    TensorRTLogger m_logger;

    std::unique_ptr<nvinfer1::IBuilder> m_builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> m_network;
    std::unique_ptr<nvinfer1::IBuilderConfig> m_builderConfig;
    std::unique_ptr<nvonnxparser::IParser> m_parser;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    // TensorRT 10.x: serialized engine (buildSerializedNetwork returns IHostMemory)
    std::unique_ptr<nvinfer1::IHostMemory> m_serializedEngine;
    // Optional calibrator (cache-only) lifetime holder
#if NV_TENSORRT_MAJOR >= 10
    std::unique_ptr<nvinfer1::IInt8Calibrator> m_int8Calibrator;
#else
    std::unique_ptr<nvinfer1::IInt8Calibrator> m_int8Calibrator;
#endif
};