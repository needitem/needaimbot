#include "engine_exporter.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>

EngineExporter::EngineExporter(const ExportConfig& config) 
    : m_config(config), m_logger(config.verbose) {
}

EngineExporter::~EngineExporter() = default;

bool EngineExporter::exportEngine() {
    std::cout << "Starting ONNX to TensorRT engine conversion...\n";
    
    if (!validateInputFile()) {
        return false;
    }
    
    if (!validateOutputPath()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create TensorRT builder
    m_builder.reset(nvinfer1::createInferBuilder(m_logger));
    if (!m_builder) {
        std::cerr << "Error: Failed to create TensorRT builder\n";
        return false;
    }
    
    if (!loadOnnxModel()) {
        return false;
    }
    
    printModelInfo();
    
    if (!buildEngine()) {
        return false;
    }
    
    if (!saveEngine()) {
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nEngine conversion completed successfully!\n";
    std::cout << "Output: " << m_config.get_output_path() << "\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
    
    return true;
}

bool EngineExporter::validateInputFile() {
    if (!std::filesystem::exists(m_config.input_onnx_path)) {
        std::cerr << "Error: Input ONNX file does not exist: " << m_config.input_onnx_path << "\n";
        return false;
    }
    
    std::string extension = std::filesystem::path(m_config.input_onnx_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension != ".onnx") {
        std::cerr << "Error: Input file must be an ONNX model (.onnx extension)\n";
        return false;
    }
    
    return true;
}

bool EngineExporter::validateOutputPath() {
    std::string output_path = m_config.get_output_path();
    std::filesystem::path output_dir = std::filesystem::path(output_path).parent_path();
    
    if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
        try {
            std::filesystem::create_directories(output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Error: Cannot create output directory: " << e.what() << "\n";
            return false;
        }
    }
    
    return true;
}

bool EngineExporter::loadOnnxModel() {
    std::cout << "Loading ONNX model: " << m_config.input_onnx_path << "\n";
    
    // Create network
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_network.reset(m_builder->createNetworkV2(explicit_batch));
    if (!m_network) {
        std::cerr << "Error: Failed to create TensorRT network\n";
        return false;
    }
    
    // Create ONNX parser
    m_parser.reset(nvonnxparser::createParser(*m_network, m_logger));
    if (!m_parser) {
        std::cerr << "Error: Failed to create ONNX parser\n";
        return false;
    }
    
    // Parse ONNX file
    if (!m_parser->parseFromFile(m_config.input_onnx_path.c_str(), 
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Error: Failed to parse ONNX file\n";
        return false;
    }
    
    return true;
}

void EngineExporter::printModelInfo() {
    if (!m_network) return;
    
    std::cout << "\nModel Information:\n";
    std::cout << "  Inputs: " << m_network->getNbInputs() << "\n";
    std::cout << "  Outputs: " << m_network->getNbOutputs() << "\n";
    
    for (int i = 0; i < m_network->getNbInputs(); ++i) {
        auto input = m_network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "  Input " << i << ": " << input->getName() << " [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << "]\n";
    }
}

bool EngineExporter::buildEngine() {
    std::cout << "\nBuilding TensorRT engine...\n";
    
    // Create builder config
    m_builderConfig.reset(m_builder->createBuilderConfig());
    if (!m_builderConfig) {
        std::cerr << "Error: Failed to create builder config\n";
        return false;
    }
    
    setupBuilderConfig();
    setupOptimizationProfile();
    
    // Build engine
    m_engine.reset(m_builder->buildEngineWithConfig(*m_network, *m_builderConfig));
    if (!m_engine) {
        std::cerr << "Error: Failed to build TensorRT engine\n";
        return false;
    }
    
    std::cout << "Engine built successfully\n";
    return true;
}

void EngineExporter::setupBuilderConfig() {
    // Set workspace size
    size_t workspace_size = static_cast<size_t>(m_config.workspace_mb) * 1024 * 1024;
    m_builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    
    std::cout << "Configuration:\n";
    std::cout << "  Workspace size: " << m_config.workspace_mb << " MB\n";
    
    // Precision settings
    if (m_config.enable_fp16 && m_builder->platformHasFastFp16()) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "  FP16 precision: Enabled\n";
    } else if (m_config.enable_fp16) {
        std::cout << "  FP16 precision: Requested but not supported on this platform\n";
    }
    
    if (m_config.enable_fp8) {
        // FP8 support check - may not be available in all TensorRT versions
        try {
            m_builderConfig->setFlag(nvinfer1::BuilderFlag::kFP8);
            std::cout << "  FP8 precision: Enabled\n";
        } catch (...) {
            std::cout << "  FP8 precision: Requested but not supported in this TensorRT version\n";
        }
    }
    
    // ========== 에임봇 최고 속도 최적화 플래그 ==========
    
    // INT8 정밀도 (cache-driven; or QAT if assume_qat_quantized)
    if (m_config.enable_int8 && m_builder->platformHasFastInt8()) {
        // 1) QAT 경로: ONNX에 Q/DQ가 있다고 가정 → 바로 INT8 활성화
        if (m_config.assume_qat_quantized) {
            m_builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
            std::cout << "  INT8 precision: Enabled (assume QAT)\n";
        }
        // 2) Calibration cache 경로: 데이터 없이 캐시만 사용
        else if (!m_config.int8_calib_cache.empty()) {
            class CacheOnlyCalibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
            public:
                explicit CacheOnlyCalibrator(const std::string& cachePath, int batch)
                    : m_cachePath(cachePath), m_batchSize(batch) {
                    std::ifstream f(m_cachePath, std::ios::binary);
                    if (f) {
                        m_cache.assign(std::istreambuf_iterator<char>(f), {});
                    }
                }
                int getBatchSize() const noexcept override { return m_batchSize; }
                bool getBatch(void*[], const char*[], int) noexcept override { return false; }
                const void* readCalibrationCache(size_t& length) noexcept override {
                    if (m_cache.empty()) { length = 0; return nullptr; }
                    length = m_cache.size();
                    return m_cache.data();
                }
                void writeCalibrationCache(const void* cache, size_t length) noexcept override {
                    try {
                        std::ofstream f(m_cachePath, std::ios::binary);
                        f.write(reinterpret_cast<const char*>(cache), static_cast<std::streamsize>(length));
                        f.close();
                    } catch (...) {}
                }
            private:
                std::string m_cachePath;
                int m_batchSize;
                std::vector<char> m_cache;
            };
            if (std::filesystem::exists(m_config.int8_calib_cache)) {
                m_int8Calibrator.reset(new CacheOnlyCalibrator(m_config.int8_calib_cache, m_config.calib_batch_size));
                m_builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
                m_builderConfig->setInt8Calibrator(m_int8Calibrator.get());
                std::cout << "  INT8 precision: Enabled (cache)\n";
            } else {
                std::cout << "  INT8 requested but no calibration cache found: " << m_config.int8_calib_cache << "\n";
                std::cout << "  -> Provide a valid cache or enable 'Assume QAT' for Q/DQ models.\n";
            }
        }
        // 3) Neither QAT nor cache → 안내만 출력
        else {
            std::cout << "  INT8 requested but neither QAT nor calibration cache provided. Skipping INT8.\n";
        }
    }
    
    // 1. 기본 최적화 플래그
    if (m_config.enable_gpu_fallback) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }
    
    if (m_config.enable_precision_constraints) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }
    
    // 2. 레이턴시 최적화 (속도 최우선)
    if (m_config.disable_timing_cache) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
    }
    m_builderConfig->setBuilderOptimizationLevel(m_config.optimization_level);
    
    // 3. 커널 선택 최적화
    if (m_config.enable_tf32 && m_builder->platformHasTf32()) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);
    }
    
    if (m_config.enable_sparse_weights) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    }
    
    if (m_config.enable_refit) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kREFIT);
    }
    
    if (m_config.enable_direct_io) {
        m_builderConfig->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
    }
    
    // 7. Tactic Sources 설정 (UI에서 선택 가능)
    uint32_t tacticSources = 0;
    if (m_config.use_cublas) {
        tacticSources |= 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS);
    }
    if (m_config.use_cublas_lt) {
        tacticSources |= 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT);
    }
    if (m_config.use_cudnn) {
        tacticSources |= 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN);
    }
    if (m_config.use_edge_mask_conv) {
        tacticSources |= 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS);
    }
    
    if (tacticSources != 0) {
        m_builderConfig->setTacticSources(tacticSources);
    }
    
    // 8. 추가 최적화 힌트 - TensorRT 10 API
    m_builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(m_config.workspace_mb) * 1024 * 1024);
    
    std::cout << "  === Performance Optimization Flags Applied ===\n";
    if (m_config.enable_int8) std::cout << "  - INT8: Enabled\n";
    if (m_config.enable_tf32) std::cout << "  - TF32: Enabled\n";
    if (m_config.enable_sparse_weights) std::cout << "  - Sparse Weights: Enabled\n";
    if (m_config.enable_direct_io) std::cout << "  - Direct I/O: Enabled\n";
    if (m_config.enable_refit) std::cout << "  - REFIT: Enabled\n";
    if (m_config.disable_timing_cache) std::cout << "  - Timing Cache: Disabled\n";
    std::cout << "  - Optimization Level: " << m_config.optimization_level << "\n";
    std::cout << "  - Tactic Sources: ";
    if (m_config.use_cublas) std::cout << "CUBLAS ";
    if (m_config.use_cublas_lt) std::cout << "CUBLAS_LT ";
    if (m_config.use_cudnn) std::cout << "CUDNN ";
    if (m_config.use_edge_mask_conv) std::cout << "EDGE_MASK ";
    std::cout << "\n";
    
    // Profiling
    if (m_config.enable_detailed_profiling) {
        m_builderConfig->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    }
}

void EngineExporter::setupOptimizationProfile() {
    if (m_network->getNbInputs() == 0) return;
    
    // Create optimization profile for dynamic inputs
    auto profile = m_builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Warning: Failed to create optimization profile\n";
        return;
    }
    
    // Set up profile for the first input (assuming it's the main input)
    auto input = m_network->getInput(0);
    const char* inputName = input->getName();
    nvinfer1::Dims inputDims = input->getDimensions();
    
    // For YOLO models or similar, typically [batch, channels, height, width]
    if (inputDims.nbDims == 4) {
        int resolution = m_config.input_resolution;
        nvinfer1::Dims dims{4, {1, 3, resolution, resolution}};
        
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
        
        std::cout << "  Input resolution: " << resolution << "x" << resolution << "\n";
    }
    
    // NMS 출력 고정 크기 설정 (NMS가 모델에 포함된 경우)
    if (m_config.fix_nms_output) {
        // 모든 출력을 확인하여 NMS 출력 찾기
        for (int i = 0; i < m_network->getNbOutputs(); ++i) {
            auto output = m_network->getOutput(i);
            const char* outputName = output->getName();
            nvinfer1::Dims outputDims = output->getDimensions();
            
            // NMS 출력은 보통 [batch, num_detections, 6] 형태
            // 동적 차원(-1)을 고정 크기로 설정
            if (outputDims.nbDims == 3 && outputDims.d[2] == 6) {
                // 사용자가 설정한 크기로 고정
                nvinfer1::Dims fixedDims{3, {1, m_config.nms_max_detections, 6}};
                
                profile->setDimensions(outputName, nvinfer1::OptProfileSelector::kMIN, fixedDims);
                profile->setDimensions(outputName, nvinfer1::OptProfileSelector::kOPT, fixedDims);
                profile->setDimensions(outputName, nvinfer1::OptProfileSelector::kMAX, fixedDims);
                
                std::cout << "  NMS output fixed to: [1, " << m_config.nms_max_detections << ", 6]\n";
            }
        }
    }
    
    m_builderConfig->addOptimizationProfile(profile);
}

bool EngineExporter::saveEngine() {
    std::cout << "Saving engine to: " << m_config.get_output_path() << "\n";
    
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(m_engine->serialize());
    if (!serializedEngine) {
        std::cerr << "Error: Failed to serialize engine\n";
        return false;
    }
    
    std::ofstream engineFile(m_config.get_output_path(), std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error: Cannot create output file: " << m_config.get_output_path() << "\n";
        return false;
    }
    
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
    
    if (!engineFile.good()) {
        std::cerr << "Error: Failed to write engine file\n";
        return false;
    }
    
    // Print file size
    auto file_size = std::filesystem::file_size(m_config.get_output_path());
    std::cout << "Engine file size: " << (file_size / 1024.0 / 1024.0) << " MB\n";
    
    return true;
}