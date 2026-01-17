#pragma once

#include <string>
#include <vector>
#include <unordered_set>

struct ExportConfig {
    std::string input_onnx_path;
    std::string output_engine_path;
    int input_resolution = 320;
    bool enable_fp16 = true;  // 에임봇 최적화: FP16 기본 활성화
    bool enable_fp8 = true;  // 기본 FP8 ON (지원 GPU에서만 활성)
    bool enable_int8 = false;  // INT8 양자화
    // INT8 calibration (cache-first; data dir optional)
    std::string int8_calib_cache;      // e.g., calib.cache
    std::string int8_calib_data_dir;   // optional; not used unless a data-driven calibrator is implemented
    int calib_batch_size = 8;
    int calib_max_batches = 200;
    bool assume_qat_quantized = false; // set true if ONNX has Q/DQ (no calibrator needed)

    int workspace_mb = 2048;  // 넉넉한 워크스페이스로 더 aggressive한 커널 선택 허용
    bool verbose = false;
    
    // TensorRT optimization settings
    bool enable_gpu_fallback = true;
    bool enable_precision_constraints = false;  // 에임봇 최적화: 속도 우선
    bool enable_detailed_profiling = false;
    
    // Advanced optimization flags
    bool enable_tf32 = true;           // TF32 (Ampere+)
    bool enable_sparse_weights = true;  // 희소 가중치 최적화
    bool enable_direct_io = true;       // Direct I/O
    bool enable_refit = false;          // Refit 가능 엔진 (dynamic shapes와 호환되지 않음)
    bool disable_timing_cache = false;   // 타이밍 캐시 사용 (빌드 느림, 성능↑)
    int optimization_level = 5;         // 최적화 레벨 (1-5)
    
    // Tactic Sources
    bool use_cublas = true;
    bool use_cublas_lt = true;
    bool use_cudnn = true;
    bool use_edge_mask_conv = true;
    
    // NMS settings
    bool fix_nms_output = true;
    int nms_max_detections = 200;
    
    // Plugin settings
    std::unordered_set<std::string> selected_plugins;
    
    // Validation
    bool is_valid() const {
        return !input_onnx_path.empty();
    }
    
    // Generate output path if not specified
    std::string get_output_path() const {
        if (!output_engine_path.empty()) {
            return output_engine_path;
        }
        
        // Generate based on input path and settings
        std::string base = input_onnx_path.substr(0, input_onnx_path.find_last_of('.'));
        base += "_" + std::to_string(input_resolution);
        if (enable_fp16) base += "_fp16";
        if (enable_fp8) base += "_fp8";
        return base + ".engine";
    }
};

class ConfigParser {
public:
    static ExportConfig parseCommandLine(int argc, char* argv[]);
    static void printUsage(const std::string& program_name);
    static void printVersion();
    
private:
    static bool isOption(const std::string& arg);
    static std::string getOptionValue(const std::vector<std::string>& args, size_t& index);
};

// Available TensorRT plugins
enum class TensorRTPlugin {
    GRID_SAMPLER,
    NORMALIZE,
    SCATTERND,
    INSTANCE_NORMALIZATION,
    CLIP,
    LEAKY_RELU,
    ELU,
    SELU,
    SOFTPLUS,
    SOFTSIGN,
    HARD_SIGMOID,
    SCALED_TANH,
    THRESH_RELU,
    PRELU,
    DETECTION_OUTPUT,
    PRIOR_BOX,
    SHUFFLE_CHANNEL,
    REGION_LAYER,
    REORG_LAYER,
    NMS_ONNX,
    EFFICIENT_NMS_ONNX
};

struct PluginInfo {
    TensorRTPlugin type;
    std::string name;
    std::string description;
    bool enabled;
    bool isCustom = false; // Flag to distinguish custom plugins
};

struct CustomPluginInfo {
    std::string name;
    std::string libraryPath;
    std::string description;
    bool enabled;
    
    CustomPluginInfo() : enabled(false) {}
    CustomPluginInfo(const std::string& n, const std::string& path, const std::string& desc = "")
        : name(n), libraryPath(path), description(desc), enabled(false) {}
};

class PluginManager {
public:
    static std::vector<PluginInfo> getAvailablePlugins();
    static std::string getPluginName(TensorRTPlugin plugin);
    static std::string getPluginDescription(TensorRTPlugin plugin);
};