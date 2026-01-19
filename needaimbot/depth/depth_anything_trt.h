#pragma once
#ifdef USE_CUDA

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

namespace depth_anything {

// OpenCV colormap types (subset)
enum ColormapTypes {
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11,
    COLORMAP_PARULA = 12,
    COLORMAP_MAGMA = 13,
    COLORMAP_INFERNO = 14,
    COLORMAP_PLASMA = 15,
    COLORMAP_VIRIDIS = 16,
    COLORMAP_CIVIDIS = 17,
    COLORMAP_TWILIGHT = 18,
    COLORMAP_TWILIGHT_SHIFTED = 19,
    COLORMAP_TURBO = 20,
    COLORMAP_DEEPGREEN = 21
};

class DepthAnythingTrt {
public:
    DepthAnythingTrt();
    ~DepthAnythingTrt();

    // Initialize with TensorRT engine file
    bool initialize(const std::string& modelPath, nvinfer1::ILogger& logger);
    
    // Run depth prediction on RGB image
    // Input: RGB uint8 data (height * width * 3)
    // Output: normalized depth map (0-255 uint8, same size as input)
    bool predictDepth(const uint8_t* inputRgb, int width, int height, 
                      uint8_t* outputDepth, int& outWidth, int& outHeight);
    
    // Alternative: predict from GPU buffer
    bool predictDepthGpu(const uint8_t* d_inputRgba, int width, int height, int inputPitch,
                         uint8_t* d_outputDepth, int& outWidth, int& outHeight, cudaStream_t stream);
    
    void setColormap(int type);
    int colormapType() const;
    bool ready() const;
    const std::string& lastError() const;
    void reset();

    int getInputWidth() const { return input_w; }
    int getInputHeight() const { return input_h; }

private:
    static constexpr int kMinInputSize = 160;
    static constexpr int kMaxInputSize = 640;
    static constexpr int kOptInputSize = 224;

    int input_w;
    int input_h;
    int min_input_size;
    int max_input_size;
    bool dynamic_input;
    
    // ImageNet normalization (BGR order)
    float mean[3];  // {123.675, 116.28, 103.53}
    float std_val[3];  // {58.395, 57.12, 57.375}
    int colormap_type;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    void* buffer[2];  // [0] = input, [1] = output
    float* d_input_float;  // Preprocessed float input on GPU
    float* d_output_float; // Raw float output on GPU
    std::vector<float> h_depth_data;  // Host buffer for depth
    cudaStream_t stream;

    bool initialized;
    std::string last_error;

    // Helper methods
    bool loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger);
    bool buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& onnxPath);
    int selectInputSize(int width, int height) const;
    bool setInputShape(int w, int h);
    
    // Preprocessing: resize and normalize RGB to float CHW
    bool preprocess(const uint8_t* inputRgb, int srcWidth, int srcHeight);
    bool preprocessGpu(const uint8_t* d_inputRgba, int srcWidth, int srcHeight, int srcPitch, cudaStream_t stream);
    
    // Postprocessing: normalize depth to 0-255
    bool postprocess(uint8_t* outputDepth, int dstWidth, int dstHeight);
};

} // namespace depth_anything

#endif // USE_CUDA
