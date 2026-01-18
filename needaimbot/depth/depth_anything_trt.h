#pragma once
#ifdef USE_CUDA

#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

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

    bool initialize(const std::string& modelPath, nvinfer1::ILogger& logger);
    
    // Predict depth and return normalized depth map (0-255, CV_8UC1)
    bool predictDepth(const float* inputRgb, int width, int height, 
                      float* outputDepth, int& outWidth, int& outHeight);
    
    void setColormap(int type);
    int colormapType() const;
    bool ready() const;
    const std::string& lastError() const;
    void reset();

    int getInputWidth() const { return input_w; }
    int getInputHeight() const { return input_h; }

private:
    int input_w;
    int input_h;
    int min_input_size;
    int max_input_size;
    bool dynamic_input;
    float mean[3];
    float std_val[3];
    int colormap_type;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    void* buffer[2];
    std::vector<float> depth_data;
    cudaStream_t stream;

    bool initialized;
    std::string last_error;

    bool loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger);
    bool setInputShape(int w, int h);
    int selectInputSize(int width, int height) const;
};

} // namespace depth_anything

#endif // USE_CUDA
