#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_H

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
private:
    // Calibration data
    std::vector<std::string> m_imageFiles;
    int m_batchSize;
    int m_currentBatch;
    int m_maxBatches;
    
    // Image dimensions
    int m_inputH;
    int m_inputW;
    int m_inputC;
    
    // GPU buffers
    void* m_deviceInput;
    std::vector<uint8_t> m_batchData;
    
    // Calibration cache
    std::string m_cacheFileName;
    std::vector<char> m_calibrationCache;
    bool m_useCache;
    
    // CUDA stream for async operations
    cudaStream_t m_stream;
    
    // Load and preprocess a batch of images
    bool loadBatch();
    
public:
    Int8Calibrator(const std::string& calibDataDir,
                   int batchSize,
                   int maxBatches,
                   int inputH,
                   int inputW,
                   int inputC = 3,
                   const std::string& cacheFile = "calibration.cache",
                   bool useCache = true);
    
    ~Int8Calibrator();
    
    // IInt8Calibrator interface
    int getBatchSize() const noexcept override { return m_batchSize; }
    
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    
    const void* readCalibrationCache(size_t& length) noexcept override;
    
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;
    
    // Helper functions
    bool imageToTensor(const std::string& imagePath, float* output, int h, int w, int c);
    void preprocessImage(const uint8_t* img, float* output, int h, int w, int c);
};

#endif // INT8_CALIBRATOR_H