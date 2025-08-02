// Windows headers must come first
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>

#include "int8_calibrator.h"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <random>

#pragma comment(lib, "gdiplus.lib")

Int8Calibrator::Int8Calibrator(const std::string& calibDataDir,
                               int batchSize,
                               int maxBatches,
                               int inputH,
                               int inputW,
                               int inputC,
                               const std::string& cacheFile,
                               bool useCache)
    : m_batchSize(batchSize),
      m_currentBatch(0),
      m_maxBatches(maxBatches),
      m_inputH(inputH),
      m_inputW(inputW),
      m_inputC(inputC),
      m_deviceInput(nullptr),
      m_cacheFileName(cacheFile),
      m_useCache(useCache),
      m_stream(nullptr) {
    
    // Initialize GDI+
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
    
    // Create CUDA stream
    cudaStreamCreate(&m_stream);
    
    // Allocate device memory for one batch
    size_t inputSize = m_batchSize * m_inputC * m_inputH * m_inputW * sizeof(float);
    cudaMalloc(&m_deviceInput, inputSize);
    
    // Allocate host memory for batch data
    m_batchData.resize(inputSize);
    
    // Collect all image files from the calibration directory
    std::filesystem::path calibPath(calibDataDir);
    if (std::filesystem::exists(calibPath) && std::filesystem::is_directory(calibPath)) {
        for (const auto& entry : std::filesystem::directory_iterator(calibPath)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                m_imageFiles.push_back(entry.path().string());
            }
        }
        
        // Shuffle images for better calibration
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(m_imageFiles.begin(), m_imageFiles.end(), g);
        
        std::cout << "[Calibrator] Found " << m_imageFiles.size() << " calibration images" << std::endl;
    } else {
        std::cerr << "[Calibrator] Calibration directory not found: " << calibDataDir << std::endl;
    }
    
    // Load calibration cache if exists and enabled
    if (m_useCache && std::filesystem::exists(m_cacheFileName)) {
        std::ifstream cacheFile(m_cacheFileName, std::ios::binary);
        if (cacheFile.is_open()) {
            cacheFile.seekg(0, std::ios::end);
            size_t cacheSize = cacheFile.tellg();
            cacheFile.seekg(0, std::ios::beg);
            m_calibrationCache.resize(cacheSize);
            cacheFile.read(m_calibrationCache.data(), cacheSize);
            cacheFile.close();
            std::cout << "[Calibrator] Loaded calibration cache from " << m_cacheFileName << std::endl;
        }
    }
}

Int8Calibrator::~Int8Calibrator() {
    if (m_deviceInput) {
        cudaFree(m_deviceInput);
    }
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
}

bool Int8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (m_currentBatch >= m_maxBatches || m_currentBatch * m_batchSize >= static_cast<int>(m_imageFiles.size())) {
        return false;
    }
    
    // Load batch of images
    if (!loadBatch()) {
        return false;
    }
    
    // Copy to device
    size_t inputSize = m_batchSize * m_inputC * m_inputH * m_inputW * sizeof(float);
    cudaMemcpyAsync(m_deviceInput, m_batchData.data(), inputSize, cudaMemcpyHostToDevice, m_stream);
    cudaStreamSynchronize(m_stream);
    
    // Set binding
    bindings[0] = m_deviceInput;
    
    m_currentBatch++;
    
    if (m_currentBatch % 10 == 0) {
        std::cout << "[Calibrator] Processing batch " << m_currentBatch << "/" << m_maxBatches << std::endl;
    }
    
    return true;
}

bool Int8Calibrator::loadBatch() {
    float* batchDataFloat = reinterpret_cast<float*>(m_batchData.data());
    
    for (int i = 0; i < m_batchSize; i++) {
        int imgIdx = m_currentBatch * m_batchSize + i;
        if (imgIdx >= static_cast<int>(m_imageFiles.size())) {
            // Repeat images if we run out
            imgIdx = imgIdx % m_imageFiles.size();
        }
        
        float* imgData = batchDataFloat + i * m_inputC * m_inputH * m_inputW;
        if (!imageToTensor(m_imageFiles[imgIdx], imgData, m_inputH, m_inputW, m_inputC)) {
            std::cerr << "[Calibrator] Failed to load image: " << m_imageFiles[imgIdx] << std::endl;
            return false;
        }
    }
    
    return true;
}

bool Int8Calibrator::imageToTensor(const std::string& imagePath, float* output, int h, int w, int c) {
    // Convert string to wide string for GDI+
    std::wstring wImagePath(imagePath.begin(), imagePath.end());
    
    // Load image using GDI+
    Gdiplus::Bitmap* bitmap = new Gdiplus::Bitmap(wImagePath.c_str());
    if (bitmap->GetLastStatus() != Gdiplus::Ok) {
        delete bitmap;
        return false;
    }
    
    // Resize if needed
    int origW = bitmap->GetWidth();
    int origH = bitmap->GetHeight();
    
    Gdiplus::Bitmap* resizedBitmap = bitmap;
    if (origW != w || origH != h) {
        resizedBitmap = new Gdiplus::Bitmap(w, h, PixelFormat24bppRGB);
        Gdiplus::Graphics graphics(resizedBitmap);
        graphics.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
        graphics.DrawImage(bitmap, 0, 0, w, h);
        delete bitmap;
    }
    
    // Extract pixel data
    Gdiplus::BitmapData bitmapData;
    Gdiplus::Rect rect(0, 0, w, h);
    resizedBitmap->LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);
    
    uint8_t* pixels = static_cast<uint8_t*>(bitmapData.Scan0);
    int stride = bitmapData.Stride;
    
    // Convert to tensor format (RGB planar, normalized to [0, 1])
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t* pixel = pixels + y * stride + x * 3;
            // GDI+ stores as BGR, convert to RGB
            output[0 * h * w + y * w + x] = pixel[2] / 255.0f; // R
            output[1 * h * w + y * w + x] = pixel[1] / 255.0f; // G
            output[2 * h * w + y * w + x] = pixel[0] / 255.0f; // B
        }
    }
    
    resizedBitmap->UnlockBits(&bitmapData);
    delete resizedBitmap;
    
    return true;
}

void Int8Calibrator::preprocessImage(const uint8_t* img, float* output, int h, int w, int c) {
    // Simple preprocessing: normalize to [0, 1]
    int channelSize = h * w;
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < channelSize; i++) {
            output[ch * channelSize + i] = img[ch * channelSize + i] / 255.0f;
        }
    }
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    if (m_useCache && !m_calibrationCache.empty()) {
        length = m_calibrationCache.size();
        return m_calibrationCache.data();
    }
    length = 0;
    return nullptr;
}

void Int8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    if (m_useCache && cache && length > 0) {
        std::ofstream cacheFile(m_cacheFileName, std::ios::binary);
        if (cacheFile.is_open()) {
            cacheFile.write(static_cast<const char*>(cache), length);
            cacheFile.close();
            std::cout << "[Calibrator] Saved calibration cache to " << m_cacheFileName << std::endl;
        }
    }
}