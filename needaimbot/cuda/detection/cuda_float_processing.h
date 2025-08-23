#pragma once

#include <cuda_runtime.h>
#include "../simple_cuda_mat.h"
#include "../cuda_resource_manager.h"

// Simple float GPU matrix class
class SimpleCudaMatFloat {
public:
    SimpleCudaMatFloat() : data_(nullptr), width_(0), height_(0), channels_(0), step_(0) {}
    
    SimpleCudaMatFloat(int height, int width, int channels) 
        : width_(width), height_(height), channels_(channels) {
        allocate();
    }
    
    ~SimpleCudaMatFloat() {
        release();
    }
    
    // Move constructor and assignment
    SimpleCudaMatFloat(SimpleCudaMatFloat&& other) noexcept 
        : data_(other.data_), width_(other.width_), height_(other.height_), 
          channels_(other.channels_), step_(other.step_) {
        other.data_ = nullptr;
        other.width_ = other.height_ = other.channels_ = 0;
        other.step_ = 0;
    }
    
    SimpleCudaMatFloat& operator=(SimpleCudaMatFloat&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            step_ = other.step_;
            other.data_ = nullptr;
            other.width_ = other.height_ = other.channels_ = 0;
        other.step_ = 0;
        }
        return *this;
    }
    
    // Delete copy constructor and assignment
    SimpleCudaMatFloat(const SimpleCudaMatFloat&) = delete;
    SimpleCudaMatFloat& operator=(const SimpleCudaMatFloat&) = delete;
    
    void create(int height, int width, int channels) {
        if (height != height_ || width != width_ || channels != channels_) {
            release();
            width_ = width;
            height_ = height;
            channels_ = channels;
            allocate();
        }
    }
    
    void release() {
        if (data_) {
            // Check if resource manager is shutting down
            if (CudaResourceManager::GetInstance().IsShuttingDown()) {
                data_ = nullptr;
                width_ = height_ = channels_ = 0;
                step_ = 0;
                return;
            }
            
            // Unregister from resource manager before freeing
            CudaResourceManager::GetInstance().UnregisterMemory(data_);
            cudaFree(data_);
            data_ = nullptr;
        }
        width_ = height_ = channels_ = 0;
        step_ = 0;
    }
    
    bool empty() const {
        return data_ == nullptr || width_ == 0 || height_ == 0;
    }
    
    int rows() const { return height_; }
    int cols() const { return width_; }
    int channels() const { return channels_; }
    size_t step() const { return step_; }
    float* data() { return data_; }
    const float* data() const { return data_; }
    
    size_t sizeInBytes() const {
        return step_ * height_ * sizeof(float);
    }
    
private:
    void allocate() {
        if (width_ > 0 && height_ > 0 && channels_ > 0) {
            // Align step to 32 floats (128 bytes) for better performance
            step_ = ((width_ * channels_ + 31) / 32) * 32;
            cudaError_t err = cudaMalloc(&data_, step_ * height_ * sizeof(float));
            
            // Register with resource manager for tracking
            if (err == cudaSuccess && data_) {
                CudaResourceManager::GetInstance().RegisterMemory(data_);
            }
        }
    }
    
    float* data_;
    int width_;
    int height_;
    int channels_;
    size_t step_;  // Step in number of floats, not bytes
};

namespace CudaFloatProcessing {

// Convert uint8 to float with scaling
void convertToFloat(const SimpleCudaMat& src, SimpleCudaMatFloat& dst, float scale, float shift, cudaStream_t stream = 0);

// Split float channels
void splitFloat(const SimpleCudaMatFloat& src, SimpleCudaMatFloat* channels, cudaStream_t stream = 0);

} // namespace CudaFloatProcessing