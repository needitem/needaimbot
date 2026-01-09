#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "../simple_cuda_mat.h"
#include "../../utils/cuda_utils.h"

// Simple float GPU matrix class with full RAII support
class SimpleCudaMatFloat {
public:
    SimpleCudaMatFloat() : width_(0), height_(0), channels_(0), step_(0) {}
    
    SimpleCudaMatFloat(int height, int width, int channels) 
        : width_(width), height_(height), channels_(channels), step_(0) {
        allocate();
    }
    
    ~SimpleCudaMatFloat() = default;  // CudaMemory handles cleanup automatically
    
    // Move constructor
    SimpleCudaMatFloat(SimpleCudaMatFloat&& other) noexcept 
        : memory_(std::move(other.memory_)), 
          width_(other.width_), 
          height_(other.height_), 
          channels_(other.channels_), 
          step_(other.step_) {
        other.width_ = other.height_ = other.channels_ = 0;
        other.step_ = 0;
    }
    
    // Move assignment
    SimpleCudaMatFloat& operator=(SimpleCudaMatFloat&& other) noexcept {
        if (this != &other) {
            memory_ = std::move(other.memory_);
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            step_ = other.step_;
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
            width_ = width;
            height_ = height;
            channels_ = channels;
            step_ = 0;
            allocate();
        }
    }
    
    void release() {
        memory_.reset();  // CudaMemory handles cleanup automatically
        width_ = height_ = channels_ = 0;
        step_ = 0;
    }
    
    bool empty() const {
        return !memory_.get() || width_ == 0 || height_ == 0 || channels_ == 0;
    }
    
    int rows() const { return height_; }
    int cols() const { return width_; }
    int channels() const { return channels_; }
    size_t step() const { return step_; }
    float* data() { return memory_.get(); }
    const float* data() const { return memory_.get(); }
    
    size_t sizeInBytes() const {
        return step_ * height_ * sizeof(float);
    }
    
private:
    void allocate() {
        if (width_ > 0 && height_ > 0 && channels_ > 0) {
            // Align step to 32 floats (128 bytes) for better performance
            step_ = ((width_ * channels_ + 31) / 32) * 32;
            size_t total_floats = step_ * height_;
            
            try {
                memory_ = CudaMemory<float>(total_floats);
            } catch (const std::exception&) {
                // Reset on allocation failure
                width_ = height_ = channels_ = 0;
                step_ = 0;
                throw;  // Re-throw the exception
            }
        }
    }
    
    CudaMemory<float> memory_;  // RAII wrapper for CUDA memory
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