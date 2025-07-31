#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <cstring>
#include "cuda_error_check.h"

// Simple GPU matrix class to replace cv::cuda::GpuMat
class SimpleCudaMat {
public:
    SimpleCudaMat() : data_(nullptr), width_(0), height_(0), channels_(0), step_(0) {}
    
    SimpleCudaMat(int height, int width, int channels) 
        : width_(width), height_(height), channels_(channels) {
        allocate();
    }
    
    ~SimpleCudaMat() {
        release();
    }
    
    // Move constructor
    SimpleCudaMat(SimpleCudaMat&& other) noexcept 
        : data_(other.data_), width_(other.width_), height_(other.height_), 
          channels_(other.channels_), step_(other.step_) {
        other.data_ = nullptr;
        other.width_ = other.height_ = other.channels_ = 0;
        other.step_ = 0;
    }
    
    // Move assignment
    SimpleCudaMat& operator=(SimpleCudaMat&& other) noexcept {
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
    SimpleCudaMat(const SimpleCudaMat&) = delete;
    SimpleCudaMat& operator=(const SimpleCudaMat&) = delete;
    
    // Create with specific dimensions
    void create(int height, int width, int channels) {
        if (height != height_ || width != width_ || channels != channels_) {
            release();
            width_ = width;
            height_ = height;
            channels_ = channels;
            allocate();
        }
    }
    
    // Release GPU memory
    void release() {
        if (data_) {
            CUDA_CHECK_WARN(cudaFree(data_));
            data_ = nullptr;
        }
        width_ = height_ = channels_ = 0;
        step_ = 0;
    }
    
    // Check if empty
    bool empty() const {
        return data_ == nullptr || width_ == 0 || height_ == 0;
    }
    
    // Get properties
    int rows() const { return height_; }
    int cols() const { return width_; }
    int channels() const { return channels_; }
    size_t step() const { return step_; }
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    
    // Get size in bytes
    size_t sizeInBytes() const {
        return step_ * height_;
    }
    
    // Copy from host memory
    void upload(const void* hostData, size_t hostStep = 0) {
        if (empty()) return;
        
        if (hostStep == 0) {
            hostStep = width_ * channels_;
        }
        
        if (hostStep == step_) {
            // Simple copy
            CUDA_CHECK(cudaMemcpy(data_, hostData, sizeInBytes(), cudaMemcpyHostToDevice));
        } else {
            // Row-by-row copy
            const uint8_t* src = static_cast<const uint8_t*>(hostData);
            uint8_t* dst = data_;
            for (int y = 0; y < height_; ++y) {
                CUDA_CHECK(cudaMemcpy(dst, src, width_ * channels_, cudaMemcpyHostToDevice));
                src += hostStep;
                dst += step_;
            }
        }
    }
    
    // Copy to host memory
    void download(void* hostData, size_t hostStep = 0) const {
        if (empty()) return;
        
        if (hostStep == 0) {
            hostStep = width_ * channels_;
        }
        
        if (hostStep == step_) {
            // Simple copy
            CUDA_CHECK(cudaMemcpy(hostData, data_, sizeInBytes(), cudaMemcpyDeviceToHost));
        } else {
            // Row-by-row copy
            uint8_t* dst = static_cast<uint8_t*>(hostData);
            const uint8_t* src = data_;
            for (int y = 0; y < height_; ++y) {
                CUDA_CHECK(cudaMemcpy(dst, src, width_ * channels_, cudaMemcpyDeviceToHost));
                dst += hostStep;
                src += step_;
            }
        }
    }
    
    // Async versions with stream
    void uploadAsync(const void* hostData, cudaStream_t stream, size_t hostStep = 0) {
        if (empty()) return;
        
        if (hostStep == 0) {
            hostStep = width_ * channels_;
        }
        
        if (hostStep == step_) {
            cudaMemcpyAsync(data_, hostData, sizeInBytes(), cudaMemcpyHostToDevice, stream);
        } else {
            const uint8_t* src = static_cast<const uint8_t*>(hostData);
            uint8_t* dst = data_;
            for (int y = 0; y < height_; ++y) {
                cudaMemcpyAsync(dst, src, width_ * channels_, cudaMemcpyHostToDevice, stream);
                src += hostStep;
                dst += step_;
            }
        }
    }
    
    void downloadAsync(void* hostData, cudaStream_t stream, size_t hostStep = 0) const {
        if (empty()) return;
        
        if (hostStep == 0) {
            hostStep = width_ * channels_;
        }
        
        if (hostStep == step_) {
            cudaMemcpyAsync(hostData, data_, sizeInBytes(), cudaMemcpyDeviceToHost, stream);
        } else {
            uint8_t* dst = static_cast<uint8_t*>(hostData);
            const uint8_t* src = data_;
            for (int y = 0; y < height_; ++y) {
                cudaMemcpyAsync(dst, src, width_ * channels_, cudaMemcpyDeviceToHost, stream);
                dst += hostStep;
                src += step_;
            }
        }
    }
    
    // Clone (deep copy)
    SimpleCudaMat clone() const {
        SimpleCudaMat result(height_, width_, channels_);
        if (!empty()) {
            CUDA_CHECK(cudaMemcpy(result.data_, data_, sizeInBytes(), cudaMemcpyDeviceToDevice));
        }
        return result;
    }
    
    // Copy from another SimpleCudaMat
    void copyFrom(const SimpleCudaMat& other) {
        if (other.empty()) {
            release();
            return;
        }
        
        create(other.height_, other.width_, other.channels_);
        
        // Handle different step sizes properly
        if (step_ == other.step_) {
            // Simple copy when steps match
            CUDA_CHECK(cudaMemcpy(data_, other.data_, sizeInBytes(), cudaMemcpyDeviceToDevice));
        } else {
            // Row-by-row copy when steps differ
            size_t copy_width = width_ * channels_;
            for (int y = 0; y < height_; ++y) {
                CUDA_CHECK(cudaMemcpy(data_ + y * step_, 
                                     other.data_ + y * other.step_, 
                                     copy_width, 
                                     cudaMemcpyDeviceToDevice));
            }
        }
    }
    
    // Set to zero
    void setZero() {
        if (!empty()) {
            CUDA_CHECK(cudaMemset(data_, 0, sizeInBytes()));
        }
    }
    
private:
    void allocate() {
        if (width_ > 0 && height_ > 0 && channels_ > 0) {
            // Align step to 32 bytes for better performance
            step_ = ((width_ * channels_ + 31) / 32) * 32;
            size_t allocSize = step_ * height_;
            
            // Check available memory before allocation
            size_t free_mem, total_mem;
            if (!checkGpuMemory(allocSize, free_mem, total_mem)) {
                throw std::runtime_error("Insufficient GPU memory for allocation");
            }
            
            cudaError_t err = cudaMalloc(&data_, allocSize);
            if (err != cudaSuccess) {
                step_ = 0;
                throw std::runtime_error("GPU allocation failed: " + getCudaErrorDescription(err));
            }
        }
    }
    
    uint8_t* data_;
    int width_;
    int height_;
    int channels_;
    size_t step_;  // Aligned row size in bytes
};

// Simple CPU matrix class to replace cv::Mat
class SimpleMat {
public:
    SimpleMat() : data_(nullptr), width_(0), height_(0), channels_(0), step_(0) {}
    
    SimpleMat(int height, int width, int channels) 
        : width_(width), height_(height), channels_(channels) {
        allocate();
    }
    
    ~SimpleMat() {
        release();
    }
    
    // Move constructor
    SimpleMat(SimpleMat&& other) noexcept 
        : data_(std::move(other.data_)), width_(other.width_), height_(other.height_), 
          channels_(other.channels_), step_(other.step_) {
        other.width_ = other.height_ = other.channels_ = 0;
        other.step_ = 0;
    }
    
    // Move assignment
    SimpleMat& operator=(SimpleMat&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            step_ = other.step_;
            other.width_ = other.height_ = other.channels_ = 0;
        other.step_ = 0;
        }
        return *this;
    }
    
    // Copy constructor
    SimpleMat(const SimpleMat& other) 
        : width_(other.width_), height_(other.height_), channels_(other.channels_), step_(other.step_),
          external_data_(other.external_data_) {
        if (other.data_) {
            allocate();
            std::memcpy(data_.get(), other.data_.get(), step_ * height_);
        }
        // If copying external data, don't copy the actual data, just the pointer
    }
    
    // Copy assignment
    SimpleMat& operator=(const SimpleMat& other) {
        if (this != &other) {
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            step_ = other.step_;
            external_data_ = other.external_data_;
            
            if (other.data_) {
                allocate();
                std::memcpy(data_.get(), other.data_.get(), step_ * height_);
            } else {
                data_.reset();
            }
        }
        return *this;
    }
    
    // Create with specific dimensions
    void create(int height, int width, int channels) {
        if (height != height_ || width != width_ || channels != channels_) {
            width_ = width;
            height_ = height;
            channels_ = channels;
            allocate();
        }
    }
    
    // Constructor from raw data (does not copy, just wraps)
    SimpleMat(int height, int width, int channels, void* data, size_t step = 0) 
        : width_(width), height_(height), channels_(channels), step_(step), 
          data_(nullptr), external_data_(static_cast<uint8_t*>(data)) {
        if (step_ == 0) {
            step_ = width_ * channels_;
        }
        // Note: This constructor doesn't own the data
    }
    
    void release() {
        data_.reset();
        external_data_ = nullptr;
        width_ = height_ = channels_ = 0;
        step_ = 0;
    }
    
    bool empty() const {
        // Check if we have any data (either owned or external) and valid dimensions
        bool hasData = (data_ != nullptr) || (external_data_ != nullptr);
        bool validDimensions = (width_ > 0) && (height_ > 0) && (channels_ > 0);
        return !hasData || !validDimensions;
    }
    
    int rows() const { return height_; }
    int cols() const { return width_; }
    int channels() const { return channels_; }
    size_t step() const { return step_; }
    
    uint8_t* data() { 
        return external_data_ ? external_data_ : data_.get(); 
    }
    
    const uint8_t* data() const { 
        return external_data_ ? external_data_ : data_.get(); 
    }
    
    size_t sizeInBytes() const {
        return step_ * height_;
    }
    
    // Clone (deep copy)
    SimpleMat clone() const {
        SimpleMat result(height_, width_, channels_);
        if (!empty()) {
            const uint8_t* src = data();
            std::memcpy(result.data_.get(), src, sizeInBytes());
        }
        return result;
    }
    
    // Set to zero
    void setZero() {
        if (!empty()) {
            std::memset(data(), 0, sizeInBytes());
        }
    }
    
    // Get pixel value (for single channel)
    uint8_t at(int y, int x) const {
        return data()[y * step_ + x * channels_];
    }
    
    // Get pixel value (for multi-channel)
    uint8_t at(int y, int x, int c) const {
        return data()[y * step_ + x * channels_ + c];
    }
    
    // Set pixel value (for single channel)
    void set(int y, int x, uint8_t value) {
        data()[y * step_ + x * channels_] = value;
    }
    
    // Set pixel value (for multi-channel)
    void set(int y, int x, int c, uint8_t value) {
        data()[y * step_ + x * channels_ + c] = value;
    }
    
private:
    void allocate() {
        if (width_ > 0 && height_ > 0 && channels_ > 0) {
            step_ = width_ * channels_;
            data_.reset(new uint8_t[step_ * height_]);
        }
    }
    
    std::unique_ptr<uint8_t[]> data_;
    uint8_t* external_data_ = nullptr;  // For wrapping external data
    int width_;
    int height_;
    int channels_;
    size_t step_;
};