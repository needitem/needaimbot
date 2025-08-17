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
        : data_(nullptr), width_(width), height_(height), channels_(channels), step_(0) {
        if (width_ > 0 && height_ > 0 && channels_ > 0) {
            try {
                allocate();
            } catch (const std::exception& e) {
                // Reset on allocation failure
                data_ = nullptr;
                width_ = 0;
                height_ = 0;
                channels_ = 0;
                step_ = 0;
                std::cerr << "[SimpleCudaMat] Allocation failed: " << e.what() << std::endl;
            }
        }
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
            // Save other's data first before releasing ours
            uint8_t* temp_data = other.data_;
            int temp_width = other.width_;
            int temp_height = other.height_;
            int temp_channels = other.channels_;
            size_t temp_step = other.step_;
            
            // Clear other's pointers immediately to prevent double-free
            other.data_ = nullptr;
            other.width_ = other.height_ = other.channels_ = 0;
            other.step_ = 0;
            
            // Now safely release our current data
            release();
            
            // Take ownership of the data
            data_ = temp_data;
            width_ = temp_width;
            height_ = temp_height;
            channels_ = temp_channels;
            step_ = temp_step;
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
            step_ = 0;
            data_ = nullptr;
            
            if (width_ > 0 && height_ > 0 && channels_ > 0) {
                try {
                    allocate();
                } catch (const std::exception& e) {
                    // Reset on allocation failure
                    data_ = nullptr;
                    width_ = 0;
                height_ = 0;
                channels_ = 0;
                step_ = 0;
                    std::cerr << "[SimpleCudaMat::create] Allocation failed: " << e.what() << std::endl;
                }
            }
        }
    }
    
    // Release GPU memory
    void release() {
        if (data_) {
            // Validate pointer before attempting to free
            // Check for obviously invalid pointers
            uintptr_t ptr_val = reinterpret_cast<uintptr_t>(data_);
            if (ptr_val == 0xffffffffffffffff ||
                ptr_val == 0xcccccccccccccccc ||
                ptr_val == 0xdddddddddddddddd ||
                ptr_val == 0xfeeefeeefeeefeee ||
                ptr_val == 0xabababababababab ||
                ptr_val == 0xcdcdcdcdcdcdcdcd ||
                ptr_val == 0xbaadf00dbaadf00d ||
                ptr_val == 0xdeadbeefdeadbeef ||
                ptr_val < 0x10000) {
                // Invalid pointer, likely already freed or corrupted
                data_ = nullptr;
                width_ = height_ = channels_ = 0;
                step_ = 0;
                return;
            }
            
            // Clear any previous errors first
            cudaGetLastError();
            
            // First check if CUDA runtime is still active
            cudaError_t runtimeErr = cudaGetLastError();
            if (runtimeErr == cudaErrorCudartUnloading) {
                // CUDA runtime is shutting down, don't try to free
                data_ = nullptr;
                width_ = height_ = channels_ = 0;
                step_ = 0;
                return;
            }
            
            // Check if pointer is valid before freeing
            cudaPointerAttributes attributes;
            memset(&attributes, 0, sizeof(attributes));
            cudaError_t queryErr = cudaPointerGetAttributes(&attributes, data_);
            
            // Only free if we can confirm it's valid CUDA memory
            if (queryErr == cudaSuccess) {
                // Additional validation - check if the memory type is valid
                if (attributes.type == cudaMemoryTypeDevice || 
                    attributes.type == cudaMemoryTypeManaged) {
                    // Valid CUDA memory, attempt to free
                    cudaError_t err = cudaFree(data_);
                    if (err != cudaSuccess && 
                        err != cudaErrorCudartUnloading && 
                        err != cudaErrorInvalidValue) {
                        // Only warn for unexpected errors
                        printf("[SimpleCudaMat] Warning: cudaFree failed: %s (error code: %d)\n", 
                               cudaGetErrorString(err), err);
                    }
                }
            }
            // For any query error or invalid memory type, just clear the pointer
            // This includes cudaErrorInvalidValue which happens during shutdown
            
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
        if (empty() || !data_) {
            return SimpleCudaMat();  // Return empty mat
        }
        
        // Validate dimensions
        if (height_ <= 0 || width_ <= 0 || channels_ <= 0) {
            return SimpleCudaMat();
        }
        
        // Validate source pointer before cloning
        uintptr_t ptr_val = reinterpret_cast<uintptr_t>(data_);
        if (ptr_val < 0x10000 || ptr_val == 0xffffffffffffffff ||
            ptr_val == 0xcccccccccccccccc || ptr_val == 0xdddddddddddddddd) {
            return SimpleCudaMat();
        }
        
        // Check if CUDA runtime is still valid
        cudaError_t lastErr = cudaGetLastError();
        if (lastErr == cudaErrorCudartUnloading) {
            return SimpleCudaMat();
        }
        
        // Validate source memory with cudaPointerGetAttributes
        cudaPointerAttributes attributes;
        memset(&attributes, 0, sizeof(attributes));
        cudaError_t queryErr = cudaPointerGetAttributes(&attributes, data_);
        
        if (queryErr != cudaSuccess) {
            // Source memory is not valid CUDA memory
            cudaGetLastError(); // Clear the error
            return SimpleCudaMat();
        }
        
        // Check if memory type is valid for copying
        if (attributes.type != cudaMemoryTypeDevice && 
            attributes.type != cudaMemoryTypeManaged) {
            return SimpleCudaMat();
        }
        
        SimpleCudaMat result(height_, width_, channels_);
        
        if (result.data_ && data_) {
            size_t size = sizeInBytes();
                   
            if (size > 0 && size < (size_t)(2048 * 2048 * 4 * sizeof(float))) {  // Sanity check - max 2048x2048x4 float
                cudaError_t err = cudaMemcpy(result.data_, data_, size, cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    result.release();
                    return SimpleCudaMat();
                }
            } else {
                result.release();
                return SimpleCudaMat();
            }
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
            // Align step to 512 bytes for optimal GPU memory coalescing
            step_ = ((width_ * channels_ + 511) / 512) * 512;
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