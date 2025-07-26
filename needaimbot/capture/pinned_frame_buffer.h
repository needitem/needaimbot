#pragma once

#include "../cuda/cuda_error_check.h"
#include "../cuda/simple_cuda_mat.h"

// Pinned memory buffer for faster CPU-GPU transfers
class PinnedFrameBuffer {
public:
    PinnedFrameBuffer(int height, int width, int channels) 
        : height_(height), width_(width), channels_(channels), data_(nullptr) {
        size_t size = height_ * width_ * channels_;
        CUDA_CHECK(cudaHostAlloc(&data_, size, cudaHostAllocDefault));
    }
    
    ~PinnedFrameBuffer() {
        if (data_) {
            cudaFreeHost(data_);
        }
    }
    
    // Delete copy
    PinnedFrameBuffer(const PinnedFrameBuffer&) = delete;
    PinnedFrameBuffer& operator=(const PinnedFrameBuffer&) = delete;
    
    // Allow move
    PinnedFrameBuffer(PinnedFrameBuffer&& other) noexcept 
        : data_(other.data_), height_(other.height_), 
          width_(other.width_), channels_(other.channels_) {
        other.data_ = nullptr;
    }
    
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    int height() const { return height_; }
    int width() const { return width_; }
    int channels() const { return channels_; }
    size_t step() const { return width_ * channels_; }
    
    // Fast async upload to GPU
    void uploadToGpuAsync(SimpleCudaMat& gpuMat, cudaStream_t stream) {
        cudaMemcpy2DAsync(
            gpuMat.data(), gpuMat.step(),
            data_, step(),
            width_ * channels_, height_,
            cudaMemcpyHostToDevice, stream
        );
    }
    
private:
    uint8_t* data_;
    int height_;
    int width_;
    int channels_;
};