#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "../cuda/cuda_resource_manager.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
               << " code=" << error << " \"" << cudaGetErrorString(error) << "\""; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

// CUDA error checking macro with cleanup
#define CUDA_CHECK_CLEANUP(call, cleanup) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cleanup; \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
               << " code=" << error << " \"" << cudaGetErrorString(error) << "\""; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

// Safe CUDA resource release
template<typename T>
inline void cudaSafeDestroy(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

// CUDA stream wrapper for RAII
class CudaStream {
public:
    CudaStream() {
        cudaError_t err = cudaStreamCreate(&stream_);
        if (err == cudaSuccess && stream_) {
            CudaResourceManager::GetInstance().RegisterStream(stream_);
        } else if (err != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed");
        }
    }
    
    ~CudaStream() {
        if (stream_) {
            if (!CudaResourceManager::GetInstance().IsShuttingDown()) {
                CudaResourceManager::GetInstance().UnregisterStream(stream_);
                cudaStreamDestroy(stream_);
            }
        }
    }
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                if (!CudaResourceManager::GetInstance().IsShuttingDown()) {
                    CudaResourceManager::GetInstance().UnregisterStream(stream_);
                    cudaStreamDestroy(stream_);
                }
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    
private:
    cudaStream_t stream_ = nullptr;
};

// CUDA memory wrapper for RAII
template<typename T>
class CudaMemory {
public:
    CudaMemory() = default;
    
    explicit CudaMemory(size_t count) : count_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err == cudaSuccess && ptr_) {
                CudaResourceManager::GetInstance().RegisterMemory(ptr_);
            } else if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
        }
    }
    
    ~CudaMemory() {
        reset();
    }
    
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
    CudaMemory(CudaMemory&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    void reset(size_t new_count = 0) {
        if (ptr_) {
            // Check if resource manager is shutting down
            if (!CudaResourceManager::GetInstance().IsShuttingDown()) {
                CudaResourceManager::GetInstance().UnregisterMemory(ptr_);
                cudaFree(ptr_);
            }
            ptr_ = nullptr;
        }
        count_ = new_count;
        if (new_count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, new_count * sizeof(T));
            if (err == cudaSuccess && ptr_) {
                CudaResourceManager::GetInstance().RegisterMemory(ptr_);
            } else if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
        }
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    
private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};