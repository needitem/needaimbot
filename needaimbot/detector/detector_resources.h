#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "../cuda/cuda_error_check.h"
#include "../core/logger.h"

// Smart pointer deleter for CUDA memory
struct CudaDeleter {
    void operator()(void* ptr) {
        if (ptr) {
            CUDA_CHECK_WARN(cudaFree(ptr));
            LOG_DEBUG("CudaDeleter", "Freed CUDA memory at ", ptr);
        }
    }
};

// Smart pointer deleter for CUDA host memory
struct CudaHostDeleter {
    void operator()(void* ptr) {
        if (ptr) {
            CUDA_CHECK_WARN(cudaFreeHost(ptr));
            LOG_DEBUG("CudaHostDeleter", "Freed CUDA host memory at ", ptr);
        }
    }
};

// Smart pointer deleter for CUDA streams
struct CudaStreamDeleter {
    void operator()(cudaStream_t* stream) {
        if (stream && *stream) {
            CUDA_CHECK_WARN(cudaStreamDestroy(*stream));
            LOG_DEBUG("CudaStreamDeleter", "Destroyed CUDA stream");
            delete stream;
        }
    }
};

// Smart pointer deleter for CUDA events
struct CudaEventDeleter {
    void operator()(cudaEvent_t* event) {
        if (event && *event) {
            CUDA_CHECK_WARN(cudaEventDestroy(*event));
            LOG_DEBUG("CudaEventDeleter", "Destroyed CUDA event");
            delete event;
        }
    }
};

// Type aliases for smart pointers
template<typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

template<typename T>
using CudaHostUniquePtr = std::unique_ptr<T, CudaHostDeleter>;

using CudaStreamPtr = std::unique_ptr<cudaStream_t, CudaStreamDeleter>;
using CudaEventPtr = std::unique_ptr<cudaEvent_t, CudaEventDeleter>;

// Helper functions to create smart pointers
template<typename T>
CudaUniquePtr<T> makeCudaUnique(size_t count) {
    T* ptr = nullptr;
    size_t size = count * sizeof(T);
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        LOG_ERROR("makeCudaUnique", "Failed to allocate ", size, " bytes: ", cudaGetErrorString(err));
        return CudaUniquePtr<T>(nullptr);
    }
    LOG_DEBUG("makeCudaUnique", "Allocated ", size, " bytes at ", ptr);
    return CudaUniquePtr<T>(ptr);
}

template<typename T>
CudaHostUniquePtr<T> makeCudaHostUnique(size_t count) {
    T* ptr = nullptr;
    size_t size = count * sizeof(T);
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        LOG_ERROR("makeCudaHostUnique", "Failed to allocate ", size, " bytes: ", cudaGetErrorString(err));
        return CudaHostUniquePtr<T>(nullptr);
    }
    LOG_DEBUG("makeCudaHostUnique", "Allocated ", size, " bytes of pinned memory at ", ptr);
    return CudaHostUniquePtr<T>(ptr);
}

inline CudaStreamPtr makeCudaStream() {
    cudaStream_t* stream = new cudaStream_t;
    cudaError_t err = cudaStreamCreate(stream);
    if (err != cudaSuccess) {
        LOG_ERROR("makeCudaStream", "Failed to create stream: ", cudaGetErrorString(err));
        delete stream;
        return CudaStreamPtr(nullptr);
    }
    LOG_DEBUG("makeCudaStream", "Created CUDA stream");
    return CudaStreamPtr(stream);
}

inline CudaEventPtr makeCudaEvent(unsigned int flags = cudaEventDefault) {
    cudaEvent_t* event = new cudaEvent_t;
    cudaError_t err = cudaEventCreateWithFlags(event, flags);
    if (err != cudaSuccess) {
        LOG_ERROR("makeCudaEvent", "Failed to create event: ", cudaGetErrorString(err));
        delete event;
        return CudaEventPtr(nullptr);
    }
    LOG_DEBUG("makeCudaEvent", "Created CUDA event");
    return CudaEventPtr(event);
}

// RAII class for CUDA graph capture
class CudaGraphCapture {
public:
    CudaGraphCapture(cudaStream_t stream) : stream_(stream), capturing_(false) {}
    
    bool begin() {
        cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            LOG_ERROR("CudaGraphCapture", "Failed to begin capture: ", cudaGetErrorString(err));
            return false;
        }
        capturing_ = true;
        return true;
    }
    
    cudaGraph_t end() {
        if (!capturing_) return nullptr;
        
        cudaGraph_t graph;
        cudaError_t err = cudaStreamEndCapture(stream_, &graph);
        capturing_ = false;
        
        if (err != cudaSuccess) {
            LOG_ERROR("CudaGraphCapture", "Failed to end capture: ", cudaGetErrorString(err));
            return nullptr;
        }
        return graph;
    }
    
    ~CudaGraphCapture() {
        if (capturing_) {
            cudaGraph_t graph;
            cudaStreamEndCapture(stream_, &graph);
            if (graph) cudaGraphDestroy(graph);
        }
    }

private:
    cudaStream_t stream_;
    bool capturing_;
};