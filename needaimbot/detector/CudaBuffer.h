#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include "GpuMemoryPool.h"

// Deleter for memory pool
struct PoolDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            getGpuMemoryPool().deallocate(ptr);
        }
    }
};

// Unique pointer for pooled memory
template<typename T>
using CudaUniquePtr = std::unique_ptr<T, PoolDeleter>;

// A wrapper for a CUDA buffer
template<typename T>
class CudaBuffer {
public:
    CudaBuffer() : m_ptr(nullptr), m_size(0) {}

    explicit CudaBuffer(size_t size) : m_ptr(nullptr), m_size(0) {
        allocate(size);
    }

    ~CudaBuffer() = default;

    // Disable copy semantics
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Enable move semantics
    CudaBuffer(CudaBuffer&& other) noexcept : m_ptr(std::move(other.m_ptr)), m_size(other.m_size) {
        other.m_size = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            m_ptr = std::move(other.m_ptr);
            m_size = other.m_size;
            other.m_size = 0;
        }
        return *this;
    }

    void allocate(size_t size, cudaStream_t stream = 0) {
        if (m_size >= size) return;

        // Use memory pool for allocation
        void* ptr = getGpuMemoryPool().allocate(size * sizeof(T), stream);
        if (!ptr) {
            std::cerr << "[CUDA] Failed to allocate buffer from pool" << std::endl;
            throw std::runtime_error("CUDA memory allocation failed");
        }
        m_ptr.reset(static_cast<T*>(ptr));
        m_size = size;
    }

    T* get() const {
        return m_ptr.get();
    }

    size_t size() const {
        return m_size;
    }
    
    // Async memory transfer methods
    cudaError_t copyToAsync(T* dst, size_t count, cudaStream_t stream = 0) const {
        if (!m_ptr || !dst || count > m_size) {
            return cudaErrorInvalidValue;
        }
        return cudaMemcpyAsync(dst, m_ptr.get(), count * sizeof(T), 
                              cudaMemcpyDeviceToHost, stream);
    }
    
    cudaError_t copyFromAsync(const T* src, size_t count, cudaStream_t stream = 0) {
        if (!m_ptr || !src || count > m_size) {
            return cudaErrorInvalidValue;
        }
        return cudaMemcpyAsync(m_ptr.get(), src, count * sizeof(T), 
                              cudaMemcpyHostToDevice, stream);
    }
    
    cudaError_t copyToDeviceAsync(T* dst, size_t count, cudaStream_t stream = 0) const {
        if (!m_ptr || !dst || count > m_size) {
            return cudaErrorInvalidValue;
        }
        return cudaMemcpyAsync(dst, m_ptr.get(), count * sizeof(T), 
                              cudaMemcpyDeviceToDevice, stream);
    }
    
    // Pinned memory support for faster async transfers
    static T* allocatePinned(size_t size) {
        T* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, size * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] Failed to allocate pinned memory: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;
    }
    
    static void freePinned(T* ptr) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }

private:
    CudaUniquePtr<T> m_ptr;
    size_t m_size;
};

#endif // CUDA_BUFFER_H
