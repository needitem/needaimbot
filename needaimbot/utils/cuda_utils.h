#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "../cuda/cuda_resource_manager.h"

// CUDA error checking macro
#ifndef CUDA_CHECK
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
#endif

// CUDA error checking macro with cleanup
#ifndef CUDA_CHECK_CLEANUP
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
#endif

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

// RAII wrapper for CUDA pinned host memory
template<typename T>
class CudaPinnedMemory {
public:
    CudaPinnedMemory() = default;
    
    explicit CudaPinnedMemory(size_t count, unsigned int flags = cudaHostAllocDefault) 
        : count_(count) {
        if (count > 0) {
            cudaError_t err = cudaHostAlloc(&ptr_, count * sizeof(T), flags);
            if (err == cudaSuccess && ptr_) {
                CudaResourceManager::GetInstance().RegisterMemory(ptr_);
            } else if (err != cudaSuccess) {
                throw std::runtime_error("cudaHostAlloc failed");
            }
        }
    }
    
    ~CudaPinnedMemory() {
        reset();
    }
    
    CudaPinnedMemory(const CudaPinnedMemory&) = delete;
    CudaPinnedMemory& operator=(const CudaPinnedMemory&) = delete;
    
    CudaPinnedMemory(CudaPinnedMemory&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    CudaPinnedMemory& operator=(CudaPinnedMemory&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    void reset(size_t new_count = 0, unsigned int flags = cudaHostAllocDefault) {
        if (ptr_) {
            if (!CudaResourceManager::GetInstance().IsShuttingDown()) {
                CudaResourceManager::GetInstance().UnregisterMemory(ptr_);
                cudaFreeHost(ptr_);
            }
            ptr_ = nullptr;
        }
        count_ = new_count;
        if (new_count > 0) {
            cudaError_t err = cudaHostAlloc(&ptr_, new_count * sizeof(T), flags);
            if (err == cudaSuccess && ptr_) {
                CudaResourceManager::GetInstance().RegisterMemory(ptr_);
            } else if (err != cudaSuccess) {
                throw std::runtime_error("cudaHostAlloc failed");
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

// RAII wrapper for CUDA events
class CudaEvent {
private:
    cudaEvent_t event_ = nullptr;
    
public:
    // Default constructor - creates event with default flags
    CudaEvent() : CudaEvent(cudaEventDefault) {}
    
    // Constructor with flags
    explicit CudaEvent(unsigned int flags) {
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        if (err == cudaSuccess && event_) {
            CudaResourceManager::GetInstance().RegisterEvent(event_);
        } else if (err != cudaSuccess) {
            throw std::runtime_error("cudaEventCreate failed");
        }
    }
    
    ~CudaEvent() {
        if (event_) {
            if (!CudaResourceManager::GetInstance().IsShuttingDown()) {
                CudaResourceManager::GetInstance().UnregisterEvent(event_);
                cudaEventDestroy(event_);
            }
        }
    }
    
    // Delete copy constructor and assignment
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // Move constructor
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    // Move assignment
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                if (!CudaResourceManager::GetInstance().IsShuttingDown()) {
                    CudaResourceManager::GetInstance().UnregisterEvent(event_);
                    cudaEventDestroy(event_);
                }
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = 0) {
        if (event_) {
            cudaEventRecord(event_, stream);
        }
    }
    
    void synchronize() {
        if (event_) {
            cudaEventSynchronize(event_);
        }
    }
    
    cudaError_t query() {
        if (event_) {
            return cudaEventQuery(event_);
        }
        return cudaErrorInvalidResourceHandle;
    }
    
    float elapsedTime(const CudaEvent& start) {
        float ms = 0.0f;
        if (event_ && start.event_) {
            cudaEventElapsedTime(&ms, start.event_, event_);
        }
        return ms;
    }
};

// CUDA memory wrapper for RAII
template<typename T>
class CudaMemory {
public:
    CudaMemory() = default;
    
    explicit CudaMemory(size_t count, bool zero_initialize = false) : count_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err == cudaSuccess && ptr_) {
                CudaResourceManager::GetInstance().RegisterMemory(ptr_);
                // Initialize memory to zero if requested (important for preventing garbage values)
                if (zero_initialize) {
                    cudaMemset(ptr_, 0, count * sizeof(T));
                }
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
    
    void reset(size_t new_count = 0, bool zero_initialize = false) {
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
                // Initialize memory to zero if requested
                if (zero_initialize) {
                    cudaMemset(ptr_, 0, new_count * sizeof(T));
                }
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

#endif // CUDA_UTILS_H