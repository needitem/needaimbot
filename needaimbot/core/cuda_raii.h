#ifndef NEEDAIMBOT_CORE_CUDA_RAII_H
#define NEEDAIMBOT_CORE_CUDA_RAII_H

#pragma warning(push)
#pragma warning(disable: 4996 4267 4244 4305 4018 4101 4800)

#include <memory>
#include <stdexcept>
#include "error_handler.h"

#ifndef __INTELLISENSE__
#include <cuda_runtime.h>
#endif

namespace CudaRAII {
    
    // CUDA error checking macro
    #define CUDA_CHECK(call) do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            ErrorHandler::Logger::error("CUDA", "CUDA error at ", __FILE__, ":", __LINE__, " - ", cudaGetErrorString(error)); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)
    
    /**
     * @brief RAII wrapper for CUDA device memory management
     * @tparam T Type of data stored in device memory
     * 
     * Provides automatic memory management for CUDA device memory with
     * proper exception safety and move semantics.
     */
    template<typename T>
    class DeviceMemory {
    private:
        T* ptr_;
        size_t size_;
        
    public:
        explicit DeviceMemory(size_t count = 1) : size_(count * sizeof(T)) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaMalloc(&ptr_, size_));
            #else
            ptr_ = nullptr;
            #endif
        }
        
        ~DeviceMemory() {
            if (ptr_) {
                #ifndef __INTELLISENSE__
                cudaFree(ptr_);
                #endif
            }
        }
        
        // Non-copyable, movable
        DeviceMemory(const DeviceMemory&) = delete;
        DeviceMemory& operator=(const DeviceMemory&) = delete;
        
        DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        
        DeviceMemory& operator=(DeviceMemory&& other) noexcept {
            if (this != &other) {
                if (ptr_) {
                    #ifndef __INTELLISENSE__
                    cudaFree(ptr_);
                    #endif
                }
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        
        T* get() const { return ptr_; }
        size_t size() const { return size_; }
        operator T*() const { return ptr_; }
        
        void copyFromHost(const T* host_ptr, size_t count = 1) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
            #endif
        }
        
        void copyToHost(T* host_ptr, size_t count = 1) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
            #endif
        }
        
        void copyFromHostAsync(const T* host_ptr, cudaStream_t stream, size_t count = 1) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
            #endif
        }
        
        void copyToHostAsync(T* host_ptr, cudaStream_t stream, size_t count = 1) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
            #endif
        }
    };
    
    // RAII wrapper for CUDA pinned (page-locked) memory
    template<typename T>
    class PinnedMemory {
    private:
        T* ptr_;
        size_t size_;
        
    public:
        explicit PinnedMemory(size_t count = 1) : size_(count * sizeof(T)) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaHostAlloc(&ptr_, size_, cudaHostAllocDefault));
            #else
            ptr_ = nullptr;
            #endif
        }
        
        ~PinnedMemory() {
            if (ptr_) {
                #ifndef __INTELLISENSE__
                cudaFreeHost(ptr_);
                #endif
            }
        }
        
        // Non-copyable, movable
        PinnedMemory(const PinnedMemory&) = delete;
        PinnedMemory& operator=(const PinnedMemory&) = delete;
        
        PinnedMemory(PinnedMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        
        PinnedMemory& operator=(PinnedMemory&& other) noexcept {
            if (this != &other) {
                if (ptr_) {
                    #ifndef __INTELLISENSE__
                    cudaFreeHost(ptr_);
                    #endif
                }
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        
        T* get() const { return ptr_; }
        size_t size() const { return size_; }
        operator T*() const { return ptr_; }
        T& operator*() const { return *ptr_; }
        T* operator->() const { return ptr_; }
    };
    
    // RAII wrapper for CUDA events
    class Event {
    private:
        cudaEvent_t event_;
        
    public:
        explicit Event(unsigned int flags = cudaEventDefault) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
            #endif
        }
        
        ~Event() {
            #ifndef __INTELLISENSE__
            if (event_) {
                cudaEventDestroy(event_);
            }
            #endif
        }
        
        // Non-copyable, movable
        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;
        
        Event(Event&& other) noexcept : event_(other.event_) {
            other.event_ = nullptr;
        }
        
        Event& operator=(Event&& other) noexcept {
            if (this != &other) {
                #ifndef __INTELLISENSE__
                if (event_) {
                    cudaEventDestroy(event_);
                }
                #endif
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }
        
        cudaEvent_t get() const { return event_; }
        operator cudaEvent_t() const { return event_; }
        
        void record(cudaStream_t stream = 0) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaEventRecord(event_, stream));
            #endif
        }
        
        void synchronize() {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaEventSynchronize(event_));
            #endif
        }
        
        float elapsedTime(const Event& start) const {
            float milliseconds = 0;
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.event_, event_));
            #endif
            return milliseconds;
        }
    };
    
    // RAII wrapper for CUDA streams
    class Stream {
    private:
        cudaStream_t stream_;
        
    public:
        explicit Stream(unsigned int flags = cudaStreamDefault) {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
            #endif
        }
        
        ~Stream() {
            #ifndef __INTELLISENSE__
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            #endif
        }
        
        // Non-copyable, movable
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
        
        Stream(Stream&& other) noexcept : stream_(other.stream_) {
            other.stream_ = nullptr;
        }
        
        Stream& operator=(Stream&& other) noexcept {
            if (this != &other) {
                #ifndef __INTELLISENSE__
                if (stream_) {
                    cudaStreamDestroy(stream_);
                }
                #endif
                stream_ = other.stream_;
                other.stream_ = nullptr;
            }
            return *this;
        }
        
        cudaStream_t get() const { return stream_; }
        operator cudaStream_t() const { return stream_; }
        
        void synchronize() {
            #ifndef __INTELLISENSE__
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            #endif
        }
        
        bool query() {
            #ifndef __INTELLISENSE__
            cudaError_t error = cudaStreamQuery(stream_);
            if (error == cudaSuccess) return true;
            if (error == cudaErrorNotReady) return false;
            CUDA_CHECK(error); // Will throw for other errors
            #endif
            return false;
        }
    };
}

#pragma warning(pop)

#endif // NEEDAIMBOT_CORE_CUDA_RAII_H