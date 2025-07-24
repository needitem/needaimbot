#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <cuda_runtime.h>

namespace NeedAimbot {

// Error handling macros
#define CHECK_CUDA_RESULT(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << msg << ": " \
                     << cudaGetErrorString(err) << " at " \
                     << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define CHECK_CUDA_THROW(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("[CUDA Error] ") + msg + ": " + \
                                   cudaGetErrorString(err) + " at " + \
                                   __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// Configuration constants
namespace Constants {
    // Buffer sizes
    constexpr int DEFAULT_SERIAL_BUFFER_SIZE = 256;
    constexpr int DEFAULT_DETECTION_BUFFER_SIZE = 1024;
    constexpr int NMS_BUFFER_RESERVE = 512;
    
    // Timeouts
    constexpr int SERIAL_READ_TIMEOUT_MS = 50;
    constexpr int SERIAL_WRITE_TIMEOUT_MS = 100;
    constexpr int GPU_SYNC_TIMEOUT_MS = 100;
    constexpr int THREAD_JOIN_TIMEOUT_MS = 500;
    
    // Performance tuning
    constexpr int GPU_STREAM_PRIORITY_HIGH = 0;
    constexpr int GPU_STREAM_PRIORITY_NORMAL = -1;
    constexpr int GPU_STREAM_PRIORITY_LOW = -2;
    
    // YOLO constants
    constexpr float YOLO_BASE_RESOLUTION = 640.0f;
    constexpr int YOLO10_NUM_CLASSES = 11;
    
    // Thread priorities
    constexpr int THREAD_PRIORITY_INFERENCE = THREAD_PRIORITY_HIGHEST;
    constexpr int THREAD_PRIORITY_MOUSE = THREAD_PRIORITY_TIME_CRITICAL;
    constexpr int THREAD_PRIORITY_CAPTURE = THREAD_PRIORITY_ABOVE_NORMAL;
    constexpr int THREAD_PRIORITY_UI = THREAD_PRIORITY_NORMAL;
}

// RAII wrapper for CUDA streams
class CudaStream {
public:
    CudaStream(int priority = 0) : stream_(nullptr) {
        int leastPriority, greatestPriority;
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        
        // Clamp priority to valid range
        priority = std::max(greatestPriority, std::min(priority, leastPriority));
        
        cudaError_t err = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
        }
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Delete copy constructor and assignment
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Move constructor and assignment
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
private:
    cudaStream_t stream_;
};

// RAII wrapper for CUDA events
class CudaEvent {
public:
    CudaEvent(unsigned int flags = cudaEventDisableTiming) : event_(nullptr) {
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA event: " + std::string(cudaGetErrorString(err)));
        }
    }
    
    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Delete copy constructor and assignment
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // Move constructor and assignment
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = 0) {
        cudaEventRecord(event_, stream);
    }
    
    void synchronize() {
        cudaEventSynchronize(event_);
    }
    
private:
    cudaEvent_t event_;
};

// Scoped timer for performance measurements
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) 
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << "[Timer] " << name_ << ": " << duration.count() << " us" << std::endl;
    }
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace NeedAimbot

#endif // COMMON_UTILS_H