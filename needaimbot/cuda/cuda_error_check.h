#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Comprehensive CUDA error checking macro with file/line info
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
               << " code=" << error << "(" << cudaGetErrorString(error) << ")"; \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

// Silent check that returns false on error (for non-critical operations)
#define CUDA_CHECK_SILENT(call) \
    (call == cudaSuccess)

// Warning-only check (logs but doesn't throw)
#define CUDA_CHECK_WARN(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "[CUDA Warning] " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << "(" << cudaGetErrorString(error) << ")" << std::endl; \
        } \
    } while(0)

// Check and return on error (for C-style functions)
#define CUDA_CHECK_RETURN(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << "(" << cudaGetErrorString(error) << ")" << std::endl; \
            return false; \
        } \
    } while(0)

// Debug-only check (only active in debug builds)
#ifdef _DEBUG
#define CUDA_CHECK_DEBUG(call) CUDA_CHECK(call)
#else
#define CUDA_CHECK_DEBUG(call) call
#endif

// Helper class for RAII CUDA resource management
class CudaResourceGuard {
public:
    CudaResourceGuard() = default;
    ~CudaResourceGuard() {
        // Ensure all CUDA operations are complete
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "[CUDA Resource Guard] Cleanup error: " << cudaGetErrorString(error) << std::endl;
        }
    }
    
    // Disable copy
    CudaResourceGuard(const CudaResourceGuard&) = delete;
    CudaResourceGuard& operator=(const CudaResourceGuard&) = delete;
};

// Helper function to get human-readable CUDA error description
inline std::string getCudaErrorDescription(cudaError_t error) {
    std::stringstream ss;
    ss << "CUDA Error " << error << ": " << cudaGetErrorString(error);
    return ss.str();
}

// Helper to check available GPU memory
inline bool checkGpuMemory(size_t requiredBytes, size_t& availableBytes, size_t& totalBytes) {
    // Clear any previous CUDA errors
    cudaError_t prevError = cudaGetLastError();
    if (prevError != cudaSuccess) {
        // Previous error exists, skip memory check
        // Assume we have enough memory and let cudaMalloc handle it
        availableBytes = requiredBytes + 1024*1024; // Add 1MB buffer
        totalBytes = availableBytes * 2;
        return true;
    }
    
    cudaError_t error = cudaMemGetInfo(&availableBytes, &totalBytes);
    if (error != cudaSuccess) {
        std::cerr << "[CUDA] Failed to get memory info: " << cudaGetErrorString(error) << std::endl;
        // Assume we have enough memory and let cudaMalloc handle it
        availableBytes = requiredBytes + 1024*1024;
        totalBytes = availableBytes * 2;
        return true;
    }
    return availableBytes >= requiredBytes;
}