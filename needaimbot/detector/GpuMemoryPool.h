#ifndef GPU_MEMORY_POOL_H
#define GPU_MEMORY_POOL_H

#include <cuda_runtime.h>
#include <queue>
#include <mutex>
#include <memory>
#include <unordered_map>

class GpuMemoryPool {
public:
    struct Buffer {
        void* ptr;
        size_t size;
        bool inUse;
        cudaStream_t stream;
    };

    GpuMemoryPool() : totalAllocated(0), maxPoolSize(1024 * 1024 * 1024) {} // 1GB max

    ~GpuMemoryPool() {
        std::lock_guard<std::mutex> lock(poolMutex);
        for (auto& [size, buffers] : bufferPools) {
            for (auto& buffer : buffers) {
                if (buffer->ptr) {
                    cudaFree(buffer->ptr);
                }
            }
        }
    }

    void* allocate(size_t size, cudaStream_t stream = 0) {
        std::lock_guard<std::mutex> lock(poolMutex);
        
        // Round up to nearest power of 2 for better reuse
        size_t alignedSize = nextPowerOf2(size);
        
        // Check if we have a free buffer of this size
        auto& pool = bufferPools[alignedSize];
        for (auto& buffer : pool) {
            if (!buffer->inUse) {
                buffer->inUse = true;
                buffer->stream = stream;
                return buffer->ptr;
            }
        }
        
        // Allocate new buffer if under limit
        if (totalAllocated + alignedSize <= maxPoolSize) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, alignedSize);
            if (err == cudaSuccess && ptr) {
                auto buffer = std::make_unique<Buffer>();
                buffer->ptr = ptr;
                buffer->size = alignedSize;
                buffer->inUse = true;
                buffer->stream = stream;
                void* returnPtr = ptr;
                pool.push_back(std::move(buffer));
                totalAllocated += alignedSize;
                return returnPtr;
            }
        }
        
        return nullptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(poolMutex);
        for (auto& [size, buffers] : bufferPools) {
            for (auto& buffer : buffers) {
                if (buffer->ptr == ptr) {
                    buffer->inUse = false;
                    buffer->stream = 0;
                    return;
                }
            }
        }
    }

    // Pre-allocate common sizes
    void preallocate(const std::vector<size_t>& sizes, cudaStream_t stream = 0) {
        for (size_t size : sizes) {
            allocate(size, stream);
            // Immediately mark as not in use for pre-allocation
            std::lock_guard<std::mutex> lock(poolMutex);
            size_t alignedSize = nextPowerOf2(size);
            auto& pool = bufferPools[alignedSize];
            if (!pool.empty()) {
                pool.back()->inUse = false;
            }
        }
    }

    size_t getTotalAllocated() const { return totalAllocated; }

private:
    size_t nextPowerOf2(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;
        return n;
    }

    std::mutex poolMutex;
    std::unordered_map<size_t, std::vector<std::unique_ptr<Buffer>>> bufferPools;
    size_t totalAllocated;
    const size_t maxPoolSize;
};

// Global instance
inline GpuMemoryPool& getGpuMemoryPool() {
    static GpuMemoryPool pool;
    return pool;
}

#endif // GPU_MEMORY_POOL_H