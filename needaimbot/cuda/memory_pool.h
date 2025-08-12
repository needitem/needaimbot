#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include "cuda_error_check.h"
#include "../core/logger.h"

// GPU Memory Pool for efficient allocation/deallocation
class GpuMemoryPool {
public:
    static GpuMemoryPool& getInstance() {
        static GpuMemoryPool instance;
        return instance;
    }

    // Allocate memory from pool
    void* allocate(size_t size, size_t alignment = 256) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Round up size to alignment
        size = ((size + alignment - 1) / alignment) * alignment;
        
        // Check if we have a free block of this size
        auto it = freeBlocks_.find(size);
        if (it != freeBlocks_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            usedBlocks_[ptr] = size;
            LOG_DEBUG("GpuMemoryPool", "Reused block of size ", size, " at ", ptr);
            return ptr;
        }
        
        // Allocate new block
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            LOG_ERROR("GpuMemoryPool", "Failed to allocate ", size, " bytes: ", cudaGetErrorString(err));
            return nullptr;
        }
        
        usedBlocks_[ptr] = size;
        totalAllocated_ += size;
        LOG_DEBUG("GpuMemoryPool", "Allocated new block of size ", size, " at ", ptr);
        return ptr;
    }

    // Free memory back to pool
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = usedBlocks_.find(ptr);
        if (it == usedBlocks_.end()) {
            LOG_WARNING("GpuMemoryPool", "Attempting to free unknown pointer ", ptr);
            return;
        }
        
        size_t size = it->second;
        usedBlocks_.erase(it);
        
        // Add to free list if pool isn't too large
        if (totalPoolSize_ < maxPoolSize_) {
            freeBlocks_[size].push_back(ptr);
            totalPoolSize_ += size;
            LOG_DEBUG("GpuMemoryPool", "Returned block of size ", size, " to pool");
        } else {
            // Actually free the memory
            CUDA_CHECK_WARN(cudaFree(ptr));
            totalAllocated_ -= size;
            LOG_DEBUG("GpuMemoryPool", "Freed block of size ", size, " (pool full)");
        }
    }

    // Clear all pooled memory
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& pair : freeBlocks_) {
            for (void* ptr : pair.second) {
                CUDA_CHECK_WARN(cudaFree(ptr));
            }
        }
        freeBlocks_.clear();
        totalPoolSize_ = 0;
        
        LOG_INFO("GpuMemoryPool", "Cleared pool, total allocated: ", totalAllocated_, " bytes");
    }

    // Get statistics
    void getStats(size_t& totalAllocated, size_t& poolSize, size_t& inUse) {
        std::lock_guard<std::mutex> lock(mutex_);
        totalAllocated = totalAllocated_;
        poolSize = totalPoolSize_;
        inUse = 0;
        for (const auto& pair : usedBlocks_) {
            inUse += pair.second;
        }
    }

    ~GpuMemoryPool() {
        clear();
        
        // Free any remaining used blocks
        for (auto& pair : usedBlocks_) {
            CUDA_CHECK_WARN(cudaFree(pair.first));
        }
    }

private:
    GpuMemoryPool() : maxPoolSize_(64 * 1024 * 1024), // 64MB max pool (reduced for game compatibility)
                      totalAllocated_(0), 
                      totalPoolSize_(0) {}

    std::unordered_map<size_t, std::vector<void*>> freeBlocks_;  // size -> list of free blocks
    std::unordered_map<void*, size_t> usedBlocks_;              // ptr -> size
    std::mutex mutex_;
    size_t maxPoolSize_;
    size_t totalAllocated_;
    size_t totalPoolSize_;
};

// RAII wrapper for pooled GPU memory
template<typename T>
class PooledGpuPtr {
public:
    PooledGpuPtr() : ptr_(nullptr), size_(0) {}
    
    explicit PooledGpuPtr(size_t count) : size_(count * sizeof(T)) {
        ptr_ = static_cast<T*>(GpuMemoryPool::getInstance().allocate(size_));
    }
    
    ~PooledGpuPtr() {
        release();
    }
    
    // Move semantics
    PooledGpuPtr(PooledGpuPtr&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PooledGpuPtr& operator=(PooledGpuPtr&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Delete copy
    PooledGpuPtr(const PooledGpuPtr&) = delete;
    PooledGpuPtr& operator=(const PooledGpuPtr&) = delete;
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T* operator->() { return ptr_; }
    const T* operator->() const { return ptr_; }
    
    bool empty() const { return ptr_ == nullptr; }
    size_t size() const { return size_; }
    
    void release() {
        if (ptr_) {
            GpuMemoryPool::getInstance().deallocate(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

private:
    T* ptr_;
    size_t size_;
};