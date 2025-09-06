#ifndef CUDA_RESOURCE_MANAGER_H
#define CUDA_RESOURCE_MANAGER_H

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <iostream>

/**
 * @brief Global CUDA Resource Manager
 * 
 * Manages all CUDA resources and ensures proper cleanup on program exit.
 * Uses RAII pattern and singleton design to guarantee cleanup even in
 * error conditions.
 */
class CudaResourceManager {
private:
    static CudaResourceManager* instance_;
    static std::mutex instance_mutex_;
    static std::atomic<bool> shutdown_initiated_;
    
    std::vector<void*> allocated_memory_;
    std::vector<cudaStream_t> created_streams_;
    std::vector<cudaEvent_t> created_events_;
    std::mutex resource_mutex_;
    
    bool is_shutting_down_ = false;
    
    CudaResourceManager() {
        // Register cleanup at exit
        std::atexit([]() {
            CudaResourceManager::Shutdown();
        });
    }
    
public:
    ~CudaResourceManager() {
        CleanupAllResources();
    }
    
    // Delete copy and move constructors
    CudaResourceManager(const CudaResourceManager&) = delete;
    CudaResourceManager& operator=(const CudaResourceManager&) = delete;
    CudaResourceManager(CudaResourceManager&&) = delete;
    CudaResourceManager& operator=(CudaResourceManager&&) = delete;
    
    static CudaResourceManager& GetInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = new CudaResourceManager();
        }
        return *instance_;
    }
    
    static void Shutdown() {
        if (shutdown_initiated_.exchange(true)) {
            return; // Already shutting down
        }
        
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (instance_) {
            delete instance_;
            instance_ = nullptr;
        }
    }
    
    // Register allocated memory for tracking
    void RegisterMemory(void* ptr) {
        if (!ptr || is_shutting_down_) return;
        
        std::lock_guard<std::mutex> lock(resource_mutex_);
        allocated_memory_.push_back(ptr);
    }
    
    // Unregister freed memory
    void UnregisterMemory(void* ptr) {
        if (!ptr || is_shutting_down_) return;
        
        std::lock_guard<std::mutex> lock(resource_mutex_);
        auto it = std::find(allocated_memory_.begin(), allocated_memory_.end(), ptr);
        if (it != allocated_memory_.end()) {
            allocated_memory_.erase(it);
        }
    }
    
    // Register stream
    void RegisterStream(cudaStream_t stream) {
        if (!stream || is_shutting_down_) return;
        
        std::lock_guard<std::mutex> lock(resource_mutex_);
        created_streams_.push_back(stream);
    }
    
    // Unregister stream
    void UnregisterStream(cudaStream_t stream) {
        if (!stream || is_shutting_down_) return;
        
        std::lock_guard<std::mutex> lock(resource_mutex_);
        auto it = std::find(created_streams_.begin(), created_streams_.end(), stream);
        if (it != created_streams_.end()) {
            created_streams_.erase(it);
        }
    }
    
    // Register event
    void RegisterEvent(cudaEvent_t event) {
        if (!event || is_shutting_down_) return;
        
        std::lock_guard<std::mutex> lock(resource_mutex_);
        created_events_.push_back(event);
    }
    
    // Unregister event
    void UnregisterEvent(cudaEvent_t event) {
        if (!event || is_shutting_down_) return;
        
        std::lock_guard<std::mutex> lock(resource_mutex_);
        auto it = std::find(created_events_.begin(), created_events_.end(), event);
        if (it != created_events_.end()) {
            created_events_.erase(it);
        }
    }
    
    // Clean up all resources
    void CleanupAllResources() {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        is_shutting_down_ = true;
        
        std::cout << "[CudaResourceManager] Starting cleanup..." << std::endl;
        
        // Synchronize all devices first
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess) {
            for (int i = 0; i < device_count; ++i) {
                cudaSetDevice(i);
                cudaDeviceSynchronize();
            }
        }
        
        // Destroy events
        for (auto& event : created_events_) {
            if (event) {
                cudaEventDestroy(event);
            }
        }
        created_events_.clear();
        
        // Destroy streams
        for (auto& stream : created_streams_) {
            if (stream) {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
        }
        created_streams_.clear();
        
        // NOTE: Memory will be freed by SimpleCudaMat destructors
        // We just clear our tracking list to avoid double-free
        allocated_memory_.clear();
        
        // Selective cleanup instead of cudaDeviceReset
        if (err == cudaSuccess) {
            for (int i = 0; i < device_count; ++i) {
                cudaSetDevice(i);
                
                // Clear L2 cache
                cudaCtxResetPersistingL2Cache();
                
                // Trim memory pools to free cached memory
                cudaMemPoolTrimTo(nullptr, 0);
                
                // Clear graph memory pool
                cudaDeviceGraphMemTrim(i);
                
                // Free default memory pool resources
                cudaMemPool_t memPool;
                if (cudaDeviceGetDefaultMemPool(&memPool, i) == cudaSuccess) {
                    cudaMemPoolTrimTo(memPool, 0);
                }
                
                // Flush GPU L2 cache
                cudaDeviceSynchronize();
                
                // Note: We intentionally DO NOT call cudaDeviceReset() here
                // to avoid reinitalization overhead on next program start
            }
        }
        
        std::cout << "[CudaResourceManager] Cleanup completed (without device reset)." << std::endl;
    }
    
    bool IsShuttingDown() const {
        return is_shutting_down_ || shutdown_initiated_.load();
    }
};

// Static member definitions
inline CudaResourceManager* CudaResourceManager::instance_ = nullptr;
inline std::mutex CudaResourceManager::instance_mutex_;
inline std::atomic<bool> CudaResourceManager::shutdown_initiated_{false};

// NOTE: Use CudaMemory and CudaStream classes from cuda_utils.h instead
// to avoid duplication. This resource manager only tracks resources for
// centralized cleanup during shutdown.

#endif // CUDA_RESOURCE_MANAGER_H