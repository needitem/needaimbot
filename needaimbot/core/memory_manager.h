#ifndef NEEDAIMBOT_CORE_MEMORY_MANAGER_H
#define NEEDAIMBOT_CORE_MEMORY_MANAGER_H

#pragma warning(push)
#pragma warning(disable: 4996 4267 4244 4305 4018 4101 4800)

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <chrono>

namespace MemoryManager {
    
    // Smart pointer type aliases for clarity
    template<typename T>
    using UniquePtr = std::unique_ptr<T>;
    
    template<typename T>
    using SharedPtr = std::shared_ptr<T>;
    
    template<typename T>
    using WeakPtr = std::weak_ptr<T>;
    
    /**
     * @brief Thread-safe object pool for frequent allocations
     * @tparam T Type of objects to pool
     * @tparam PoolSize Maximum number of objects to keep in pool
     * 
     * Reduces allocation overhead by reusing objects. Thread-safe
     * implementation suitable for high-performance applications.
     */
    template<typename T, size_t PoolSize = 1024>
    class ObjectPool {
    private:
        std::vector<std::unique_ptr<T>> pool_;
        std::mutex pool_mutex_;
        std::atomic<size_t> allocated_count_{0};
        
    public:
        ObjectPool() {
            pool_.reserve(PoolSize);
        }
        
        template<typename... Args>
        std::unique_ptr<T> acquire(Args&&... args) {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            
            if (!pool_.empty()) {
                auto obj = std::move(pool_.back());
                pool_.pop_back();
                allocated_count_.fetch_add(1);
                return obj;
            }
            
            allocated_count_.fetch_add(1);
            return std::make_unique<T>(std::forward<Args>(args)...);
        }
        
        void release(std::unique_ptr<T> obj) {
            if (!obj) return;
            
            std::lock_guard<std::mutex> lock(pool_mutex_);
            if (pool_.size() < PoolSize) {
                pool_.push_back(std::move(obj));
            }
            allocated_count_.fetch_sub(1);
        }
        
        size_t getAllocatedCount() const {
            return allocated_count_.load();
        }
        
        size_t getPoolSize() const {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            return pool_.size();
        }
    };
    
    // RAII buffer management for CUDA operations
    template<typename T>
    class ManagedBuffer {
    private:
        T* data_;
        size_t size_;
        bool is_device_memory_;
        
    public:
        ManagedBuffer(size_t count, bool device = false) 
            : size_(count * sizeof(T)), is_device_memory_(device) {
            
            if (device) {
                #ifndef __INTELLISENSE__
                cudaMalloc(&data_, size_);
                #else
                data_ = nullptr;
                #endif
            } else {
                data_ = new T[count];
            }
        }
        
        ~ManagedBuffer() {
            if (data_) {
                if (is_device_memory_) {
                    #ifndef __INTELLISENSE__
                    cudaFree(data_);
                    #endif
                } else {
                    delete[] data_;
                }
            }
        }
        
        // Non-copyable, movable
        ManagedBuffer(const ManagedBuffer&) = delete;
        ManagedBuffer& operator=(const ManagedBuffer&) = delete;
        
        ManagedBuffer(ManagedBuffer&& other) noexcept 
            : data_(other.data_), size_(other.size_), is_device_memory_(other.is_device_memory_) {
            other.data_ = nullptr;
        }
        
        ManagedBuffer& operator=(ManagedBuffer&& other) noexcept {
            if (this != &other) {
                if (data_) {
                    if (is_device_memory_) {
                        #ifndef __INTELLISENSE__
                        cudaFree(data_);
                        #endif
                    } else {
                        delete[] data_;
                    }
                }
                
                data_ = other.data_;
                size_ = other.size_;
                is_device_memory_ = other.is_device_memory_;
                other.data_ = nullptr;
            }
            return *this;
        }
        
        T* get() const { return data_; }
        size_t size() const { return size_; }
        bool isDeviceMemory() const { return is_device_memory_; }
    };
    
    // Memory usage tracking
    class MemoryTracker {
    private:
        static std::atomic<size_t> total_allocated_;
        static std::atomic<size_t> peak_usage_;
        static std::mutex tracking_mutex_;
        static std::unordered_map<std::string, size_t> component_usage_;
        
    public:
        static void recordAllocation(const std::string& component, size_t bytes) {
            size_t new_total = total_allocated_.fetch_add(bytes) + bytes;
            
            // Update peak usage
            size_t current_peak = peak_usage_.load();
            while (new_total > current_peak && 
                   !peak_usage_.compare_exchange_weak(current_peak, new_total)) {
                // Retry until successful or no longer needed
            }
            
            // Update component usage
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            component_usage_[component] += bytes;
        }
        
        static void recordDeallocation(const std::string& component, size_t bytes) {
            total_allocated_.fetch_sub(bytes);
            
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            auto it = component_usage_.find(component);
            if (it != component_usage_.end()) {
                it->second = (it->second > bytes) ? it->second - bytes : 0;
            }
        }
        
        static size_t getTotalAllocated() { return total_allocated_.load(); }
        static size_t getPeakUsage() { return peak_usage_.load(); }
        
        static std::unordered_map<std::string, size_t> getComponentUsage() {
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            return component_usage_;
        }
    };
    
    // Static member definitions
    inline std::atomic<size_t> MemoryTracker::total_allocated_{0};
    inline std::atomic<size_t> MemoryTracker::peak_usage_{0};
    inline std::mutex MemoryTracker::tracking_mutex_;
    inline std::unordered_map<std::string, size_t> MemoryTracker::component_usage_;
}

#pragma warning(pop)

#endif // NEEDAIMBOT_CORE_MEMORY_MANAGER_H