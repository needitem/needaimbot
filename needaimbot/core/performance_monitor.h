#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <Windows.h>
#include <Psapi.h>

class PerformanceMonitor {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
    struct PerformanceMetrics {
        float avg_time_ms = 0.0f;
        float min_time_ms = (std::numeric_limits<float>::max)();
        float max_time_ms = 0.0f;
        float fps = 0.0f;
        size_t sample_count = 0;
    };
    
    struct SystemMetrics {
        float cpu_usage_percent = 0.0f;
        size_t memory_usage_mb = 0;
        size_t gpu_memory_usage_mb = 0;
        float gpu_usage_percent = 0.0f;
    };
    
    static PerformanceMonitor& getInstance() {
        static PerformanceMonitor instance;
        return instance;
    }
    
    // Scoped timer for automatic measurement
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& name) 
            : name_(name), start_(Clock::now()) {}
        
        ~ScopedTimer() {
            auto end = Clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            float ms = duration.count() / 1000.0f;
            PerformanceMonitor::getInstance().recordTime(name_, ms);
        }
        
    private:
        std::string name_;
        TimePoint start_;
    };
    
    void recordTime(const std::string& name, float time_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& metrics = metrics_map_[name];
        metrics.sample_count++;
        
        // Update min/max
        metrics.min_time_ms = (std::min)(metrics.min_time_ms, time_ms);
        metrics.max_time_ms = (std::max)(metrics.max_time_ms, time_ms);
        
        // Update rolling average
        const float alpha = 0.1f; // Smoothing factor
        if (metrics.sample_count == 1) {
            metrics.avg_time_ms = time_ms;
        } else {
            metrics.avg_time_ms = metrics.avg_time_ms * (1.0f - alpha) + time_ms * alpha;
        }
        
        // Calculate FPS if applicable
        if (time_ms > 0) {
            metrics.fps = 1000.0f / time_ms;
        }
    }
    
    PerformanceMetrics getMetrics(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_map_.find(name);
        if (it != metrics_map_.end()) {
            return it->second;
        }
        return PerformanceMetrics{};
    }
    
    std::vector<std::pair<std::string, PerformanceMetrics>> getAllMetrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, PerformanceMetrics>> result;
        result.reserve(metrics_map_.size());  // Pre-allocate to avoid reallocations
        for (const auto& pair : metrics_map_) {
            result.push_back(pair);
        }
        // Sort by average time (slowest first)
        std::sort(result.begin(), result.end(), 
            [](const auto& a, const auto& b) {
                return a.second.avg_time_ms > b.second.avg_time_ms;
            });
        return result;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_map_.clear();
    }
    
    SystemMetrics getSystemMetrics() {
        SystemMetrics metrics;
        
        // CPU usage
        static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
        static int numProcessors = 0;
        static HANDLE self = GetCurrentProcess();
        
        if (numProcessors == 0) {
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            numProcessors = sysInfo.dwNumberOfProcessors;
        }
        
        FILETIME ftime, fsys, fuser;
        ULARGE_INTEGER now, sys, user;
        
        GetSystemTimeAsFileTime(&ftime);
        memcpy(&now, &ftime, sizeof(FILETIME));
        
        GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
        memcpy(&sys, &fsys, sizeof(FILETIME));
        memcpy(&user, &fuser, sizeof(FILETIME));
        
        if (lastCPU.QuadPart != 0) {
            metrics.cpu_usage_percent = static_cast<float>(
                (sys.QuadPart - lastSysCPU.QuadPart) + 
                (user.QuadPart - lastUserCPU.QuadPart)
            );
            metrics.cpu_usage_percent /= (now.QuadPart - lastCPU.QuadPart);
            metrics.cpu_usage_percent /= numProcessors;
            metrics.cpu_usage_percent *= 100.0f;
        }
        
        lastCPU = now;
        lastUserCPU = user;
        lastSysCPU = sys;
        
        // Memory usage
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(self, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            metrics.memory_usage_mb = static_cast<size_t>(pmc.WorkingSetSize / (1024 * 1024));
        }
        
        return metrics;
    }
    
    // Check if performance is degraded
    bool isPerformanceDegraded(const std::string& metric_name, float threshold_ms) const {
        auto metrics = getMetrics(metric_name);
        return metrics.avg_time_ms > threshold_ms;
    }
    
    void logSlowOperations(float threshold_ms = 16.0f) const {
        auto all_metrics = getAllMetrics();
        for (const auto& [name, metrics] : all_metrics) {
            // Skip intentional wait operations
            if (name.find("Wait") != std::string::npos || 
                name.find("wait") != std::string::npos) {
                continue;
            }
            
            if (metrics.avg_time_ms > threshold_ms) {
                std::cout << "[Performance] Slow operation detected: " << name 
                          << " avg=" << metrics.avg_time_ms << "ms"
                          << " min=" << metrics.min_time_ms << "ms"
                          << " max=" << metrics.max_time_ms << "ms"
                          << std::endl;
            }
        }
    }
    
private:
    PerformanceMonitor() {
        metrics_map_.reserve(50);  // Pre-allocate for typical number of metrics
    }
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, PerformanceMetrics> metrics_map_;
};

// Convenience macro for easy performance measurement
#define PERF_TIMER(name) PerformanceMonitor::ScopedTimer _perf_timer_##__LINE__(name)

#endif // PERFORMANCE_MONITOR_H