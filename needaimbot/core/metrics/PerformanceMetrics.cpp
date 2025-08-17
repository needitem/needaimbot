#include "PerformanceMetrics.h"
#include <algorithm>
#include <numeric>

namespace Core {
    void PerformanceMetrics::Metric::update(float value) {
        current = value;
        std::lock_guard<std::mutex> lock(historyMutex);
        history.push_back(value);
        if (history.size() > maxHistorySize) {
            history.erase(history.begin());
        }
    }
    
    float PerformanceMetrics::Metric::getAverage() const {
        std::lock_guard<std::mutex> lock(historyMutex);
        if (history.empty()) return 0.0f;
        return std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
    }
    
    float PerformanceMetrics::Metric::getMin() const {
        std::lock_guard<std::mutex> lock(historyMutex);
        if (history.empty()) return 0.0f;
        return *std::min_element(history.begin(), history.end());
    }
    
    float PerformanceMetrics::Metric::getMax() const {
        std::lock_guard<std::mutex> lock(historyMutex);
        if (history.empty()) return 0.0f;
        return *std::max_element(history.begin(), history.end());
    }
    
    PerformanceMetrics::PerformanceMetrics() {
        // 기본 메트릭들을 미리 등록
        registerMetric("frame_acquisition_time");
        registerMetric("capture_fps");
        registerMetric("capture_cycle_time");
        registerMetric("process_frame_time");
        registerMetric("detector_cycle_time");
        registerMetric("inference_time");
        registerMetric("input_send_time");
        registerMetric("detection_to_movement_time");
        registerMetric("fps_delay_time");
        registerMetric("total_cycle_time");
    }
    
    void PerformanceMetrics::registerMetric(const std::string& name) {
        std::lock_guard<std::mutex> lock(metricsMapMutex_);
        if (metrics_.find(name) == metrics_.end()) {
            metrics_[name] = std::make_unique<Metric>();
        }
    }
    
    void PerformanceMetrics::updateMetric(const std::string& name, float value) {
        std::lock_guard<std::mutex> lock(metricsMapMutex_);
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            it->second->update(value);
        }
    }
    
    float PerformanceMetrics::getCurrentValue(const std::string& name) const {
        std::lock_guard<std::mutex> lock(metricsMapMutex_);
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            return it->second->current.load();
        }
        return 0.0f;
    }
    
    std::vector<float> PerformanceMetrics::getHistory(const std::string& name) const {
        std::lock_guard<std::mutex> lock(metricsMapMutex_);
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            std::lock_guard<std::mutex> historyLock(it->second->historyMutex);
            return it->second->history;
        }
        return {};
    }
    
    PerformanceMetrics::Stats PerformanceMetrics::getStats(const std::string& name) const {
        std::lock_guard<std::mutex> lock(metricsMapMutex_);
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            Stats stats;
            stats.current = it->second->current.load();
            stats.average = it->second->getAverage();
            stats.min = it->second->getMin();
            stats.max = it->second->getMax();
            return stats;
        }
        return {};
    }
    
    PerformanceMetrics::ScopedTimer::ScopedTimer(PerformanceMetrics& metrics, const std::string& name)
        : metrics_(metrics), metricName_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    PerformanceMetrics::ScopedTimer::~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        float ms = duration.count() / 1000.0f;
        metrics_.updateMetric(metricName_, ms);
    }
}