#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <mutex>
#include "../AppContext.h"

class AutoTuner {
public:
    struct TuningParameter {
        std::string name;
        float min_value;
        float max_value;
        float step;
        float current_value;
        float best_value;
        std::function<void(float)> setter;
    };
    
    struct PerformanceMetrics {
        float accuracy;      // Hit rate
        float precision;     // How close to target center
        float response_time; // Time to acquire target
        float stability;     // Movement smoothness
        
        float getScore() const {
            // Weighted combination of metrics
            return accuracy * 0.4f + precision * 0.3f + 
                   (1.0f / (response_time + 0.001f)) * 0.2f + 
                   stability * 0.1f;
        }
    };

private:
    std::vector<TuningParameter> parameters;
    std::map<std::string, std::vector<float>> performance_history;
    std::mutex tuning_mutex;
    bool is_tuning = false;
    std::chrono::steady_clock::time_point tuning_start_time;
    static constexpr int SAMPLES_PER_SETTING = 100;
    int current_sample_count = 0;
    PerformanceMetrics current_metrics;
    
public:
    static AutoTuner& getInstance() {
        static AutoTuner instance;
        return instance;
    }
    
    void initializeParameters() {
        auto& ctx = AppContext::getInstance();
        
        // PID parameters
        parameters.push_back({"kp_x", 0.1f, 2.0f, 0.1f, 
                            static_cast<float>(ctx.config.kp_x), 
                            static_cast<float>(ctx.config.kp_x),
                            [&ctx](float v) { ctx.config.kp_x = v; }});
        
        parameters.push_back({"ki_x", 0.0f, 0.1f, 0.01f,
                            static_cast<float>(ctx.config.ki_x),
                            static_cast<float>(ctx.config.ki_x),
                            [&ctx](float v) { ctx.config.ki_x = v; }});
        
        parameters.push_back({"kd_x", 0.0f, 0.1f, 0.01f,
                            static_cast<float>(ctx.config.kd_x),
                            static_cast<float>(ctx.config.kd_x),
                            [&ctx](float v) { ctx.config.kd_x = v; }});
        
        // Movement smoothing
        parameters.push_back({"movement_smoothing", 0.0f, 0.6f, 0.05f,
                            ctx.config.movement_smoothing,
                            ctx.config.movement_smoothing,
                            [&ctx](float v) { ctx.config.movement_smoothing = v; }});
        
        // Prediction factor
        parameters.push_back({"prediction_time_factor", 0.0001f, 0.01f, 0.0001f,
                            ctx.config.prediction_time_factor,
                            ctx.config.prediction_time_factor,
                            [&ctx](float v) { ctx.config.prediction_time_factor = v; }});
    }
    
    void startAutoTuning() {
        std::lock_guard<std::mutex> lock(tuning_mutex);
        is_tuning = true;
        tuning_start_time = std::chrono::steady_clock::now();
        current_sample_count = 0;
        resetMetrics();
    }
    
    void stopAutoTuning() {
        std::lock_guard<std::mutex> lock(tuning_mutex);
        is_tuning = false;
        applyBestParameters();
        saveResults();
    }
    
    void updateMetrics(float error_x, float error_y, bool hit_target, float response_time_ms) {
        if (!is_tuning) return;
        
        std::lock_guard<std::mutex> lock(tuning_mutex);
        
        // Update running averages
        float precision = 1.0f / (1.0f + std::sqrt(error_x * error_x + error_y * error_y));
        current_metrics.precision = (current_metrics.precision * current_sample_count + precision) / 
                                   (current_sample_count + 1);
        
        if (hit_target) {
            current_metrics.accuracy = (current_metrics.accuracy * current_sample_count + 1.0f) / 
                                     (current_sample_count + 1);
        } else {
            current_metrics.accuracy = (current_metrics.accuracy * current_sample_count) / 
                                     (current_sample_count + 1);
        }
        
        current_metrics.response_time = (current_metrics.response_time * current_sample_count + 
                                       response_time_ms) / (current_sample_count + 1);
        
        // Simple stability metric based on error variance
        static float last_error_x = 0, last_error_y = 0;
        float error_change = std::sqrt((error_x - last_error_x) * (error_x - last_error_x) + 
                                     (error_y - last_error_y) * (error_y - last_error_y));
        current_metrics.stability = 1.0f / (1.0f + error_change);
        last_error_x = error_x;
        last_error_y = error_y;
        
        current_sample_count++;
        
        // Check if we have enough samples for this parameter setting
        if (current_sample_count >= SAMPLES_PER_SETTING) {
            evaluateCurrentSetting();
            moveToNextSetting();
        }
    }
    
    bool isTuning() const {
        return is_tuning;
    }
    
    float getTuningProgress() const {
        if (!is_tuning) return 0.0f;
        
        int total_steps = 0;
        int current_step = 0;
        
        for (const auto& param : parameters) {
            int steps = static_cast<int>((param.max_value - param.min_value) / param.step) + 1;
            total_steps += steps;
            
            int param_step = static_cast<int>((param.current_value - param.min_value) / param.step);
            current_step += param_step;
        }
        
        return static_cast<float>(current_step) / static_cast<float>(total_steps);
    }

private:
    AutoTuner() = default;
    
    void resetMetrics() {
        current_metrics = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    void evaluateCurrentSetting() {
        float score = current_metrics.getScore();
        
        // Store performance for this parameter combination
        std::string config_key = getCurrentConfigString();
        performance_history[config_key].push_back(score);
        
        // Update best values if this is better
        static float best_score = 0.0f;
        if (score > best_score) {
            best_score = score;
            for (auto& param : parameters) {
                param.best_value = param.current_value;
            }
        }
    }
    
    void moveToNextSetting() {
        // Grid search through parameter space
        bool carry = true;
        for (auto& param : parameters) {
            if (carry) {
                param.current_value += param.step;
                if (param.current_value > param.max_value) {
                    param.current_value = param.min_value;
                } else {
                    carry = false;
                }
            }
        }
        
        // Apply new settings
        for (const auto& param : parameters) {
            param.setter(param.current_value);
        }
        
        // Reset for next round
        current_sample_count = 0;
        resetMetrics();
        
        // Check if we've tested all combinations
        if (carry) {
            stopAutoTuning();
        }
    }
    
    void applyBestParameters() {
        auto& ctx = AppContext::getInstance();
        
        for (const auto& param : parameters) {
            param.setter(param.best_value);
        }
        
        ctx.config.saveConfig();
    }
    
    std::string getCurrentConfigString() const {
        std::string result;
        for (const auto& param : parameters) {
            result += param.name + "=" + std::to_string(param.current_value) + ";";
        }
        return result;
    }
    
    void saveResults() {
        std::ofstream file("tuning_results.txt");
        if (!file.is_open()) return;
        
        file << "Auto-tuning results:\n";
        file << "Best parameters:\n";
        
        for (const auto& param : parameters) {
            file << param.name << ": " << param.best_value << "\n";
        }
        
        file << "\nPerformance history:\n";
        for (const auto& [config, scores] : performance_history) {
            float avg_score = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
            file << config << " -> avg_score: " << avg_score << "\n";
        }
        
        file.close();
    }
};

// Helper class for automatic performance tracking
class PerformanceTracker {
private:
    std::chrono::high_resolution_clock::time_point target_acquired_time;
    bool tracking_target = false;
    
public:
    void onTargetAcquired() {
        target_acquired_time = std::chrono::high_resolution_clock::now();
        tracking_target = true;
    }
    
    void onTargetLost() {
        tracking_target = false;
    }
    
    void onMouseMove(float error_x, float error_y) {
        if (!tracking_target) return;
        
        auto& tuner = AutoTuner::getInstance();
        if (!tuner.isTuning()) return;
        
        auto now = std::chrono::high_resolution_clock::now();
        float response_time = std::chrono::duration<float, std::milli>(
            now - target_acquired_time).count();
        
        bool hit_target = (std::abs(error_x) < 5.0f && std::abs(error_y) < 5.0f);
        
        tuner.updateMetrics(error_x, error_y, hit_target, response_time);
    }
};