#ifndef APP_CONTEXT_H
#define APP_CONTEXT_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "cuda/simple_cuda_mat.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <array>
#include <memory>
#include "config/config.h"
#include "cuda/detection/postProcess.h"
#include <queue>
#include <chrono>

// Mouse movement event structure
struct MouseEvent {
    float dx;
    float dy;
    bool has_target;
    Target target;
    std::chrono::steady_clock::time_point timestamp;
};

struct AppContext {
public:
    AppContext(const AppContext&) = delete;
    AppContext& operator=(const AppContext&) = delete;


    static AppContext& getInstance() {
        static AppContext instance;
        return instance;
    }

    // Config
    Config config;

    // State management - use regular bool with mutex for less frequently accessed
    std::atomic<bool> detection_paused{false};
    bool preview_enabled{false};  // Not hot path - regular bool is fine
    bool crosshair_offset_changed{false};  // Rare change
    bool model_changed{false};  // Rare change
    
    // Target data - Optimize with fixed-size array for cache locality
    static constexpr size_t MAX_TARGETS = 64;
    mutable std::mutex target_mutex;
    std::array<Target, MAX_TARGETS> all_targets_;  // Fixed size for better cache
    size_t num_targets_ = 0;
    Target current_target_;
    std::atomic<bool> has_target_{false};

    // Frame synchronization
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    std::atomic<bool> new_frame_available{false};

    // Target synchronization
    std::mutex detection_mutex;
    std::condition_variable detection_cv;

    // Application state
    std::atomic<bool> should_exit{false};
    std::atomic<bool> aiming{false};
    std::atomic<bool> shooting{false};  // Track if auto_shoot button is pressed
    std::atomic<bool> single_shot_requested{false};  // Single shot trigger (F8)
    std::atomic<bool> single_shot_mode{false};  // Flag to indicate if currently in single shot mode
    bool input_method_changed{false};  // Rare change - no need for atomic
    
    // Aiming state synchronization for CPU efficiency
    std::mutex aiming_mutex;
    std::condition_variable aiming_cv;
    
    // Pipeline activation for event-driven idle (CPU optimization)
    std::mutex pipeline_activation_mutex;
    std::condition_variable pipeline_activation_cv;
    
    bool detection_resolution_changed{false};  // Rare change - no need for atomic
    
    
    // Application control
    std::mutex configMutex;
    
    std::atomic<float> g_movementDeltaX{0.0f};
    std::atomic<float> g_movementDeltaY{0.0f};
    
    // Event-based mouse control - Ring buffer for lock-free access
    static constexpr size_t MOUSE_EVENT_BUFFER_SIZE = 32;
    std::array<MouseEvent, MOUSE_EVENT_BUFFER_SIZE> mouse_event_buffer;
    std::atomic<size_t> mouse_event_write_pos{0};
    std::atomic<size_t> mouse_event_read_pos{0};
    std::condition_variable mouse_event_cv;
    
    // GPU 직접 계산된 마우스 이동량 - Lock-free structure
    struct GPUMouseMovement {
        std::atomic<int> dx{0};
        std::atomic<int> dy{0};
        std::atomic<float> confidence{0.0f};
        std::atomic<bool> hasTarget{false};
    };
    GPUMouseMovement latestMouseMovement;
    std::atomic<bool> mouseDataReady{false};
    std::condition_variable mouseDataCV;
    std::mutex mouseDataMutex;  // Keep mutex for CV, but data access is lock-free
    
    // Event-based inference control
    std::mutex inference_frame_mutex;
    std::condition_variable inference_frame_cv;
    std::atomic<bool> inference_frame_ready{false};

    // CUDA Graph optimization
    std::atomic<bool> use_cuda_graph{false};

    // Overlay Target Data
    
    // Target management helpers
    void updateTargets(const std::vector<Target>& targets) {
        std::lock_guard<std::mutex> lock(target_mutex);
        // Copy to fixed array
        num_targets_ = (targets.size() < MAX_TARGETS) ? targets.size() : MAX_TARGETS;
        for (size_t i = 0; i < num_targets_; ++i) {
            all_targets_[i] = targets[i];
        }
        has_target_ = !targets.empty();
        
        if (!targets.empty()) {
            auto bestTarget = std::max_element(targets.begin(), targets.end(),
                [](const Target& a, const Target& b) {
                    return a.confidence < b.confidence;
                });
            current_target_ = *bestTarget;
        }
    }
    
    Target getBestTarget() const {
        std::lock_guard<std::mutex> lock(target_mutex);
        return current_target_;
    }
    
    std::vector<Target> getAllTargets() const {
        std::lock_guard<std::mutex> lock(target_mutex);
        // Convert array to vector for compatibility
        return std::vector<Target>(all_targets_.begin(), all_targets_.begin() + num_targets_);
    }
    
    void clearTargets() {
        std::lock_guard<std::mutex> lock(target_mutex);
        num_targets_ = 0;
        has_target_ = false;
        current_target_ = Target{};
    }
    
    bool hasValidTarget() const {
        return has_target_.load();
    }

private:
    AppContext() : new_frame_available(false) {
        // State variables are initialized with their default values
    }
};

#endif // APP_CONTEXT_H