#ifndef APP_CONTEXT_H
#define APP_CONTEXT_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "cuda/simple_cuda_mat.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include "config/config.h"
#include "cuda/detection/postProcess.h"
#include <queue>
#include <chrono>


class MouseThread; // Forward declaration
class RecoilControlThread; // Forward declaration

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

    // State management - atomic variables replacing State classes
    std::atomic<bool> detection_paused{false};
    std::atomic<bool> preview_enabled{false};
    std::atomic<bool> capture_method_changed{false};
    std::atomic<bool> crosshair_offset_changed{false};
    std::atomic<bool> model_changed{false};
    
    // Target data
    mutable std::mutex target_mutex;
    std::vector<Target> all_targets_;
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
    std::atomic<bool> input_method_changed{false};
    
    std::atomic<bool> detection_resolution_changed{false};
    
    
    // Application control
    std::mutex configMutex;
    
    std::atomic<float> g_movementDeltaX{0.0f};
    std::atomic<float> g_movementDeltaY{0.0f};
    
    // Event-based mouse control
    std::queue<MouseEvent> mouse_event_queue;
    std::mutex mouse_event_mutex;
    std::condition_variable mouse_event_cv;
    std::atomic<bool> mouse_events_available{false};
    
    // GPU 직접 계산된 마우스 이동량
    struct GPUMouseMovement {
        int dx = 0;
        int dy = 0;
        float confidence = 0.0f;
        bool hasTarget = false;
    };
    GPUMouseMovement latestMouseMovement;
    std::atomic<bool> mouseDataReady{false};
    std::condition_variable mouseDataCV;
    std::mutex mouseDataMutex;
    
    // Event-based inference control
    std::mutex inference_frame_mutex;
    std::condition_variable inference_frame_cv;
    std::atomic<bool> inference_frame_ready{false};

    // CUDA Graph optimization
    std::atomic<bool> use_cuda_graph{false};
    
    // Modules
    MouseThread* mouseThread = nullptr;  // Stack allocated in main, so using raw pointer
    // Detector removed - TensorRT is now integrated into UnifiedGraphPipeline

    // Overlay Target Data
    
    // Target management helpers
    void updateTargets(const std::vector<Target>& targets) {
        std::lock_guard<std::mutex> lock(target_mutex);
        all_targets_ = targets;
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
        return all_targets_;
    }
    
    void clearTargets() {
        std::lock_guard<std::mutex> lock(target_mutex);
        all_targets_.clear();
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