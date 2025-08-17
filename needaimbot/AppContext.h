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
#include "core/states/CaptureState.h"
#include "core/states/DetectionState.h"
#include "core/metrics/PerformanceMetrics.h"


class MouseThread; // Forward declaration
class Detector;
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

#include "core/states/CaptureState.h"

    static AppContext& getInstance() {
        static AppContext instance;
        return instance;
    }

    // Config
    Config config;

    // State management - 새로운 상태 관리 시스템
    std::unique_ptr<Core::CaptureState> captureState_;
    std::unique_ptr<Core::DetectionState> detectionState_;
    std::unique_ptr<Core::PerformanceMetrics> performanceMetrics_;

    // Legacy frame synchronization (will be moved to CaptureState gradually)
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
    
    // Detection resolution changes (TODO: move to DetectionState)
    std::atomic<bool> detection_resolution_changed{false};
    
    // Performance metrics (TODO: PerformanceMetrics로 이동됨)
    
    // Application control (TODO: 일부 기능은 DetectionState로 이동됨)
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
    Detector* detector = nullptr;  // Raw pointer, lifetime managed externally

    // Overlay Target Data (TODO: DetectionState로 이동됨)
    
    // State 접근자
    Core::CaptureState& getCaptureState() { return *captureState_; }
    const Core::CaptureState& getCaptureState() const { return *captureState_; }
    
    Core::DetectionState& getDetectionState() { return *detectionState_; }
    const Core::DetectionState& getDetectionState() const { return *detectionState_; }
    
    Core::PerformanceMetrics& getPerformanceMetrics() { return *performanceMetrics_; }
    const Core::PerformanceMetrics& getPerformanceMetrics() const { return *performanceMetrics_; }

    // Helper functions
    void add_to_history(std::vector<float>& history, float value, std::mutex& mutex) {
        std::lock_guard<std::mutex> lock(mutex);
        history.push_back(value);
        if (history.size() > 100) {
            history.erase(history.begin());
        }
    }

private:
    AppContext() : new_frame_available(false) {
        // Initialize new state management
        captureState_ = std::make_unique<Core::CaptureState>(4);
        detectionState_ = std::make_unique<Core::DetectionState>();
        performanceMetrics_ = std::make_unique<Core::PerformanceMetrics>();
        
        // Initialize capture_method from config after config is loaded
        captureState_->setCaptureMethod(config.gpu_capture_method);
    }
};

#endif // APP_CONTEXT_H