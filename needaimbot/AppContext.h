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

    static AppContext& getInstance() {
        static AppContext instance;
        return instance;
    }

    // Config
    Config config;

    // Capture buffers
    std::vector<SimpleCudaMat> captureGpuBuffer;
    std::atomic<int> captureGpuWriteIdx{0};
    std::vector<SimpleMat> captureCpuBuffer;
    std::atomic<int> captureCpuWriteIdx{0};

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
    
    // Capture state changes
    std::atomic<bool> capture_fps_changed{false};
    std::atomic<bool> detection_resolution_changed{false};
    std::atomic<bool> capture_cursor_changed{false};
    std::atomic<bool> capture_borders_changed{false};
    std::atomic<bool> capture_timeout_changed{false};
    std::atomic<bool> capture_method_changed{false};
    std::atomic<bool> crosshair_offset_changed{false};
    
    // Performance metrics
    std::atomic<float> g_current_frame_acquisition_time_ms{0.0f};
    std::atomic<float> g_current_capture_fps{0.0f};
    std::atomic<float> g_current_capture_cycle_time_ms{0.0f};
    std::vector<float> g_frame_acquisition_time_history;
    std::vector<float> g_capture_fps_history;
    std::vector<float> g_capture_cycle_time_history;
    std::mutex g_frame_acquisition_history_mutex;
    std::mutex g_capture_history_mutex;
    std::mutex g_capture_cycle_history_mutex;
    
    // Detector performance metrics
    std::atomic<float> g_current_process_frame_time_ms{0.0f};
    std::atomic<float> g_current_detector_cycle_time_ms{0.0f};
    std::atomic<float> g_current_inference_time_ms{0.0f};
    std::atomic<float> g_current_input_send_time_ms{0.0f};
    std::atomic<float> g_current_detection_to_movement_time_ms{0.0f};
    std::atomic<float> g_current_fps_delay_time_ms{0.0f};
    std::atomic<float> g_current_total_cycle_time_ms{0.0f};
    std::vector<float> g_process_frame_time_history;
    std::vector<float> g_detector_cycle_time_history;
    std::vector<float> g_inference_time_history;
    std::vector<float> g_input_send_time_history;
    std::vector<float> g_detection_to_movement_time_history;
    std::vector<float> g_fps_delay_time_history;
    std::vector<float> g_total_cycle_time_history;
    std::mutex g_process_frame_history_mutex;
    std::mutex g_detector_cycle_history_mutex;
    std::mutex g_inference_history_mutex;
    std::mutex g_input_send_history_mutex;
    std::mutex g_detection_to_movement_history_mutex;
    std::mutex g_fps_delay_history_mutex;
    std::mutex g_total_cycle_history_mutex;
    
    // Application control
    std::atomic<bool> detectionPaused{false};
    std::atomic<bool> detector_model_changed{false};
    std::mutex configMutex;
    
    // GPU-calculated movement deltas (deprecated - use event queue instead)
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

    // Overlay Target Data (Synchronized for UI)
    std::atomic<bool> overlay_has_target{false};
    Target overlay_target_info{}; // Zero-initialized
    std::mutex overlay_target_mutex;
    
    // Helper functions
    void add_to_history(std::vector<float>& history, float value, std::mutex& mutex) {
        std::lock_guard<std::mutex> lock(mutex);
        history.push_back(value);
        if (history.size() > 100) {
            history.erase(history.begin());
        }
    }

private:
    AppContext() : captureGpuWriteIdx(0), captureCpuWriteIdx(0), new_frame_available(false) {
        // Initialize capture buffers
        captureGpuBuffer.resize(4); // Use literal instead of FRAME_BUFFER_COUNT to avoid circular include
        captureCpuBuffer.resize(4);
    }
};

#endif // APP_CONTEXT_H