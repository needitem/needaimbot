#ifndef APP_CONTEXT_H
#define APP_CONTEXT_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include "config/config.h"
#include "detector/postProcess.h"

class MouseThread; // Forward declaration
class Detector;

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
    std::vector<cv::cuda::GpuMat> captureGpuBuffer;
    std::atomic<int> captureGpuWriteIdx{0};
    std::vector<cv::Mat> captureCpuBuffer;
    std::atomic<int> captureCpuWriteIdx{0};

    // Frame synchronization
    std::mutex frameMutex;
    std::condition_variable frameCV;
    std::atomic<bool> newFrameAvailable{false};

    // Detection synchronization
    std::mutex detectionMutex;
    std::condition_variable detectionCV;

    // Application state
    std::atomic<bool> shouldExit{false};
    std::atomic<bool> aiming{false};
    std::atomic<bool> input_method_changed{false};
    
    // Capture state changes
    std::atomic<bool> capture_fps_changed{false};
    std::atomic<bool> detection_resolution_changed{false};
    std::atomic<bool> capture_cursor_changed{false};
    std::atomic<bool> capture_borders_changed{false};
    std::atomic<bool> capture_timeout_changed{false};
    std::atomic<bool> capture_method_changed{false};
    
    // Performance metrics
    std::atomic<float> g_current_frame_acquisition_time_ms{0.0f};
    std::atomic<float> g_current_capture_fps{0.0f};
    std::vector<float> g_frame_acquisition_time_history;
    std::vector<float> g_capture_fps_history;
    std::mutex g_frame_acquisition_history_mutex;
    std::mutex g_capture_history_mutex;
    
    // Detector performance metrics
    std::atomic<float> g_current_process_frame_time_ms{0.0f};
    std::atomic<float> g_current_detector_cycle_time_ms{0.0f};
    std::atomic<float> g_current_inference_time_ms{0.0f};
    std::atomic<float> g_current_pid_calc_time_ms{0.0f};
    std::atomic<float> g_current_input_send_time_ms{0.0f};
    std::atomic<float> g_current_detection_to_movement_time_ms{0.0f};
    std::atomic<float> g_current_fps_delay_time_ms{0.0f};
    std::atomic<float> g_current_total_cycle_time_ms{0.0f};
    std::vector<float> g_process_frame_time_history;
    std::vector<float> g_detector_cycle_time_history;
    std::vector<float> g_inference_time_history;
    std::vector<float> g_pid_calc_time_history;
    std::vector<float> g_input_send_time_history;
    std::vector<float> g_detection_to_movement_time_history;
    std::vector<float> g_fps_delay_time_history;
    std::vector<float> g_total_cycle_time_history;
    std::mutex g_process_frame_history_mutex;
    std::mutex g_detector_cycle_history_mutex;
    std::mutex g_inference_history_mutex;
    std::mutex g_pid_calc_history_mutex;
    std::mutex g_input_send_history_mutex;
    std::mutex g_detection_to_movement_history_mutex;
    std::mutex g_fps_delay_history_mutex;
    std::mutex g_total_cycle_history_mutex;
    
    // Application control
    std::atomic<bool> detectionPaused{false};
    std::atomic<bool> detector_model_changed{false};
    std::mutex configMutex;

    // Modules
    MouseThread* globalMouseThread = nullptr;  // Stack allocated in main, so using raw pointer
    Detector* detector = nullptr;  // Changed back to raw pointer to avoid incomplete type issues

    // Overlay Target Data (Synchronized for UI)
    std::atomic<bool> overlay_has_target{false};
    Detection overlay_target_info{}; // Zero-initialized
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
    AppContext() : captureGpuWriteIdx(0), captureCpuWriteIdx(0), newFrameAvailable(false) {
        // Initialize capture buffers
        captureGpuBuffer.resize(4); // Use literal instead of FRAME_BUFFER_COUNT to avoid circular include
        captureCpuBuffer.resize(4);
    }
};

#endif // APP_CONTEXT_H
