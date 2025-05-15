#ifndef NEEDAIMBOT_H
#define NEEDAIMBOT_H

#include "config.h"
#include "detector.h"
#include "mouse.h"
#include "mouse/input_drivers/SerialConnection.h"
#include <vector>
#include <atomic>
#include <mutex>
#include "capture/optical_flow.h"

extern Config config;
extern Detector detector;
extern MouseThread* globalMouseThread;
extern SerialConnection* arduinoSerial;
extern std::atomic<bool> input_method_changed;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;
extern std::atomic<bool> capture_timeout_changed;
extern OpticalFlow opticalFlow;
extern std::atomic<bool> config_optical_flow_changed;

// Maximum history size for plots
const int STATS_HISTORY_SIZE = 100;

// Inference Time
extern std::atomic<float> g_current_inference_time_ms;
extern std::vector<float> g_inference_time_history;
extern std::mutex g_inference_history_mutex;

// Capture FPS
extern std::atomic<float> g_current_capture_fps;
extern std::vector<float> g_capture_fps_history;
extern std::mutex g_capture_history_mutex;

// Detector Loop Cycle Time
extern std::atomic<float> g_current_detector_cycle_time_ms;
extern std::vector<float> g_detector_cycle_time_history;
extern std::mutex g_detector_cycle_history_mutex;

// Frame Acquisition Time
extern std::atomic<float> g_current_frame_acquisition_time_ms;
extern std::vector<float> g_frame_acquisition_time_history;
extern std::mutex g_frame_acquisition_history_mutex;

// PID Calculation Time
extern std::atomic<float> g_current_pid_calc_time_ms;
extern std::vector<float> g_pid_calc_time_history;
extern std::mutex g_pid_calc_history_mutex;

// Predictor Calculation Time
extern std::atomic<float> g_current_predictor_calc_time_ms;
extern std::vector<float> g_predictor_calc_time_history;
extern std::mutex g_predictor_calc_history_mutex;

// Input Send Time
extern std::atomic<float> g_current_input_send_time_ms;
extern std::vector<float> g_input_send_time_history;
extern std::mutex g_input_send_history_mutex;

// Function to update history
void add_to_history(std::vector<float>& history, float value, std::mutex& mtx, int max_size = STATS_HISTORY_SIZE);

#endif // NEEDAIMBOT_H