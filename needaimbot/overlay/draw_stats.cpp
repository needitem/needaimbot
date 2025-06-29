#include "imgui.h"
#include "needaimbot.h" 
#include "overlay/draw_settings.h" 
#include "../config/config.h" 

#include <string>   
#include <vector>   
#include <cstdio>   
#include <algorithm> 
#include <cmath>     
#include <fstream>
#include <numeric>

// Static variables for stats logging
static bool stats_log_header_written = false;
static int stats_log_frame_counter = 0;
static const int stats_log_interval = 60; // log every 60 frames

static std::vector<float> get_history_copy(const std::vector<float>& history, std::mutex& mtx) {
    std::lock_guard<std::mutex> lock(mtx);
    return history; 
}


static void get_plot_scale(const std::vector<float>& history, float& min_y, float& max_y, bool is_fps = false) {
    min_y = 0.0f;
    if (is_fps) {
        max_y = 60.0f; 
    } else {
        max_y = 10.0f; 
    }

    if (!history.empty()) {
        float current_max = 0.0f;
        bool first = true;
        for (size_t i = 0; i < history.size(); ++i) {
            if (first) {
                current_max = history[i];
                first = false;
            } else if (history[i] > current_max) {
                current_max = history[i];
            }
        }
        if (current_max > max_y) {
            max_y = current_max;
        }
    }
    max_y *= 1.1f; 
    if (is_fps && max_y < 60.0f) max_y = 60.0f; 
    else if (!is_fps && max_y < 10.0f) max_y = 10.0f; 
}

void draw_stats() {
    ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x, 60.0f);
    float min_y, max_y;

    
    ImGui::Text("Detector Cycle Time");
    float current_cycle_time = g_current_detector_cycle_time_ms.load(std::memory_order_relaxed);
    std::vector<float> cycle_history = get_history_copy(g_detector_cycle_time_history, g_detector_cycle_history_mutex);
    char cycle_label[128];
    snprintf(cycle_label, sizeof(cycle_label), "Current: %.2f ms", current_cycle_time);
    get_plot_scale(cycle_history, min_y, max_y);
    if (!cycle_history.empty()) {
        ImGui::PlotLines("##DetectorCyclePlot", cycle_history.data(), static_cast<int>(cycle_history.size()), 
                         0, nullptr, min_y, max_y, plot_size); 
    } else {
        ImGui::Dummy(plot_size);
        ImGui::Text("No cycle data yet.");
    }
    ImGui::TextUnformatted(cycle_label);
    ImGui::Separator();

    
    ImGui::Text("Frame Acquisition Time");
    float current_acq_time = g_current_frame_acquisition_time_ms.load(std::memory_order_relaxed);
    std::vector<float> acq_history = get_history_copy(g_frame_acquisition_time_history, g_frame_acquisition_history_mutex);
    char acq_label[128];
    snprintf(acq_label, sizeof(acq_label), "Current: %.2f ms", current_acq_time);
    get_plot_scale(acq_history, min_y, max_y);
    if (!acq_history.empty()) {
        ImGui::PlotLines("##FrameAcquisitionPlot", acq_history.data(), static_cast<int>(acq_history.size()), 
                         0, nullptr, min_y, max_y, plot_size); 
    } else {
        ImGui::Dummy(plot_size); 
        ImGui::Text("No acquisition data yet.");
    }
    ImGui::TextUnformatted(acq_label);
    ImGui::Separator();

    
    ImGui::Text("Inference Time");
    float current_inference_time = g_current_inference_time_ms.load(std::memory_order_relaxed);
    std::vector<float> inference_history = get_history_copy(g_inference_time_history, g_inference_history_mutex);
    char inference_label[128];
    snprintf(inference_label, sizeof(inference_label), "Current: %.2f ms", current_inference_time);
    get_plot_scale(inference_history, min_y, max_y);
    if (!inference_history.empty()) {
        ImGui::PlotLines("##InferenceTimePlot", inference_history.data(), static_cast<int>(inference_history.size()), 
                         0, nullptr, min_y, max_y, plot_size);
    } else {
        ImGui::Dummy(plot_size);
        ImGui::Text("No inference data yet.");
    }
    ImGui::TextUnformatted(inference_label);
    ImGui::Separator();

    
    ImGui::Text("Capture FPS");
    float current_fps = g_current_capture_fps.load(std::memory_order_relaxed);
    std::vector<float> fps_history = get_history_copy(g_capture_fps_history, g_capture_history_mutex);
    char fps_label[128];
    snprintf(fps_label, sizeof(fps_label), "Current: %.1f FPS", current_fps);
    get_plot_scale(fps_history, min_y, max_y, true);
    if (!fps_history.empty()) {
        ImGui::PlotLines("##CaptureFPSPlot", fps_history.data(), static_cast<int>(fps_history.size()), 
                         0, nullptr, min_y, max_y, plot_size);
    } else {
        ImGui::Dummy(plot_size);
        ImGui::Text("No FPS data yet.");
    }
    ImGui::TextUnformatted(fps_label);
    ImGui::Separator();

    
    ImGui::Text("PID Calculation Time");
    float current_pid_time = g_current_pid_calc_time_ms.load(std::memory_order_relaxed);
    std::vector<float> pid_history = get_history_copy(g_pid_calc_time_history, g_pid_calc_history_mutex);
    char pid_label[128];
    snprintf(pid_label, sizeof(pid_label), "Current: %.3f ms", current_pid_time);
    get_plot_scale(pid_history, min_y, max_y);
    if (!pid_history.empty()) {
        ImGui::PlotLines("##PIDCalcTimePlot", pid_history.data(), static_cast<int>(pid_history.size()),
                         0, nullptr, min_y, max_y, plot_size);
    } else {
        ImGui::Dummy(plot_size);
        ImGui::Text("No PID data yet.");
    }
    ImGui::TextUnformatted(pid_label);
    ImGui::Separator();

    
    

    
    ImGui::Text("Input Send Time");
    float current_input_send_time = g_current_input_send_time_ms.load(std::memory_order_relaxed);
    std::vector<float> input_send_history = get_history_copy(g_input_send_time_history, g_input_send_history_mutex);
    char input_send_label[128];
    snprintf(input_send_label, sizeof(input_send_label), "Current: %.3f ms", current_input_send_time);
    get_plot_scale(input_send_history, min_y, max_y);
    if (!input_send_history.empty()) {
        ImGui::PlotLines("##InputSendTimePlot", input_send_history.data(), static_cast<int>(input_send_history.size()),
                         0, nullptr, min_y, max_y, plot_size);
    } else {
        ImGui::Dummy(plot_size);
        ImGui::Text("No input send data yet.");
    }
    ImGui::TextUnformatted(input_send_label);
    
    // Log stats to CSV file periodically
    stats_log_frame_counter++;
    if (stats_log_frame_counter >= stats_log_interval) {
        stats_log_frame_counter = 0;
        double avg_cycle = 0.0;
        if (!cycle_history.empty()) {
            avg_cycle = std::accumulate(cycle_history.begin(), cycle_history.end(), 0.0) / cycle_history.size();
        }
        double avg_acq = 0.0;
        if (!acq_history.empty()) {
            avg_acq = std::accumulate(acq_history.begin(), acq_history.end(), 0.0) / acq_history.size();
        }
        double avg_inf = 0.0;
        if (!inference_history.empty()) {
            avg_inf = std::accumulate(inference_history.begin(), inference_history.end(), 0.0) / inference_history.size();
        }
        double avg_pid = 0.0;
        if (!pid_history.empty()) {
            avg_pid = std::accumulate(pid_history.begin(), pid_history.end(), 0.0) / pid_history.size();
        }
        double avg_input = 0.0;
        if (!input_send_history.empty()) {
            avg_input = std::accumulate(input_send_history.begin(), input_send_history.end(), 0.0) / input_send_history.size();
        }
        double avg_fps = 0.0;
        if (!fps_history.empty()) {
            avg_fps = std::accumulate(fps_history.begin(), fps_history.end(), 0.0) / fps_history.size();
        }
    }
}

