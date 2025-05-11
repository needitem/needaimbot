#include "imgui.h"
#include "../sunone_aimbot_cpp.h" // For accessing global stats variables
#include "overlay/draw_settings.h" // For the function declaration
#include "../config/config.h" // For accessing config for verbose mode etc.

#include <string>   // For snprintf
#include <vector>   // For local copy of history
#include <cstdio>   // For snprintf
#include <algorithm> // For std::min_element, std::max_element
#include <cmath>     // For std::abs

// Helper to get a copy of the history for thread-safe plotting
static std::vector<float> get_history_copy(const std::vector<float>& history, std::mutex& mtx) {
    std::lock_guard<std::mutex> lock(mtx);
    return history; // Return a copy
}

// Helper to determine plot Y scale
static void get_plot_scale(const std::vector<float>& history, float& min_y, float& max_y, bool is_fps = false) {
    min_y = 0.0f;
    if (is_fps) {
        max_y = 60.0f; // Default sensible max for FPS
    } else {
        max_y = 10.0f; // Default sensible max for ms times
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
    max_y *= 1.1f; // Add 10% headroom
    if (is_fps && max_y < 60.0f) max_y = 60.0f; 
    else if (!is_fps && max_y < 10.0f) max_y = 10.0f; 
}

void draw_stats() {
    ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x, 60.0f);
    float min_y, max_y;

    // Detector Cycle Time (Previously Detector Loop Cycle Time)
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

    // Frame Acquisition Time
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

    // Inference Time
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

    // Capture FPS
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
}
