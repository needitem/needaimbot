#include "imgui.h"
#include "../sunone_aimbot_cpp.h" // For accessing global stats variables
#include "overlay/draw_settings.h" // For the function declaration

#include <string>   // For snprintf
#include <vector>   // For local copy of history
#include <cstdio>   // For snprintf
#include <algorithm> // For std::min/max if needed for plot scaling

// Helper to get a copy of the history for thread-safe plotting
static std::vector<float> get_history_copy(const std::vector<float>& history, std::mutex& mtx) {
    std::lock_guard<std::mutex> lock(mtx);
    return history; // Return a copy
}

void draw_stats() {
    // Inference Time
    if (ImGui::CollapsingHeader("Inference Time", ImGuiTreeNodeFlags_DefaultOpen)) {
        float current_inference_time = g_current_inference_time_ms.load(std::memory_order_relaxed);
        std::vector<float> inference_history = get_history_copy(g_inference_time_history, g_inference_history_mutex);

        char inference_label[128];
        snprintf(inference_label, sizeof(inference_label), "Inference: %.2f ms", current_inference_time);
        
        ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x, 80.0f);
        if (!inference_history.empty()) {
            ImGui::PlotLines("##InferenceTimePlot", inference_history.data(), static_cast<int>(inference_history.size()), 
                             0, nullptr, 0.0f, 50.0f, plot_size); // Adjust max Y (50ms) as needed
        } else {
            ImGui::Dummy(plot_size); // Keep layout consistent if no data
            ImGui::Text("No inference data yet.");
        }
        ImGui::TextUnformatted(inference_label);
    }

    ImGui::Separator();

    // Capture FPS
    if (ImGui::CollapsingHeader("Capture FPS", ImGuiTreeNodeFlags_DefaultOpen)) {
        float current_fps = g_current_capture_fps.load(std::memory_order_relaxed);
        std::vector<float> fps_history = get_history_copy(g_capture_fps_history, g_capture_history_mutex);

        char fps_label[128];
        snprintf(fps_label, sizeof(fps_label), "FPS: %.1f", current_fps);

        ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x, 80.0f);
        if (!fps_history.empty()) {
            ImGui::PlotLines("##CaptureFPSPlot", fps_history.data(), static_cast<int>(fps_history.size()), 
                             0, nullptr, 0.0f, 150.0f, plot_size); // Adjust max Y (150 FPS) as needed
        } else {
            ImGui::Dummy(plot_size); // Keep layout consistent
            ImGui::Text("No FPS data yet.");
        }
        ImGui::TextUnformatted(fps_label);
    }
}
