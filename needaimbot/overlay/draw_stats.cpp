#include "../AppContext.h"
#include "../imgui/imgui.h"
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

void draw_stat_plot(const char* label, const std::vector<float>& history, float current_value, const char* unit, bool is_fps = false) {
    auto& ctx = AppContext::getInstance();
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", label);

    ImGui::TableSetColumnIndex(1);
    ImGui::PushItemWidth(-1);

    float min_y, max_y;
    get_plot_scale(history, min_y, max_y, is_fps);

    ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x, 50.0f);

    if (!history.empty()) {
        // Create unique ID for each plot using the label
        std::string unique_id = "##plot_" + std::string(label);
        ImGui::PlotLines(unique_id.c_str(), history.data(), static_cast<int>(history.size()), 0, nullptr, min_y, max_y, plot_size);
    } else {
        ImGui::Dummy(plot_size);
        ImGui::Text("No data yet.");
    }

    ImGui::Text("Current: %.2f %s", current_value, unit);
    ImGui::PopItemWidth();
}

void draw_stats() {
    auto& ctx = AppContext::getInstance();
    if (ImGui::BeginTable("stats_table", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthStretch, 0.4f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.6f);

        draw_stat_plot("Capture Time", get_history_copy(ctx.g_frame_acquisition_time_history, ctx.g_frame_acquisition_history_mutex), ctx.g_current_frame_acquisition_time_ms.load(std::memory_order_relaxed), "ms");
        draw_stat_plot("Inference Time", get_history_copy(ctx.g_inference_time_history, ctx.g_inference_history_mutex), ctx.g_current_inference_time_ms.load(std::memory_order_relaxed), "ms");
        draw_stat_plot("Mouse Movement Time", get_history_copy(ctx.g_input_send_time_history, ctx.g_input_send_history_mutex), ctx.g_current_input_send_time_ms.load(std::memory_order_relaxed), "ms");
        draw_stat_plot("Total Cycle Time", get_history_copy(ctx.g_total_cycle_time_history, ctx.g_total_cycle_history_mutex), ctx.g_current_total_cycle_time_ms.load(std::memory_order_relaxed), "ms");
        draw_stat_plot("FPS Delay", get_history_copy(ctx.g_fps_delay_time_history, ctx.g_fps_delay_history_mutex), ctx.g_current_fps_delay_time_ms.load(std::memory_order_relaxed), "ms");
        

        ImGui::EndTable();
    }
}

