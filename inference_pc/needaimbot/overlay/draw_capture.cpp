#include "../core/windows_headers.h"
#include <Psapi.h>

#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>

#include "AppContext.h"
#include "../core/constants.h"
#include "config/config.h"
#include "needaimbot.h"
// #include "../capture/capture.h" - removed, using GPU capture now
#include "include/other_tools.h"
#include "draw_settings.h"
#include "ui_helpers.h"
#include "../cuda/unified_graph_pipeline.h"

// Monitor count is now simplified - just count all monitors
int monitors = GetSystemMetrics(SM_CMONITORS);

static void draw_capture_area_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Capture Area");

    // Resolution slider
    float detection_res_float = static_cast<float>(ctx.config.profile().detection_resolution);
    int old_resolution = ctx.config.profile().detection_resolution;

    if (UIHelpers::EnhancedSliderFloat("Detection Resolution", &detection_res_float, 50.0f, 1280.0f, "%.0f px",
                                       "Size of the capture area around cursor.\nSmaller = faster, larger = wider FOV")) {
        int new_resolution = static_cast<int>(detection_res_float);
        if (new_resolution != old_resolution) {
            ctx.config.profile().detection_resolution = new_resolution;
            SAVE_PROFILE();
            extern std::atomic<bool> detection_resolution_changed;
            detection_resolution_changed.store(true);
            ctx.model_changed = true;
        }
    }

    // Performance warning
    if (ctx.config.profile().detection_resolution >= Constants::DETECTION_RESOLUTION_HIGH_PERF) {
        UIHelpers::CompactSpacer();
        ImGui::TextColored(UIHelpers::GetWarningColor(), "High resolution may impact FPS");
    }

    UIHelpers::CompactSpacer();

    if (UIHelpers::BeautifulToggle("Circle Mask", &ctx.config.profile().circle_mask,
                                   "Ignore corners of capture area")) {
        SAVE_PROFILE();
    }

    UIHelpers::EndCard();

    // Capture Debug Card
    UIHelpers::CompactSpacer();
    UIHelpers::BeginCard("Capture Info");
    {
        auto* pipeline = gpa::PipelineManager::getInstance().getPipeline();
        if (pipeline) {
            gpa::UnifiedGraphPipeline::CaptureStats stats{};
            pipeline->getCaptureStats(stats);

            if (ImGui::BeginTable("##capture_info", 2, ImGuiTableFlags_NoBordersInBody)) {
                ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                auto addRow = [](const char* label, const char* fmt, auto... args) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s", label);
                    ImGui::TableNextColumn();
                    ImGui::Text(fmt, args...);
                };

                addRow("Backend", "%s", stats.backend ? stats.backend : "N/A");
                addRow("Frame", "%dx%d", stats.lastWidth, stats.lastHeight);
                addRow("Mode", "%s", stats.gpuDirect ? "GPU Direct" : "CPU Copy");
                addRow("Region", "%d,%d (%dpx)", stats.roiLeft, stats.roiTop, stats.roiSize);

                ImGui::EndTable();
            }
        } else {
            ImGui::TextDisabled("Pipeline not initialized");
        }
    }
    UIHelpers::EndCard();
}

static void draw_capture_behavior_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Performance Tuning");

    // Pipeline loop delay with better presentation
    int delay = ctx.config.profile().pipeline_loop_delay_ms;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("##loop_delay", &delay, 0, 50, "Loop Delay: %d ms")) {
        ctx.config.profile().pipeline_loop_delay_ms = delay;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Delay per loop iteration.\n0 = fastest (may drop game FPS)\n1-5 = recommended\n10+ = safe for weak systems");
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        delay = std::clamp(delay, 0, 50);
        ctx.config.profile().pipeline_loop_delay_ms = delay;
        if (delay == 0) {
            ImGui::OpenPopup("Warning##LoopDelayWarning");
        }
        SAVE_PROFILE();
    }

    // Warning popup
    if (ImGui::BeginPopup("Warning##LoopDelayWarning")) {
        ImGui::TextColored(UIHelpers::GetWarningColor(), "Warning!");
        ImGui::Text("Loop delay 0 may cause game FPS drops.");
        ImGui::Text("Recommended: 1-5ms");
        if (UIHelpers::BeautifulButton("OK", ImVec2(80, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    UIHelpers::CompactSpacer();

    // Timeout scale (advanced)
    float scale = ctx.config.profile().capture_timeout_scale;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##timeout_scale", &scale, 0.50f, 0.70f, "Timeout Scale: %.2f")) {
        // Just preview
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Frame acquisition timing.\nLower = more responsive, higher = more stable");
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        scale = std::clamp(scale, 0.50f, 0.80f);
        if (fabsf(scale - ctx.config.profile().capture_timeout_scale) > 1e-3f) {
            ctx.config.profile().capture_timeout_scale = scale;
            SAVE_PROFILE();
        }
    }

    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    UIHelpers::BeginCard("Capture Options");

    if (ImGui::BeginTable("##capture_opts", 2, ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (UIHelpers::BeautifulToggle("Include Borders", &ctx.config.profile().capture_borders)) {
            SAVE_PROFILE();
        }
        ImGui::TableNextColumn();
        if (UIHelpers::BeautifulToggle("Include Cursor", &ctx.config.profile().capture_cursor)) {
            SAVE_PROFILE();
        }
        ImGui::EndTable();
    }

    UIHelpers::EndCard();
}

static void draw_capture_source_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Capture Source (CUDA Only)");
    
    std::vector<std::string> monitorNames;
    if (monitors == -1)
    {
        monitorNames.push_back("Monitor 1");
    }
    else
    {
        for (int i = -1; i < monitors; ++i)
        {
            monitorNames.push_back("Monitor " + std::to_string(i + 1));
        }
    }

    std::vector<const char*> monitorItems;
    for (const auto& name : monitorNames)
    {
        monitorItems.push_back(name.c_str());
    }

    UIHelpers::CompactCombo("Capture Monitor", &ctx.config.profile().monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size()));
    UIHelpers::InfoTooltip("Select which monitor to capture from when using CUDA-based screen capture.");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        SAVE_PROFILE();
    }
    
    UIHelpers::EndCard();
}


void draw_capture_settings()
{
    draw_capture_area_settings();
    UIHelpers::Spacer();

    draw_capture_behavior_settings();
    UIHelpers::Spacer();

    draw_capture_source_settings();
}

