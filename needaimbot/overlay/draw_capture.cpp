#include "../core/windows_headers.h"
#include <Psapi.h>

#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"

#include "AppContext.h"
#include "../core/constants.h"
#include "config/config.h"
#include "needaimbot.h"
// #include "../capture/capture.h" - removed, using GPU capture now
#include "include/other_tools.h"
#include "draw_settings.h"
#include "ui_helpers.h"

// Monitor count is now simplified - just count all monitors
int monitors = GetSystemMetrics(SM_CMONITORS);

static void draw_capture_area_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Capture Area & Resolution");
    
    // Use temporary float for the slider
    float detection_res_float = static_cast<float>(ctx.config.detection_resolution);
    
    // Store the old value to detect changes
    int old_resolution = ctx.config.detection_resolution;
    
    UIHelpers::CompactSlider("Detection Resolution", &detection_res_float, 50.0f, 1280.0f, "%.0f");
    UIHelpers::InfoTooltip("Size (in pixels) of the square area around the cursor to capture for detection.\nSmaller values improve performance but may miss targets further from the crosshair.");
    
    // Check if value changed
    int new_resolution = static_cast<int>(detection_res_float);
    if (new_resolution != old_resolution) {
        ctx.config.detection_resolution = new_resolution;
        SAVE_PROFILE();
        
        // Force update flags
        extern std::atomic<bool> detection_resolution_changed;
        // detector_model_changed is now managed by DetectionState
        detection_resolution_changed.store(true);
        ctx.model_changed = true;
    }
    
    if (ctx.config.detection_resolution >= Constants::DETECTION_RESOLUTION_HIGH_PERF)
    {
        UIHelpers::BeautifulText("WARNING: Large detection resolution can impact performance.", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulToggle("Circle Mask", &ctx.config.circle_mask, "Applies a circular mask to the captured area, ignoring corners."))
    {
        SAVE_PROFILE();
    }
    
    UIHelpers::EndCard();
}

static void draw_capture_behavior_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Capture Behavior");

    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Desktop Duplication Capture Active");
    UIHelpers::InfoTooltip("The Windows Desktop Duplication API (DDA) powers high-quality capture with low latency.\nThis path delivers consistent results across modern GPUs without requiring vendor-specific drivers.");
    
    UIHelpers::CompactSpacer();
    
    
    ImGui::Columns(2, "capture_options", false);
    
    if (UIHelpers::BeautifulToggle("Capture Borders", &ctx.config.capture_borders, "Includes window borders in the screen capture (if applicable)."))
    {
        SAVE_PROFILE();
    }
    
    ImGui::NextColumn();
    
    if (UIHelpers::BeautifulToggle("Capture Cursor", &ctx.config.capture_cursor, "Includes the mouse cursor in the screen capture."))
    {
        SAVE_PROFILE();
    }
    
    ImGui::Columns(1);
    
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

    UIHelpers::CompactCombo("Capture Monitor", &ctx.config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size()));
    UIHelpers::InfoTooltip("Select which monitor to capture from when using CUDA-based screen capture.");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        SAVE_PROFILE();
    }
    
    UIHelpers::EndCard();
}


void draw_capture_settings()
{
    auto& ctx = AppContext::getInstance();

    draw_capture_area_settings();
    UIHelpers::Spacer();

    draw_capture_behavior_settings();
    UIHelpers::Spacer();

    draw_capture_source_settings();
}
