#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <Psapi.h>

#include <imgui/imgui.h>
#include "imgui/imgui_internal.h"

#include "AppContext.h"
#include "config/config.h"
#include "needaimbot.h"
#include "../capture/capture.h"
#include "include/other_tools.h"
#include "draw_settings.h"
#include "ui_helpers.h"

int monitors = get_active_monitors();

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
        ctx.config.saveConfig();
        
        // Force update flags
        extern std::atomic<bool> detection_resolution_changed;
        extern std::atomic<bool> detector_model_changed;
        detection_resolution_changed.store(true);
        detector_model_changed.store(true);
    }
    
    if (ctx.config.detection_resolution >= 400)
    {
        UIHelpers::BeautifulText("WARNING: Large detection resolution can impact performance.", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulToggle("Circle Mask", &ctx.config.circle_mask, "Applies a circular mask to the captured area, ignoring corners."))
    {
        ctx.config.saveConfig();
    }
    
    UIHelpers::EndCard();
}

static void draw_capture_behavior_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Capture Behavior");
    
    // Use temporary float for the slider
    float capture_fps_float = static_cast<float>(ctx.config.capture_fps);
    
    // Store the old value to detect changes
    int old_fps = ctx.config.capture_fps;
    
    UIHelpers::CompactSlider("Lock FPS", &capture_fps_float, 0.0f, 240.0f, "%.0f");
    UIHelpers::InfoTooltip("Limits the screen capture rate. 0 = Unlocked (fastest possible).\nLower values reduce CPU usage but increase detection latency.");
    
    // Check if value changed
    int new_fps = static_cast<int>(capture_fps_float);
    if (new_fps != old_fps) {
        ctx.config.capture_fps = new_fps;
        ctx.config.saveConfig();
        
        // Force update flags
        extern std::atomic<bool> capture_fps_changed;
        capture_fps_changed.store(true);
    }
    
    if (ctx.config.capture_fps == 0)
    {
        ImGui::SameLine();
        UIHelpers::BeautifulText("-> Unlocked", UIHelpers::GetErrorColor());
    }
    
    if (ctx.config.capture_fps == 0 || ctx.config.capture_fps >= 61)
    {
        UIHelpers::BeautifulText("WARNING: High or unlocked FPS can significantly impact performance.", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::CompactSpacer();
    
    const char* capture_methods[] = { "Simple (BitBlt)", "Desktop Duplication API", "Virtual Camera", "NDI Stream" };
    static int current_method = 0;
    static bool first_time = true;
    
    // Initialize on first run
    if (first_time) {
        if (ctx.config.capture_method == "simple") current_method = 0;
        else if (ctx.config.capture_method == "duplication") current_method = 1;
        else if (ctx.config.capture_method == "virtual_camera") current_method = 2;
        else if (ctx.config.capture_method == "ndi") current_method = 3;
        first_time = false;
    }
    
    int previous_method = current_method;
    UIHelpers::CompactCombo("Capture Method", &current_method, capture_methods, IM_ARRAYSIZE(capture_methods));
    UIHelpers::InfoTooltip("Simple: Fast BitBlt screen capture\nDuplication API: Windows Desktop Duplication API\nVirtual Camera: 2PC setup via virtual camera devices\nNDI Stream: Network video streaming for 2PC setup");
    
    // Check if the value actually changed
    if (current_method != previous_method)
    {
        if (current_method == 0) ctx.config.capture_method = "simple";
        else if (current_method == 1) ctx.config.capture_method = "duplication";
        else if (current_method == 2) ctx.config.capture_method = "virtual_camera";
        else if (current_method == 3) ctx.config.capture_method = "ndi";
        ctx.config.saveConfig();
        ctx.capture_method_changed = true;
    }
    
    UIHelpers::CompactSpacer();
    
    ImGui::Columns(2, "capture_options", false);
    
    if (UIHelpers::BeautifulToggle("Capture Borders", &ctx.config.capture_borders, "Includes window borders in the screen capture (if applicable)."))
    {
        ctx.config.saveConfig();
    }
    
    ImGui::NextColumn();
    
    if (UIHelpers::BeautifulToggle("Capture Cursor", &ctx.config.capture_cursor, "Includes the mouse cursor in the screen capture."))
    {
        ctx.config.saveConfig();
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
        ctx.config.saveConfig();
    }
    
    UIHelpers::EndCard();
}

static void draw_2pc_capture_settings()
{
    auto& ctx = AppContext::getInstance();
    
    if (ctx.config.capture_method == "virtual_camera" || ctx.config.capture_method == "ndi")
    {
        UIHelpers::BeginCard("2PC Capture Settings");
        
        if (ctx.config.capture_method == "virtual_camera")
        {
            UIHelpers::BeautifulText("Virtual Camera Configuration", UIHelpers::GetAccentColor());
            UIHelpers::CompactSpacer();
            
            // Virtual camera device selection
            static int selected_camera = 0;
            const char* camera_devices[] = { "OBS Virtual Camera", "XSplit VCam", "Streamlabs Virtual Camera", "NVIDIA Broadcast", "Custom Device 1", "Custom Device 2" };
            
            UIHelpers::CompactCombo("Virtual Camera Device", &selected_camera, camera_devices, IM_ARRAYSIZE(camera_devices));
            UIHelpers::InfoTooltip("Select the virtual camera device to capture from your streaming PC.\nMake sure the virtual camera is enabled in your streaming software.");
            
            UIHelpers::CompactSpacer();
            
            ImGui::BulletText("Enable virtual camera in streaming software");
            ImGui::BulletText("Select matching device above");
            ImGui::BulletText("Start capture on both PCs");
        }
        else if (ctx.config.capture_method == "ndi")
        {
            UIHelpers::BeautifulText("NDI Stream Configuration", UIHelpers::GetAccentColor());
            UIHelpers::CompactSpacer();
            
            // NDI source selection
            static char ndi_source_name[256] = "";
            static bool first_init = true;
            if (first_init) {
                strncpy_s(ndi_source_name, ctx.config.ndi_source_name.c_str(), sizeof(ndi_source_name) - 1);
                first_init = false;
            }
            
            if (ImGui::InputText("NDI Source Name", ndi_source_name, sizeof(ndi_source_name))) {
                ctx.config.ndi_source_name = std::string(ndi_source_name);
                ctx.config.saveConfig();
            }
            UIHelpers::InfoTooltip("Enter the NDI source name from your streaming PC.\nLeave empty to auto-detect first available source.");
            
            UIHelpers::CompactSpacer();
            
            // Network fallback URL
            static char network_url[512] = "";
            static bool first_url_init = true;
            if (first_url_init) {
                strncpy_s(network_url, ctx.config.ndi_network_url.empty() ? "http://localhost:8080/video.mjpg" : ctx.config.ndi_network_url.c_str(), sizeof(network_url) - 1);
                first_url_init = false;
            }
            
            if (ImGui::InputText("Network Stream URL", network_url, sizeof(network_url))) {
                ctx.config.ndi_network_url = std::string(network_url);
                ctx.config.saveConfig();
            }
            UIHelpers::InfoTooltip("Fallback network stream URL when NDI is not available.\nCommon formats: HTTP MJPEG, RTMP, UDP streams.");
            
            UIHelpers::CompactSpacer();
            
            if (UIHelpers::BeautifulToggle("Low Latency Mode", &ctx.config.ndi_low_latency, "Enables high bandwidth mode for lowest possible latency."))
            {
                ctx.config.saveConfig();
            }
            
            UIHelpers::CompactSpacer();
            
            ImGui::BulletText("Install NDI Tools on both PCs");
            ImGui::BulletText("Enable NDI output in streaming software");
            ImGui::BulletText("Ensure network connectivity");
        }
        
        UIHelpers::EndCard();
    }
}

void draw_capture_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.6f);
    
    // Left column - Main capture settings
    draw_capture_area_settings();
    UIHelpers::CompactSpacer();
    
    draw_capture_behavior_settings();
    
    UIHelpers::NextColumn();
    
    // Right column - Source and info
    draw_capture_source_settings();
    UIHelpers::CompactSpacer();
    
    draw_2pc_capture_settings();
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("Performance Tips", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Lower detection resolution for better performance");
    ImGui::BulletText("Use circle mask to focus on center targets");
    ImGui::BulletText("Limit FPS to 60 for stable performance");
    ImGui::BulletText("Simple capture method is usually fastest");
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulText("Capture Method Guide", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Simple: Best for windowed games");
    ImGui::BulletText("Duplication: Good for fullscreen apps");
    ImGui::BulletText("Virtual Camera: For 2PC streaming setup");
    ImGui::BulletText("NDI Stream: Network capture for 2PC setup");
    
    UIHelpers::EndInfoPanel();
    
    UIHelpers::EndTwoColumnLayout();
}
