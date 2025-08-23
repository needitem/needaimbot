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
#include "../utils/window_enum.h"

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
    
    // FPS limiting removed - running at maximum performance when active
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Maximum Performance Mode");
    UIHelpers::InfoTooltip("No FPS limiting - runs at maximum speed when aimbot is active.\nGPU usage: 0%% when OFF, 100%% when ON.");
    
    UIHelpers::CompactSpacer();
    
    const char* capture_methods[] = { 
        "Desktop Duplication (Full GPU)", 
        "Virtual Camera (OBS/XSplit)",
        "OBS Game Hook (Ultra Low Latency)"
    };
    
    // Initialize from config
    int current_method = ctx.config.gpu_capture_method;
    
    // Debug log
    static bool logged = false;
    if (!logged) {
        std::cout << "[UI] Current gpu_capture_method from config: " << current_method << std::endl;
        logged = true;
    }
    
    int previous_method = current_method;
    UIHelpers::CompactCombo("Capture Method", &current_method, capture_methods, IM_ARRAYSIZE(capture_methods));
    UIHelpers::InfoTooltip("Desktop Duplication: Captures full screen then crops to target area (current method)\nRegion Capture: Directly captures only the target region using Windows Graphics Capture API (Windows 10 1903+)\n\nRegion Capture uses less GPU memory and may have better performance.");
    
    // Check if the value actually changed
    if (current_method != previous_method)
    {
        ctx.config.gpu_capture_method = current_method;  // Save to config
        std::cout << "[UI] GPU capture method changed from " << previous_method << " to " << current_method << std::endl;
        SAVE_PROFILE();  // This will save to file
        ctx.capture_method_changed = true;
    }
    
    UIHelpers::CompactSpacer();
    
    // Show game window selection for OBS Hook mode
    if (current_method == 2) { // OBS Game Hook
        UIHelpers::BeautifulText("Game Window Selection", UIHelpers::GetAccentColor());
        UIHelpers::CompactSpacer();
        
        // Get list of windows
        static std::vector<std::string> windowTitles;
        static std::vector<const char*> windowItems;
        static bool windowsLoaded = false;
        static int selectedWindow = -1;
        static char customWindowName[256] = "";
        static bool useCustomName = false;
        
        // Initialize custom window name with current config value
        static bool firstInit = true;
        if (firstInit) {
            strncpy_s(customWindowName, ctx.config.game_window_name.c_str(), sizeof(customWindowName) - 1);
            firstInit = false;
        }
        
        // Refresh window list button
        if (ImGui::Button("Refresh Window List")) {
            windowTitles = WindowEnumerator::GetWindowTitles();
            windowItems.clear();
            windowItems.reserve(windowTitles.size());
            
            // Find current selection
            selectedWindow = -1;
            for (size_t i = 0; i < windowTitles.size(); ++i) {
                windowItems.push_back(windowTitles[i].c_str());
                if (windowTitles[i] == ctx.config.game_window_name) {
                    selectedWindow = static_cast<int>(i);
                }
            }
            windowsLoaded = true;
        }
        
        ImGui::SameLine();
        UIHelpers::InfoTooltip("Click to refresh the list of available windows");
        
        // Load windows on first display
        if (!windowsLoaded) {
            windowTitles = WindowEnumerator::GetWindowTitles();
            windowItems.clear();
            windowItems.reserve(windowTitles.size());
            
            // Find current selection
            selectedWindow = -1;
            for (size_t i = 0; i < windowTitles.size(); ++i) {
                windowItems.push_back(windowTitles[i].c_str());
                if (windowTitles[i] == ctx.config.game_window_name) {
                    selectedWindow = static_cast<int>(i);
                }
            }
            windowsLoaded = true;
        }
        
        // Window selection combo
        if (!windowItems.empty()) {
            int prev_selected = selectedWindow;
            UIHelpers::CompactCombo("Select Window", &selectedWindow, windowItems.data(), static_cast<int>(windowItems.size()));
            UIHelpers::InfoTooltip("Select the game window from running applications");
            
            if (selectedWindow != prev_selected) {
                if (selectedWindow >= 0 && selectedWindow < static_cast<int>(windowTitles.size())) {
                    ctx.config.game_window_name = windowTitles[selectedWindow];
                    strncpy_s(customWindowName, windowTitles[selectedWindow].c_str(), sizeof(customWindowName) - 1);
                    useCustomName = false;
                    SAVE_PROFILE();
                    std::cout << "[UI] Game window changed to: " << ctx.config.game_window_name << std::endl;
                }
            }
        }
        
        UIHelpers::CompactSpacer();
        
        // Custom window name option
        if (UIHelpers::BeautifulToggle("Use Custom Window Name", &useCustomName, "Manually enter a window title instead of selecting from the list")) {
            if (useCustomName) {
                selectedWindow = -1;
            }
        }
        
        if (useCustomName) {
            if (ImGui::InputText("Window Title", customWindowName, sizeof(customWindowName))) {
                ctx.config.game_window_name = std::string(customWindowName);
                SAVE_PROFILE();
                std::cout << "[UI] Game window changed to custom: " << ctx.config.game_window_name << std::endl;
            }
            UIHelpers::InfoTooltip("Enter the exact window title of the game you want to capture");
        }
        
        UIHelpers::CompactSpacer();
        
        // Show current selection
        ImGui::Text("Current Target: %s", ctx.config.game_window_name.c_str());
        
        // Check if window exists
        HWND hwnd = FindWindowA(nullptr, ctx.config.game_window_name.c_str());
        if (hwnd) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "(Found)");
        } else {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "(Not Found)");
        }
        
        UIHelpers::CompactSpacer();
        ImGui::Separator();
        UIHelpers::CompactSpacer();
    }
    
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
                SAVE_PROFILE();
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
                SAVE_PROFILE();
            }
            UIHelpers::InfoTooltip("Fallback network stream URL when NDI is not available.\nCommon formats: HTTP MJPEG, RTMP, UDP streams.");
            
            UIHelpers::CompactSpacer();
            
            if (UIHelpers::BeautifulToggle("Low Latency Mode", &ctx.config.ndi_low_latency, "Enables high bandwidth mode for lowest possible latency."))
            {
                SAVE_PROFILE();
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
    
    draw_capture_area_settings();
    UIHelpers::Spacer();
    
    draw_capture_behavior_settings();
    UIHelpers::Spacer();
    
    draw_capture_source_settings();
    UIHelpers::Spacer();
    
    draw_2pc_capture_settings();
}
