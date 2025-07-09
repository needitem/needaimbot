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
#include "capture.h"
#include "include/other_tools.h"
#include "draw_settings.h"
#include "ui_helpers.h"

int monitors = get_active_monitors();

static void draw_capture_area_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Capture Area & Resolution");
    
    UIHelpers::CompactSlider("Detection Resolution", reinterpret_cast<float*>(&ctx.config.detection_resolution), 50.0f, 1280.0f, "%.0f");
    UIHelpers::InfoTooltip("Size (in pixels) of the square area around the cursor to capture for detection.\nSmaller values improve performance but may miss targets further from the crosshair.");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    if (ctx.config.detection_resolution >= 400)
    {
        UIHelpers::BeautifulText("WARNING: Large detection resolution can impact performance.", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::Spacer(5.0f);
    
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
    
    UIHelpers::CompactSlider("Lock FPS", reinterpret_cast<float*>(&ctx.config.capture_fps), 0.0f, 240.0f, "%.0f");
    UIHelpers::InfoTooltip("Limits the screen capture rate. 0 = Unlocked (fastest possible).\nLower values reduce CPU usage but increase detection latency.");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
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
    
    UIHelpers::Spacer(5.0f);
    
    const char* capture_methods[] = { "Simple (BitBlt)", "Desktop Duplication API", "UnknownCheats Game Capture" };
    int current_method = 0;
    if (ctx.config.capture_method == "simple") current_method = 0;
    else if (ctx.config.capture_method == "duplication") current_method = 1;
    else if (ctx.config.capture_method == "game_capture") current_method = 2;
    
    UIHelpers::CompactCombo("Capture Method", &current_method, capture_methods, IM_ARRAYSIZE(capture_methods));
    UIHelpers::InfoTooltip("Simple: Fast BitBlt screen capture\nDuplication API: Windows Desktop Duplication API\nGame Capture: UnknownCheats method for game capturing");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        if (current_method == 0) ctx.config.capture_method = "simple";
        else if (current_method == 1) ctx.config.capture_method = "duplication";
        else if (current_method == 2) ctx.config.capture_method = "game_capture";
        ctx.config.saveConfig();
        ctx.capture_method_changed = true;
    }
    
    // Show game selection dropdown when GameCapture is selected
    if (ctx.config.capture_method == "game_capture") {
        UIHelpers::Spacer(5.0f);
        
        // Get list of available windows
        static std::vector<std::string> window_titles;
        static std::vector<std::string> window_display_names;
        static bool need_refresh = true;
        
        if (UIHelpers::BeautifulButton("Refresh Windows", ImVec2(-1, 0)) || need_refresh) {
            window_titles.clear();
            window_display_names.clear();
            window_titles.push_back(""); // Empty option
            window_display_names.push_back("-- Select a window --");
            
            // Enumerate windows
            struct WindowData {
                std::vector<std::string>* titles;
                std::vector<std::string>* display_names;
            } window_data = { &window_titles, &window_display_names };
            
            EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
                auto* data = reinterpret_cast<WindowData*>(lParam);
                
                if (IsWindowVisible(hwnd)) {
                    char window_title[256];
                    if (GetWindowTextA(hwnd, window_title, sizeof(window_title)) > 0) {
                        std::string title = window_title;
                        if (!title.empty() && title != "Program Manager") {
                            // Get process name for better identification
                            DWORD pid;
                            GetWindowThreadProcessId(hwnd, &pid);
                            HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
                            if (hProcess) {
                                char process_name[MAX_PATH];
                                if (GetModuleBaseNameA(hProcess, NULL, process_name, sizeof(process_name))) {
                                    std::string display_name = title + " (" + process_name + ")";
                                    data->titles->push_back(title);
                                    data->display_names->push_back(display_name);
                                }
                                CloseHandle(hProcess);
                            } else {
                                data->titles->push_back(title);
                                data->display_names->push_back(title);
                            }
                        }
                    }
                }
                return TRUE;
            }, reinterpret_cast<LPARAM>(&window_data));
            
            need_refresh = false;
        }
        
        // Find current selection index
        int current_selection = 0;
        for (size_t i = 0; i < window_titles.size(); ++i) {
            if (window_titles[i] == ctx.config.target_game_name) {
                current_selection = static_cast<int>(i);
                break;
            }
        }
        
        // Create dropdown
        UIHelpers::CompactCombo("Target Game Window", &current_selection, 
                        [](void* data, int idx, const char** out_text) -> bool {
                            auto* names = static_cast<std::vector<std::string>*>(data);
                            if (idx >= 0 && idx < static_cast<int>(names->size())) {
                                *out_text = (*names)[idx].c_str();
                                return true;
                            }
                            return false;
                        }, &window_display_names, static_cast<int>(window_display_names.size()));
        UIHelpers::InfoTooltip("Select the window you want to capture from the list of visible windows");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            if (current_selection >= 0 && current_selection < static_cast<int>(window_titles.size())) {
                ctx.config.target_game_name = window_titles[current_selection];
                ctx.config.saveConfig();
                ctx.capture_method_changed = true;
            }
        }
        
        if (ctx.config.target_game_name.empty()) {
            UIHelpers::BeautifulText("Warning: Please select a target window for Game Capture!", UIHelpers::GetWarningColor());
        }
    }
    
    UIHelpers::Spacer(5.0f);
    
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

void draw_capture_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.6f);
    
    // Left column - Main capture settings
    draw_capture_area_settings();
    UIHelpers::Spacer();
    
    draw_capture_behavior_settings();
    
    UIHelpers::NextColumn();
    
    // Right column - Source and info
    draw_capture_source_settings();
    UIHelpers::Spacer();
    
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("Performance Tips", UIHelpers::GetAccentColor());
    UIHelpers::Spacer(5.0f);
    
    ImGui::BulletText("Lower detection resolution for better performance");
    ImGui::BulletText("Use circle mask to focus on center targets");
    ImGui::BulletText("Limit FPS to 60 for stable performance");
    ImGui::BulletText("Simple capture method is usually fastest");
    
    UIHelpers::Spacer();
    
    UIHelpers::BeautifulText("Capture Method Guide", UIHelpers::GetAccentColor());
    UIHelpers::Spacer(5.0f);
    
    ImGui::BulletText("Simple: Best for windowed games");
    ImGui::BulletText("Duplication: Good for fullscreen apps");
    ImGui::BulletText("Game Capture: For specific game windows");
    
    UIHelpers::EndInfoPanel();
    
    UIHelpers::EndTwoColumnLayout();
}
