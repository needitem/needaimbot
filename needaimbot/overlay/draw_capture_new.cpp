#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <Psapi.h>

#include <imgui/imgui.h>
#include "AppContext.h"
#include "needaimbot.h"
#include "capture.h"
#include "ui_helpers_new.h"
#include "include/other_tools.h"

void draw_capture_settings_new()
{
    auto& ctx = AppContext::getInstance();
    
    UI::Space();
    
    // Detection Resolution
    UI::Section("Detection Area");
    
    float detection_res = static_cast<float>(ctx.config.detection_resolution);
    if (UI::Slider("Detection Size##capture", &detection_res, 50.0f, 1280.0f, "%.0f px")) {
        ctx.config.detection_resolution = static_cast<int>(detection_res);
        ctx.config.saveConfig();
    }
    UI::Tip("Square area size around cursor for detection");
    
    if (ctx.config.detection_resolution >= 400) {
        UI::Warning("Large detection size impacts performance");
    }
    
    UI::SmallSpace();
    
    if (UI::Toggle("Circle Mask##capture", &ctx.config.circle_mask)) {
        ctx.config.saveConfig();
    }
    
    UI::Space();
    UI::Space();
    
    // Capture Settings
    UI::Section("Capture Settings");
    
    float capture_fps = static_cast<float>(ctx.config.capture_fps);
    if (UI::Slider("FPS Limit##capture", &capture_fps, 0.0f, 240.0f, "%.0f")) {
        ctx.config.capture_fps = static_cast<int>(capture_fps);
        ctx.config.saveConfig();
    }
    UI::Tip("0 = Unlimited (may impact performance)");
    
    UI::SmallSpace();
    
    // Capture Method
    const char* methods[] = { "Simple (BitBlt)", "Desktop Duplication", "Game Capture" };
    int method = 0;
    if (ctx.config.capture_method == "simple") method = 0;
    else if (ctx.config.capture_method == "duplication") method = 1;
    else if (ctx.config.capture_method == "game_capture") method = 2;
    
    int prev_method = method;
    if (UI::Combo("Method##capture", &method, methods, 3)) {
        if (method == 0) ctx.config.capture_method = "simple";
        else if (method == 1) ctx.config.capture_method = "duplication";
        else if (method == 2) ctx.config.capture_method = "game_capture";
        ctx.config.saveConfig();
        ctx.capture_method_changed = true;
    }
    
    // Game capture specific settings
    if (ctx.config.capture_method == "game_capture") {
        UI::SmallSpace();
        
        static std::vector<std::string> window_titles;
        static std::vector<const char*> window_items;
        static bool need_refresh = true;
        
        if (UI::Button("Refresh Windows##capture", -1) || need_refresh) {
            window_titles.clear();
            window_items.clear();
            window_titles.push_back("-- Select Window --");
            
            EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
                auto* titles = reinterpret_cast<std::vector<std::string>*>(lParam);
                if (IsWindowVisible(hwnd)) {
                    char title[256];
                    if (GetWindowTextA(hwnd, title, sizeof(title)) > 0) {
                        std::string str = title;
                        if (!str.empty() && str != "Program Manager") {
                            titles->push_back(str);
                        }
                    }
                }
                return TRUE;
            }, reinterpret_cast<LPARAM>(&window_titles));
            
            window_items.reserve(window_titles.size());
            for (const auto& title : window_titles) {
                window_items.push_back(title.c_str());
            }
            need_refresh = false;
        }
        
        int current = 0;
        for (size_t i = 0; i < window_titles.size(); ++i) {
            if (window_titles[i] == ctx.config.target_game_name) {
                current = static_cast<int>(i);
                break;
            }
        }
        
        if (UI::Combo("Target##window", &current, window_items.data(), static_cast<int>(window_items.size()))) {
            ctx.config.target_game_name = window_titles[current];
            ctx.config.saveConfig();
            ctx.capture_method_changed = true;
        }
        
        if (ctx.config.target_game_name.empty() || current == 0) {
            UI::Warning("Please select a target window!");
        }
    }
    
    UI::SmallSpace();
    
    // Options in two columns
    UI::BeginColumns(0.5f);
    
    if (UI::Toggle("Capture Borders##capture", &ctx.config.capture_borders)) {
        ctx.config.saveConfig();
    }
    
    UI::NextColumn();
    
    if (UI::Toggle("Capture Cursor##capture", &ctx.config.capture_cursor)) {
        ctx.config.saveConfig();
    }
    
    UI::EndColumns();
    
    UI::Space();
    UI::Space();
    
    // Monitor Selection
    UI::Section("Monitor Selection");
    
    int monitors = get_active_monitors();
    std::vector<const char*> monitor_items;
    std::vector<std::string> monitor_names;
    
    for (int i = -1; i < monitors; ++i) {
        monitor_names.push_back("Monitor " + std::to_string(i + 2));
    }
    
    for (const auto& name : monitor_names) {
        monitor_items.push_back(name.c_str());
    }
    
    if (UI::Combo("Monitor##capture", &ctx.config.monitor_idx, monitor_items.data(), static_cast<int>(monitor_items.size()))) {
        ctx.config.saveConfig();
    }
    
    UI::Space();
}