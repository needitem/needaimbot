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

int monitors = get_active_monitors();

void draw_capture_settings()
{
    auto& ctx = AppContext::getInstance();
    
    ImGui::SeparatorText("Capture Area & Resolution");
    ImGui::Spacing();

    if (ImGui::SliderInt("Detection Resolution", &ctx.config.detection_resolution, 50, 1280)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Size (in pixels) of the square area around the cursor to capture for detection.\nSmaller values improve performance but may miss targets further from the crosshair.");
    }
    if (ctx.config.detection_resolution >= 400)
    {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: Large detection resolution can impact performance.");
    }

    ImGui::Spacing();
    if (ImGui::Checkbox("Circle mask", &ctx.config.circle_mask))
    {
        ctx.config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Applies a circular mask to the captured area, ignoring corners.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Capture Behavior");
    ImGui::Spacing();

    if (ImGui::SliderInt("Lock FPS", &ctx.config.capture_fps, 0, 240)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Limits the screen capture rate. 0 = Unlocked (fastest possible).\nLower values reduce CPU usage but increase detection latency.");
    }
    if (ctx.config.capture_fps == 0)
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "-> Unlocked");
    }

    if (ctx.config.capture_fps == 0 || ctx.config.capture_fps >= 61)
    {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: High or unlocked FPS can significantly impact performance.");
    }

    ImGui::Spacing();

    // Capture Method Selection
    const char* capture_methods[] = { "Simple (BitBlt)", "Desktop Duplication API", "UnknownCheats Game Capture" };
    int current_method = 0;
    if (ctx.config.capture_method == "simple") current_method = 0;
    else if (ctx.config.capture_method == "duplication") current_method = 1;
    else if (ctx.config.capture_method == "game_capture") current_method = 2;

    if (ImGui::Combo("Capture Method", &current_method, capture_methods, IM_ARRAYSIZE(capture_methods)))
    {
        if (current_method == 0) ctx.config.capture_method = "simple";
        else if (current_method == 1) ctx.config.capture_method = "duplication";
        else if (current_method == 2) ctx.config.capture_method = "game_capture";
        ctx.config.saveConfig();
        ctx.capture_method_changed = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Simple: Fast BitBlt screen capture (current method)\nDuplication API: Windows Desktop Duplication API\nGame Capture: UnknownCheats method for game capturing");
    }

    // Show game selection dropdown when GameCapture is selected
    if (ctx.config.capture_method == "game_capture") {
        ImGui::Spacing();
        
        // Get list of available windows
        static std::vector<std::string> window_titles;
        static std::vector<std::string> window_display_names;
        static bool need_refresh = true;
        
        if (need_refresh || ImGui::Button("Refresh Windows")) {
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
        if (ImGui::Combo("Target Game Window", &current_selection, 
                        [](void* data, int idx, const char** out_text) -> bool {
                            auto* names = static_cast<std::vector<std::string>*>(data);
                            if (idx >= 0 && idx < static_cast<int>(names->size())) {
                                *out_text = (*names)[idx].c_str();
                                return true;
                            }
                            return false;
                        }, &window_display_names, static_cast<int>(window_display_names.size()))) {
            
            if (current_selection >= 0 && current_selection < static_cast<int>(window_titles.size())) {
                ctx.config.target_game_name = window_titles[current_selection];
                ctx.config.saveConfig();
                ctx.capture_method_changed = true;
            }
        }
        
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Select the window you want to capture from the list of visible windows");
        }
        
        if (ctx.config.target_game_name.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Warning: Please select a target window for Game Capture!");
        }
    }

    ImGui::Spacing();

    if (ImGui::SliderInt("Acquire Timeout (ms)", &ctx.config.capture_timeout_ms, 1, 100))
    {
        ctx.config.saveConfig();
        ctx.capture_timeout_changed = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Timeout for AcquireNextFrame in milliseconds (1-100ms).\nLower values can make the application feel more responsive if frames are ready quickly,\nbut may lead to more timeouts (frame drops) if the system is slow to provide frames.\nHigher values give the system more time but can increase perceived latency if waiting for a slow frame.");
    }

    ImGui::Spacing();
    if (ImGui::Checkbox("Capture Borders", &ctx.config.capture_borders)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Includes window borders in the screen capture (if applicable).");
    }
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    if (ImGui::Checkbox("Capture Cursor", &ctx.config.capture_cursor)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Includes the mouse cursor in the screen capture.");
    }


    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Capture Source (CUDA Only)");
    ImGui::Spacing();
    {
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

        if (ImGui::Combo("Capture Monitor", &ctx.config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size())))
        {
            ctx.config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Select which monitor to capture from when using CUDA-based screen capture.");
        }
    }
    ImGui::Spacing();
}
