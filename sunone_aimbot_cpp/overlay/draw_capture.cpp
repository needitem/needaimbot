#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <imgui/imgui.h>
#include "imgui/imgui_internal.h"

#include "config.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "include/other_tools.h"
#include "draw_settings.h"

int monitors = get_active_monitors();

void draw_capture_settings()
{
    ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 1280);
    if (config.detection_resolution >= 400)
    {
        ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: A large screen capture size can negatively affect performance.");
    }

    ImGui::SliderInt("Lock FPS", &config.capture_fps, 0, 240);
    if (config.capture_fps == 0)
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(255, 0, 0, 255), "-> Disabled");
    }

    if (config.capture_fps == 0 || config.capture_fps >= 61)
    {
        ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: A large number of FPS can negatively affect performance.");
    }

    if (ImGui::Checkbox("Circle mask", &config.circle_mask))
    {
        config.saveConfig();
    }

    ImGui::Checkbox("Capture Borders", &config.capture_borders);
    ImGui::Checkbox("Capture Cursor", &config.capture_cursor);

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

        if (ImGui::Combo("Capture monitor (CUDA GPU)", &config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size())))
        {
            config.saveConfig();
        }
    }
}