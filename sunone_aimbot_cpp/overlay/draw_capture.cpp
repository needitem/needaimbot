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
    ImGui::SeparatorText("Capture Area & Resolution");
    ImGui::Spacing();

    if (ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 1280)) { config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Size (in pixels) of the square area around the cursor to capture for detection.\nSmaller values improve performance but may miss targets further from the crosshair.");
    }
    if (config.detection_resolution >= 400)
    {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: Large detection resolution can impact performance.");
    }

    ImGui::Spacing();
    if (ImGui::Checkbox("Circle mask", &config.circle_mask))
    {
        config.saveConfig();
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

    if (ImGui::SliderInt("Lock FPS", &config.capture_fps, 0, 240)) { config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Limits the screen capture rate. 0 = Unlocked (fastest possible).\nLower values reduce CPU usage but increase detection latency.");
    }
    if (config.capture_fps == 0)
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "-> Unlocked");
    }

    if (config.capture_fps == 0 || config.capture_fps >= 61)
    {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: High or unlocked FPS can significantly impact performance.");
    }

    ImGui::Spacing();
    if (ImGui::Checkbox("Capture Borders", &config.capture_borders)) { config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Includes window borders in the screen capture (if applicable).");
    }
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    if (ImGui::Checkbox("Capture Cursor", &config.capture_cursor)) { config.saveConfig(); }
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

        if (ImGui::Combo("Capture Monitor", &config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size())))
        {
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Select which monitor to capture from when using CUDA-based screen capture.");
        }
    }
    ImGui::Spacing();
}