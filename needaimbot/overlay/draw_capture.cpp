#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

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
    if (ImGui::Checkbox("Use 1ms Capture", &ctx.config.use_1ms_capture)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Uses 1ms interval capture method for game capture (alternative to duplication API).");
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
