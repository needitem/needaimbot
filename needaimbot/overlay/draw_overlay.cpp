#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "AppContext.h"
#include "../imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"

void draw_overlay()
{
    auto& ctx = AppContext::getInstance();
    
    ImGui::SeparatorText("Overlay Appearance");
    ImGui::Spacing();

    ImGui::SliderInt("Overlay Opacity", &ctx.config.overlay_opacity, 40, 255);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Adjusts the transparency of the overlay window (settings menu)."); }

    ImGui::Spacing();

    static float ui_scale = ctx.config.overlay_ui_scale;

    if (ImGui::SliderFloat("UI Scale", &ui_scale, 0.5f, 3.0f, "%.2f"))
    {
        ImGui::GetIO().FontGlobalScale = ui_scale;

        ctx.config.overlay_ui_scale = ui_scale;
        // Config will be saved by batch processing in overlay.cpp

        extern const int BASE_OVERLAY_WIDTH;
        extern const int BASE_OVERLAY_HEIGHT;
        overlayWidth = static_cast<int>(BASE_OVERLAY_WIDTH * ui_scale);
        overlayHeight = static_cast<int>(BASE_OVERLAY_HEIGHT * ui_scale);

        SetWindowPos(g_hwnd, NULL, 0, 0, overlayWidth, overlayHeight, SWP_NOMOVE | SWP_NOZORDER);
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Scales the size of the entire settings UI. May require restart for some elements."); }
    
    ImGui::Spacing();
    ImGui::SeparatorText("Performance");
    ImGui::Spacing();
    
    // Display current overlay FPS setting
    float clamped_fps = (std::max)(15.0f, (std::min)(240.0f, ctx.config.target_fps));
    ImGui::Text("Overlay FPS: %.1f", clamped_fps);
    if (ImGui::IsItemHovered()) { 
        ImGui::SetTooltip("Overlay frame rate is controlled by Target FPS setting in Capture tab.\nClamped between 15-240 FPS for stability."); 
    }
    
    // Show current frame time
    float frame_time_ms = 1000.0f / clamped_fps;
    ImGui::Text("Frame Time: %.2f ms", frame_time_ms);
    
    ImGui::Spacing();
}
