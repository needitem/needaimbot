#include "../core/windows_headers.h"

#include "AppContext.h"
#include "../core/constants.h"
#include "../imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "../include/other_tools.h"
#include "draw_settings.h"
#include <iostream>

void draw_overlay()
{
    auto& ctx = AppContext::getInstance();
    
    ImGui::SeparatorText("Console");
    ImGui::Spacing();
    
    // Console toggle button
    static bool console_visible = IsConsoleVisible();
    
    // Update console state if it was changed externally
    bool current_console_state = IsConsoleVisible();
    if (current_console_state != console_visible) {
        console_visible = current_console_state;
    }
    
    if (ImGui::Checkbox("Show Console Window", &console_visible))
    {
        if (console_visible) {
            ShowConsole();
            std::cout << "[Console] Console window restored" << std::endl;
        } else {
            HideConsole();
        }
    }
    if (ImGui::IsItemHovered()) { 
        ImGui::SetTooltip("Toggle the console window on/off. Useful for debugging and monitoring."); 
    }
    
    ImGui::Spacing();
    ImGui::SeparatorText("Overlay Appearance");
    ImGui::Spacing();

    if (ImGui::SliderInt("Overlay Opacity", &ctx.config.global().overlay_opacity, 40, 255)) {
        MARK_CONFIG_DIRTY();  // Auto-save after delay
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Adjusts the transparency of the overlay window (settings menu)."); }

    ImGui::Spacing();

    static float ui_scale = ctx.config.global().overlay_ui_scale;

    if (ImGui::SliderFloat("UI Scale", &ui_scale, 0.5f, 3.0f, "%.2f"))
    {
        ImGui::GetIO().FontGlobalScale = ui_scale;

        ctx.config.global().overlay_ui_scale = ui_scale;
        MARK_CONFIG_DIRTY();  // Auto-save after delay

        overlayWidth = static_cast<int>(Constants::BASE_OVERLAY_WIDTH * ui_scale);
        overlayHeight = static_cast<int>(Constants::BASE_OVERLAY_HEIGHT * ui_scale);

        SetWindowPos(g_hwnd, NULL, 0, 0, overlayWidth, overlayHeight, SWP_NOMOVE | SWP_NOZORDER);
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Scales the size of the entire settings UI. May require restart for some elements."); }
    
    ImGui::Spacing();
}
