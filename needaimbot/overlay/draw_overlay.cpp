#include "../core/windows_headers.h"

#include "AppContext.h"
#include "../core/constants.h"
#include "../imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "../include/other_tools.h"
#include <iostream>

void draw_overlay()
{
    auto& ctx = AppContext::getInstance();
    
    ImGui::SeparatorText("Console");
    ImGui::Spacing();
    
    // Console toggle button
    static bool console_visible = IsConsoleVisible();
    bool console_state_changed = false;
    
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
        console_state_changed = true;
    }
    if (ImGui::IsItemHovered()) { 
        ImGui::SetTooltip("Toggle the console window on/off. Useful for debugging and monitoring."); 
    }
    
    ImGui::Spacing();
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

        overlayWidth = static_cast<int>(Constants::BASE_OVERLAY_WIDTH * ui_scale);
        overlayHeight = static_cast<int>(Constants::BASE_OVERLAY_HEIGHT * ui_scale);

        SetWindowPos(g_hwnd, NULL, 0, 0, overlayWidth, overlayHeight, SWP_NOMOVE | SWP_NOZORDER);
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Scales the size of the entire settings UI. May require restart for some elements."); }
    
    ImGui::Spacing();
    ImGui::SeparatorText("Performance");
    ImGui::Spacing();
    
    // Display overlay FPS
    ImGui::Text("Overlay FPS: 30.0");
    if (ImGui::IsItemHovered()) { 
        ImGui::SetTooltip("Overlay runs at fixed 30 FPS for UI responsiveness."); 
    }
    
    // Show current frame time
    ImGui::Text("Frame Time: 33.33 ms");
    
    ImGui::Spacing();
}
