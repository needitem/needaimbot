#include "../core/windows_headers.h"

#include <iostream>
#include "d3d11.h"
#include "../imgui/imgui.h"

#include "AppContext.h"
#include "overlay.h"
#include "draw_settings.h"
#include "needaimbot.h"
#include "other_tools.h"
 

 

void draw_target()
{
    auto& ctx = AppContext::getInstance();
    
    // Display pause status prominently at the top
    if (ctx.detection_paused.load()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f)); // Red color
        ImGui::Text("AIMBOT PAUSED (Press %s to resume)", 
                    ctx.config.button_pause.empty() ? "F3" : ctx.config.button_pause[0].c_str());
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f)); // Green color
        ImGui::Text("AIMBOT ACTIVE");
        ImGui::PopStyleColor();
    }
    
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::Checkbox("Auto Aim", &ctx.config.auto_aim)) {
        SAVE_PROFILE();
    }

    // Target selection controls
    ImGui::Separator();
    ImGui::Text("Target Selection");
    if (ImGui::SliderFloat("Sticky Threshold", &ctx.config.sticky_target_threshold, 0.0f, 1.0f, "%.2f")) {
        if (ctx.config.sticky_target_threshold < 0.0f) ctx.config.sticky_target_threshold = 0.0f;
        if (ctx.config.sticky_target_threshold > 1.0f) ctx.config.sticky_target_threshold = 1.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("How much better a new target must be to switch.\n0.00 = always switch to closest, 0.30~0.50 = moderate stickiness.");
    }    
}

 
