#include "../core/windows_headers.h"

#include <shellapi.h>
#include <algorithm>

#include "AppContext.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "ui_helpers.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include "../mouse/mouse.h" 

static void draw_movement_controls()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Mouse Movement Settings");
    
    UIHelpers::BeautifulText("Simple proportional controller for aiming.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    UIHelpers::SettingsSubHeader("Proportional Gain Settings");
    
    // Simple P controller gains in two columns
    if (ImGui::BeginTable("GainsTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("X-Axis (Horizontal)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y-Axis (Vertical)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Gain (Kp)");
        if (ImGui::SliderFloat("##KpX", &ctx.config.pd_kp_x, 0.1f, 2.0f, "%.2f")) {
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Gain (Kp)");
        if (ImGui::SliderFloat("##KpY", &ctx.config.pd_kp_y, 0.1f, 2.0f, "%.2f")) {
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::EndTable();
    }
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Movement Info");
    
    if (ImGui::BeginTable("CurrentSettingsTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("X-Axis Settings:");
        ImGui::BulletText("Gain: %.2f", ctx.config.pd_kp_x);
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Y-Axis Settings:");
        ImGui::BulletText("Gain: %.2f", ctx.config.pd_kp_y);
        
        ImGui::EndTable();
    }
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    draw_movement_controls();
}