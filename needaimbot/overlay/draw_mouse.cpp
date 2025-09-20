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
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 60);
        if (ImGui::InputFloat("##KpX", &ctx.config.pd_kp_x, 0.0f, 0.0f, "%.3f")) {
            if (ctx.config.pd_kp_x < 0.0f) ctx.config.pd_kp_x = 0.0f;
            if (ctx.config.pd_kp_x > 100.0f) ctx.config.pd_kp_x = 100.0f;
            SAVE_PROFILE();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("-##KpXMinus", ImVec2(25, 0))) {
            ctx.config.pd_kp_x -= 0.01f;
            if (ctx.config.pd_kp_x < 0.0f) ctx.config.pd_kp_x = 0.0f;
            SAVE_PROFILE();
        }
        ImGui::SameLine();
        if (ImGui::Button("+##KpXPlus", ImVec2(25, 0))) {
            ctx.config.pd_kp_x += 0.01f;
            if (ctx.config.pd_kp_x > 100.0f) ctx.config.pd_kp_x = 100.0f;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Gain (Kp)");
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 60);
        if (ImGui::InputFloat("##KpY", &ctx.config.pd_kp_y, 0.0f, 0.0f, "%.3f")) {
            if (ctx.config.pd_kp_y < 0.0f) ctx.config.pd_kp_y = 0.0f;
            if (ctx.config.pd_kp_y > 100.0f) ctx.config.pd_kp_y = 100.0f;
            SAVE_PROFILE();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("-##KpYMinus", ImVec2(25, 0))) {
            ctx.config.pd_kp_y -= 0.01f;
            if (ctx.config.pd_kp_y < 0.0f) ctx.config.pd_kp_y = 0.0f;
            SAVE_PROFILE();
        }
        ImGui::SameLine();
        if (ImGui::Button("+##KpYPlus", ImVec2(25, 0))) {
            ctx.config.pd_kp_y += 0.01f;
            if (ctx.config.pd_kp_y > 100.0f) ctx.config.pd_kp_y = 100.0f;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::EndTable();
    }
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    draw_movement_controls();
}