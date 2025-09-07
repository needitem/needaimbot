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
            if (ctx.config.pd_kp_x > 10.0f) ctx.config.pd_kp_x = 10.0f;
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
            if (ctx.config.pd_kp_x > 10.0f) ctx.config.pd_kp_x = 10.0f;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Gain (Kp)");
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 60);
        if (ImGui::InputFloat("##KpY", &ctx.config.pd_kp_y, 0.0f, 0.0f, "%.3f")) {
            if (ctx.config.pd_kp_y < 0.0f) ctx.config.pd_kp_y = 0.0f;
            if (ctx.config.pd_kp_y > 10.0f) ctx.config.pd_kp_y = 10.0f;
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
            if (ctx.config.pd_kp_y > 10.0f) ctx.config.pd_kp_y = 10.0f;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::EndTable();
    }
    
    UIHelpers::EndCard();
}

static void draw_dead_zone_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Dead Zone Settings");
    
    UIHelpers::BeautifulText("Prevent micro-adjustments when target is very close", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    // Dead zone
    ImGui::Text("Dead Zone (pixels)");
    ImGui::SameLine(150);
    ImGui::PushItemWidth(200);
    if (ImGui::SliderFloat("##DeadZone", &ctx.config.movement_dead_zone, 0.0f, 10.0f, "%.1f")) {
        SAVE_PROFILE();
    }
    ImGui::PopItemWidth();
    UIHelpers::HelpMarker("No movement when target is closer than this distance. Helps prevent jittery aiming.");
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    draw_movement_controls();
    draw_dead_zone_settings();
}