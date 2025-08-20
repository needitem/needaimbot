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
    
    UIHelpers::BeautifulText("PD controller for precise and stable aiming.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    UIHelpers::SettingsSubHeader("Deadzone Settings");
    
    // Deadzone settings in two columns
    if (ImGui::BeginTable("DeadzoneTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("X-Axis (Horizontal)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y-Axis (Vertical)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        if (ImGui::SliderFloat("##DeadzoneX", &ctx.config.min_movement_threshold_x, 0.0f, 5.0f, "%.1f px")) {
            ctx.config.min_movement_threshold_x = (std::max)(0.0f, (std::min)(5.0f, ctx.config.min_movement_threshold_x));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Ignore movements smaller than this");
        
        ImGui::TableSetColumnIndex(1);
        if (ImGui::SliderFloat("##DeadzoneY", &ctx.config.min_movement_threshold_y, 0.0f, 5.0f, "%.1f px")) {
            ctx.config.min_movement_threshold_y = (std::max)(0.0f, (std::min)(5.0f, ctx.config.min_movement_threshold_y));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Ignore movements smaller than this");
        
        ImGui::EndTable();
    }
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("PD Controller Settings");
    
    UIHelpers::BeautifulText("Proportional-Derivative controller for precise and stable aiming", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    // PD Gains in a table
    if (ImGui::BeginTable("PDGainsTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("Proportional (P) Gain", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Derivative (D) Gain", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        
        // X-axis gains
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("X-Axis (Kp)");
        if (ImGui::SliderFloat("##KpX", &ctx.config.pd_kp_x, 0.1f, 2.0f, "%.2f")) {
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to position error");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("X-Axis (Kd)");
        if (ImGui::SliderFloat("##KdX", &ctx.config.pd_kd_x, 0.0f, 0.5f, "%.3f")) {
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Dampens movement to prevent overshooting");
        
        // Y-axis gains
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Y-Axis (Kp)");
        if (ImGui::SliderFloat("##KpY", &ctx.config.pd_kp_y, 0.1f, 2.0f, "%.2f")) {
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to position error");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Y-Axis (Kd)");
        if (ImGui::SliderFloat("##KdY", &ctx.config.pd_kd_y, 0.0f, 0.5f, "%.3f")) {
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Dampens movement to prevent overshooting");
        
        ImGui::EndTable();
    }
    
    UIHelpers::CompactSpacer();
    
    // Derivative filter
    ImGui::Text("Derivative Filter");
    if (ImGui::SliderFloat("##DFilter", &ctx.config.pd_derivative_filter, 0.0f, 0.95f, "%.2f")) {
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker(
        "Filters noise from derivative calculation\n"
        "0.0 = No filtering (responsive but noisy)\n"
        "0.7 = Balanced (default)\n"
        "0.95 = Heavy filtering (smooth but less responsive)"
    );
    
    // Reset button placeholder - can be used for future functionality
    // if (ImGui::Button("Reset States")) {
    //     // Reset functionality can be added here
    // }
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Movement Info");
    
    if (ImGui::BeginTable("CurrentSettingsTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("X-Axis Settings:");
        ImGui::BulletText("Movement: %.0f%%", ctx.config.movement_factor_x * 100);
        ImGui::BulletText("Sensitivity: %.2fx", ctx.config.mouse_sensitivity_x);
        ImGui::BulletText("Deadzone: %.1f px", ctx.config.min_movement_threshold_x);
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Y-Axis Settings:");
        ImGui::BulletText("Movement: %.0f%%", ctx.config.movement_factor_y * 100);
        ImGui::BulletText("Sensitivity: %.2fx", ctx.config.mouse_sensitivity_y);
        ImGui::BulletText("Deadzone: %.1f px", ctx.config.min_movement_threshold_y);
        
        ImGui::EndTable();
    }
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    draw_movement_controls();
}