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
    
    UIHelpers::BeautifulText("Configure mouse movement behavior for smooth and accurate aiming.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    UIHelpers::SettingsSubHeader("Movement Parameters");
    
    // Create two columns for X and Y parameters
    if (ImGui::BeginTable("MouseMovementTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("X-Axis (Horizontal)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y-Axis (Vertical)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        
        // Movement Factor Row
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Movement Factor");
        if (ImGui::SliderFloat("##MovementX", &ctx.config.movement_factor_x, 0.1f, 1.0f, "%.1f%%")) {
            ctx.config.movement_factor_x = (std::max)(0.1f, (std::min)(1.0f, ctx.config.movement_factor_x));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("X-axis: % of error per frame");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Movement Factor");
        if (ImGui::SliderFloat("##MovementY", &ctx.config.movement_factor_y, 0.1f, 1.0f, "%.1f%%")) {
            ctx.config.movement_factor_y = (std::max)(0.1f, (std::min)(1.0f, ctx.config.movement_factor_y));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Y-axis: % of error per frame");
        
        ImGui::Separator();
        
        // Sensitivity Row
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Sensitivity");
        if (ImGui::SliderFloat("##SensitivityX", &ctx.config.mouse_sensitivity_x, 0.1f, 10.0f, "%.2fx")) {
            ctx.config.mouse_sensitivity_x = (std::max)(0.1f, (std::min)(10.0f, ctx.config.mouse_sensitivity_x));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("X-axis sensitivity multiplier");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Sensitivity");
        if (ImGui::SliderFloat("##SensitivityY", &ctx.config.mouse_sensitivity_y, 0.1f, 10.0f, "%.2fx")) {
            ctx.config.mouse_sensitivity_y = (std::max)(0.1f, (std::min)(10.0f, ctx.config.mouse_sensitivity_y));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Y-axis sensitivity multiplier");
        
        ImGui::Separator();
        
        // Deadzone Row
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Deadzone");
        if (ImGui::SliderFloat("##DeadzoneX", &ctx.config.min_movement_threshold_x, 0.0f, 5.0f, "%.1f px")) {
            ctx.config.min_movement_threshold_x = (std::max)(0.0f, (std::min)(5.0f, ctx.config.min_movement_threshold_x));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("X-axis minimum movement");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Deadzone");
        if (ImGui::SliderFloat("##DeadzoneY", &ctx.config.min_movement_threshold_y, 0.0f, 5.0f, "%.1f px")) {
            ctx.config.min_movement_threshold_y = (std::max)(0.0f, (std::min)(5.0f, ctx.config.min_movement_threshold_y));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Y-axis minimum movement");
        
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