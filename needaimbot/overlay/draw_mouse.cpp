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
    
    if (UIHelpers::EnhancedSliderFloat("Movement Factor", &ctx.config.movement_factor, 0.1f, 1.0f, "%.1f%%",
                                      "Percentage of error to move per frame (lower = smoother, higher = faster)")) {
        ctx.config.movement_factor = (std::max)(0.1f, (std::min)(1.0f, ctx.config.movement_factor));
        SAVE_PROFILE();
    }
    
    UIHelpers::HelpMarker("Controls how much of the distance to target is covered per frame.\n"
                          "0.3 = Move 30% of the distance each frame\n"
                          "Lower values give smoother movement but may converge slowly\n"
                          "Higher values are faster but may overshoot");
    
    UIHelpers::Spacer();
    
    if (UIHelpers::EnhancedSliderFloat("Mouse Sensitivity", &ctx.config.mouse_sensitivity, 0.1f, 10.0f, "%.2f",
                                      "Overall mouse sensitivity multiplier")) {
        ctx.config.mouse_sensitivity = (std::max)(0.1f, (std::min)(10.0f, ctx.config.mouse_sensitivity));
        SAVE_PROFILE();
    }
    
    UIHelpers::HelpMarker("Multiplies the final movement values.\n"
                          "Adjust based on your game's sensitivity settings.\n"
                          "Start with 1.0 and adjust as needed.");
    
    UIHelpers::Spacer();
    
    if (UIHelpers::EnhancedSliderFloat("Min Movement Threshold", &ctx.config.min_movement_threshold, 0.0f, 5.0f, "%.1f px",
                                      "Minimum pixel distance to trigger movement (deadzone)")) {
        ctx.config.min_movement_threshold = (std::max)(0.0f, (std::min)(5.0f, ctx.config.min_movement_threshold));
        SAVE_PROFILE();
    }
    
    UIHelpers::HelpMarker("Prevents micro-adjustments when target is very close to center.\n"
                          "0 = No deadzone, always move\n"
                          "1-2 = Small deadzone for stability");
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Movement Info");
    
    ImGui::Text("Current Settings:");
    ImGui::BulletText("Movement: %.0f%% of error per frame", ctx.config.movement_factor * 100);
    ImGui::BulletText("Sensitivity: %.2fx", ctx.config.mouse_sensitivity);
    ImGui::BulletText("Deadzone: %.1f pixels", ctx.config.min_movement_threshold);
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    draw_movement_controls();
}