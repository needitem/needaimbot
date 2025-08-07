#include "AppContext.h"
#include "draw_settings.h"
#include "needaimbot.h" 
#include "overlay.h" 
#include "ui_helpers.h"
#include <vector> 
#include <string> 
#include <algorithm> 
#include <cstdio>

 


void draw_rcs_settings() {
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::PushStyleColors();
    
    UIHelpers::BeautifulSection("Recoil Control System");
    
    UIHelpers::BeautifulToggle("Enable Recoil Compensation", &ctx.config.easynorecoil, 
                               "Enables automatic recoil compensation. Adjust the strength to match your game's recoil patterns.");
    
    if (ctx.config.easynorecoil)
    {
        ImGui::Spacing();
        UIHelpers::BeautifulSeparator("Weapon Profiles");
        
        // Weapon dropdown selection
        UIHelpers::WeaponProfileDropdown();
        ImGui::SameLine();
        UIHelpers::HelpMarker("Right-click on a weapon to delete or duplicate it");
        
        WeaponRecoilProfile* current_profile = ctx.config.getCurrentWeaponProfile();
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Current Weapon Settings Section
        if (current_profile) {
            UIHelpers::BeginCard("Weapon Settings");
            
            UIHelpers::BeautifulText(current_profile->weapon_name.c_str(), UIHelpers::GetAccentColor());
            
            if (UIHelpers::BeautifulSlider("Base Strength", &current_profile->base_strength, 0.1f, 10.0f, "%.2f")) {
                SAVE_WEAPON_PROFILE();
            }
            if (UIHelpers::BeautifulSlider("Fire Rate Multiplier", &current_profile->fire_rate_multiplier, 0.1f, 3.0f, "%.2f")) {
                SAVE_WEAPON_PROFILE();
            }
            if (UIHelpers::BeautifulSlider("Recoil Delay (ms)", &current_profile->recoil_ms, 0.1f, 50.0f, "%.1f")) {
                SAVE_WEAPON_PROFILE();
            }
            if (ImGui::SliderInt("Start Delay (ms)", &current_profile->start_delay_ms, 0, 500)) {
                SAVE_WEAPON_PROFILE();
            }
            if (ImGui::SliderInt("End Delay (ms)", &current_profile->end_delay_ms, 0, 500)) {
                SAVE_WEAPON_PROFILE();
            }
            
            UIHelpers::EndCard();
        } else {
            UIHelpers::BeautifulText("No weapon profile selected", UIHelpers::GetWarningColor());
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Scope Multipliers Section
        if (current_profile) {
            UIHelpers::BeginCard("Scope Multipliers");
            
            // Manual Scope Selection
            const char* scope_options[] = { "1x", "2x", "3x", "4x", "6x", "8x" };
            int scope_values[] = { 1, 2, 3, 4, 6, 8 };
            int current_scope_index = 0;
            
            // Find current scope index
            for (int i = 0; i < 6; i++) {
                if (scope_values[i] == ctx.config.active_scope_magnification) {
                    current_scope_index = i;
                    break;
                }
            }
            
            ImGui::Text("Select Scope:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::Combo("##ScopeSelector", &current_scope_index, scope_options, 6)) {
                ctx.config.active_scope_magnification = scope_values[current_scope_index];
                SAVE_PROFILE();
            }
            ImGui::PopItemWidth();
            
            ImGui::SameLine();
            ImGui::Text("Current: %dx", ctx.config.active_scope_magnification == 0 ? 1 : ctx.config.active_scope_magnification);
            
            ImGui::Spacing();
            
            ImGui::Columns(3, "ScopeMultipliers", false);
            
            // 1x Multiplier
            ImGui::Text("1x Multi: %.2f", current_profile->scope_mult_1x);
            ImGui::SameLine();
            if (ImGui::Button("-##1x")) {
                current_profile->scope_mult_1x = (std::max)(0.1f, current_profile->scope_mult_1x - 0.1f);
                SAVE_WEAPON_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("+##1x")) {
                current_profile->scope_mult_1x += 0.1f;
                SAVE_WEAPON_PROFILE();
            }
            ImGui::NextColumn();
            
            // 2x Multiplier
            ImGui::Text("2x Multi: %.2f", current_profile->scope_mult_2x);
            ImGui::SameLine();
            if (ImGui::Button("-##2x")) {
                current_profile->scope_mult_2x = (std::max)(0.1f, current_profile->scope_mult_2x - 0.1f);
                SAVE_WEAPON_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("+##2x")) {
                current_profile->scope_mult_2x += 0.1f;
                SAVE_WEAPON_PROFILE();
            }
            ImGui::NextColumn();
            
            // 3x Multiplier
            ImGui::Text("3x Multi: %.2f", current_profile->scope_mult_3x);
            ImGui::SameLine();
            if (ImGui::Button("-##3x")) {
                current_profile->scope_mult_3x = (std::max)(0.1f, current_profile->scope_mult_3x - 0.1f);
                SAVE_WEAPON_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("+##3x")) {
                current_profile->scope_mult_3x += 0.1f;
                SAVE_WEAPON_PROFILE();
            }
            ImGui::NextColumn();
            
            // 4x Multiplier
            ImGui::Text("4x Multi: %.2f", current_profile->scope_mult_4x);
            ImGui::SameLine();
            if (ImGui::Button("-##4x")) {
                current_profile->scope_mult_4x = (std::max)(0.1f, current_profile->scope_mult_4x - 0.1f);
                SAVE_WEAPON_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("+##4x")) {
                current_profile->scope_mult_4x += 0.1f;
                SAVE_WEAPON_PROFILE();
            }
            ImGui::NextColumn();
            
            // 6x Multiplier
            ImGui::Text("6x Multi: %.2f", current_profile->scope_mult_6x);
            ImGui::SameLine();
            if (ImGui::Button("-##6x")) {
                current_profile->scope_mult_6x = (std::max)(0.1f, current_profile->scope_mult_6x - 0.1f);
                SAVE_WEAPON_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("+##6x")) {
                current_profile->scope_mult_6x += 0.1f;
                SAVE_WEAPON_PROFILE();
            }
            ImGui::NextColumn();
            
            // 8x Multiplier
            ImGui::Text("8x Multi: %.2f", current_profile->scope_mult_8x);
            ImGui::SameLine();
            if (ImGui::Button("-##8x")) {
                current_profile->scope_mult_8x = (std::max)(0.1f, current_profile->scope_mult_8x - 0.1f);
                SAVE_WEAPON_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("+##8x")) {
                current_profile->scope_mult_8x += 0.1f;
                SAVE_WEAPON_PROFILE();
            }
            
            ImGui::Columns(1);
            
            UIHelpers::EndCard();
        }
        
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Crouch Recoil Reduction Section
        UIHelpers::BeginCard("Crouch Recoil Reduction");
        
        UIHelpers::BeautifulToggle("Enable Crouch Compensation Adjustment", &ctx.config.crouch_recoil_enabled,
                                  "Adjust recoil compensation strength when crouching");
        
        if (ctx.config.crouch_recoil_enabled) {
            ImGui::Spacing();
            
            // Reduction percentage slider
            ImGui::Text("Compensation Adjustment:");
            ImGui::SameLine();
            if (ctx.config.crouch_recoil_reduction > 0) {
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "+%.0f%% (More compensation)", ctx.config.crouch_recoil_reduction);
            } else if (ctx.config.crouch_recoil_reduction < 0) {
                ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "%.0f%% (Less compensation)", ctx.config.crouch_recoil_reduction);
            } else {
                ImGui::Text("0%% (No change)");
            }
            
            if (UIHelpers::BeautifulSlider("##CrouchReduction", &ctx.config.crouch_recoil_reduction, -100.0f, 100.0f, "%.0f%%")) {
                SAVE_PROFILE();
            }
            
            ImGui::Spacing();
            ImGui::Text("Crouch Key: Left Control");
            ImGui::SameLine();
            UIHelpers::HelpMarker("Hold Left Control while shooting to adjust compensation\n"
                                 "-50% = Apply only 50% of recoil compensation\n"
                                 "0% = Normal compensation\n"
                                 "+50% = Apply 150% of recoil compensation");
        }
        
        UIHelpers::EndCard();
        
        ImGui::Spacing();
        
        // Information Section
        UIHelpers::BeginCard("Quick Reference");
        
        if (current_profile && current_profile->base_strength >= 10.0f) {
            UIHelpers::BeautifulText("WARNING: High recoil strength may be detected", UIHelpers::GetWarningColor());
            ImGui::Spacing();
        }
        
        UIHelpers::BeautifulText("Key Bindings:", UIHelpers::GetAccentColor());
        ImGui::BulletText("Left/Right Arrow: Adjust recoil strength");
        ImGui::BulletText("Page Up/Down: Switch weapon profiles");
        ImGui::BulletText("Left Control: Apply crouch recoil reduction");
        
        ImGui::Spacing();
        
        UIHelpers::BeautifulText("Recoil Methods:", UIHelpers::GetAccentColor());
        ImGui::BulletText("Pattern-based: Uses weapon-specific recoil patterns");
        ImGui::BulletText("Optical Flow: Uses real-time camera movement detection");
        
        UIHelpers::EndCard();
    }
    
    UIHelpers::PopStyleColors();
}