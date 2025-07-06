#include "AppContext.h"
#include "draw_settings.h"
#include "needaimbot.h" 
#include "overlay.h" 
#include "ui_helpers.h"
#include <vector> 
#include <string> 
#include <algorithm> 
#include <cstdio>

 

void SetWrappedTooltipRCS(const char* text)
{
    ImGui::BeginTooltip();
    ImVec2 window_size = ImGui::GetIO().DisplaySize;
    ImVec2 mouse_pos = ImGui::GetMousePos();
    float max_width = window_size.x * 0.5f;
    ImGui::PushTextWrapPos(max_width);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

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
        
        // Weapon Selection Section
        UIHelpers::BeautifulText("Available Weapons:", UIHelpers::GetAccentColor());
        
        auto weapon_names = ctx.config.getWeaponProfileNames();
        WeaponRecoilProfile* current_profile = ctx.config.getCurrentWeaponProfile();
        
        ImGui::BeginChild("WeaponList", ImVec2(0, 100), true);
        {
            for (size_t i = 0; i < weapon_names.size(); ++i) {
                bool is_selected = (ctx.config.active_weapon_profile_index == static_cast<int>(i));
                
                if (ImGui::Selectable(weapon_names[i].c_str(), is_selected)) {
                    ctx.config.setActiveWeaponProfile(weapon_names[i]);
                    ctx.config.saveWeaponProfiles();
                }
                
                // Right-click menu for weapon management
                if (ImGui::BeginPopupContextItem()) {
                    if (weapon_names[i] != "Default" && ImGui::MenuItem("Delete Weapon")) {
                        ctx.config.removeWeaponProfile(weapon_names[i]);
                        ctx.config.saveWeaponProfiles();
                    }
                    ImGui::EndPopup();
                }
            }
        }
        ImGui::EndChild();
        
        // Add new weapon
        static char weapon_name_buffer[64] = "";
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("##NewWeaponName", weapon_name_buffer, sizeof(weapon_name_buffer));
        ImGui::SameLine();
        if (ImGui::Button("Add Weapon") && strlen(weapon_name_buffer) > 0) {
            WeaponRecoilProfile new_profile(weapon_name_buffer, 3.0f, 1.0f);
            if (ctx.config.addWeaponProfile(new_profile)) {
                ctx.config.saveWeaponProfiles();
                weapon_name_buffer[0] = '\0';
            }
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Current Weapon Settings Section
        if (current_profile) {
            UIHelpers::BeautifulText("Current Weapon Settings:", UIHelpers::GetAccentColor());
            ImGui::Text("Weapon: %s", current_profile->weapon_name.c_str());
            
            ImGui::Spacing();
            
            if (UIHelpers::BeautifulSlider("Base Strength", &current_profile->base_strength, 0.1f, 10.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            if (UIHelpers::BeautifulSlider("Fire Rate Multiplier", &current_profile->fire_rate_multiplier, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            if (UIHelpers::BeautifulSlider("Recoil Delay (ms)", &current_profile->recoil_ms, 0.1f, 50.0f, "%.1f")) {
                ctx.config.saveWeaponProfiles();
            }
            if (ImGui::SliderInt("Start Delay (ms)", &current_profile->start_delay_ms, 0, 500)) {
                ctx.config.saveWeaponProfiles();
            }
            if (ImGui::SliderInt("End Delay (ms)", &current_profile->end_delay_ms, 0, 500)) {
                ctx.config.saveWeaponProfiles();
            }
        } else {
            UIHelpers::BeautifulText("No weapon profile selected", UIHelpers::GetWarningColor());
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Scope Multipliers Section
        if (current_profile) {
            UIHelpers::BeautifulSeparator("Scope Multipliers");
            
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
                ctx.config.saveConfig();
            }
            ImGui::PopItemWidth();
            
            ImGui::SameLine();
            ImGui::Text("Current: %dx", ctx.config.active_scope_magnification == 0 ? 1 : ctx.config.active_scope_magnification);
            
            ImGui::Spacing();
            
            ImGui::Columns(3, "ScopeMultipliers", false);
            if (UIHelpers::BeautifulSlider("1x Multi", &current_profile->scope_mult_1x, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("2x Multi", &current_profile->scope_mult_2x, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("3x Multi", &current_profile->scope_mult_3x, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            
            if (UIHelpers::BeautifulSlider("4x Multi", &current_profile->scope_mult_4x, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("6x Multi", &current_profile->scope_mult_6x, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("8x Multi", &current_profile->scope_mult_8x, 0.1f, 3.0f, "%.2f")) {
                ctx.config.saveWeaponProfiles();
            }
            
            ImGui::Columns(1);
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
        
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Information Section
        UIHelpers::BeautifulSeparator("Information");
        
        if (current_profile && current_profile->base_strength >= 10.0f) {
            UIHelpers::BeautifulText("WARNING: High recoil strength may be detected", UIHelpers::GetWarningColor());
        }
        
        ImGui::BeginChild("KeyBindings", ImVec2(0, 80), true, ImGuiWindowFlags_AlwaysUseWindowPadding);
        {
            UIHelpers::BeautifulText("Key Bindings:", UIHelpers::GetAccentColor());
            ImGui::Text("Left/Right Arrow: Adjust recoil strength");
            ImGui::Text("Page Up/Down: Switch weapon profiles");
            ImGui::Text("Right-click weapon: Delete weapon profile");
            ImGui::Spacing();
            UIHelpers::BeautifulText("Recoil Methods:", UIHelpers::GetAccentColor());
            ImGui::Text("Pattern-based: Uses weapon-specific recoil patterns");
            ImGui::Text("Optical Flow: Uses real-time camera movement detection");
        }
        ImGui::EndChild();
    }
    
    UIHelpers::PopStyleColors();
}