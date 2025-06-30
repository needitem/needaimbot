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
    UIHelpers::PushStyleColors();
    
    UIHelpers::BeautifulSection("Recoil Control System");
    
    UIHelpers::BeautifulToggle("Enable Recoil Compensation", &config.easynorecoil, 
                               "Enables automatic recoil compensation. Adjust the strength to match your game's recoil patterns.");
    
    if (config.easynorecoil)
    {
        ImGui::Spacing();
        UIHelpers::BeautifulSeparator("Weapon Profiles");
        
        // Weapon Selection Section
        UIHelpers::BeautifulText("Available Weapons:", UIHelpers::GetAccentColor());
        
        auto weapon_names = config.getWeaponProfileNames();
        WeaponRecoilProfile* current_profile = config.getCurrentWeaponProfile();
        
        ImGui::BeginChild("WeaponList", ImVec2(0, 100), true);
        {
            for (size_t i = 0; i < weapon_names.size(); ++i) {
                bool is_selected = (config.active_weapon_profile_index == static_cast<int>(i));
                
                if (ImGui::Selectable(weapon_names[i].c_str(), is_selected)) {
                    config.setActiveWeaponProfile(weapon_names[i]);
                }
                
                // Right-click menu for weapon management
                if (ImGui::BeginPopupContextItem()) {
                    if (weapon_names[i] != "Default" && ImGui::MenuItem("Delete Weapon")) {
                        config.removeWeaponProfile(weapon_names[i]);
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
            if (config.addWeaponProfile(new_profile)) {
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
            
            UIHelpers::BeautifulSlider("Base Strength", &current_profile->base_strength, 0.1f, 10.0f, "%.2f");
            UIHelpers::BeautifulSlider("Fire Rate Multiplier", &current_profile->fire_rate_multiplier, 0.1f, 3.0f, "%.2f");
            UIHelpers::BeautifulSlider("Recoil Delay (ms)", &current_profile->recoil_ms, 0.1f, 50.0f, "%.1f");
            ImGui::SliderInt("Start Delay (ms)", &current_profile->start_delay_ms, 0, 500);
            ImGui::SliderInt("End Delay (ms)", &current_profile->end_delay_ms, 0, 500);
        } else {
            UIHelpers::BeautifulText("No weapon profile selected", UIHelpers::GetWarningColor());
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Scope Multipliers Section
        if (current_profile) {
            UIHelpers::BeautifulSeparator("Scope Multipliers");
            
            ImGui::Text("Current Scope: %dx", config.active_scope_magnification == 0 ? 1 : config.active_scope_magnification);
            
            ImGui::Spacing();
            
            ImGui::Columns(3, "ScopeMultipliers", false);
            UIHelpers::BeautifulSlider("1x Multi", &current_profile->scope_mult_1x, 0.1f, 3.0f, "%.2f");
            ImGui::NextColumn();
            UIHelpers::BeautifulSlider("2x Multi", &current_profile->scope_mult_2x, 0.1f, 3.0f, "%.2f");
            ImGui::NextColumn();
            UIHelpers::BeautifulSlider("3x Multi", &current_profile->scope_mult_3x, 0.1f, 3.0f, "%.2f");
            ImGui::NextColumn();
            
            UIHelpers::BeautifulSlider("4x Multi", &current_profile->scope_mult_4x, 0.1f, 3.0f, "%.2f");
            ImGui::NextColumn();
            UIHelpers::BeautifulSlider("6x Multi", &current_profile->scope_mult_6x, 0.1f, 3.0f, "%.2f");
            ImGui::NextColumn();
            UIHelpers::BeautifulSlider("8x Multi", &current_profile->scope_mult_8x, 0.1f, 3.0f, "%.2f");
            
            ImGui::Columns(1);
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
        
        // Information Section
        UIHelpers::BeautifulSeparator("Information");
        
        if (current_profile && current_profile->base_strength >= 10.0f) {
            UIHelpers::BeautifulText("WARNING: High recoil strength may be detected", UIHelpers::GetWarningColor());
        }
        
        ImGui::BeginChild("KeyBindings", ImVec2(0, 60), true, ImGuiWindowFlags_AlwaysUseWindowPadding);
        {
            UIHelpers::BeautifulText("Key Bindings:", UIHelpers::GetAccentColor());
            ImGui::Text("Left/Right Arrow: Adjust recoil strength");
            ImGui::Text("Page Up/Down: Switch weapon profiles");
            ImGui::Text("Right-click weapon: Delete weapon profile");
        }
        ImGui::EndChild();
    }
    
    UIHelpers::PopStyleColors();
}