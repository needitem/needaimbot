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
        
        ImGui::BeginChild("WeaponProfilesSection", ImVec2(0, 220), true, ImGuiWindowFlags_AlwaysUseWindowPadding);
        {
            auto weapon_names = config.getWeaponProfileNames();
            WeaponRecoilProfile* current_profile = config.getCurrentWeaponProfile();
            
            ImGui::Columns(2, "WeaponProfilesLayout", false);
            ImGui::SetColumnWidth(0, 280);
            
            // Left column: Weapon selection
            UIHelpers::BeautifulText("Available Weapons:", UIHelpers::GetAccentColor());
            
            ImGui::BeginChild("WeaponList", ImVec2(0, 140), true);
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
            
            // Add new weapon button
            static char weapon_name_buffer[64] = "";
            ImGui::SetNextItemWidth(180);
            ImGui::InputText("##NewWeaponName", weapon_name_buffer, sizeof(weapon_name_buffer));
            ImGui::SameLine();
            if (ImGui::Button("Add Weapon") && strlen(weapon_name_buffer) > 0) {
                WeaponRecoilProfile new_profile(weapon_name_buffer, 3.0f, 1.0f);
                if (config.addWeaponProfile(new_profile)) {
                            weapon_name_buffer[0] = '\0'; // Clear input
                }
            }
            
            ImGui::NextColumn();
            
            // Right column: Current weapon info
            if (current_profile) {
                UIHelpers::BeautifulText("Current Weapon Settings:", UIHelpers::GetAccentColor());
                
                ImGui::BeginChild("CurrentWeaponSettings", ImVec2(0, 180), true);
                {
                    ImGui::Text("Weapon: %s", current_profile->weapon_name.c_str());
                    
                    UIHelpers::BeautifulSlider("Base Strength", &current_profile->base_strength, 0.1f, 10.0f, "%.2f");
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                    }
                    
                    UIHelpers::BeautifulSlider("Fire Rate Multiplier", &current_profile->fire_rate_multiplier, 0.1f, 3.0f, "%.2f");
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                    }
                    
                    UIHelpers::BeautifulSlider("Recoil Delay (ms)", &current_profile->recoil_ms, 0.1f, 50.0f, "%.1f");
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                    }
                    
                    ImGui::SliderInt("Start Delay (ms)", &current_profile->start_delay_ms, 0, 500);
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                    }
                    
                    ImGui::SliderInt("End Delay (ms)", &current_profile->end_delay_ms, 0, 500);
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                    }
                }
                ImGui::EndChild();
            } else {
                UIHelpers::BeautifulText("No weapon profile selected", UIHelpers::GetWarningColor());
            }
            
            ImGui::Columns(1);
        }
        ImGui::EndChild();
        
        UIHelpers::BeautifulSeparator("Global Settings");
        
        UIHelpers::BeautifulSlider("Legacy Compensation Strength", &config.easynorecoilstrength, 0.1f, 10.0f, "%.1f");
        UIHelpers::InfoTooltip("Fallback strength when no weapon profile is active");
        
        if (ImGui::IsItemDeactivatedAfterEdit()) {
        }
        
        ImGui::Spacing();
        UIHelpers::BeautifulSlider("Adjustment Step Size", &config.norecoil_step, 0.1f, 50.0f, "%.1f");
        UIHelpers::InfoTooltip("Step size for adjusting recoil with arrow keys (0.1 - 50.0)");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
        }
        
        UIHelpers::BeautifulSlider("Recoil Delay (ms)", &config.norecoil_ms, 0.0f, 100.0f, "%.1f");
        UIHelpers::InfoTooltip("Delay between recoil compensation movements (0.0 - 100.0)");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
        }
        
        ImGui::Spacing();
        UIHelpers::BeautifulSeparator("Timing Controls");
        
        static int start_delay = config.easynorecoil_start_delay_ms;
        static int end_delay = config.easynorecoil_end_delay_ms;
        
        ImGui::SliderInt("Start Delay (ms)", &start_delay, 0, 500);
        UIHelpers::InfoTooltip("Delay before recoil compensation starts after shoot key press");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            config.easynorecoil_start_delay_ms = start_delay;
        }
        
        ImGui::SliderInt("End Delay (ms)", &end_delay, 0, 500);
        UIHelpers::InfoTooltip("Delay before recoil compensation stops after shoot key release");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            config.easynorecoil_end_delay_ms = end_delay;
        }

        ImGui::Spacing();
        UIHelpers::BeautifulSeparator("Active Scope Settings");
        
        ImGui::BeginChild("ScopeSettings", ImVec2(0, 200), true, ImGuiWindowFlags_AlwaysUseWindowPadding);
        {
            UIHelpers::BeautifulText("Current Scope Magnification", UIHelpers::GetAccentColor());
            
            bool scope_changed = false;
            
            ImGui::Columns(3, "ScopeColumns", false);
            
            if (ImGui::RadioButton("1x / None", &config.active_scope_magnification, 0)) { scope_changed = true; }
            ImGui::NextColumn();
            if (ImGui::RadioButton("2x Scope", &config.active_scope_magnification, 2)) { scope_changed = true; }
            ImGui::NextColumn();
            if (ImGui::RadioButton("3x Scope", &config.active_scope_magnification, 3)) { scope_changed = true; }
            ImGui::NextColumn();
            
            if (ImGui::RadioButton("4x Scope", &config.active_scope_magnification, 4)) { scope_changed = true; }
            ImGui::NextColumn();
            if (ImGui::RadioButton("6x Scope", &config.active_scope_magnification, 6)) { scope_changed = true; }
            ImGui::NextColumn();
            if (ImGui::RadioButton("8x Scope", &config.active_scope_magnification, 8)) { scope_changed = true; }
            
            ImGui::Columns(1);
            
            if (scope_changed) {
                }
            
            ImGui::Spacing();
            UIHelpers::BeautifulSeparator("Scope Multipliers");
            
            ImGui::Columns(2, "MultiplierColumns", false);
            ImGui::SetColumnWidth(0, 120);
            
            UIHelpers::BeautifulSlider("2x Multi", &config.recoil_mult_2x, 0.1f, 3.0f, "%.2f");
            UIHelpers::BeautifulSlider("3x Multi", &config.recoil_mult_3x, 0.1f, 3.0f, "%.2f");
            UIHelpers::BeautifulSlider("4x Multi", &config.recoil_mult_4x, 0.1f, 3.0f, "%.2f");
            
            ImGui::NextColumn();
            
            UIHelpers::BeautifulSlider("6x Multi", &config.recoil_mult_6x, 0.1f, 3.0f, "%.2f");
            
            WeaponRecoilProfile* current_profile = config.getCurrentWeaponProfile();
            if (current_profile) {
                UIHelpers::BeautifulSlider("8x Multi", &current_profile->scope_mult_8x, 0.1f, 3.0f, "%.2f");
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                        }
            }
            
        }
        ImGui::EndChild();
        
        ImGui::Spacing();
        UIHelpers::BeautifulSeparator("Information");
        
        if (config.easynorecoilstrength >= 10.0f) {
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
