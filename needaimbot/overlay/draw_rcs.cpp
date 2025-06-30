#include "draw_settings.h"
#include "needaimbot.h" 
#include "overlay.h" 
#include "ui_helpers.h"
#include <vector> 
#include <string> 
#include <algorithm> 
#include <cstdio>

extern std::atomic<bool> g_config_optical_flow_changed; 

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
                    config.saveWeaponProfiles();
                }
                
                // Right-click menu for weapon management
                if (ImGui::BeginPopupContextItem()) {
                    if (weapon_names[i] != "Default" && ImGui::MenuItem("Delete Weapon")) {
                        config.removeWeaponProfile(weapon_names[i]);
                        config.saveWeaponProfiles();
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
                config.saveWeaponProfiles();
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
                config.saveWeaponProfiles();
            }
            if (UIHelpers::BeautifulSlider("Fire Rate Multiplier", &current_profile->fire_rate_multiplier, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            if (UIHelpers::BeautifulSlider("Recoil Delay (ms)", &current_profile->recoil_ms, 0.1f, 50.0f, "%.1f")) {
                config.saveWeaponProfiles();
            }
            if (ImGui::SliderInt("Start Delay (ms)", &current_profile->start_delay_ms, 0, 500)) {
                config.saveWeaponProfiles();
            }
            if (ImGui::SliderInt("End Delay (ms)", &current_profile->end_delay_ms, 0, 500)) {
                config.saveWeaponProfiles();
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
                if (scope_values[i] == config.active_scope_magnification) {
                    current_scope_index = i;
                    break;
                }
            }
            
            ImGui::Text("Select Scope:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::Combo("##ScopeSelector", &current_scope_index, scope_options, 6)) {
                config.active_scope_magnification = scope_values[current_scope_index];
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            
            ImGui::SameLine();
            ImGui::Text("Current: %dx", config.active_scope_magnification == 0 ? 1 : config.active_scope_magnification);
            
            ImGui::Spacing();
            
            ImGui::Columns(3, "ScopeMultipliers", false);
            if (UIHelpers::BeautifulSlider("1x Multi", &current_profile->scope_mult_1x, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("2x Multi", &current_profile->scope_mult_2x, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("3x Multi", &current_profile->scope_mult_3x, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            
            if (UIHelpers::BeautifulSlider("4x Multi", &current_profile->scope_mult_4x, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("6x Multi", &current_profile->scope_mult_6x, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            ImGui::NextColumn();
            if (UIHelpers::BeautifulSlider("8x Multi", &current_profile->scope_mult_8x, 0.1f, 3.0f, "%.2f")) {
                config.saveWeaponProfiles();
            }
            
            ImGui::Columns(1);
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
        
        // Advanced Optical Flow NoRecoil Section
        UIHelpers::BeautifulSeparator("Advanced Optical Flow NoRecoil");
        
        if (ImGui::Checkbox("Enable Optical Flow Processing", &config.enable_optical_flow)) {
            g_config_optical_flow_changed.store(true);
            config.saveConfig();
        }
        if (ImGui::IsItemHovered()) {
            SetWrappedTooltipRCS("Enables advanced optical flow-based recoil detection and compensation. Works independently from pattern-based recoil control.");
        }
        
        if (config.enable_optical_flow) {
            ImGui::Indent();
            
            if (ImGui::Checkbox("Enable OF NoRecoil", &config.optical_flow_norecoil)) {
                g_config_optical_flow_changed.store(true);
                config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltipRCS("Enable optical flow based recoil compensation. Uses camera movement detection to counteract recoil automatically.");
            }
            
            if (config.optical_flow_norecoil) {
                ImGui::Indent();
                
                if (UIHelpers::BeautifulSlider("OF NoRecoil Strength", &config.optical_flow_norecoil_strength, 0.1f, 10.0f, "%.2f")) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Strength of optical flow based recoil compensation. Higher values = stronger compensation.");
                }
                
                if (UIHelpers::BeautifulSlider("OF NoRecoil Threshold", &config.optical_flow_norecoil_threshold, 0.1f, 5.0f, "%.2f")) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Minimum optical flow magnitude to trigger recoil compensation. Lower = more sensitive.");
                }
                
                if (ImGui::SliderInt("OF NoRecoil Frames", &config.optical_flow_norecoil_frames, 1, 10)) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Number of consecutive frames above threshold required to trigger compensation. Higher = less false positives.");
                }
                
                ImGui::Unindent();
            }
            
            // Debug Options
            if (ImGui::CollapsingHeader("Debug Options##OpticalFlow")) {
                if (ImGui::Checkbox("Draw Optical Flow Vectors", &config.draw_optical_flow)) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Show calculated optical flow vectors on debug preview window for troubleshooting.");
                }
                
                if (ImGui::SliderInt("Flow Vis Steps", &config.draw_optical_flow_steps, 1, 32)) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Density of drawn flow vectors. Higher values = less dense visualization.");
                }
                
                if (ImGui::SliderFloat("Flow Mag Threshold", &config.optical_flow_magnitudeThreshold, 0.0f, 10.0f, "%.2f")) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Minimum magnitude for a flow vector to be drawn/considered in calculations.");
                }
                
                if (ImGui::SliderFloat("Static Frame Threshold", &config.staticFrameThreshold, 0.01f, 10.0f, "%.2f")) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Threshold for mean pixel difference to consider a frame static (may pause OF processing).");
                }
                
                if (ImGui::SliderFloat("FOV X (degrees)", &config.fovX, 30.0f, 150.0f, "%.1f")) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Horizontal Field of View. Used in optical flow calculations for accuracy.");
                }
                
                if (ImGui::SliderFloat("FOV Y (degrees)", &config.fovY, 30.0f, 120.0f, "%.1f")) {
                    g_config_optical_flow_changed.store(true);
                    config.saveConfig();
                }
                if (ImGui::IsItemHovered()) {
                    SetWrappedTooltipRCS("Vertical Field of View. Used in optical flow calculations for accuracy.");
                }
            }
            
            ImGui::Unindent();
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