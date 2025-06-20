#include "draw_settings.h"
#include "needaimbot.h" 
#include "imgui/imgui.h" 
#include "overlay.h" 
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
    
    if (ImGui::CollapsingHeader("Recoil Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        
        ImGui::Checkbox("Enable Recoil Compensation", &config.easynorecoil);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltipRCS("Enables automatic recoil compensation. Adjust the strength to match your game's recoil patterns.");
        }
        
        if (config.easynorecoil)
        {
            
            ImGui::Indent(10.0f);
            
            
            if (ImGui::InputFloat("Compensation Strength", &config.easynorecoilstrength, 0.1f, 1.0f, "%.1f")) {
                config.saveConfig(); 
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Adjusts the base intensity of recoil compensation.");
            }
            
            
            if (ImGui::InputFloat("Adjustment Step Size", &config.norecoil_step, 0.0f, 0.0f, "%.1f"))
            {
                config.norecoil_step = (std::max)(0.1f, (std::min)(config.norecoil_step, 50.0f));
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Step size for adjusting recoil compensation strength with left/right arrow keys (0.1 - 50.0)");
            }
            
            
            if (ImGui::InputFloat("Recoil Delay (ms)", &config.norecoil_ms, 0.0f, 0.0f, "%.1f"))
            {
                config.norecoil_ms = (std::max)(0.0f, (std::min)(config.norecoil_ms, 100.0f));
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Delay in milliseconds between recoil compensation movements (0.0 - 100.0)");
            }

            
            if (ImGui::InputInt("Start Delay (ms)", &config.easynorecoil_start_delay_ms, 1, 10))
            {
                config.easynorecoil_start_delay_ms = (std::max)(0, config.easynorecoil_start_delay_ms);
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Delay in milliseconds before recoil compensation starts after the shoot key is pressed. Set to 0 to disable.");
            }

            
            if (ImGui::InputInt("End Delay (ms)", &config.easynorecoil_end_delay_ms, 1, 10))
            {
                config.easynorecoil_end_delay_ms = (std::max)(0, config.easynorecoil_end_delay_ms);
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Delay in milliseconds before recoil compensation stops after the shoot key is released. Set to 0 to disable.");
            }

            ImGui::Spacing();
            ImGui::SeparatorText("Active Scope Recoil");
            
            bool scope_changed = false;
            if (ImGui::RadioButton("None##Scope", &config.active_scope_magnification, 0)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltipRCS("Use base recoil strength (no multiplier).");
            ImGui::SameLine();
            if (ImGui::RadioButton("2x##Scope", &config.active_scope_magnification, 2)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltipRCS("Apply 2x scope recoil multiplier.");
            ImGui::SameLine();
            if (ImGui::RadioButton("3x##Scope", &config.active_scope_magnification, 3)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltipRCS("Apply 3x scope recoil multiplier.");
            ImGui::SameLine();
            if (ImGui::RadioButton("4x##Scope", &config.active_scope_magnification, 4)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltipRCS("Apply 4x scope recoil multiplier.");
            ImGui::SameLine();
            if (ImGui::RadioButton("6x##Scope", &config.active_scope_magnification, 6)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltipRCS("Apply 6x scope recoil multiplier.");

            if (scope_changed) {
                config.saveConfig();
            }

            ImGui::Spacing();
            ImGui::SeparatorText("Scope Multiplier Values");
            
            
            struct ScopeMultiplierValue {
                const char* label;
                float* multiplier; 
                int id; 
            };

            std::vector<ScopeMultiplierValue> scope_values = {
                {"2x Multiplier:", &config.recoil_mult_2x, 2},
                {"3x Multiplier:", &config.recoil_mult_3x, 3},
                {"4x Multiplier:", &config.recoil_mult_4x, 4},
                {"6x Multiplier:", &config.recoil_mult_6x, 6}
            };

            for (const auto& scope_val : scope_values) {
                ImGui::Text("%s", scope_val.label);
                ImGui::SameLine();
                ImGui::PushItemWidth(100); 
                std::string input_label = "##MultVal" + std::to_string(scope_val.id);
                if (ImGui::InputFloat(input_label.c_str(), scope_val.multiplier, 0.01f, 0.1f, "%.2f")) {
                    *scope_val.multiplier = std::max(0.0f, *scope_val.multiplier); 
                    config.saveConfig();
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    char tooltip[128];
                    snprintf(tooltip, sizeof(tooltip), "Recoil multiplier factor for %dx scope. Base strength is multiplied by this value.", scope_val.id);
                    SetWrappedTooltipRCS(tooltip);
                }
            }

            ImGui::Spacing(); 

            ImGui::Unindent(10.0f);
            
            
            if (config.easynorecoilstrength >= 100.0f)
            {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: High recoil strength may be detected."); 
            }
            
            
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Left/Right Arrow keys: Adjust recoil strength");
        }
        ImGui::Spacing(); 
    }
}
