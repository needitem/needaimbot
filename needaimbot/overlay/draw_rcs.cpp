#include "draw_settings.h"
#include "needaimbot.h" // For config
#include "imgui/imgui.h" // For ImGui
#include "overlay.h" // For key_names, key_names_cstrs
#include <vector> // For std::vector
#include <string> // For std::string
#include <algorithm> // For std::max
#include <cstdio> // For snprintf

// Helper function to show a tooltip with word wrapping and prevention of screen cutoff
void SetWrappedTooltipRCS(const char* text)
{
    ImGui::BeginTooltip();

    // Get window size and position
    ImVec2 window_size = ImGui::GetIO().DisplaySize;
    ImVec2 mouse_pos = ImGui::GetMousePos();

    // Calculate max width for tooltip (use 50% of screen width)
    float max_width = window_size.x * 0.5f;

    // Set text wrapping
    ImGui::PushTextWrapPos(max_width);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();

    ImGui::EndTooltip();
}

void draw_rcs_settings() {
    ImGui::Begin("RCS Settings");

    // Create a collapsible section for Recoil Control
    if (ImGui::CollapsingHeader("Recoil Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // No recoil settings
        ImGui::Checkbox("Enable Recoil Compensation", &config.easynorecoil);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltipRCS("Enables automatic recoil compensation. Adjust the strength to match your game's recoil patterns.");
        }
        
        if (config.easynorecoil)
        {
            // Add spacing for better visual separation
            ImGui::Indent(10.0f);
            
            // Recoil strength input field
            if (ImGui::InputFloat("Compensation Strength", &config.easynorecoilstrength, 0.1f, 1.0f, "%.1f")) {
                config.saveConfig(); // Save on change
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Adjusts the base intensity of recoil compensation.");
            }
            
            // Recoil adjustment step size
            if (ImGui::InputFloat("Adjustment Step Size", &config.norecoil_step, 0.0f, 0.0f, "%.1f"))
            {
                config.norecoil_step = (std::max)(0.1f, (std::min)(config.norecoil_step, 50.0f));
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Step size for adjusting recoil compensation strength with left/right arrow keys (0.1 - 50.0)");
            }
            
            // Recoil delay in milliseconds
            if (ImGui::InputFloat("Recoil Delay (ms)", &config.norecoil_ms, 0.0f, 0.0f, "%.1f"))
            {
                config.norecoil_ms = (std::max)(0.0f, (std::min)(config.norecoil_ms, 100.0f));
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltipRCS("Delay in milliseconds between recoil compensation movements (0.0 - 100.0)");
            }

            ImGui::Spacing();
            ImGui::SeparatorText("Active Scope Recoil");
            // TODO: Add int active_scope_magnification = 0; to your config struct.
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
            // TODO: Ensure these float variables exist in your config struct:
            // recoil_mult_2x, recoil_mult_3x, recoil_mult_4x, recoil_mult_6x
            struct ScopeMultiplierValue {
                const char* label;
                float* multiplier; // Pointer to config.recoil_mult_X
                int id; // Unique ID for ImGui
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
                ImGui::PushItemWidth(100); // Set a fixed width for the input field
                std::string input_label = "##MultVal" + std::to_string(scope_val.id);
                if (ImGui::InputFloat(input_label.c_str(), scope_val.multiplier, 0.01f, 0.1f, "%.2f")) {
                    *scope_val.multiplier = std::max(0.0f, *scope_val.multiplier); // Ensure non-negative
                    config.saveConfig();
                }
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    char tooltip[128];
                    snprintf(tooltip, sizeof(tooltip), "Recoil multiplier factor for %dx scope. Base strength is multiplied by this value.", scope_val.id);
                    SetWrappedTooltipRCS(tooltip);
                }
            }

            ImGui::Spacing(); // Add spacing after scope multipliers

            ImGui::Unindent(10.0f);
            
            // Warning for high recoil strength
            if (config.easynorecoilstrength >= 100.0f)
            {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: High recoil strength may be detected."); // ImVec4(255,255,0,255) is not correct for ImGui colors
            }
            
            // Hotkey information
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Left/Right Arrow keys: Adjust recoil strength");
        }
        ImGui::Spacing(); // Add spacing after Recoil settings
    }

    ImGui::End();
}