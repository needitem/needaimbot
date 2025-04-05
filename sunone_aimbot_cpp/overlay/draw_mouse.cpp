#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "include/other_tools.h"

std::string ghub_version = get_ghub_version();

// Helper function to show a tooltip with word wrapping and prevention of screen cutoff
void SetWrappedTooltip(const char* text) 
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

void draw_mouse()
{
    ImGui::Columns(2, "MouseSettingsColumns", false); // Start 2-column layout

    // --- Column 1 Start ---
    if (ImGui::CollapsingHeader("Display & Sensitivity", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("DPI", &config.dpi, 800, 5000);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Mouse DPI (Dots Per Inch). Higher values increase mouse sensitivity.");
        }
        
        ImGui::SliderInt("FOV X", &config.fovX, 60, 120);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Horizontal Field of View in degrees. Should match your game's settings.");
        }
        
        ImGui::SliderInt("FOV Y", &config.fovY, 40, 100);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Vertical Field of View in degrees. Should match your game's settings.");
        }
        ImGui::Spacing(); // Add spacing at the end of the group
    }

    // No Separator needed here due to column layout
    ImGui::Text("PID Controller Settings");
    ImGui::Spacing(); // Add spacing before the first PID header
    
    // X-axis PID Settings
    if (ImGui::CollapsingHeader("Horizontal (X-axis) PID", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Cast to float for ImGui slider but preserve double precision
        float kp_x_display = static_cast<float>(config.kp_x);
        if (ImGui::SliderFloat("Proportional X (Kp)", &kp_x_display, 0.0f, 20.0f, "%.3f"))
        {
            config.kp_x = static_cast<double>(kp_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Affects the immediate horizontal response. Higher values make aiming more responsive but can cause overshooting.");
        }

        float ki_x_display = static_cast<float>(config.ki_x);
        if (ImGui::SliderFloat("Integral X (Ki)", &ki_x_display, 0.0f, 20.0f, "%.3f"))
        {
            config.ki_x = static_cast<double>(ki_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Accounts for accumulated horizontal error over time. Higher values help eliminate persistent offset but can cause oscillation.");
        }

        float kd_x_display = static_cast<float>(config.kd_x);
        if (ImGui::SliderFloat("Derivative X (Kd)", &kd_x_display, 0.0f, 5.0f, "%.3f"))
        {
            config.kd_x = static_cast<double>(kd_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Predicts future horizontal error based on rate of change. Higher values add dampening to reduce overshooting.");
        }
        ImGui::Spacing(); // Add spacing after the X-axis PID settings
    }
    
    // Y-axis PID Settings
    if (ImGui::CollapsingHeader("Vertical (Y-axis) PID", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float kp_y_display = static_cast<float>(config.kp_y);
        if (ImGui::SliderFloat("Proportional Y (Kp)", &kp_y_display, 0.0f, 20.0f, "%.3f"))
        {
            config.kp_y = static_cast<double>(kp_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Affects the immediate vertical response. Higher values make aiming more responsive but can cause overshooting.");
        }

        float ki_y_display = static_cast<float>(config.ki_y);
        if (ImGui::SliderFloat("Integral Y (Ki)", &ki_y_display, 0.0f, 20.0f, "%.3f"))
        {
            config.ki_y = static_cast<double>(ki_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Accounts for accumulated vertical error over time. Higher values help eliminate persistent offset but can cause oscillation.");
        }

        float kd_y_display = static_cast<float>(config.kd_y);
        if (ImGui::SliderFloat("Derivative Y (Kd)", &kd_y_display, 0.0f, 5.0f, "%.3f"))
        {
            config.kd_y = static_cast<double>(kd_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Predicts future vertical error based on rate of change. Higher values add dampening to reduce overshooting.");
        }
        ImGui::Spacing(); // Add spacing after the Y-axis PID settings
    }

    // No Separator needed here
    // ImGui::Text("Kalman Filter Settings"); // Text might be redundant if header is descriptive
    ImGui::Spacing(); // Add spacing before the Kalman header

    // Group Kalman Filter Settings
    if (ImGui::CollapsingHeader("Kalman Filter Smoothing", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Process Noise (Q)", &config.process_noise_q, 0.001f, 100.0f, "%.3f");
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Represents the uncertainty in the process model. Higher values make the filter more responsive to new measurements but noisier. Lower values make aim smoother but less responsive to sudden changes.");
        }
        ImGui::SliderFloat("Measurement Noise (R)", &config.measurement_noise_r, 0.001f, 100.0f, "%.3f");
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Represents noise in the measurements. Higher values make the filter trust measurements less, resulting in smoother but potentially slower aiming. Lower values increase responsiveness but may cause jitter.");
        }
        ImGui::SliderFloat("Estimation Error (P)", &config.estimation_error_p, 0.001f, 50.0f, "%.3f");
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Initial uncertainty in the state estimate. Higher values cause the filter to give more weight to initial measurements. Affects how quickly the filter converges to stable tracking.");
        }
        ImGui::Spacing(); // Add spacing at the end of the group
    }
    // --- Column 1 End ---

    ImGui::NextColumn(); // Move to the second column

    // --- Column 2 Start ---
    // No Separator needed here
    ImGui::Spacing(); // Add spacing before the Recoil header

    // Create a collapsible section for Recoil Control
    if (ImGui::CollapsingHeader("Recoil Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // No recoil settings
        ImGui::Checkbox("Enable Recoil Compensation", &config.easynorecoil);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Enables automatic recoil compensation. Adjust the strength to match your game's recoil patterns.");
        }
        
        if (config.easynorecoil)
        {
            // Add spacing for better visual separation
            ImGui::Indent(10.0f);
            
            // Recoil strength slider
            ImGui::SliderFloat("Compensation Strength", &config.easynorecoilstrength, 0.1f, 500.0f, "%.1f");
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Adjusts the intensity of recoil compensation. Higher values mean stronger compensation.");
            }
            
            // Recoil adjustment step size
            if (ImGui::InputFloat("Adjustment Step Size", &config.norecoil_step, 0.0f, 0.0f, "%.1f"))
            {
                config.norecoil_step = std::max(0.1f, std::min(config.norecoil_step, 50.0f));
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Step size for adjusting recoil compensation strength with left/right arrow keys (0.1 - 50.0)");
            }
            ImGui::Unindent(10.0f);
            
            // Warning for high recoil strength
            if (config.easynorecoilstrength >= 100.0f)
            {
                ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: High recoil strength may be detected.");
            }
            
            // Hotkey information
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Left/Right Arrow keys: Adjust recoil strength");
        }
        ImGui::Spacing(); // Add spacing after Recoil settings
    }

    // No Separator needed here
    ImGui::Spacing(); // Add spacing before Aiming settings header

    // Create a collapsible section for Aiming Settings
    if (ImGui::CollapsingHeader("Aiming Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Auto Shoot", &config.auto_shoot);
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Automatically fires when aiming at a target. Use with caution as this may be detected in some games.");
        }
        
        if (config.auto_shoot)
        {
            ImGui::Indent(10.0f);
            ImGui::SliderFloat("Scope Multiplier", &config.bScope_multiplier, 0.5f, 2.0f, "%.1f");
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Adjusts aiming when using scoped weapons. Lower values for more precise scopes.");
            }
            ImGui::Unindent(10.0f);
        }
        ImGui::Spacing(); // Add spacing after Aiming settings
    }

    // INPUT METHODS - Put in its own collapsible section
    // No Separator needed here
    ImGui::Spacing(); // Add spacing before Input Method settings header
    if (ImGui::CollapsingHeader("Input Method Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        std::vector<std::string> input_methods = { "WIN32", "GHUB", "ARDUINO" };
        std::vector<const char*> method_items;
        method_items.reserve(input_methods.size());
        for (const auto& item : input_methods)
        {
            method_items.push_back(item.c_str());
        }

        int input_method_index = 0;
        for (size_t i = 0; i < input_methods.size(); ++i)
        {
            if (input_methods[i] == config.input_method)
            {
                input_method_index = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Input Method", &input_method_index, method_items.data(), method_items.size()))
        {
            config.input_method = input_methods[input_method_index];
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Select the input method for sending mouse movements:\n"
                             "WIN32: Standard Windows SendInput. May be detected.\n"
                             "GHUB: Logitech G Hub driver (if installed and supported). Generally safer.\n"
                             "ARDUINO: Requires a connected Arduino board flashed with appropriate firmware.");
        }

        // Display GHUB version if GHUB method is selected or potentially usable
        if (config.input_method == "GHUB" || !ghub_version.empty())
        {
            ImGui::Text("GHUB Version: %s", ghub_version.c_str());
            if (ghub_version.empty())
            {
                 ImGui::SameLine();
                 ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "(Not Found/Error)");
            }
        }

        // Optional: Add ARDUINO COM port selection if needed
        // if (config.input_method == "ARDUINO") { ... }

        ImGui::Spacing(); // Add spacing at the end
    }
    // --- Column 2 End ---

    ImGui::Columns(1); // End column layout and return to 1 column

    // Optionally add a separator at the very end if needed before other global UI elements
    // ImGui::Separator();
}