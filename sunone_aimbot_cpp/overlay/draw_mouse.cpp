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

    // No Separator needed here due to column layout
    ImGui::Text("PID Controller Settings");
    ImGui::Spacing(); // Add spacing before the first PID header
    
    // X-axis PID Settings
    if (ImGui::CollapsingHeader("Horizontal (X-axis) PID", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Cast to float for ImGui input but preserve double precision
        float kp_x_display = static_cast<float>(config.kp_x);
        if (ImGui::InputFloat("Proportional X (Kp)", &kp_x_display, 0.01f, 0.1f, "%.3f"))
        {
            config.kp_x = static_cast<double>(kp_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Affects the immediate horizontal response. Higher values make aiming more responsive but can cause overshooting.");
        }

        float ki_x_display = static_cast<float>(config.ki_x);
        if (ImGui::InputFloat("Integral X (Ki)", &ki_x_display, 0.01f, 0.1f, "%.3f"))
        {
            config.ki_x = static_cast<double>(ki_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Accounts for accumulated horizontal error over time. Higher values help eliminate persistent offset but can cause oscillation.");
        }

        float kd_x_display = static_cast<float>(config.kd_x);
        if (ImGui::InputFloat("Derivative X (Kd)", &kd_x_display, 0.01f, 0.1f, "%.3f"))
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
        if (ImGui::InputFloat("Proportional Y (Kp)", &kp_y_display, 0.01f, 0.1f, "%.3f"))
        {
            config.kp_y = static_cast<double>(kp_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Affects the immediate vertical response. Higher values make aiming more responsive but can cause overshooting.");
        }

        float ki_y_display = static_cast<float>(config.ki_y);
        if (ImGui::InputFloat("Integral Y (Ki)", &ki_y_display, 0.01f, 0.1f, "%.3f"))
        {
            config.ki_y = static_cast<double>(ki_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Accounts for accumulated vertical error over time. Higher values help eliminate persistent offset but can cause oscillation.");
        }

        float kd_y_display = static_cast<float>(config.kd_y);
        if (ImGui::InputFloat("Derivative Y (Kd)", &kd_y_display, 0.01f, 0.1f, "%.3f"))
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
    if (ImGui::CollapsingHeader("Prediction & Scope"))
    {
        // Existing Scope Multiplier if auto_shoot is enabled (moved here)
        // Consider if this should always be visible or only with auto_shoot
        // if (config.auto_shoot)
        // {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Scope Settings");
            if (ImGui::SliderFloat("Scope Multiplier", &config.bScope_multiplier, 1.0f, 10.0f, "%.2f")) {
                 config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Sensitivity reduction factor when target is within scope range. (1.0 = no reduction, >1.0 = reduction)");
            }
            ImGui::Unindent(10.0f);
            ImGui::Spacing(); // Add spacing after scope setting
        // } else {
        //    ImGui::TextDisabled("Scope Multiplier (requires Auto Shoot)");
        // }

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
            
            // Recoil strength input field
            if (ImGui::InputFloat("Compensation Strength", &config.easynorecoilstrength, 0.1f, 1.0f, "%.1f")) {
                config.saveConfig(); // Save on change
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Adjusts the base intensity of recoil compensation.");
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
            
            // Recoil delay in milliseconds
            if (ImGui::InputFloat("Recoil Delay (ms)", &config.norecoil_ms, 0.0f, 0.0f, "%.1f"))
            {
                config.norecoil_ms = std::max(0.0f, std::min(config.norecoil_ms, 100.0f));
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Delay in milliseconds between recoil compensation movements (0.0 - 100.0)");
            }

            ImGui::Spacing();
            ImGui::SeparatorText("Active Scope Recoil");
            // TODO: Add int active_scope_magnification = 0; to your config struct.
            bool scope_changed = false;
            if (ImGui::RadioButton("None##Scope", &config.active_scope_magnification, 0)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Use base recoil strength (no multiplier).");
            ImGui::SameLine();
            if (ImGui::RadioButton("2x##Scope", &config.active_scope_magnification, 2)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Apply 2x scope recoil multiplier.");
            ImGui::SameLine();
            if (ImGui::RadioButton("3x##Scope", &config.active_scope_magnification, 3)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Apply 3x scope recoil multiplier.");
            ImGui::SameLine();
            if (ImGui::RadioButton("4x##Scope", &config.active_scope_magnification, 4)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Apply 4x scope recoil multiplier.");
            ImGui::SameLine();
            if (ImGui::RadioButton("6x##Scope", &config.active_scope_magnification, 6)) { scope_changed = true; }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Apply 6x scope recoil multiplier.");

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
                    SetWrappedTooltip(tooltip);
                }
            }

            ImGui::Spacing(); // Add spacing after scope multipliers

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
            // ImGui::SliderFloat("Scope Multiplier", &config.bScope_multiplier, 1.0f, 10.0f, "%.2f");
            // if (ImGui::IsItemHovered())
            // {
            //     SetWrappedTooltip("Sensitivity reduction factor when target is within scope range. (1.0 = no reduction, >1.0 = reduction)");
            // }
            ImGui::Unindent(10.0f);
        }
        ImGui::Spacing(); // Add spacing after Aiming settings
    }

    // INPUT METHODS - Put in its own collapsible section
    // No Separator needed here
    ImGui::Spacing(); // Add spacing before Input Method settings header
    if (ImGui::CollapsingHeader("Input Method Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Add "RAZER" and potentially "KMBOX" if applicable
        std::vector<std::string> input_methods = { "WIN32", "GHUB", "ARDUINO", "RAZER", "KMBOX" }; 
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
                             "ARDUINO: Requires a connected Arduino board flashed with appropriate firmware.\n"
                             "RAZER: Uses a specific Razer driver DLL (rzctl.dll). Requires DLL path.\n"
                             "KMBOX: Uses kmBoxNet library (requires B box hardware).");
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
        if (config.input_method == "ARDUINO")
        {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Arduino Settings");

            // COM Port Input
            char port_buffer[64]; // Buffer to hold the COM port string
            strncpy_s(port_buffer, sizeof(port_buffer), config.arduino_port.c_str(), _TRUNCATE);

            ImGui::Text("COM Port:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100); // Adjust width as needed
            if (ImGui::InputText("##ArduinoPort", port_buffer, sizeof(port_buffer)))
            {
                config.arduino_port = port_buffer;
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Enter the COM port your Arduino is connected to (e.g., COM3, /dev/ttyACM0).");
            }

            // Baud Rate Input
            ImGui::Text("Baud Rate:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::InputInt("##ArduinoBaud", &config.arduino_baudrate, 0)) // No step buttons
            {
                // Add validation if necessary (e.g., ensure it's a standard rate)
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Serial communication speed (e.g., 9600, 115200). Must match Arduino firmware.");
            }

            // 16-bit Mouse Checkbox
            if (ImGui::Checkbox("Use 16-bit Mouse Movement", &config.arduino_16_bit_mouse))
            {
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Send mouse movement data as 16-bit values (requires compatible firmware). Otherwise, uses 8-bit.");
            }

            // Enable Keys Checkbox (Optional, add if needed)
            // if (ImGui::Checkbox("Enable Key Sending", &config.arduino_enable_keys))
            // {
            //     config.saveConfig();
            // }
            // if (ImGui::IsItemHovered())
            // {
            //     SetWrappedTooltip("Allow sending key presses via Arduino (requires compatible firmware).");
            // }

            // TODO: Add other Arduino settings here if needed (Baud rate, 16-bit mouse, enable keys) <-- REMOVED
            ImGui::Unindent(10.0f);
        }

        // Kmbox Settings (assuming kmboxNet library is integrated elsewhere)
        if (config.input_method == "KMBOX") 
        {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Kmbox Settings");

            // Example: Display connection status or allow config changes
            // You might need functions like kmNet_is_connected() or similar
            // bool connected = kmNet_is_connected(); // Hypothetical function
            // ImGui::Text("Status: %s", connected ? "Connected" : "Disconnected");
            // Add inputs for IP/Port if needed
            ImGui::TextWrapped("KmboxNet selected. Ensure B-box hardware is connected and kmNet library is initialized.");

            ImGui::Unindent(10.0f);
        }

        ImGui::Spacing(); // Add spacing at the end
    }
    // --- Column 2 End ---

    ImGui::Columns(1); // End column layout and return to 1 column

    // Optionally add a separator at the very end if needed before other global UI elements
    // ImGui::Separator();
}