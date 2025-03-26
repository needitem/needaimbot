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
    
    ImGui::Separator();
    ImGui::Text("PID Controller Settings");
    
    // X-axis PID Settings
    if (ImGui::CollapsingHeader("Horizontal (X-axis) PID", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Cast to float for ImGui slider but preserve double precision
        float kp_x_display = static_cast<float>(config.kp_x);
        if (ImGui::SliderFloat("Proportional X (Kp)", &kp_x_display, 0.0f, 3.0f, "%.3f"))
        {
            config.kp_x = static_cast<double>(kp_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Affects the immediate horizontal response. Higher values make aiming more responsive but can cause overshooting.");
        }

        float ki_x_display = static_cast<float>(config.ki_x);
        if (ImGui::SliderFloat("Integral X (Ki)", &ki_x_display, 0.0f, 5.0f, "%.3f"))
        {
            config.ki_x = static_cast<double>(ki_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Accounts for accumulated horizontal error over time. Higher values help eliminate persistent offset but can cause oscillation.");
        }

        float kd_x_display = static_cast<float>(config.kd_x);
        if (ImGui::SliderFloat("Derivative X (Kd)", &kd_x_display, 0.0f, 1.0f, "%.3f"))
        {
            config.kd_x = static_cast<double>(kd_x_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Predicts future horizontal error based on rate of change. Higher values add dampening to reduce overshooting.");
        }
    }
    
    // Y-axis PID Settings
    if (ImGui::CollapsingHeader("Vertical (Y-axis) PID", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float kp_y_display = static_cast<float>(config.kp_y);
        if (ImGui::SliderFloat("Proportional Y (Kp)", &kp_y_display, 0.0f, 3.0f, "%.3f"))
        {
            config.kp_y = static_cast<double>(kp_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Affects the immediate vertical response. Higher values make aiming more responsive but can cause overshooting.");
        }

        float ki_y_display = static_cast<float>(config.ki_y);
        if (ImGui::SliderFloat("Integral Y (Ki)", &ki_y_display, 0.0f, 5.0f, "%.3f"))
        {
            config.ki_y = static_cast<double>(ki_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Accounts for accumulated vertical error over time. Higher values help eliminate persistent offset but can cause oscillation.");
        }

        float kd_y_display = static_cast<float>(config.kd_y);
        if (ImGui::SliderFloat("Derivative Y (Kd)", &kd_y_display, 0.0f, 1.0f, "%.3f"))
        {
            config.kd_y = static_cast<double>(kd_y_display);
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Predicts future vertical error based on rate of change. Higher values add dampening to reduce overshooting.");
        }
    }
    
    // Legacy combined PID settings (for backward compatibility)
    if (ImGui::CollapsingHeader("Combined PID (Legacy)"))
    {
        float kp_display = static_cast<float>(config.kp);
        if (ImGui::SliderFloat("Proportional (Kp)", &kp_display, 0.0f, 3.0f, "%.3f"))
        {
            config.kp = static_cast<double>(kp_display);
            // Sync legacy value to both axes if user adjusts it
            config.kp_x = config.kp;
            config.kp_y = config.kp;
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Legacy setting that affects both axes. Prefer using the separate X and Y controls above.");
        }
        
        float ki_display = static_cast<float>(config.ki);
        if (ImGui::SliderFloat("Integral (Ki)", &ki_display, 0.0f, 5.0f, "%.3f"))
        {
            config.ki = static_cast<double>(ki_display);
            // Sync legacy value to both axes
            config.ki_x = config.ki;
            config.ki_y = config.ki;
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Legacy setting that affects both axes. Prefer using the separate X and Y controls above.");
        }
        
        float kd_display = static_cast<float>(config.kd);
        if (ImGui::SliderFloat("Derivative (Kd)", &kd_display, 0.0f, 1.0f, "%.3f"))
        {
            config.kd = static_cast<double>(kd_display);
            // Sync legacy value to both axes
            config.kd_x = config.kd;
            config.kd_y = config.kd;
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Legacy setting that affects both axes. Prefer using the separate X and Y controls above.");
        }
    }

    ImGui::Separator();
    ImGui::Text("Kalman Filter Settings");
    ImGui::SliderFloat("Process Noise (Q)", &config.process_noise_q, 0.001f, 5.0f, "%.3f");
    if (ImGui::IsItemHovered())
    {
        SetWrappedTooltip("Represents the uncertainty in the process model. Higher values make the filter more responsive to new measurements but noisier. Lower values make aim smoother but less responsive to sudden changes.");
    }
    ImGui::SliderFloat("Measurement Noise (R)", &config.measurement_noise_r, 0.001f, 1.0f, "%.3f");
    if (ImGui::IsItemHovered())
    {
        SetWrappedTooltip("Represents noise in the measurements. Higher values make the filter trust measurements less, resulting in smoother but potentially slower aiming. Lower values increase responsiveness but may cause jitter.");
    }
    ImGui::SliderFloat("Estimation Error (P)", &config.estimation_error_p, 0.001f, 10.0f, "%.3f");
    if (ImGui::IsItemHovered())
    {
        SetWrappedTooltip("Initial uncertainty in the state estimate. Higher values cause the filter to give more weight to initial measurements. Affects how quickly the filter converges to stable tracking.");
    }

    ImGui::Separator();

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
            
            // Add a slider for recoil reduction when aiming
            // Convert the 0.0-1.0 range to -100 to 100 percent for display
            int reduction_percent = static_cast<int>((config.recoil_reduction_while_aiming * 2.0f - 1.0f) * 100.0f);
            if (ImGui::SliderInt("Recoil Adjust When Aiming", &reduction_percent, -100, 100, "%d%%"))
            {
                // Convert percent back to 0.0-1.0 range
                config.recoil_reduction_while_aiming = static_cast<float>(reduction_percent) / 100.0f * 0.5f + 0.5f;
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Adjusts recoil compensation when aiming at an enemy:\n"
                                 "0% = Normal compensation (unchanged)\n"
                                 "-100% = Reduces recoil compensation when aiming\n"
                                 "100% = Increases recoil compensation when aiming\n\n"
                                 "Use negative values if recoil compensation is too strong when aiming at enemies.");
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
    }

    ImGui::Separator();

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
    }

    // INPUT METHODS - Put in its own collapsible section
    ImGui::Separator();
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

        if (ImGui::Combo("Input Method", &input_method_index, method_items.data(), static_cast<int>(method_items.size())))
        {
            std::string new_input_method = input_methods[input_method_index];

            if (new_input_method != config.input_method)
            {
                config.input_method = new_input_method;
                config.saveConfig();
                input_method_changed.store(true);
            }
        }
        
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Select method for mouse control. Each method has different compatibility with games.");
        }

        // Add some indent for method-specific settings
        ImGui::Indent(10.0f);
        
        if (config.input_method == "ARDUINO")
        {
            if (arduinoSerial)
            {
                if (arduinoSerial->isOpen())
                {
                    ImGui::TextColored(ImVec4(0, 255, 0, 255), "Arduino connected");
                }
                else
                {
                    ImGui::TextColored(ImVec4(255, 0, 0, 255), "Arduino not connected");
                }
            }

            std::vector<std::string> port_list;
            for (int i = 1; i <= 30; ++i)
            {
                port_list.push_back("COM" + std::to_string(i));
            }

            std::vector<const char*> port_items;
            port_items.reserve(port_list.size());
            for (const auto& port : port_list)
            {
                port_items.push_back(port.c_str());
            }

            int port_index = 0;
            for (size_t i = 0; i < port_list.size(); ++i)
            {
                if (port_list[i] == config.arduino_port)
                {
                    port_index = static_cast<int>(i);
                    break;
                }
            }

            if (ImGui::Combo("Port", &port_index, port_items.data(), static_cast<int>(port_items.size())))
            {
                config.arduino_port = port_list[port_index];
                config.saveConfig();
                input_method_changed.store(true);
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Select the COM port your Arduino is connected to. Check Device Manager if unsure.");
            }

            std::vector<int> baud_rate_list = { 9600, 19200, 38400, 57600, 115200 };
            std::vector<std::string> baud_rate_str_list;
            for (const auto& rate : baud_rate_list)
            {
                baud_rate_str_list.push_back(std::to_string(rate));
            }

            std::vector<const char*> baud_rate_items;
            baud_rate_items.reserve(baud_rate_str_list.size());
            for (const auto& rate_str : baud_rate_str_list)
            {
                baud_rate_items.push_back(rate_str.c_str());
            }

            int baud_rate_index = 0;
            for (size_t i = 0; i < baud_rate_list.size(); ++i)
            {
                if (baud_rate_list[i] == config.arduino_baudrate)
                {
                    baud_rate_index = static_cast<int>(i);
                    break;
                }
            }

            if (ImGui::Combo("Baudrate", &baud_rate_index, baud_rate_items.data(), static_cast<int>(baud_rate_items.size())))
            {
                config.arduino_baudrate = baud_rate_list[baud_rate_index];
                config.saveConfig();
                input_method_changed.store(true);
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Baud rate for Arduino communication. Must match the rate in your Arduino sketch.");
            }

            if (ImGui::Checkbox("16-bit Mouse Precision", &config.arduino_16_bit_mouse))
            {
                config.saveConfig();
                input_method_changed.store(true);
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Enable 16-bit precision for mouse movements (recommended for smoother control).");
            }
            
            if (ImGui::Checkbox("Enable Keyboard Keys", &config.arduino_enable_keys))
            {
                config.saveConfig();
                input_method_changed.store(true);
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Enable keyboard key simulation through Arduino. Required if using keybinds for any functions.");
            }
        }
        else if (config.input_method == "GHUB")
        {
            if (ghub_version == "13.1.4")
            {
                ImGui::TextColored(ImVec4(0, 255, 0, 255), "Correct Ghub version installed: %s", ghub_version.c_str());
            }
            else
            {
                if (ghub_version == "")
                {
                    ghub_version = "unknown";
                }

                ImGui::TextColored(ImVec4(255, 165, 0, 255), "Installed Ghub version: %s", ghub_version.c_str());
                ImGui::TextWrapped("The wrong version of Ghub is installed or the path to Ghub is not set by default.\nDefault system path: C:\\Program Files\\LGHUB");
                if (ImGui::Button("View GHub Documentation"))
                {
                    ShellExecute(0, 0, L"https://github.com/SunOner/sunone_aimbot_docs/blob/main/tips/ghub.md", 0, 0, SW_SHOW);
                }
                if (ImGui::IsItemHovered())
                {
                    SetWrappedTooltip("Open documentation for GHUB configuration in your web browser.");
                }
            }
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Use at your own risk, this method may be detected in some games.");
        }
        else if (config.input_method == "WIN32")
        {
            ImGui::TextWrapped("This is a standard mouse input method. It may not work in most games.");
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Use at your own risk, this method may be detected in some games.");
        }
        
        ImGui::Unindent(10.0f);
    }
}