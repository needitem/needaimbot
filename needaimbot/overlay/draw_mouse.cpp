#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h" 

std::string ghub_version = get_ghub_version();

static void SetWrappedTooltip(const char* text) 
{
    ImGui::BeginTooltip();
    
    ImVec2 window_size = ImGui::GetIO().DisplaySize;
    
    float max_width = window_size.x * 0.5f;
    
    ImGui::PushTextWrapPos(max_width);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    
    ImGui::EndTooltip();
}

void draw_mouse()
{
    ImGui::Columns(2, "MouseSettingsColumns", false); 

    
    ImGui::Text("Controller Settings");
    ImGui::Spacing();
    
    if (ImGui::Checkbox("Use Predictive Controller", &config.use_predictive_controller)) { config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        SetWrappedTooltip("Enable advanced Kalman filter + PID controller for better target tracking and prediction. Recommended for moving targets.");
    }
    ImGui::Spacing(); 
    
    
    if (ImGui::CollapsingHeader("Horizontal (X-axis) PID", ImGuiTreeNodeFlags_DefaultOpen))
    {
        
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
        ImGui::Spacing(); 
    }
    
    
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
        ImGui::Spacing(); 
    }

    
    if (config.use_predictive_controller && ImGui::CollapsingHeader("Predictive Controller Settings"))
    {
        float prediction_time_display = config.prediction_time_ms;
        if (ImGui::InputFloat("Prediction Time (ms)", &prediction_time_display, 1.0f, 10.0f, "%.1f"))
        {
            config.prediction_time_ms = prediction_time_display;
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("How far ahead to predict target movement in milliseconds. Higher values work better for fast-moving targets.");
        }

        float process_noise_display = config.kalman_process_noise;
        if (ImGui::InputFloat("Process Noise", &process_noise_display, 0.1f, 1.0f, "%.1f"))
        {
            config.kalman_process_noise = process_noise_display;
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Kalman filter process noise. Higher values make the tracker adapt faster to sudden target movements.");
        }

        float measurement_noise_display = config.kalman_measurement_noise;
        if (ImGui::InputFloat("Measurement Noise", &measurement_noise_display, 0.1f, 1.0f, "%.1f"))
        {
            config.kalman_measurement_noise = measurement_noise_display;
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Kalman filter measurement noise. Higher values smooth out detection jitter but reduce responsiveness.");
        }
        ImGui::Spacing();
    }

    
    ImGui::Spacing(); 
    if (ImGui::CollapsingHeader("Input Method Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        
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

        
        if (config.input_method == "GHUB" || !ghub_version.empty())
        {
            ImGui::Text("GHUB Version: %s", ghub_version.c_str());
            if (ghub_version.empty())
            {
                 ImGui::SameLine();
                 ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "(Not Found/Error)");
            }
        }

        
        if (config.input_method == "ARDUINO")
        {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Arduino Settings");

            
            char port_buffer[64]; 
            strncpy_s(port_buffer, sizeof(port_buffer), config.arduino_port.c_str(), _TRUNCATE);

            ImGui::Text("COM Port:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100); 
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

            
            ImGui::Text("Baud Rate:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::InputInt("##ArduinoBaud", &config.arduino_baudrate, 0)) 
            {
                
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Serial communication speed (e.g., 9600, 115200). Must match Arduino firmware.");
            }

            
            if (ImGui::Checkbox("Use 16-bit Mouse Movement", &config.arduino_16_bit_mouse))
            {
                config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Send mouse movement data as 16-bit values (requires compatible firmware). Otherwise, uses 8-bit.");
            }

            ImGui::Unindent(10.0f);
        }

        
        if (config.input_method == "KMBOX") 
        {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Kmbox Settings");

            ImGui::TextWrapped("KmboxNet selected. Ensure B-box hardware is connected and kmNet library is initialized.");

            ImGui::Unindent(10.0f);
        }

        ImGui::Spacing(); 
    }
    

    
    
    ImGui::Spacing(); 

    
    ImGui::NextColumn(); 

    
    
    ImGui::Spacing(); 

    
    ImGui::Spacing(); 

    
    if (ImGui::CollapsingHeader("Aiming Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        
        
        ImGui::SeparatorText("Auto Shoot Hotkeys");

        for (size_t i = 0; i < config.button_auto_shoot.size(); )
        {
            std::string& current_key_name = config.button_auto_shoot[i];

            int current_index = -1;
            for (size_t k = 0; k < key_names.size(); ++k)
            {
                if (key_names[k] == current_key_name)
                {
                    current_index = static_cast<int>(k);
                    break;
                }
            }

            if (current_index == -1)
            {
                current_index = 0; 
            }

            std::string combo_label = "Auto Shoot Hotkey " + std::to_string(i);

            ImGui::PushID(combo_label.c_str()); 
            ImGui::PushItemWidth(150); 
            if (ImGui::Combo("", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                current_key_name = key_names[current_index];
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID();

            ImGui::SameLine();
            std::string remove_button_label = "Remove##auto_shoot_hotkey_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (config.button_auto_shoot.size() <= 1)
                {
                    config.button_auto_shoot[0] = std::string("None");
                    config.saveConfig();
                    
                }
                else
                {
                    config.button_auto_shoot.erase(config.button_auto_shoot.begin() + i);
                    config.saveConfig();
                    
                }
                continue; 
            }

            ++i;
        }

        if (ImGui::Button("Add Auto Shoot Hotkey##auto_shoot_hotkey"))
        {
            config.button_auto_shoot.push_back("None");
            config.saveConfig();
        }
        

        
        ImGui::SeparatorText("Scope Settings");
        ImGui::PushItemWidth(150); 
        if (ImGui::SliderFloat("Triggerbot Area Size", &config.bScope_multiplier, 0.1f, 2.0f, "%.2f")) { 
             config.saveConfig();
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area, Larger value = smaller area.\n(1.0 = default area)");
        }
        ImGui::Spacing(); 
        

        ImGui::SeparatorText("Targeting Buttons"); 

        for (size_t i = 0; i < config.button_targeting.size(); )
        {
            std::string& current_key_name = config.button_targeting[i];

            int current_index = -1;
            for (size_t k = 0; k < key_names.size(); ++k)
            {
                if (key_names[k] == current_key_name)
                {
                    current_index = static_cast<int>(k);
                    break;
                }
            }

            if (current_index == -1)
            {
                current_index = 0; 
            }

            std::string combo_label = "Targeting Button " + std::to_string(i);

            ImGui::PushID(combo_label.c_str()); 
            ImGui::PushItemWidth(150); 
            if (ImGui::Combo("", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                current_key_name = key_names[current_index];
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID(); 

            ImGui::SameLine();
            std::string remove_button_label = "Remove##targeting_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (config.button_targeting.size() <= 1)
                {
                    config.button_targeting[0] = std::string("None");
                    config.saveConfig();
                    
                }
                else
                {
                    config.button_targeting.erase(config.button_targeting.begin() + i);
                    config.saveConfig();
                    
                }
                continue; 
            }

            ++i;
        }

        if (ImGui::Button("Add Targeting Button##targeting"))
        {
            config.button_targeting.push_back("None");
            config.saveConfig();
        }

        ImGui::SeparatorText("Disable Upward Aim"); 

        for (size_t i = 0; i < config.button_disable_upward_aim.size(); )
        {
            std::string& current_key_name = config.button_disable_upward_aim[i];

            int current_index = -1;
            for (size_t k = 0; k < key_names.size(); ++k)
            {
                if (key_names[k] == current_key_name)
                {
                    current_index = static_cast<int>(k);
                    break;
                }
            }

            if (current_index == -1)
            {
                current_index = 0;
            }

            std::string combo_label = "Disable Upward Button " + std::to_string(i);

            ImGui::PushID(combo_label.c_str()); 
            ImGui::PushItemWidth(150); 
            if (ImGui::Combo("", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                current_key_name = key_names[current_index];
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID(); 

            ImGui::SameLine();
            std::string remove_button_label = "Remove##disable_upward_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (config.button_disable_upward_aim.size() <= 1)
                {
                    config.button_disable_upward_aim[0] = std::string("None");
                    config.saveConfig();
                    
                }
                else
                {
                    config.button_disable_upward_aim.erase(config.button_disable_upward_aim.begin() + i);
                    config.saveConfig();
                    
                }
                continue; 
            }

            ++i;
        }

        if (ImGui::Button("Add Disable Upward Button##disable_upward_aim"))
        {
            config.button_disable_upward_aim.push_back("None");
            config.saveConfig();
        }
        
        ImGui::Spacing(); 
    }

    
    ImGui::SeparatorText("Silent Aim Hotkeys");

    for (size_t i = 0; i < config.button_silent_aim.size(); )
    {
        std::string& current_key_name = config.button_silent_aim[i];

        int current_index = -1;
        for (size_t k = 0; k < key_names.size(); ++k)
        {
            if (key_names[k] == current_key_name)
            {
                current_index = static_cast<int>(k);
                break;
            }
        }

        if (current_index == -1)
        {
            current_index = 0; 
        }

        std::string combo_label = "Silent Aim Hotkey " + std::to_string(i);

        ImGui::PushID(combo_label.c_str()); 
        ImGui::PushItemWidth(150); 
        if (ImGui::Combo("", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }
        ImGui::PopItemWidth();
        ImGui::PopID();

        ImGui::SameLine();
        std::string remove_button_label = "Remove##silent_aim_hotkey_" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_silent_aim.size() <= 1)
            {
                config.button_silent_aim[0] = std::string("None");
                config.saveConfig();
                
            }
            else
            {
                config.button_silent_aim.erase(config.button_silent_aim.begin() + i);
                config.saveConfig();
                
            }
            continue; 
        }

        ++i;
    }

    if (ImGui::Button("Add Silent Aim Hotkey##silent_aim_hotkey"))
    {
        config.button_silent_aim.push_back("None");
        config.saveConfig();
    }
    

    
    
    
    
    
    
    
    

    ImGui::Columns(1); 

    
    
}
