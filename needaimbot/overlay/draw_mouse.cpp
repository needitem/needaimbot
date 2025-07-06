#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>

#include "AppContext.h"
#include "imgui/imgui.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "ui_helpers.h" 

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
    auto& ctx = AppContext::getInstance();
    
    ImGui::Columns(2, "MouseSettingsColumns", false); 

    
    ImGui::Text("Controller Settings");
    ImGui::Spacing();
    
    if (ImGui::Checkbox("Use Predictive Controller", &ctx.config.use_predictive_controller)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered())
    {
        SetWrappedTooltip("Enable advanced Kalman filter + PID controller for better target tracking and prediction. Recommended for moving targets.");
    }
    ImGui::Spacing(); 
    
    
    if (ImGui::CollapsingHeader("PID Controller Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("PID parameters control how the aimbot tracks targets.");
        ImGui::Spacing();
        
        // X-axis PID
        if (ImGui::TreeNode("X-Axis (Horizontal)"))
        {
            float kp_x_display = static_cast<float>(ctx.config.kp_x);
            if (ImGui::InputFloat("Kp X", &kp_x_display, 0.01f, 0.1f, "%.3f"))
            {
                ctx.config.kp_x = static_cast<double>(kp_x_display);
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Proportional gain for X-axis. Higher values = faster response but may cause oscillation.");
            }
            
            float ki_x_display = static_cast<float>(ctx.config.ki_x);
            if (ImGui::InputFloat("Ki X", &ki_x_display, 0.01f, 0.1f, "%.3f"))
            {
                ctx.config.ki_x = static_cast<double>(ki_x_display);
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Integral gain for X-axis. Helps eliminate steady-state error.");
            }
            
            float kd_x_display = static_cast<float>(ctx.config.kd_x);
            if (ImGui::InputFloat("Kd X", &kd_x_display, 0.01f, 0.1f, "%.3f"))
            {
                ctx.config.kd_x = static_cast<double>(kd_x_display);
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Derivative gain for X-axis. Reduces overshoot and oscillation.");
            }
            
            ImGui::TreePop();
        }
        
        ImGui::Spacing();
        
        // Y-axis PID
        if (ImGui::TreeNode("Y-Axis (Vertical)"))
        {
            float kp_y_display = static_cast<float>(ctx.config.kp_y);
            if (ImGui::InputFloat("Kp Y", &kp_y_display, 0.01f, 0.1f, "%.3f"))
            {
                ctx.config.kp_y = static_cast<double>(kp_y_display);
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Proportional gain for Y-axis. Higher values = faster response but may cause oscillation.");
            }
            
            float ki_y_display = static_cast<float>(ctx.config.ki_y);
            if (ImGui::InputFloat("Ki Y", &ki_y_display, 0.01f, 0.1f, "%.3f"))
            {
                ctx.config.ki_y = static_cast<double>(ki_y_display);
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Integral gain for Y-axis. Helps eliminate steady-state error.");
            }
            
            float kd_y_display = static_cast<float>(ctx.config.kd_y);
            if (ImGui::InputFloat("Kd Y", &kd_y_display, 0.01f, 0.1f, "%.3f"))
            {
                ctx.config.kd_y = static_cast<double>(kd_y_display);
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Derivative gain for Y-axis. Reduces overshoot and oscillation.");
            }
            
            ImGui::TreePop();
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Advanced Settings
        if (ImGui::TreeNode("Advanced Settings"))
        {
            if (UIHelpers::BeautifulToggle("Enable Adaptive PID", &ctx.config.enable_adaptive_pid, 
                                           "Uses distance-based PID adjustment for better stability at different ranges.")) {
                ctx.config.saveConfig();
            }
            
            if (UIHelpers::BeautifulSlider("Derivative Smoothing", &ctx.config.pid_derivative_smoothing, 0.0f, 0.8f, "%.3f")) {
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Smooths derivative calculation to reduce noise. Higher values = more smoothing but slower response to rapid changes.");
            }
            
            if (UIHelpers::BeautifulSlider("Movement Smoothing", &ctx.config.movement_smoothing, 0.0f, 0.6f, "%.3f")) {
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered()) {
                SetWrappedTooltip("Smooths final mouse movement. Higher values = less jitter but may reduce responsiveness.");
            }
            
            ImGui::TreePop();
        }
        
        ImGui::Spacing();
        
        // Info section
        if (ImGui::TreeNode("Tuning Tips"))
        {
            ImGui::BeginChild("PIDInfo", ImVec2(0, 80), true, ImGuiWindowFlags_AlwaysUseWindowPadding);
            {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Tuning Tips:");
                ImGui::Text("• Start with Kp, then add Kd to reduce oscillation");
                ImGui::Text("• Use Ki sparingly - too much causes overshoot");
                ImGui::Text("• Enable Adaptive PID for better long-range stability");
                ImGui::Text("• Increase smoothing if you experience jitter");
            }
            ImGui::EndChild();
            
            ImGui::TreePop();
        }
        
        ImGui::Spacing();
    }

    
    if (ctx.config.use_predictive_controller && ImGui::CollapsingHeader("Predictive Controller Settings"))
    {
        float prediction_time_display = ctx.config.prediction_time_ms;
        if (ImGui::InputFloat("Prediction Time (ms)", &prediction_time_display, 1.0f, 10.0f, "%.1f"))
        {
            ctx.config.prediction_time_ms = prediction_time_display;
            ctx.config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("How far ahead to predict target movement in milliseconds. Higher values work better for fast-moving targets.");
        }

        float process_noise_display = ctx.config.kalman_process_noise;
        if (ImGui::InputFloat("Process Noise", &process_noise_display, 0.1f, 1.0f, "%.1f"))
        {
            ctx.config.kalman_process_noise = process_noise_display;
            ctx.config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Kalman filter process noise. Higher values make the tracker adapt faster to sudden target movements.");
        }

        float measurement_noise_display = ctx.config.kalman_measurement_noise;
        if (ImGui::InputFloat("Measurement Noise", &measurement_noise_display, 0.1f, 1.0f, "%.1f"))
        {
            ctx.config.kalman_measurement_noise = measurement_noise_display;
            ctx.config.saveConfig();
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
            if (input_methods[i] == ctx.config.input_method)
            {
                input_method_index = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Input Method", &input_method_index, method_items.data(), method_items.size()))
        {
            ctx.config.input_method = input_methods[input_method_index];
            ctx.config.saveConfig();
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

        
        if (ctx.config.input_method == "GHUB" || !ghub_version.empty())
        {
            ImGui::Text("GHUB Version: %s", ghub_version.c_str());
            if (ghub_version.empty())
            {
                 ImGui::SameLine();
                 ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "(Not Found/Error)");
            }
        }

        
        if (ctx.config.input_method == "ARDUINO")
        {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Arduino Settings");

            
            char port_buffer[64]; 
            strncpy_s(port_buffer, sizeof(port_buffer), ctx.config.arduino_port.c_str(), _TRUNCATE);

            ImGui::Text("COM Port:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100); 
            if (ImGui::InputText("##ArduinoPort", port_buffer, sizeof(port_buffer)))
            {
                ctx.config.arduino_port = port_buffer;
                ctx.config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Enter the COM port your Arduino is connected to (e.g., COM3, /dev/ttyACM0).");
            }

            
            ImGui::Text("Baud Rate:");
            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::InputInt("##ArduinoBaud", &ctx.config.arduino_baudrate, 0)) 
            {
                
                ctx.config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Serial communication speed (e.g., 9600, 115200). Must match Arduino firmware.");
            }

            
            if (ImGui::Checkbox("Use 16-bit Mouse Movement", &ctx.config.arduino_16_bit_mouse))
            {
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Send mouse movement data as 16-bit values (requires compatible firmware). Otherwise, uses 8-bit.");
            }

            ImGui::Unindent(10.0f);
        }

        
        if (ctx.config.input_method == "KMBOX") 
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

        for (size_t i = 0; i < ctx.config.button_auto_shoot.size(); )
        {
            std::string& current_key_name = ctx.config.button_auto_shoot[i];

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
                ctx.config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID();

            ImGui::SameLine();
            std::string remove_button_label = "Remove##auto_shoot_hotkey_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (ctx.config.button_auto_shoot.size() <= 1)
                {
                    ctx.config.button_auto_shoot[0] = std::string("None");
                    ctx.config.saveConfig();
                    
                }
                else
                {
                    ctx.config.button_auto_shoot.erase(ctx.config.button_auto_shoot.begin() + i);
                    ctx.config.saveConfig();
                    
                }
                continue; 
            }

            ++i;
        }

        if (ImGui::Button("Add Auto Shoot Hotkey##auto_shoot_hotkey"))
        {
            ctx.config.button_auto_shoot.push_back("None");
            ctx.config.saveConfig();
        }
        

        
        ImGui::SeparatorText("Scope Settings");
        ImGui::PushItemWidth(150); 
        if (ImGui::SliderFloat("Triggerbot Area Size", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f")) { 
             ctx.config.saveConfig();
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area, Larger value = smaller area.\n(1.0 = default area)");
        }
        ImGui::Spacing(); 
        

        ImGui::SeparatorText("Targeting Buttons"); 

        for (size_t i = 0; i < ctx.config.button_targeting.size(); )
        {
            std::string& current_key_name = ctx.config.button_targeting[i];

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
                ctx.config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID(); 

            ImGui::SameLine();
            std::string remove_button_label = "Remove##targeting_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (ctx.config.button_targeting.size() <= 1)
                {
                    ctx.config.button_targeting[0] = std::string("None");
                    ctx.config.saveConfig();
                    
                }
                else
                {
                    ctx.config.button_targeting.erase(ctx.config.button_targeting.begin() + i);
                    ctx.config.saveConfig();
                    
                }
                continue; 
            }

            ++i;
        }

        if (ImGui::Button("Add Targeting Button##targeting"))
        {
            ctx.config.button_targeting.push_back("None");
            ctx.config.saveConfig();
        }

        ImGui::SeparatorText("Disable Upward Aim"); 

        for (size_t i = 0; i < ctx.config.button_disable_upward_aim.size(); )
        {
            std::string& current_key_name = ctx.config.button_disable_upward_aim[i];

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
                ctx.config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID(); 

            ImGui::SameLine();
            std::string remove_button_label = "Remove##disable_upward_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (ctx.config.button_disable_upward_aim.size() <= 1)
                {
                    ctx.config.button_disable_upward_aim[0] = std::string("None");
                    ctx.config.saveConfig();
                    
                }
                else
                {
                    ctx.config.button_disable_upward_aim.erase(ctx.config.button_disable_upward_aim.begin() + i);
                    ctx.config.saveConfig();
                    
                }
                continue; 
            }

            ++i;
        }

        if (ImGui::Button("Add Disable Upward Button##disable_upward_aim"))
        {
            ctx.config.button_disable_upward_aim.push_back("None");
            ctx.config.saveConfig();
        }
        
        ImGui::Spacing(); 
    }

    

    
    
    
    
    
    
    
    

    ImGui::Columns(1); 

    
    
}
