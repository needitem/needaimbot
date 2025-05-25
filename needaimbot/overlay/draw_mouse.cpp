#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h" // Include necessary header for key_names etc.

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

    // --- Input Method Settings (Moved to Column 1) ---
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

            ImGui::Unindent(10.0f);
        }

        // Kmbox Settings (assuming kmboxNet library is integrated elsewhere)
        if (config.input_method == "KMBOX") 
        {
            ImGui::Indent(10.0f);
            ImGui::SeparatorText("Kmbox Settings");

            ImGui::TextWrapped("KmboxNet selected. Ensure B-box hardware is connected and kmNet library is initialized.");

            ImGui::Unindent(10.0f);
        }

        ImGui::Spacing(); // Add spacing at the end
    }
    // --- End Input Method Settings ---

    // --- WindMouse Settings (New Section) ---
    ImGui::Spacing(); // Add spacing before WindMouse settings header
    if (ImGui::CollapsingHeader("WindMouse Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Checkbox("Enable WindMouse", &config.wind_mouse_enabled))
        {
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Enables more human-like mouse movements using the WindMouse algorithm. Adjust G, W, M, D parameters below.");
        }

        ImGui::Indent(10.0f);
        ImGui::PushItemWidth(100);

        if (ImGui::InputFloat("Gravity (G)", &config.wind_G, 0.1f, 1.0f, "%.1f"))
        {
            config.wind_G = std::max(0.0f, config.wind_G); // Ensure non-negative
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Strength of the force pulling the cursor towards the target. Higher values mean more direct movement.");
        }

        if (ImGui::InputFloat("Wind (W)", &config.wind_W, 0.1f, 1.0f, "%.1f"))
        {
            config.wind_W = std::max(0.0f, config.wind_W); // Ensure non-negative
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Magnitude of random mouse movements (shake/tremor). Higher values make movement more erratic.");
        }

        if (ImGui::InputFloat("Mean (M)", &config.wind_M, 0.1f, 1.0f, "%.1f"))
        {
            config.wind_M = std::max(1.0f, config.wind_M); // Ensure M is at least 1 to avoid issues with division by zero or very small numbers in the algorithm
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Mean distance for random mouse movements. Influences the scale of erratic jumps.");
        }

        if (ImGui::InputFloat("Deviation (D)", &config.wind_D, 0.1f, 1.0f, "%.1f"))
        {
            config.wind_D = std::max(1.0f, config.wind_D); // Ensure D is at least 1 for similar reasons to M
            config.saveConfig();
        }
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Standard deviation for random mouse movements. Controls the variability of the erratic jumps.");
        }

        ImGui::PopItemWidth();
        ImGui::Unindent(10.0f);
        ImGui::Spacing(); // Add spacing at the end of WindMouse settings
    }
    // --- End WindMouse Settings ---

    // No Separator needed here
    // ImGui::Text("Kalman Filter Settings"); // Text might be redundant if header is descriptive
    ImGui::Spacing(); // Add spacing before the Kalman header

    // Group Kalman Filter Settings
    if (ImGui::CollapsingHeader("Prediction & Scope"))
    {
        ImGui::Indent(10.0f);

        // --- Prediction Algorithm Selection ---
        ImGui::SeparatorText("Prediction Algorithm");

        // TODO: Add std::string prediction_algorithm = "None"; to your config struct.
        // Non-AI prediction methods + Kalman
        const char* prediction_algorithms[] = {
            "None",
            "Velocity Based",
            "Linear Regression",
            "Exponential Smoothing",
            "Kalman Filter"
        };
        int current_algorithm_index = 0;
        // Map string config to index
        std::string current_algo = config.prediction_algorithm;
        if (current_algo == "Velocity Based")          { current_algorithm_index = 1; }
        else if (current_algo == "Linear Regression")    { current_algorithm_index = 2; }
        else if (current_algo == "Exponential Smoothing"){ current_algorithm_index = 3; }
        else if (current_algo == "Kalman Filter")        { current_algorithm_index = 4; }
        else { current_algorithm_index = 0; } // Default to None

        ImGui::PushItemWidth(150); // Adjust width as needed
        if (ImGui::Combo("Algorithm", &current_algorithm_index, prediction_algorithms, IM_ARRAYSIZE(prediction_algorithms)))
        {
            config.prediction_algorithm = prediction_algorithms[current_algorithm_index];
            config.saveConfig();
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Select the prediction method (non-AI options):\n"
                             "None: No prediction.\n"
                             "Velocity Based: Simple prediction based on current velocity.\n"
                             "Linear Regression: Predicts based on linear fit of past movement.\n"
                             "Exponential Smoothing: Weighted average prediction, favors recent data.\n"
                             "Kalman Filter: Statistical filter for noisy data (requires tuning).");
        }
        ImGui::Spacing();

        // --- Algorithm Specific Settings ---
        if (config.prediction_algorithm == "Velocity Based")
        {
            // TODO: Add float velocity_prediction_ms = 16.0f; to your config struct.
            ImGui::SeparatorText("Velocity Prediction Settings");
            ImGui::PushItemWidth(100);
            if (ImGui::InputFloat("Prediction Time (ms)", &config.velocity_prediction_ms, 1.0f, 5.0f, "%.1f"))
            {
                config.velocity_prediction_ms = std::max(0.0f, config.velocity_prediction_ms); // Ensure non-negative
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("How far ahead in milliseconds to predict the target's position based on its current velocity.");
            }
        }
        else if (config.prediction_algorithm == "Linear Regression")
        {
            // TODO: Add settings for Linear Regression if needed (e.g., number of past points)
            ImGui::SeparatorText("Linear Regression Settings");
            ImGui::PushItemWidth(100);
            ImGui::Text("Past Points (N):"); ImGui::SameLine();
            if (ImGui::InputInt("##LRPastPoints", &config.lr_past_points, 1, 5))
            {
                config.lr_past_points = std::max(2, config.lr_past_points); // Need at least 2 points for a line
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Number of past target positions to use for calculating the regression line. More points = smoother but less responsive.");
            }
        }
        else if (config.prediction_algorithm == "Exponential Smoothing")
        {
            // TODO: Add settings for Exponential Smoothing if needed (e.g., alpha parameter)
            ImGui::SeparatorText("Exponential Smoothing Settings");
            ImGui::PushItemWidth(100);
            ImGui::Text("Factor (Alpha):"); ImGui::SameLine();
            // Using SliderFloat for easier adjustment between 0 and 1
            if (ImGui::SliderFloat("##ESAlpha", &config.es_alpha, 0.01f, 1.0f, "%.2f"))
            {
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            if (ImGui::IsItemHovered())
            {
                SetWrappedTooltip("Smoothing factor (alpha) between 0.01 and 1.0. Controls weighting of recent vs. past data.\nCloser to 1.0 = more weight on recent data (more responsive, less smooth).\nCloser to 0.01 = more weight on past data (smoother, less responsive).");
            }
        }
        else if (config.prediction_algorithm == "Kalman Filter")
        {
            // TODO: Add float kalman_q = 0.1f;, kalman_r = 0.1f;, kalman_p = 0.1f; to your config struct.
            ImGui::SeparatorText("Kalman Filter Settings");

            ImGui::PushItemWidth(100);
            ImGui::Text("Process Noise (Q):"); ImGui::SameLine();
            float temp_q = static_cast<float>(config.kalman_q);
            if (ImGui::InputFloat("##KalmanQ", &temp_q, 0.001f, 0.01f, "%.3f")) {
                config.kalman_q = static_cast<double>(temp_q);
                if (config.kalman_q < 0) config.kalman_q = 0; // Ensure non-negative
                config.saveConfig();
            }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Represents the uncertainty in the target's movement model. Higher values trust measurements less.");

            ImGui::Text("Measurement Noise (R):"); ImGui::SameLine();
            float temp_r = static_cast<float>(config.kalman_r);
            if (ImGui::InputFloat("##KalmanR", &temp_r, 0.001f, 0.01f, "%.3f")) {
                config.kalman_r = static_cast<double>(temp_r);
                if (config.kalman_r < 0) config.kalman_r = 0; // Ensure non-negative
                config.saveConfig();
            }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Represents the uncertainty in the measurements (detection). Higher values trust the model prediction more.");

            ImGui::Text("Estimate Error (P):"); ImGui::SameLine();
            float temp_p = static_cast<float>(config.kalman_p);
            if (ImGui::InputFloat("##KalmanP", &temp_p, 0.001f, 0.01f, "%.3f")) {
                 config.kalman_p = static_cast<double>(temp_p);
                if (config.kalman_p < 0) config.kalman_p = 0; // Ensure non-negative
                config.saveConfig();
            }
            if (ImGui::IsItemHovered()) SetWrappedTooltip("Initial estimate of the state covariance. Represents the initial uncertainty about the target's state.");
            ImGui::PopItemWidth();
        }

        ImGui::Unindent(10.0f); // Unindent the whole "Prediction & Scope" section content
        ImGui::Spacing(); // Add spacing at the end of the group
    }
    // --- Column 1 End ---

    ImGui::NextColumn(); // Move to the second column

    // --- Column 2 Start ---
    // No Separator needed here
    ImGui::Spacing(); // Add spacing before the Recoil header

    // No Separator needed here
    ImGui::Spacing(); // Add spacing before Aiming settings header

    // Create a collapsible section for Aiming Settings
    if (ImGui::CollapsingHeader("Aiming Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // --- Auto Shoot Hotkey Settings (Keep this section) --- 
        // TODO: Add `std::vector<std::string> button_auto_shoot = {"None"};` to your config struct.
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
                current_index = 0; // Default to "None"
            }

            std::string combo_label = "Auto Shoot Hotkey " + std::to_string(i);

            ImGui::PushID(combo_label.c_str()); // Unique ID for Combo
            ImGui::PushItemWidth(150); // Adjust width
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
                    // Do not increment i
                }
                else
                {
                    config.button_auto_shoot.erase(config.button_auto_shoot.begin() + i);
                    config.saveConfig();
                    // Do not increment i
                }
                continue; // Skip incrementing i
            }

            ++i;
        }

        if (ImGui::Button("Add Auto Shoot Hotkey##auto_shoot_hotkey"))
        {
            config.button_auto_shoot.push_back("None");
            config.saveConfig();
        }
        // --- End Auto Shoot Hotkey Settings ---

        // --- Scope Settings (Moved Here) ---
        ImGui::SeparatorText("Scope Settings");
        ImGui::PushItemWidth(150); // Make slider wider
        if (ImGui::SliderFloat("Triggerbot Area Size", &config.bScope_multiplier, 0.1f, 2.0f, "%.2f")) { 
             config.saveConfig();
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area, Larger value = smaller area.\n(1.0 = default area)");
        }
        ImGui::Spacing(); // Add spacing after scope setting
        // --- End Scope Settings ---

        ImGui::SeparatorText("Targeting Buttons"); // Separator for clarity

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
                current_index = 0; // Default to "None" if not found
            }

            std::string combo_label = "Targeting Button " + std::to_string(i);

            ImGui::PushID(combo_label.c_str()); // Push unique ID for Combo
            ImGui::PushItemWidth(150); // Adjust width as needed
            if (ImGui::Combo("", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                current_key_name = key_names[current_index];
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID(); // Pop ID

            ImGui::SameLine();
            std::string remove_button_label = "Remove##targeting_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (config.button_targeting.size() <= 1)
                {
                    config.button_targeting[0] = std::string("None");
                    config.saveConfig();
                    // Do not increment i, loop continues with the modified vector
                }
                else
                {
                    config.button_targeting.erase(config.button_targeting.begin() + i);
                    config.saveConfig();
                    // Do not increment i, loop continues with the modified vector
                }
                continue; // Skip incrementing i
            }

            ++i;
        }

        if (ImGui::Button("Add Targeting Button##targeting"))
        {
            config.button_targeting.push_back("None");
            config.saveConfig();
        }

        ImGui::SeparatorText("Disable Upward Aim"); // Separator for clarity

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

            ImGui::PushID(combo_label.c_str()); // Push unique ID for Combo
            ImGui::PushItemWidth(150); // Adjust width as needed
            if (ImGui::Combo("", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                current_key_name = key_names[current_index];
                config.saveConfig();
            }
            ImGui::PopItemWidth();
            ImGui::PopID(); // Pop ID

            ImGui::SameLine();
            std::string remove_button_label = "Remove##disable_upward_" + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str()))
            {
                if (config.button_disable_upward_aim.size() <= 1)
                {
                    config.button_disable_upward_aim[0] = std::string("None");
                    config.saveConfig();
                    // Do not increment i
                }
                else
                {
                    config.button_disable_upward_aim.erase(config.button_disable_upward_aim.begin() + i);
                    config.saveConfig();
                    // Do not increment i
                }
                continue; // Skip incrementing i
            }

            ++i;
        }

        if (ImGui::Button("Add Disable Upward Button##disable_upward_aim"))
        {
            config.button_disable_upward_aim.push_back("None");
            config.saveConfig();
        }
        
        ImGui::Spacing(); // Add spacing after Aiming settings
    }

    // --- Silent Aim Hotkey Settings (New Section) ---
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
            current_index = 0; // Default to "None"
        }

        std::string combo_label = "Silent Aim Hotkey " + std::to_string(i);

        ImGui::PushID(combo_label.c_str()); // Unique ID for Combo
        ImGui::PushItemWidth(150); // Adjust width
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
                // Do not increment i
            }
            else
            {
                config.button_silent_aim.erase(config.button_silent_aim.begin() + i);
                config.saveConfig();
                // Do not increment i
            }
            continue; // Skip incrementing i
        }

        ++i;
    }

    if (ImGui::Button("Add Silent Aim Hotkey##silent_aim_hotkey"))
    {
        config.button_silent_aim.push_back("None");
        config.saveConfig();
    }
    // --- End Silent Aim Hotkey Settings ---

    // INPUT METHODS - REMOVED FROM HERE
    // // No Separator needed here
    // ImGui::Spacing(); // Add spacing before Input Method settings header
    // if (ImGui::CollapsingHeader("Input Method Settings", ImGuiTreeNodeFlags_DefaultOpen))
    // {
    //    // ... (Content of Input Method Settings was here) ...
    // }
    // --- Column 2 End ---

    ImGui::Columns(1); // End column layout and return to 1 column

    // Optionally add a separator at the very end if needed before other global UI elements
    // ImGui::Separator();
}