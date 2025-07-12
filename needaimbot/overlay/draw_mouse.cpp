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

static void draw_pid_controls()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("PID Controller Settings");
    
    UIHelpers::BeautifulText("PID parameters control how the aimbot tracks targets.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    // Combined PID settings in a more compact layout
    ImGui::Columns(2, "pid_columns", false);
    
    // X-axis
    UIHelpers::SettingsSubHeader("X-Axis (Horizontal)");
    
    float kp_x = static_cast<float>(ctx.config.kp_x);
    if (UIHelpers::EnhancedSliderFloat("Kp X", &kp_x, 0.0f, 2.0f, "%.3f", "Proportional gain for X-axis. Higher values = faster response but may cause oscillation.")) {
        ctx.config.kp_x = static_cast<double>(kp_x);
        ctx.config.saveConfig();
    }
    
    float ki_x = static_cast<float>(ctx.config.ki_x);
    if (UIHelpers::EnhancedSliderFloat("Ki X", &ki_x, 0.0f, 1.0f, "%.3f", "Integral gain for X-axis. Helps eliminate steady-state error.")) {
        ctx.config.ki_x = static_cast<double>(ki_x);
        ctx.config.saveConfig();
    }
    
    float kd_x = static_cast<float>(ctx.config.kd_x);
    if (UIHelpers::EnhancedSliderFloat("Kd X", &kd_x, 0.0f, 1.0f, "%.3f", "Derivative gain for X-axis. Reduces overshoot and oscillation.")) {
        ctx.config.kd_x = static_cast<double>(kd_x);
        ctx.config.saveConfig();
    }
    
    ImGui::NextColumn();
    
    // Y-axis
    UIHelpers::SettingsSubHeader("Y-Axis (Vertical)");
    
    float kp_y = static_cast<float>(ctx.config.kp_y);
    if (UIHelpers::EnhancedSliderFloat("Kp Y", &kp_y, 0.0f, 2.0f, "%.3f", "Proportional gain for Y-axis. Higher values = faster response but may cause oscillation.")) {
        ctx.config.kp_y = static_cast<double>(kp_y);
        ctx.config.saveConfig();
    }
    
    float ki_y = static_cast<float>(ctx.config.ki_y);
    if (UIHelpers::EnhancedSliderFloat("Ki Y", &ki_y, 0.0f, 1.0f, "%.3f", "Integral gain for Y-axis. Helps eliminate steady-state error.")) {
        ctx.config.ki_y = static_cast<double>(ki_y);
        ctx.config.saveConfig();
    }
    
    float kd_y = static_cast<float>(ctx.config.kd_y);
    if (UIHelpers::EnhancedSliderFloat("Kd Y", &kd_y, 0.0f, 1.0f, "%.3f", "Derivative gain for Y-axis. Reduces overshoot and oscillation.")) {
        ctx.config.kd_y = static_cast<double>(kd_y);
        ctx.config.saveConfig();
    }
    
    ImGui::Columns(1);
    
    UIHelpers::EndCard();
}

static void draw_advanced_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Advanced Controller Settings");
    
    if (UIHelpers::EnhancedCheckbox("Enable Adaptive PID", &ctx.config.enable_adaptive_pid, 
                                   "Uses distance-based PID adjustment for better stability at different ranges.")) {
        ctx.config.saveConfig();
    }
    
    if (UIHelpers::EnhancedSliderFloat("Derivative Smoothing", &ctx.config.pid_derivative_smoothing, 0.0f, 1.0f, "%.3f",
                                      "Smooths derivative calculation to reduce noise. Higher values = more smoothing but slower response to rapid changes.")) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Sub-pixel Movement");
    
    if (UIHelpers::EnhancedCheckbox("Enable Sub-pixel Dithering", &ctx.config.enable_subpixel_dithering, 
                                   "Adds small random variations to improve movement smoothness and reduce stepping artifacts.")) {
        ctx.config.saveConfig();
    }
    
    if (ctx.config.enable_subpixel_dithering) {
        if (UIHelpers::EnhancedSliderFloat("Dither Strength", &ctx.config.dither_strength, 0.0f, 1.0f, "%.3f",
                                          "Strength of dithering effect. Higher values = more smoothness but may reduce precision.")) {
            ctx.config.saveConfig();
        }
    }
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Target Prediction");
    
    if (UIHelpers::EnhancedCheckbox("Enable Velocity History", &ctx.config.enable_velocity_history, 
                                   "Uses target velocity history for more accurate prediction of moving targets.")) {
        ctx.config.saveConfig();
    }
    
    if (ctx.config.enable_velocity_history) {
        // Enhanced velocity history size slider
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, UIHelpers::GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, UIHelpers::GetAccentColor(1.0f));
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("Velocity History Size", &ctx.config.velocity_history_size, 2, 10, "%d samples")) {
            ctx.config.saveConfig();
        }
        ImGui::PopStyleColor(4);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Number of past velocity samples to use for prediction.");
            ImGui::Text("More samples = smoother but slower adaptation to target changes.");
            ImGui::EndTooltip();
        }
    }
    
    if (UIHelpers::EnhancedSliderFloat("Prediction Factor", &ctx.config.prediction_time_factor, 0.0001f, 0.01f, "%.4f",
                                      "How much prediction is applied based on target distance. Higher values = more aggressive prediction.")) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::EndCard();
}


static void draw_input_method_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Input Method Configuration");
    
    UIHelpers::BeautifulText("Choose how mouse movements are sent to the system", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::Spacer(8.0f);
    
    // Method selection with cleaner layout
    std::vector<std::string> input_methods = { "WIN32", "GHUB", "ARDUINO", "RAZER", "KMBOX" };
    std::vector<const char*> method_items;
    method_items.reserve(input_methods.size());
    for (const auto& item : input_methods) {
        method_items.push_back(item.c_str());
    }
    
    int input_method_index = 0;
    for (size_t i = 0; i < input_methods.size(); ++i) {
        if (input_methods[i] == ctx.config.input_method) {
            input_method_index = static_cast<int>(i);
            break;
        }
    }
    
    // Enhanced method selector with cleaner styling
    UIHelpers::SettingsSubHeader("Input Method");
    if (UIHelpers::EnhancedCombo("##input_method", &input_method_index, method_items.data(), method_items.size(),
                                "WIN32: Standard API (detectable)\nGHUB: Logitech driver (safer)\nARDUINO: Hardware device\nRAZER: Razer driver\nKMBOX: Hardware solution")) {
        ctx.config.input_method = input_methods[input_method_index];
        ctx.config.saveConfig();
        ctx.input_method_changed.store(true);
    }
    
    UIHelpers::Spacer(12.0f);
    
    // Method-specific settings in organized sections
    if (ctx.config.input_method == "GHUB") {
        UIHelpers::SettingsSubHeader("Logitech G HUB Status");
        
        if (!ghub_version.empty()) {
            UIHelpers::StatusIndicator("G HUB Version", true, ("Version: " + ghub_version).c_str());
        } else {
            UIHelpers::StatusIndicator("G HUB Version", false, "G HUB not detected or error occurred");
            UIHelpers::BeautifulText("Install Logitech G HUB for this input method to work", UIHelpers::GetWarningColor());
        }
    }
    
    if (ctx.config.input_method == "ARDUINO") {
        UIHelpers::SettingsSubHeader("Arduino Configuration");
        
        // COM Port setting with cleaner layout
        ImGui::Text("COM Port");
        char port_buffer[64];
        strncpy_s(port_buffer, sizeof(port_buffer), ctx.config.arduino_port.c_str(), _TRUNCATE);
        
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##arduino_port", port_buffer, sizeof(port_buffer))) {
            ctx.config.arduino_port = port_buffer;
            ctx.config.saveConfig();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enter the COM port your Arduino is connected to (e.g., COM3, /dev/ttyACM0)");
        }
        
        UIHelpers::CompactSpacer();
        
        // Baud rate with cleaner presentation
        ImGui::Text("Baud Rate");
        const char* baud_rates[] = { "9600", "115200", "250000", "500000", "1000000" };
        int current_baud_index = -1;
        char custom_baud[32];
        snprintf(custom_baud, sizeof(custom_baud), "%d", ctx.config.arduino_baudrate);
        
        for (int i = 0; i < IM_ARRAYSIZE(baud_rates); i++) {
            if (std::string(baud_rates[i]) == custom_baud) {
                current_baud_index = i;
                break;
            }
        }
        
        ImGui::SetNextItemWidth(-1);
        if (current_baud_index >= 0) {
            if (UIHelpers::EnhancedCombo("##arduino_baud_combo", &current_baud_index, baud_rates, IM_ARRAYSIZE(baud_rates),
                                        "115200: Standard speed\n250000-1000000: High speed (requires compatible firmware)")) {
                ctx.config.arduino_baudrate = std::stoi(baud_rates[current_baud_index]);
                ctx.config.saveConfig();
                ctx.input_method_changed.store(true);
            }
        } else {
            if (ImGui::InputInt("##arduino_baud", &ctx.config.arduino_baudrate, 0)) {
                ctx.config.saveConfig();
                ctx.input_method_changed.store(true);
            }
        }
        
        UIHelpers::Spacer(8.0f);
        
        if (UIHelpers::EnhancedCheckbox("Use 16-bit Mouse Movement", &ctx.config.arduino_16_bit_mouse,
                                       "Send mouse data as 16-bit values (requires compatible firmware). Higher precision but needs firmware support.")) {
            ctx.config.saveConfig();
        }
    }
    
    if (ctx.config.input_method == "KMBOX") {
        UIHelpers::SettingsSubHeader("KmboxNet Hardware");
        UIHelpers::StatusIndicator("Hardware Required", true, "Ensure B-box hardware is connected and kmNet library is initialized");
        UIHelpers::BeautifulText("Hardware-based input simulation for maximum safety", UIHelpers::GetSuccessColor());
    }
    
    if (ctx.config.input_method == "RAZER") {
        UIHelpers::SettingsSubHeader("Razer Synapse");
        UIHelpers::StatusIndicator("Driver Required", true, "Requires Razer Synapse to be installed and running");
    }
    
    if (ctx.config.input_method == "WIN32") {
        UIHelpers::SettingsSubHeader("Windows API Warning");
        UIHelpers::StatusIndicator("Detection Risk", false, "This method uses standard Windows API which may be detected by anti-cheat systems");
        UIHelpers::BeautifulText("Consider using hardware-based methods for better safety", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::EndCard();
}

static void draw_hotkey_section(const char* title, std::vector<std::string>& hotkeys, const char* add_id)
{
    // Create child window for better space management
    ImGui::BeginChild((std::string("hotkey_section_") + add_id).c_str(), ImVec2(0, 0), false);
    
    for (size_t i = 0; i < hotkeys.size(); )
    {
        std::string& current_key_name = hotkeys[i];
        
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
        
        // Use unique ID combining section name and index
        std::string unique_id = std::string(add_id) + "_" + std::to_string(i);
        ImGui::PushID(unique_id.c_str());
        
        // Calculate proper button width
        float remove_button_width = 80.0f;
        float available_width = ImGui::GetContentRegionAvail().x;
        float combo_width = available_width - remove_button_width - ImGui::GetStyle().ItemSpacing.x;
        
        // Enhanced key selector with better styling and wider width
        ImGui::SetNextItemWidth(combo_width);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Button, UIHelpers::GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, UIHelpers::GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_Header, UIHelpers::GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, UIHelpers::GetAccentColor(0.8f));
        
        std::string combo_label = "##hotkey_combo_" + unique_id;
        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            AppContext::getInstance().config.saveConfig();
        }
        ImGui::PopStyleColor(6);
        
        ImGui::SameLine();
        std::string remove_button_label = "Remove##" + unique_id;
        if (UIHelpers::BeautifulButton(remove_button_label.c_str(), ImVec2(remove_button_width, 0)))
        {
            if (hotkeys.size() <= 1)
            {
                hotkeys[0] = std::string("None");
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            }
            else
            {
                hotkeys.erase(hotkeys.begin() + i);
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            }
        }
        
        ImGui::PopID();
        ++i;
    }
    
    UIHelpers::CompactSpacer();
    std::string add_button_label = "Add Key##" + std::string(add_id);
    if (UIHelpers::BeautifulButton(add_button_label.c_str(), ImVec2(-1, 0)))
    {
        hotkeys.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
    
    ImGui::EndChild();
}

static void draw_aiming_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Aiming Controls");
    
    // Targeting Section
    UIHelpers::SettingsSubHeader("Targeting Controls");
    UIHelpers::BeautifulText("Configure keys for aimbot activation", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    draw_hotkey_section("Aimbot Activation Keys", ctx.config.button_targeting, "targeting_keys");
    
    UIHelpers::Spacer(8.0f);
    
    // Auto Shoot Section
    UIHelpers::SettingsSubHeader("Auto Shooting");
    UIHelpers::BeautifulText("Automatically shoot when targeting enemies", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    draw_hotkey_section("Auto Shoot Keys", ctx.config.button_auto_shoot, "auto_shoot_keys");
    
    UIHelpers::Spacer(8.0f);
    
    // Movement Restrictions
    UIHelpers::SettingsSubHeader("Movement Restrictions");
    UIHelpers::BeautifulText("Control when aimbot should avoid certain movements", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    draw_hotkey_section("Disable Upward Aim Keys", ctx.config.button_disable_upward_aim, "disable_upward_keys");
    
    UIHelpers::EndCard();
    
    UIHelpers::CompactSpacer();
    
    // Separate card for Triggerbot to give more space
    UIHelpers::BeginCard("Triggerbot Configuration");
    
    UIHelpers::BeautifulText("Configure the screen area where triggerbot activates", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    ImGui::Text("Area Size Multiplier");
    if (UIHelpers::EnhancedSliderFloat("##triggerbot_area", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f",
                                      "Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area\nLarger value = smaller area\n(1.0 = default area)")) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.65f);
    
    // Left column - Controller settings
    UIHelpers::BeginCard("Controller Settings");
    
    if (UIHelpers::BeautifulToggle("Use Predictive Controller", &ctx.config.use_predictive_controller,
                                   "Enable advanced Kalman filter + PID controller for better target tracking and prediction. Recommended for moving targets.")) {
        ctx.config.saveConfig();
    }
    
    // Show predictive settings directly under the toggle when enabled
    if (ctx.config.use_predictive_controller) {
        UIHelpers::Spacer(8.0f);
        UIHelpers::SettingsSubHeader("Predictive Settings");
        
        // Prediction Time
        ImGui::Text("Prediction Time (ms)");
        if (UIHelpers::EnhancedSliderFloat("##prediction_time", &ctx.config.prediction_time_ms, 1.0f, 100.0f, "%.1f",
                                          "How far ahead to predict target movement. Higher values work better for fast-moving targets.")) {
            ctx.config.saveConfig();
        }
        
        UIHelpers::CompactSpacer();
        
        // Kalman Filter settings in compact layout
        ImGui::Text("Process Noise");
        if (UIHelpers::EnhancedSliderFloat("##process_noise", &ctx.config.kalman_process_noise, 0.001f, 1.0f, "%.3f",
                                          "Kalman filter process noise. Higher values make tracker adapt faster to sudden movements.")) {
            ctx.config.saveConfig();
        }
        
        UIHelpers::CompactSpacer();
        
        ImGui::Text("Measurement Noise");
        if (UIHelpers::EnhancedSliderFloat("##measurement_noise", &ctx.config.kalman_measurement_noise, 0.001f, 1.0f, "%.3f",
                                          "Kalman filter measurement noise. Higher values smooth out jitter but reduce responsiveness.")) {
            ctx.config.saveConfig();
        }
    }
    
    UIHelpers::EndCard();
    UIHelpers::CompactSpacer();
    
    draw_pid_controls();
    UIHelpers::CompactSpacer();
    
    draw_advanced_settings();
    UIHelpers::CompactSpacer();
    
    draw_input_method_settings();
    
    UIHelpers::NextColumn();
    
    // Right column - Aiming settings and tips
    draw_aiming_settings();
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("PID Tuning Tips", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Start with Kp, then add Kd to reduce oscillation");
    ImGui::BulletText("Use Ki sparingly - too much causes overshoot");
    ImGui::BulletText("Enable Adaptive PID for better long-range stability");
    ImGui::BulletText("Increase smoothing if you experience jitter");
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulText("Input Method Guide", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("WIN32: Standard method, may be detected");
    ImGui::BulletText("GHUB: Logitech driver, generally safer");
    ImGui::BulletText("ARDUINO: Hardware-based, requires setup");
    ImGui::BulletText("KMBOX: Hardware solution, very safe");
    
    UIHelpers::EndInfoPanel();
    
    UIHelpers::EndTwoColumnLayout();
}
