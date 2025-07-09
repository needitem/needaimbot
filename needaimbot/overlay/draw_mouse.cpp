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
    UIHelpers::Spacer(5.0f);
    
    // Combined PID settings in a more compact layout
    ImGui::Columns(2, "pid_columns", false);
    
    // X-axis
    UIHelpers::BeautifulText("X-Axis (Horizontal)", UIHelpers::GetAccentColor());
    
    float kp_x = static_cast<float>(ctx.config.kp_x);
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##kp_x", &kp_x, 0.01f, 0.1f, "%.3f")) {
        ctx.config.kp_x = static_cast<double>(kp_x);
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Kp X");
    UIHelpers::InfoTooltip("Proportional gain for X-axis. Higher values = faster response but may cause oscillation.");
    
    float ki_x = static_cast<float>(ctx.config.ki_x);
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##ki_x", &ki_x, 0.01f, 0.1f, "%.3f")) {
        ctx.config.ki_x = static_cast<double>(ki_x);
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Ki X");
    UIHelpers::InfoTooltip("Integral gain for X-axis. Helps eliminate steady-state error.");
    
    float kd_x = static_cast<float>(ctx.config.kd_x);
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##kd_x", &kd_x, 0.01f, 0.1f, "%.3f")) {
        ctx.config.kd_x = static_cast<double>(kd_x);
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Kd X");
    UIHelpers::InfoTooltip("Derivative gain for X-axis. Reduces overshoot and oscillation.");
    
    ImGui::NextColumn();
    
    // Y-axis
    UIHelpers::BeautifulText("Y-Axis (Vertical)", UIHelpers::GetAccentColor());
    
    float kp_y = static_cast<float>(ctx.config.kp_y);
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##kp_y", &kp_y, 0.01f, 0.1f, "%.3f")) {
        ctx.config.kp_y = static_cast<double>(kp_y);
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Kp Y");
    UIHelpers::InfoTooltip("Proportional gain for Y-axis. Higher values = faster response but may cause oscillation.");
    
    float ki_y = static_cast<float>(ctx.config.ki_y);
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##ki_y", &ki_y, 0.01f, 0.1f, "%.3f")) {
        ctx.config.ki_y = static_cast<double>(ki_y);
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Ki Y");
    UIHelpers::InfoTooltip("Integral gain for Y-axis. Helps eliminate steady-state error.");
    
    float kd_y = static_cast<float>(ctx.config.kd_y);
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##kd_y", &kd_y, 0.01f, 0.1f, "%.3f")) {
        ctx.config.kd_y = static_cast<double>(kd_y);
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Kd Y");
    UIHelpers::InfoTooltip("Derivative gain for Y-axis. Reduces overshoot and oscillation.");
    
    ImGui::Columns(1);
    
    UIHelpers::EndCard();
}

static void draw_advanced_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Advanced Controller Settings");
    
    if (UIHelpers::BeautifulToggle("Enable Adaptive PID", &ctx.config.enable_adaptive_pid, 
                                   "Uses distance-based PID adjustment for better stability at different ranges.")) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::CompactSlider("Derivative Smoothing", &ctx.config.pid_derivative_smoothing, 0.0f, 0.8f, "%.3f");
    UIHelpers::InfoTooltip("Smooths derivative calculation to reduce noise. Higher values = more smoothing but slower response to rapid changes.");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::CompactSlider("Movement Smoothing", &ctx.config.movement_smoothing, 0.0f, 0.6f, "%.3f");
    UIHelpers::InfoTooltip("Smooths final mouse movement. Higher values = less jitter but may reduce responsiveness.");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::BeautifulSeparator("Sub-pixel Movement");
    
    if (UIHelpers::BeautifulToggle("Enable Sub-pixel Dithering", &ctx.config.enable_subpixel_dithering, 
                                   "Adds small random variations to improve movement smoothness and reduce stepping artifacts.")) {
        ctx.config.saveConfig();
    }
    
    if (ctx.config.enable_subpixel_dithering) {
        UIHelpers::CompactSlider("Dither Strength", &ctx.config.dither_strength, 0.0f, 1.0f, "%.3f");
        UIHelpers::InfoTooltip("Strength of dithering effect. Higher values = more smoothness but may reduce precision.");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            ctx.config.saveConfig();
        }
    }
    
    UIHelpers::BeautifulSeparator("Target Prediction");
    
    if (UIHelpers::BeautifulToggle("Enable Velocity History", &ctx.config.enable_velocity_history, 
                                   "Uses target velocity history for more accurate prediction of moving targets.")) {
        ctx.config.saveConfig();
    }
    
    if (ctx.config.enable_velocity_history) {
        ImGui::PushItemWidth(-1);
        if (ImGui::SliderInt("##velocity_history", &ctx.config.velocity_history_size, 2, 10)) {
            ctx.config.saveConfig();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Text("Velocity History Size");
        UIHelpers::InfoTooltip("Number of past velocity samples to use for prediction. More samples = smoother but slower adaptation.");
    }
    
    UIHelpers::CompactSlider("Prediction Factor", &ctx.config.prediction_time_factor, 0.0001f, 0.01f, "%.4f");
    UIHelpers::InfoTooltip("How much prediction is applied based on target distance. Higher values = more aggressive prediction.");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::EndCard();
}

static void draw_predictive_settings()
{
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.config.use_predictive_controller) return;
    
    UIHelpers::BeginCard("Predictive Controller Settings");
    
    float prediction_time = ctx.config.prediction_time_ms;
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##prediction_time", &prediction_time, 1.0f, 10.0f, "%.1f")) {
        ctx.config.prediction_time_ms = prediction_time;
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Prediction Time (ms)");
    UIHelpers::InfoTooltip("How far ahead to predict target movement in milliseconds. Higher values work better for fast-moving targets.");
    
    float process_noise = ctx.config.kalman_process_noise;
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##process_noise", &process_noise, 0.1f, 1.0f, "%.1f")) {
        ctx.config.kalman_process_noise = process_noise;
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Process Noise");
    UIHelpers::InfoTooltip("Kalman filter process noise. Higher values make the tracker adapt faster to sudden target movements.");
    
    float measurement_noise = ctx.config.kalman_measurement_noise;
    ImGui::PushItemWidth(-1);
    if (ImGui::InputFloat("##measurement_noise", &measurement_noise, 0.1f, 1.0f, "%.1f")) {
        ctx.config.kalman_measurement_noise = measurement_noise;
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Measurement Noise");
    UIHelpers::InfoTooltip("Kalman filter measurement noise. Higher values smooth out detection jitter but reduce responsiveness.");
    
    UIHelpers::EndCard();
}

static void draw_input_method_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Input Method Settings");
    
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
    
    UIHelpers::CompactCombo("Input Method", &input_method_index, method_items.data(), method_items.size());
    UIHelpers::InfoTooltip("Select the input method for sending mouse movements:\n"
                          "WIN32: Standard Windows SendInput. May be detected.\n"
                          "GHUB: Logitech G Hub driver (if installed and supported). Generally safer.\n"
                          "ARDUINO: Requires a connected Arduino board flashed with appropriate firmware.\n"
                          "RAZER: Uses a specific Razer driver DLL (rzctl.dll). Requires DLL path.\n"
                          "KMBOX: Uses kmBoxNet library (requires B box hardware).");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.input_method = input_methods[input_method_index];
        ctx.config.saveConfig();
    }
    
    if (ctx.config.input_method == "GHUB" || !ghub_version.empty()) {
        UIHelpers::Spacer(5.0f);
        ImGui::Text("GHUB Version: %s", ghub_version.c_str());
        if (ghub_version.empty()) {
            ImGui::SameLine();
            UIHelpers::BeautifulText("(Not Found/Error)", UIHelpers::GetErrorColor());
        }
    }
    
    if (ctx.config.input_method == "ARDUINO") {
        UIHelpers::BeautifulSeparator("Arduino Settings");
        
        char port_buffer[64];
        strncpy_s(port_buffer, sizeof(port_buffer), ctx.config.arduino_port.c_str(), _TRUNCATE);
        
        ImGui::PushItemWidth(-1);
        if (ImGui::InputText("##arduino_port", port_buffer, sizeof(port_buffer))) {
            ctx.config.arduino_port = port_buffer;
            ctx.config.saveConfig();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Text("COM Port");
        UIHelpers::InfoTooltip("Enter the COM port your Arduino is connected to (e.g., COM3, /dev/ttyACM0).");
        
        ImGui::PushItemWidth(-1);
        if (ImGui::InputInt("##arduino_baud", &ctx.config.arduino_baudrate, 0)) {
            ctx.config.saveConfig();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Text("Baud Rate");
        UIHelpers::InfoTooltip("Serial communication speed (e.g., 9600, 115200). Must match Arduino firmware.");
        
        if (UIHelpers::BeautifulToggle("Use 16-bit Mouse Movement", &ctx.config.arduino_16_bit_mouse,
                                       "Send mouse movement data as 16-bit values (requires compatible firmware). Otherwise, uses 8-bit.")) {
            ctx.config.saveConfig();
        }
    }
    
    if (ctx.config.input_method == "KMBOX") {
        UIHelpers::BeautifulSeparator("Kmbox Settings");
        UIHelpers::BeautifulText("KmboxNet selected. Ensure B-box hardware is connected and kmNet library is initialized.", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::EndCard();
}

static void draw_hotkey_section(const char* title, std::vector<std::string>& hotkeys, const char* add_id)
{
    UIHelpers::BeautifulSeparator(title);
    
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
        
        ImGui::PushID(static_cast<int>(i));
        
        float button_width = 70.0f;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button_width - ImGui::GetStyle().ItemSpacing.x);
        
        if (ImGui::Combo("##hotkey", &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            AppContext::getInstance().config.saveConfig();
        }
        
        ImGui::SameLine();
        if (UIHelpers::BeautifulButton("Remove", ImVec2(button_width, 0)))
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
    
    UIHelpers::Spacer(5.0f);
    std::string add_button_label = "Add " + std::string(title) + " Hotkey";
    if (UIHelpers::BeautifulButton(add_button_label.c_str(), ImVec2(-1, 0)))
    {
        hotkeys.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
}

static void draw_aiming_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Aiming Settings");
    
    draw_hotkey_section("Auto Shoot", ctx.config.button_auto_shoot, "auto_shoot");
    
    UIHelpers::BeautifulSeparator("Scope Settings");
    
    UIHelpers::CompactSlider("Triggerbot Area Size", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f");
    UIHelpers::InfoTooltip("Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area, Larger value = smaller area.\n(1.0 = default area)");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    draw_hotkey_section("Targeting", ctx.config.button_targeting, "targeting");
    
    draw_hotkey_section("Disable Upward Aim", ctx.config.button_disable_upward_aim, "disable_upward_aim");
    
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
    
    UIHelpers::EndCard();
    UIHelpers::Spacer();
    
    draw_pid_controls();
    UIHelpers::Spacer();
    
    draw_advanced_settings();
    UIHelpers::Spacer();
    
    draw_predictive_settings();
    UIHelpers::Spacer();
    
    draw_input_method_settings();
    
    UIHelpers::NextColumn();
    
    // Right column - Aiming settings and tips
    draw_aiming_settings();
    UIHelpers::Spacer();
    
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("PID Tuning Tips", UIHelpers::GetAccentColor());
    UIHelpers::Spacer(5.0f);
    
    ImGui::BulletText("Start with Kp, then add Kd to reduce oscillation");
    ImGui::BulletText("Use Ki sparingly - too much causes overshoot");
    ImGui::BulletText("Enable Adaptive PID for better long-range stability");
    ImGui::BulletText("Increase smoothing if you experience jitter");
    
    UIHelpers::Spacer();
    
    UIHelpers::BeautifulText("Input Method Guide", UIHelpers::GetAccentColor());
    UIHelpers::Spacer(5.0f);
    
    ImGui::BulletText("WIN32: Standard method, may be detected");
    ImGui::BulletText("GHUB: Logitech driver, generally safer");
    ImGui::BulletText("ARDUINO: Hardware-based, requires setup");
    ImGui::BulletText("KMBOX: Hardware solution, very safe");
    
    UIHelpers::EndInfoPanel();
    
    UIHelpers::EndTwoColumnLayout();
}
