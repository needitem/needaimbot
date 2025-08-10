#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>
#include <algorithm>

#include "AppContext.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "ui_helpers.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include "../mouse/mouse.h" 

// GHub version check removed - not needed

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
        SAVE_PROFILE();
    }
    
    float ki_x = static_cast<float>(ctx.config.ki_x);
    if (UIHelpers::EnhancedSliderFloat("Ki X", &ki_x, 0.0f, 0.02f, "%.4f", "Integral gain for X-axis. Helps eliminate steady-state error.")) {
        ctx.config.ki_x = static_cast<double>(ki_x);
        SAVE_PROFILE();
    }
    
    float kd_x = static_cast<float>(ctx.config.kd_x);
    if (UIHelpers::EnhancedSliderFloat("Kd X", &kd_x, 0.0f, 0.05f, "%.4f", "Derivative gain for X-axis. Reduces overshoot and oscillation.")) {
        ctx.config.kd_x = static_cast<double>(kd_x);
        SAVE_PROFILE();
    }
    
    ImGui::NextColumn();
    
    // Y-axis
    UIHelpers::SettingsSubHeader("Y-Axis (Vertical)");
    
    float kp_y = static_cast<float>(ctx.config.kp_y);
    if (UIHelpers::EnhancedSliderFloat("Kp Y", &kp_y, 0.0f, 2.0f, "%.3f", "Proportional gain for Y-axis. Higher values = faster response but may cause oscillation.")) {
        ctx.config.kp_y = static_cast<double>(kp_y);
        SAVE_PROFILE();
    }
    
    float ki_y = static_cast<float>(ctx.config.ki_y);
    if (UIHelpers::EnhancedSliderFloat("Ki Y", &ki_y, 0.0f, 0.02f, "%.4f", "Integral gain for Y-axis. Helps eliminate steady-state error.")) {
        ctx.config.ki_y = static_cast<double>(ki_y);
        SAVE_PROFILE();
    }
    
    float kd_y = static_cast<float>(ctx.config.kd_y);
    if (UIHelpers::EnhancedSliderFloat("Kd Y", &kd_y, 0.0f, 0.05f, "%.4f", "Derivative gain for Y-axis. Reduces overshoot and oscillation.")) {
        ctx.config.kd_y = static_cast<double>(kd_y);
        SAVE_PROFILE();
    }
    
    ImGui::Columns(1);
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Quick Presets");
    
    if (ImGui::Button("Conservative", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3, 0))) {
        ctx.config.kp_x = 0.3; ctx.config.ki_x = 0.001; ctx.config.kd_x = 0.005;
        ctx.config.kp_y = 0.3; ctx.config.ki_y = 0.001; ctx.config.kd_y = 0.005;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Safe values for most situations");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Balanced", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2, 0))) {
        ctx.config.kp_x = 0.5; ctx.config.ki_x = 0.003; ctx.config.kd_x = 0.01;
        ctx.config.kp_y = 0.5; ctx.config.ki_y = 0.003; ctx.config.kd_y = 0.01;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Balanced between speed and stability");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Aggressive", ImVec2(-1, 0))) {
        ctx.config.kp_x = 0.8; ctx.config.ki_x = 0.005; ctx.config.kd_x = 0.02;
        ctx.config.kp_y = 0.8; ctx.config.ki_y = 0.005; ctx.config.kd_y = 0.02;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fast tracking, may oscillate");
    }
    
    UIHelpers::EndCard();

    // Derivative stabilization advanced controls
    UIHelpers::BeginCard("PID Derivative Stabilization");
    UIHelpers::SettingsSubHeader("Noise Guards and Limits");

    if (UIHelpers::EnhancedSliderFloat("D Deadband (px)", &ctx.config.pid_d_deadband, 0.0f, 2.0f, "%.3f",
                                      "Ignore tiny error deltas below this")) {
        SAVE_PROFILE();
    }
    if (UIHelpers::EnhancedSliderFloat("Disable D Near Error (px)", &ctx.config.pid_d_disable_error, 0.0f, 5.0f, "%.3f",
                                      "Turn off D when |error| is small")) {
        SAVE_PROFILE();
    }
    // Removed D delta/output clamps from UI
    if (UIHelpers::EnhancedSliderFloat("Output Deadzone (px)", &ctx.config.pid_output_deadzone, 0.0f, 2.0f, "%.2f",
                                      "Zero very small outputs to avoid jitter")) {
        SAVE_PROFILE();
    }
    {
        int warm = ctx.config.pid_d_warmup_frames;
        if (ImGui::SliderInt("D Warmup Frames", &warm, 0, 10, "%d frames")) {
            ctx.config.pid_d_warmup_frames = warm;
            SAVE_PROFILE();
        }
    }

    UIHelpers::EndCard();
}

// Advanced settings removed - keeping code simple


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
    if (UIHelpers::EnhancedCombo("##input_method", &input_method_index, method_items.data(), static_cast<int>(method_items.size()),
                                "WIN32: Standard API (detectable)\nGHUB: Logitech driver (safer)\nARDUINO: Hardware device\nRAZER: Razer driver\nKMBOX: Hardware solution")) {
        ctx.config.input_method = input_methods[input_method_index];
        SAVE_PROFILE();
        ctx.input_method_changed.store(true);
    }
    
    UIHelpers::Spacer(12.0f);
    
    // Method-specific settings in organized sections
    if (ctx.config.input_method == "GHUB") {
        UIHelpers::SettingsSubHeader("Logitech G HUB Status");
        
        // Version check removed - just show status
        UIHelpers::StatusIndicator("G HUB", true, "Selected as input method");
        UIHelpers::BeautifulText("Make sure Logitech G HUB is installed for this to work", UIHelpers::GetAccentColor(0.7f));
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
            SAVE_PROFILE();
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
                SAVE_PROFILE();
                ctx.input_method_changed.store(true);
            }
        } else {
            if (ImGui::InputInt("##arduino_baud", &ctx.config.arduino_baudrate, 0)) {
                SAVE_PROFILE();
                ctx.input_method_changed.store(true);
            }
        }
        
        UIHelpers::Spacer(8.0f);
        
        if (UIHelpers::EnhancedCheckbox("Use 16-bit Mouse Movement", &ctx.config.arduino_16_bit_mouse,
                                       "Send mouse data as 16-bit values (requires compatible firmware). Higher precision but needs firmware support.")) {
            SAVE_PROFILE();
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



void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    // Add triggerbot and rapidfire toggles at the top
    UIHelpers::BeginCard("Combat Features");
    
    if (UIHelpers::EnhancedCheckbox("Enable Triggerbot", &ctx.config.enable_triggerbot,
                                   "Automatically fire when crosshair is on target")) {
        SAVE_PROFILE();
    }
    
    ImGui::SameLine();
    
    if (UIHelpers::EnhancedCheckbox("Enable Rapidfire", &ctx.config.enable_rapidfire,
                                   "Rapid fire mode for semi-automatic weapons")) {
        SAVE_PROFILE();
        // Update rapidfire state in mouse thread
        if (ctx.global_mouse_thread) {
            ctx.global_mouse_thread->updateRapidFire();
        }
    }
    
    // CPS slider for rapidfire (only show when rapidfire is enabled)
    if (ctx.config.enable_rapidfire) {
        UIHelpers::CompactSpacer();
        ImGui::Text("Clicks Per Second");
        if (ImGui::SliderInt("##rapidfire_cps", &ctx.config.rapidfire_cps, 1, 30, "%d CPS")) {
            SAVE_PROFILE();
            // Update rapidfire CPS in mouse thread
            if (ctx.global_mouse_thread) {
                ctx.global_mouse_thread->updateRapidFire();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Adjust the rapid fire speed (clicks per second)");
        }
    }
    
    UIHelpers::EndCard();
    UIHelpers::Spacer();
    
    draw_pid_controls();
    UIHelpers::Spacer();
    
    draw_input_method_settings();
}
