#include "../core/windows_headers.h"

#include <shellapi.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "AppContext.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "ui_helpers.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include "../mouse/mouse.h" 

static void draw_movement_controls()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Mouse Movement Settings");
    
    UIHelpers::BeautifulText("Simple proportional controller for aiming.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    UIHelpers::SettingsSubHeader("Proportional Gain Settings");
    
    // Simple P controller gains in two columns
    if (ImGui::BeginTable("GainsTable", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("X-Axis (Horizontal)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y-Axis (Vertical)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Gain (Kp)");
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 60);
        if (ImGui::InputFloat("##KpX", &ctx.config.pd_kp_x, 0.0f, 0.0f, "%.3f")) {
            if (ctx.config.pd_kp_x < 0.0f) ctx.config.pd_kp_x = 0.0f;
            if (ctx.config.pd_kp_x > 100.0f) ctx.config.pd_kp_x = 100.0f;
            SAVE_PROFILE();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("-##KpXMinus", ImVec2(25, 0))) {
            ctx.config.pd_kp_x -= 0.01f;
            if (ctx.config.pd_kp_x < 0.0f) ctx.config.pd_kp_x = 0.0f;
            SAVE_PROFILE();
        }
        ImGui::SameLine();
        if (ImGui::Button("+##KpXPlus", ImVec2(25, 0))) {
            ctx.config.pd_kp_x += 0.01f;
            if (ctx.config.pd_kp_x > 100.0f) ctx.config.pd_kp_x = 100.0f;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Gain (Kp)");
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 60);
        if (ImGui::InputFloat("##KpY", &ctx.config.pd_kp_y, 0.0f, 0.0f, "%.3f")) {
            if (ctx.config.pd_kp_y < 0.0f) ctx.config.pd_kp_y = 0.0f;
            if (ctx.config.pd_kp_y > 100.0f) ctx.config.pd_kp_y = 100.0f;
            SAVE_PROFILE();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("-##KpYMinus", ImVec2(25, 0))) {
            ctx.config.pd_kp_y -= 0.01f;
            if (ctx.config.pd_kp_y < 0.0f) ctx.config.pd_kp_y = 0.0f;
            SAVE_PROFILE();
        }
        ImGui::SameLine();
        if (ImGui::Button("+##KpYPlus", ImVec2(25, 0))) {
            ctx.config.pd_kp_y += 0.01f;
            if (ctx.config.pd_kp_y > 100.0f) ctx.config.pd_kp_y = 100.0f;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("How strongly to respond to target distance");

        ImGui::EndTable();
    }

    UIHelpers::CompactSpacer();
    UIHelpers::SettingsSubHeader("Smoothing Filter Tuning");
    UIHelpers::HelpMarker(
        "Applies an exponential smoothing step with a configurable acceleration cap. "
        "Higher rates and step limits chase moving targets faster, lower values make small "
        "adjustments steadier."
    );

    if (ImGui::SliderFloat("Response Rate (Hz)", &ctx.config.smoothing_rate, 0.1f, 60.0f, "%.2f")) {
        if (ctx.config.smoothing_rate < 0.1f) ctx.config.smoothing_rate = 0.1f;
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("Base frequency for the exponential lerp – higher means quicker convergence.");

    if (ImGui::SliderFloat("Minimum Alpha", &ctx.config.smoothing_min_alpha, 0.0f, 1.0f, "%.3f")) {
        ctx.config.smoothing_min_alpha = std::clamp(ctx.config.smoothing_min_alpha, 0.0f, 1.0f);
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("Lower bound for interpolation so micro-corrections still move the cursor.");

    if (ImGui::SliderFloat("Boost Scale (px)", &ctx.config.smoothing_alpha_boost_scale, 1.0f, 60.0f, "%.1f")) {
        if (ctx.config.smoothing_alpha_boost_scale < 1.0f) ctx.config.smoothing_alpha_boost_scale = 1.0f;
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("How many pixels of error are needed before alpha starts accelerating.");

    if (ImGui::SliderFloat("Boost Limit (x)", &ctx.config.smoothing_alpha_boost_limit, 0.0f, 4.0f, "%.2f")) {
        if (ctx.config.smoothing_alpha_boost_limit < 0.0f) ctx.config.smoothing_alpha_boost_limit = 0.0f;
        SAVE_PROFILE();
    }

    if (ImGui::SliderInt("Base Step (px)", &ctx.config.smoothing_step_base, 0, 64)) {
        if (ctx.config.smoothing_step_base < 0) ctx.config.smoothing_step_base = 0;
        if (ctx.config.smoothing_step_cap < ctx.config.smoothing_step_base) {
            ctx.config.smoothing_step_cap = ctx.config.smoothing_step_base;
        }
        SAVE_PROFILE();
    }

    if (ImGui::SliderFloat("Step per Second (px/s)", &ctx.config.smoothing_step_per_second, 0.0f, 600.0f, "%.1f")) {
        if (ctx.config.smoothing_step_per_second < 0.0f) ctx.config.smoothing_step_per_second = 0.0f;
        SAVE_PROFILE();
    }

    if (ImGui::SliderInt("Step Cap (px)", &ctx.config.smoothing_step_cap, 0, 256)) {
        if (ctx.config.smoothing_step_cap < ctx.config.smoothing_step_base) {
            ctx.config.smoothing_step_cap = ctx.config.smoothing_step_base;
        }
        SAVE_PROFILE();
    }

    if (ImGui::SliderFloat("Burst Multiplier", &ctx.config.smoothing_burst_multiplier, 0.0f, 2.0f, "%.3f")) {
        if (ctx.config.smoothing_burst_multiplier < 0.0f) ctx.config.smoothing_burst_multiplier = 0.0f;
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("Extra allowance = request * multiplier, helps catch sudden large deltas.");

    if (ImGui::SliderInt("Rest Deadzone (px)", &ctx.config.smoothing_rest_deadzone, 0, 5)) {
        if (ctx.config.smoothing_rest_deadzone < 0) ctx.config.smoothing_rest_deadzone = 0;
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("Residual pixels below this threshold snap to 0 so the cursor stops drifting.");

    UIHelpers::EndCard();
}

static void draw_input_device_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Mouse Input Device");

    static constexpr const char* INPUT_METHODS[] = {
        "WIN32",
        "GHUB",
        "ARDUINO",
        "KMBOX",
        "MAKCU",
        "RAZER"
    };

    int method_index = 0;
    for (int i = 0; i < IM_ARRAYSIZE(INPUT_METHODS); ++i) {
        if (ctx.config.input_method == INPUT_METHODS[i]) {
            method_index = i;
            break;
        }
    }

    const int previous_method = method_index;
    if (UIHelpers::EnhancedCombo("Input Method", &method_index, INPUT_METHODS, IM_ARRAYSIZE(INPUT_METHODS),
        "Select which driver handles mouse movement"))
    {
        ctx.config.input_method = INPUT_METHODS[method_index];
        ctx.input_method_changed = true;
        SAVE_PROFILE();
    }

    UIHelpers::CompactSpacer();

    const char* active_method = INPUT_METHODS[method_index];

    if (std::strcmp(active_method, "ARDUINO") == 0) {
        UIHelpers::SettingsSubHeader("Arduino Serial Settings");

        static char arduino_port_buffer[64] = "";
        static char arduino_baud_buffer[64] = "";
        static bool buffers_initialized = false;

        if (!buffers_initialized || previous_method != method_index || ctx.config.arduino_port != arduino_port_buffer) {
            std::snprintf(arduino_port_buffer, IM_ARRAYSIZE(arduino_port_buffer), "%s", ctx.config.arduino_port.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || std::to_string(ctx.config.arduino_baudrate) != arduino_baud_buffer) {
            std::snprintf(arduino_baud_buffer, IM_ARRAYSIZE(arduino_baud_buffer), "%d", ctx.config.arduino_baudrate);
        }
        buffers_initialized = true;

        if (ImGui::InputText("Serial Port", arduino_port_buffer, IM_ARRAYSIZE(arduino_port_buffer))) {
            ctx.config.arduino_port = arduino_port_buffer;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("COM port that the Arduino is connected to");

        if (ImGui::InputText("Baud Rate", arduino_baud_buffer, IM_ARRAYSIZE(arduino_baud_buffer), ImGuiInputTextFlags_CharsDecimal)) {
            ctx.config.arduino_baudrate = std::max(0, std::atoi(arduino_baud_buffer));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Serial speed used for communicating with the Arduino");

        if (UIHelpers::EnhancedCheckbox("Enable Key Passthrough", &ctx.config.arduino_enable_keys,
            "Forward keyboard events to the Arduino for on-board handling"))
        {
            SAVE_PROFILE();
        }
    }
    else if (std::strcmp(active_method, "KMBOX") == 0) {
        UIHelpers::SettingsSubHeader("KMBOX Network Settings");

        static char kmbox_ip_buffer[64] = "";
        static char kmbox_port_buffer[16] = "";
        static char kmbox_mac_buffer[64] = "";
        static bool buffers_initialized = false;

        if (!buffers_initialized || previous_method != method_index || ctx.config.kmbox_ip != kmbox_ip_buffer) {
            std::snprintf(kmbox_ip_buffer, IM_ARRAYSIZE(kmbox_ip_buffer), "%s", ctx.config.kmbox_ip.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || ctx.config.kmbox_port != kmbox_port_buffer) {
            std::snprintf(kmbox_port_buffer, IM_ARRAYSIZE(kmbox_port_buffer), "%s", ctx.config.kmbox_port.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || ctx.config.kmbox_mac != kmbox_mac_buffer) {
            std::snprintf(kmbox_mac_buffer, IM_ARRAYSIZE(kmbox_mac_buffer), "%s", ctx.config.kmbox_mac.c_str());
        }
        buffers_initialized = true;

        if (ImGui::InputText("Device IP", kmbox_ip_buffer, IM_ARRAYSIZE(kmbox_ip_buffer))) {
            ctx.config.kmbox_ip = kmbox_ip_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::InputText("Device Port", kmbox_port_buffer, IM_ARRAYSIZE(kmbox_port_buffer), ImGuiInputTextFlags_CharsDecimal)) {
            ctx.config.kmbox_port = kmbox_port_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::InputText("Device MAC", kmbox_mac_buffer, IM_ARRAYSIZE(kmbox_mac_buffer), ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_CharsUppercase)) {
            ctx.config.kmbox_mac = kmbox_mac_buffer;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Enter MAC without separators, e.g. 46405C53");
    }
    else if (std::strcmp(active_method, "MAKCU") == 0) {
        UIHelpers::SettingsSubHeader("MAKCU Serial Settings");

        static char makcu_port_buffer[64] = "";
        static char makcu_baud_buffer[64] = "";
        static bool buffers_initialized = false;

        if (!buffers_initialized || previous_method != method_index || ctx.config.makcu_port != makcu_port_buffer) {
            std::snprintf(makcu_port_buffer, IM_ARRAYSIZE(makcu_port_buffer), "%s", ctx.config.makcu_port.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || std::to_string(ctx.config.makcu_baudrate) != makcu_baud_buffer) {
            std::snprintf(makcu_baud_buffer, IM_ARRAYSIZE(makcu_baud_buffer), "%d", ctx.config.makcu_baudrate);
        }
        buffers_initialized = true;

        if (ImGui::InputText("Serial Port", makcu_port_buffer, IM_ARRAYSIZE(makcu_port_buffer))) {
            ctx.config.makcu_port = makcu_port_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::InputText("Baud Rate", makcu_baud_buffer, IM_ARRAYSIZE(makcu_baud_buffer), ImGuiInputTextFlags_CharsDecimal)) {
            ctx.config.makcu_baudrate = std::max(0, std::atoi(makcu_baud_buffer));
            SAVE_PROFILE();
        }
    }
    else if (std::strcmp(active_method, "GHUB") == 0) {
        UIHelpers::SettingsSubHeader("G HUB Integration");
        UIHelpers::BeautifulText("Logitech G HUB must be running for this mode to work.", UIHelpers::GetWarningColor());
    }
    else if (std::strcmp(active_method, "RAZER") == 0) {
        UIHelpers::SettingsSubHeader("Razer Synapse");
        UIHelpers::BeautifulText("Requires Razer Synapse with the SDK enabled.", UIHelpers::GetWarningColor());
    }
    else {
        UIHelpers::SettingsSubHeader("Windows API");
        UIHelpers::BeautifulText("Uses the default Win32 mouse events. No extra setup required.", UIHelpers::GetAccentColor());
    }

    UIHelpers::EndCard();
}

void draw_mouse()
{
    draw_input_device_settings();
    UIHelpers::Spacer();
    draw_movement_controls();
}