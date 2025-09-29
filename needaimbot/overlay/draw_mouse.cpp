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

    UIHelpers::SettingsSubHeader("Rate Normalization & Filter");

    if (UIHelpers::EnhancedCheckbox("Normalize by Frame Time", &ctx.config.normalize_movement_rate,
        "Keep response per second consistent across FPS"))
    {
        SAVE_PROFILE();
    }

    ImGui::BeginDisabled(!ctx.config.normalize_movement_rate);
    {
        float alpha = ctx.config.movement_rate_ema_alpha;
        if (ImGui::SliderFloat("EMA Alpha", &alpha, 0.01f, 0.5f, "%.2f")) {
            ctx.config.movement_rate_ema_alpha = alpha;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Smoothing factor for dt estimate (lower = smoother, higher = quicker)");

        int warmup = ctx.config.movement_warmup_frames;
        if (ImGui::SliderInt("Warmup Frames", &warmup, 0, 60)) {
            ctx.config.movement_warmup_frames = warmup;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Frames to establish baseline dt when not using fixed FPS");

        if (UIHelpers::EnhancedCheckbox("Use Fixed Reference FPS", &ctx.config.rate_use_fixed_reference_fps,
            "Bypass warmup and assume this FPS as the baseline"))
        {
            SAVE_PROFILE();
        }

        ImGui::BeginDisabled(!ctx.config.rate_use_fixed_reference_fps);
        float ref_fps = ctx.config.rate_fixed_reference_fps;
        if (ImGui::InputFloat("Reference FPS", &ref_fps, 0.0f, 0.0f, "%.1f")) {
            if (ref_fps < 1.0f) ref_fps = 1.0f;
            ctx.config.rate_fixed_reference_fps = ref_fps;
            SAVE_PROFILE();
        }
        ImGui::EndDisabled();
    }
    ImGui::EndDisabled();

    UIHelpers::CompactSpacer();

    float deadzone = ctx.config.movement_deadzone;
    if (ImGui::SliderFloat("Deadzone (px)", &deadzone, 0.0f, 5.0f, "%.2f")) {
        ctx.config.movement_deadzone = deadzone;
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("Suppress tiny oscillations near zero movement");

    int max_step = ctx.config.movement_max_step;
    if (ImGui::SliderInt("Max Step (px)", &max_step, 1, 100)) {
        ctx.config.movement_max_step = max_step;
        SAVE_PROFILE();
    }
    UIHelpers::HelpMarker("Clamp per-dispatch mouse step to limit spikes");

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
