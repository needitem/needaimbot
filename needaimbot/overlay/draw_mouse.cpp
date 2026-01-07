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
#include "../cuda/unified_graph_pipeline.h"
#include "../mouse/mouse.h" 

static void draw_movement_controls()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("PID Controller Settings");

    UIHelpers::BeautifulText("PID controller provides smooth, accurate tracking with oscillation suppression.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::BeautifulText("P = Responsiveness, I = Tracking moving targets, D = Dampening oscillation", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
    UIHelpers::CompactSpacer();

    if (ImGui::BeginTabBar("PIDTabs")) {
        if (ImGui::BeginTabItem("Proportional (P)")) {
            UIHelpers::SettingsSubHeader("Proportional Gain (Kp)");
            UIHelpers::BeautifulText("Controls response speed. Higher = faster aiming, but may overshoot.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::BeginTable("KpTable", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KpX", &ctx.config.profile().pid_kp_x, 0.0f, 2.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KpY", &ctx.config.profile().pid_kp_y, 0.0f, 2.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            UIHelpers::HelpMarker("Recommended: 0.3-0.8. Higher values = faster response but more overshoot.");
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Integral (I)")) {
            UIHelpers::SettingsSubHeader("Integral Gain (Ki)");
            UIHelpers::BeautifulText("Eliminates steady-state error. Essential for tracking moving targets.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::BeautifulText("Set to 0 for PD controller (no integral). Use small values (0.01-0.1).", UIHelpers::GetWarningColor());
            UIHelpers::CompactSpacer();

            if (ImGui::BeginTable("KiTable", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KiX", &ctx.config.profile().pid_ki_x, 0.0f, 0.3f, "%.4f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KiY", &ctx.config.profile().pid_ki_y, 0.0f, 0.3f, "%.4f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            UIHelpers::HelpMarker("Start with 0 (PD mode). Increase gradually to 0.05-0.1 if target tracking lags.");

            UIHelpers::CompactSpacer();
            UIHelpers::SettingsSubHeader("Anti-Windup Limit");
            ImGui::PushItemWidth(-1);
            if (ImGui::SliderFloat("##IntegralMax", &ctx.config.profile().pid_integral_max, 10.0f, 500.0f, "%.0f px")) {
                SAVE_PROFILE();
            }
            ImGui::PopItemWidth();
            UIHelpers::HelpMarker("Prevents integral from accumulating too much (windup protection). Default: 100px.");

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Derivative (D)")) {
            UIHelpers::SettingsSubHeader("Derivative Gain (Kd)");
            UIHelpers::BeautifulText("Suppresses oscillation by damping sudden changes. Higher = smoother.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::BeginTable("KdTable", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KdX", &ctx.config.profile().pid_kd_x, 0.0f, 1.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KdY", &ctx.config.profile().pid_kd_y, 0.0f, 1.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            UIHelpers::HelpMarker("Recommended: 0.2-0.5. Higher values reduce oscillation but may slow response.");

            UIHelpers::CompactSpacer();
            UIHelpers::SettingsSubHeader("Derivative Clamp Limit");
            ImGui::PushItemWidth(-1);
            if (ImGui::SliderFloat("##DerivativeMax", &ctx.config.profile().pid_derivative_max, 10.0f, 200.0f, "%.0f px")) {
                SAVE_PROFILE();
            }
            ImGui::PopItemWidth();
            UIHelpers::HelpMarker("Limits maximum derivative value to prevent excessive oscillation from large movements. Default: 50px.");

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    UIHelpers::EndCard();
}

static void draw_input_device_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Mouse Input Device");

    UIHelpers::BeautifulText("Select the hardware/driver used to send mouse movements.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::BeautifulText("WIN32 works for most cases. Hardware options may bypass detection.", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
    UIHelpers::CompactSpacer();

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
        if (ctx.config.global().input_method == INPUT_METHODS[i]) {
            method_index = i;
            break;
        }
    }

    const int previous_method = method_index;
    if (UIHelpers::EnhancedCombo("Input Method", &method_index, INPUT_METHODS, IM_ARRAYSIZE(INPUT_METHODS),
        "Select which driver handles mouse movement"))
    {
        ctx.config.global().input_method = INPUT_METHODS[method_index];
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

        if (!buffers_initialized || previous_method != method_index || ctx.config.global().arduino_port != arduino_port_buffer) {
            std::snprintf(arduino_port_buffer, IM_ARRAYSIZE(arduino_port_buffer), "%s", ctx.config.global().arduino_port.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || std::to_string(ctx.config.global().arduino_baudrate) != arduino_baud_buffer) {
            std::snprintf(arduino_baud_buffer, IM_ARRAYSIZE(arduino_baud_buffer), "%d", ctx.config.global().arduino_baudrate);
        }
        buffers_initialized = true;

        if (ImGui::InputText("Serial Port", arduino_port_buffer, IM_ARRAYSIZE(arduino_port_buffer))) {
            ctx.config.global().arduino_port = arduino_port_buffer;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("COM port that the Arduino is connected to");

        if (ImGui::InputText("Baud Rate", arduino_baud_buffer, IM_ARRAYSIZE(arduino_baud_buffer), ImGuiInputTextFlags_CharsDecimal)) {
            ctx.config.global().arduino_baudrate = std::max(0, std::atoi(arduino_baud_buffer));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Serial speed used for communicating with the Arduino");

        if (UIHelpers::EnhancedCheckbox("Enable Key Passthrough", &ctx.config.global().arduino_enable_keys,
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

        if (!buffers_initialized || previous_method != method_index || ctx.config.global().kmbox_ip != kmbox_ip_buffer) {
            std::snprintf(kmbox_ip_buffer, IM_ARRAYSIZE(kmbox_ip_buffer), "%s", ctx.config.global().kmbox_ip.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || ctx.config.global().kmbox_port != kmbox_port_buffer) {
            std::snprintf(kmbox_port_buffer, IM_ARRAYSIZE(kmbox_port_buffer), "%s", ctx.config.global().kmbox_port.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || ctx.config.global().kmbox_mac != kmbox_mac_buffer) {
            std::snprintf(kmbox_mac_buffer, IM_ARRAYSIZE(kmbox_mac_buffer), "%s", ctx.config.global().kmbox_mac.c_str());
        }
        buffers_initialized = true;

        if (ImGui::InputText("Device IP", kmbox_ip_buffer, IM_ARRAYSIZE(kmbox_ip_buffer))) {
            ctx.config.global().kmbox_ip = kmbox_ip_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::InputText("Device Port", kmbox_port_buffer, IM_ARRAYSIZE(kmbox_port_buffer), ImGuiInputTextFlags_CharsDecimal)) {
            ctx.config.global().kmbox_port = kmbox_port_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::InputText("Device MAC", kmbox_mac_buffer, IM_ARRAYSIZE(kmbox_mac_buffer), ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_CharsUppercase)) {
            ctx.config.global().kmbox_mac = kmbox_mac_buffer;
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Enter MAC without separators, e.g. 46405C53");
    }
    else if (std::strcmp(active_method, "MAKCU") == 0) {
        UIHelpers::SettingsSubHeader("MAKCU Network Settings (2PC)");
        UIHelpers::BeautifulText("Sends movement over UDP to a second PC running MakcuRelay.", UIHelpers::GetAccentColor(0.8f));

        static char makcu_ip_buffer[64] = "";
        static char makcu_port_buffer[16] = "";
        static bool buffers_initialized = false;

        if (!buffers_initialized || previous_method != method_index || ctx.config.global().makcu_remote_ip != makcu_ip_buffer) {
            std::snprintf(makcu_ip_buffer, IM_ARRAYSIZE(makcu_ip_buffer), "%s", ctx.config.global().makcu_remote_ip.c_str());
        }
        if (!buffers_initialized || previous_method != method_index || std::to_string(ctx.config.global().makcu_remote_port) != makcu_port_buffer) {
            std::snprintf(makcu_port_buffer, IM_ARRAYSIZE(makcu_port_buffer), "%d", ctx.config.global().makcu_remote_port);
        }
        buffers_initialized = true;

        if (ImGui::InputText("Second PC IP", makcu_ip_buffer, IM_ARRAYSIZE(makcu_ip_buffer))) {
            ctx.config.global().makcu_remote_ip = makcu_ip_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::InputText("UDP Port", makcu_port_buffer, IM_ARRAYSIZE(makcu_port_buffer), ImGuiInputTextFlags_CharsDecimal)) {
            ctx.config.global().makcu_remote_port = std::max(0, std::atoi(makcu_port_buffer));
            SAVE_PROFILE();
        }
        UIHelpers::HelpMarker("Set this to the IP and UDP port where MakcuRelay.exe is listening on the second PC.");
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
    UIHelpers::Spacer();

    // Deadband / jitter filter controls
    auto& ctx = AppContext::getInstance();
    UIHelpers::BeginCard("Jitter Filter (Deadband)");
    UIHelpers::BeautifulText("Suppress micro-oscillation near target", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();

    if (ImGui::BeginTable("##deadband_table", 2, ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupColumn("X-Axis");
        ImGui::TableSetupColumn("Y-Axis");

        // Headers
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(UIHelpers::GetAccentColor(), "X-Axis");
        ImGui::TableNextColumn();
        ImGui::TextColored(UIHelpers::GetAccentColor(), "Y-Axis");

        // Enter thresholds
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##enter_x", &ctx.config.profile().deadband_enter_x, 0, 10, "Enter: %d px")) {
            ctx.config.profile().deadband_enter_x = std::max(0, std::min(ctx.config.profile().deadband_enter_x, ctx.config.profile().deadband_exit_x));
            SAVE_PROFILE();
            if (auto* p = gpa::PipelineManager::getInstance().getPipeline()) p->markPidConfigDirty();
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##enter_y", &ctx.config.profile().deadband_enter_y, 0, 10, "Enter: %d px")) {
            ctx.config.profile().deadband_enter_y = std::max(0, std::min(ctx.config.profile().deadband_enter_y, ctx.config.profile().deadband_exit_y));
            SAVE_PROFILE();
            if (auto* p = gpa::PipelineManager::getInstance().getPipeline()) p->markPidConfigDirty();
        }

        // Exit thresholds
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##exit_x", &ctx.config.profile().deadband_exit_x, 1, 20, "Exit: %d px")) {
            ctx.config.profile().deadband_exit_x = std::max(ctx.config.profile().deadband_exit_x, ctx.config.profile().deadband_enter_x);
            SAVE_PROFILE();
            if (auto* p = gpa::PipelineManager::getInstance().getPipeline()) p->markPidConfigDirty();
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##exit_y", &ctx.config.profile().deadband_exit_y, 1, 20, "Exit: %d px")) {
            ctx.config.profile().deadband_exit_y = std::max(ctx.config.profile().deadband_exit_y, ctx.config.profile().deadband_enter_y);
            SAVE_PROFILE();
            if (auto* p = gpa::PipelineManager::getInstance().getPipeline()) p->markPidConfigDirty();
        }

        ImGui::EndTable();
    }

    UIHelpers::CompactSpacer();
    ImGui::TextDisabled("Enter = start suppressing | Exit = stop suppressing");

    UIHelpers::CompactSpacer();
    if (UIHelpers::BeautifulButton("Reset to Defaults", ImVec2(-1, 0))) {
        ctx.config.profile().deadband_enter_x = 2;
        ctx.config.profile().deadband_exit_x  = 5;
        ctx.config.profile().deadband_enter_y = 2;
        ctx.config.profile().deadband_exit_y  = 5;
        SAVE_PROFILE();
        if (auto* p = gpa::PipelineManager::getInstance().getPipeline()) p->markPidConfigDirty();
    }

    UIHelpers::EndCard();
}
