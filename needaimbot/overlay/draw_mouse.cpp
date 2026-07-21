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

    UIHelpers::BeginCard("Aim Controller (Nonlinear P+D + One Euro)");

    UIHelpers::BeautifulText("Nonlinear P+D with a One Euro center filter and detection-gap coast (ported from the 2pc build).", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::BeautifulText("Kp = response, Softness = gentleness near target, Kd = damping. (Integral removed: it amplifies dead-time lag.)", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
    UIHelpers::CompactSpacer();

    if (ImGui::BeginTabBar("AimTabs")) {
        if (ImGui::BeginTabItem("Proportional (P)")) {
            UIHelpers::SettingsSubHeader("Proportional Gain (Kp)");
            UIHelpers::BeautifulText("Response speed. Higher = faster aiming, but may overshoot.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::BeginTable("KpTable", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KpX", &ctx.config.profile().aim_kp_x, 0.0f, 2.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KpY", &ctx.config.profile().aim_kp_y, 0.0f, 2.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            UIHelpers::HelpMarker("Recommended: 0.4-0.7.");

            UIHelpers::CompactSpacer();
            UIHelpers::SettingsSubHeader("Softness (nonlinear knee)");
            UIHelpers::BeautifulText("Higher = gentler correction near target (suppresses detector-noise buzz).", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::BeginTable("SoftTable", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##SoftX", &ctx.config.profile().aim_softness_x, 1.0f, 40.0f, "%.1f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##SoftY", &ctx.config.profile().aim_softness_y, 1.0f, 40.0f, "%.1f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            UIHelpers::HelpMarker("Recommended: 8-14. Lower = snappier but noisier near target.");
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Derivative (D)")) {
            UIHelpers::SettingsSubHeader("Derivative Gain (Kd)");
            UIHelpers::BeautifulText("Damps oscillation on the smoothed error rate. Higher = smoother.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::BeginTable("KdTable", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("X-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Y-Axis", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KdX", &ctx.config.profile().aim_kd_x, 0.0f, 1.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(-1);
                if (ImGui::SliderFloat("##KdY", &ctx.config.profile().aim_kd_y, 0.0f, 1.0f, "%.3f")) {
                    SAVE_PROFILE();
                }
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            UIHelpers::HelpMarker("Recommended: 0.15-0.35.");
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("One Euro Filter")) {
            UIHelpers::SettingsSubHeader("Center Jitter Filter");
            UIHelpers::BeautifulText("Adaptive low-pass on the target center: heavy smoothing at rest, light when moving fast.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::Checkbox("Enabled##OneEuro", &ctx.config.profile().oneeuro_enabled)) {
                SAVE_PROFILE();
            }
            ImGui::PushItemWidth(-1);
            if (ImGui::SliderFloat("##MinCutoff", &ctx.config.profile().oneeuro_min_cutoff, 0.01f, 1.0f, "min_cutoff %.3f")) {
                SAVE_PROFILE();
            }
            UIHelpers::HelpMarker("Base cutoff at rest. Lower = smoother / more lag. Default 0.10.");
            if (ImGui::SliderFloat("##Beta", &ctx.config.profile().oneeuro_beta, 0.0f, 0.5f, "beta %.3f")) {
                SAVE_PROFILE();
            }
            UIHelpers::HelpMarker("Speed coefficient. Higher = less lag when moving fast. Default 0.02.");
            if (ImGui::SliderFloat("##DCutoff", &ctx.config.profile().oneeuro_dcutoff, 0.1f, 2.0f, "dcutoff %.2f")) {
                SAVE_PROFILE();
            }
            UIHelpers::HelpMarker("Derivative cutoff for the speed estimate. Default 0.50.");
            ImGui::PopItemWidth();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Coast / Limits")) {
            UIHelpers::SettingsSubHeader("Detection-gap Coast");
            UIHelpers::BeautifulText("Bridges brief detection gaps by gliding on the last drift instead of freezing.", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            UIHelpers::CompactSpacer();

            if (ImGui::Checkbox("Coast enabled", &ctx.config.profile().coast_enabled)) {
                SAVE_PROFILE();
            }
            ImGui::PushItemWidth(-1);
            if (ImGui::SliderFloat("##CoastDecay", &ctx.config.profile().coast_decay, 0.0f, 1.0f, "decay %.2f")) {
                SAVE_PROFILE();
            }
            UIHelpers::HelpMarker("Per-missed-frame glide decay (nearer 1 = glides longer). Default 0.85.");
            if (ImGui::SliderInt("##Persist", &ctx.config.profile().track_persistence_frames, 0, 15, "persistence %d frames")) {
                SAVE_PROFILE();
            }
            UIHelpers::HelpMarker("Missed frames to bridge before dropping the target. Default 3.");
            ImGui::PopItemWidth();

            UIHelpers::CompactSpacer();
            UIHelpers::SettingsSubHeader("Per-frame Max Step");
            ImGui::PushItemWidth(-1);
            if (ImGui::SliderFloat("##MaxStep", &ctx.config.profile().aim_max_step, 0.0f, 100.0f, "%.0f px")) {
                SAVE_PROFILE();
            }
            ImGui::PopItemWidth();
            UIHelpers::HelpMarker("Caps the per-frame move so a large error is crossed smoothly. 0 = unbounded. Default 30px.");
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
