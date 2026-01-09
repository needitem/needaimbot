#include "draw_settings.h"
#include "../AppContext.h"
#include "ui_helpers.h"
#include "imgui.h"
#include <algorithm>

static void draw_profile_selector()
{
    auto& ctx = AppContext::getInstance();
    auto& profiles = ctx.config.profile().input_profiles;
    int& active_idx = ctx.config.profile().active_input_profile_index;

    // Ensure valid index
    if (active_idx < 0 || active_idx >= static_cast<int>(profiles.size())) {
        active_idx = 0;
    }

    UIHelpers::BeginCard("Input Profile");

    UIHelpers::BeautifulText("Select or manage input profiles.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();

    if (!profiles.empty()) {
        // Profile selector row
        if (ImGui::BeginTable("##profile_selector", 3, ImGuiTableFlags_NoBordersInBody)) {
            ImGui::TableSetupColumn("Combo", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Add", ImGuiTableColumnFlags_WidthFixed, 30.0f);
            ImGui::TableSetupColumn("Del", ImGuiTableColumnFlags_WidthFixed, 30.0f);

            ImGui::TableNextRow();

            // Profile combo
            ImGui::TableNextColumn();
            const char* preview = profiles[active_idx].profile_name.c_str();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##ProfileCombo", preview)) {
                for (int i = 0; i < static_cast<int>(profiles.size()); i++) {
                    bool is_selected = (active_idx == i);
                    if (ImGui::Selectable(profiles[i].profile_name.c_str(), is_selected)) {
                        active_idx = i;
                        SAVE_PROFILE();
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            // Add button
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.2f, 0.9f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.3f, 1.0f));
            if (ImGui::Button("+##add", ImVec2(-1, 0))) {
                InputProfile new_profile("New Profile", 3.0f, 1.0f);
                profiles.push_back(new_profile);
                active_idx = static_cast<int>(profiles.size()) - 1;
                SAVE_PROFILE();
            }
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Add new profile");

            // Delete button
            ImGui::TableNextColumn();
            ImGui::BeginDisabled(profiles.size() <= 1);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.2f, 0.2f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("-##del", ImVec2(-1, 0))) {
                profiles.erase(profiles.begin() + active_idx);
                if (active_idx >= static_cast<int>(profiles.size())) {
                    active_idx = static_cast<int>(profiles.size()) - 1;
                }
                SAVE_PROFILE();
            }
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete current profile");
            ImGui::EndDisabled();

            ImGui::EndTable();
        }

        UIHelpers::CompactSpacer();

        // Profile name edit
        InputProfile& profile = profiles[active_idx];
        char name_buf[64];
        strncpy(name_buf, profile.profile_name.c_str(), sizeof(name_buf) - 1);
        name_buf[sizeof(name_buf) - 1] = '\0';

        ImGui::Text("Profile Name");
        ImGui::SameLine();
        UIHelpers::HelpMarker("Name for this input profile");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##profile_name", name_buf, sizeof(name_buf))) {
            profile.profile_name = name_buf;
            SAVE_PROFILE();
        }
    }
    else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "No input profiles. Click + to add one.");
    }

    UIHelpers::EndCard();
}

static void draw_stabilizer_settings()
{
    auto& ctx = AppContext::getInstance();
    auto& profiles = ctx.config.profile().input_profiles;
    int active_idx = ctx.config.profile().active_input_profile_index;

    if (profiles.empty() || active_idx < 0 || active_idx >= static_cast<int>(profiles.size())) {
        return;
    }

    InputProfile& profile = profiles[active_idx];

    UIHelpers::BeginCard("Stabilizer Settings");

    UIHelpers::BeautifulText("Adjust input stabilization strength and timing.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();

    // Stabilizer settings table
    if (ImGui::BeginTable("##stabilizer_settings", 2, ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 130.0f);
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);

        // Base Strength
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Base Strength");
        ImGui::SameLine();
        UIHelpers::HelpMarker("Base stabilization strength (pixels per tick)");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float base_strength = profile.base_strength;
        if (ImGui::SliderFloat("##base_strength", &base_strength, 0.0f, 20.0f, "%.1f px")) {
            profile.base_strength = base_strength;
            SAVE_PROFILE();
        }

        // Fire Rate Multiplier
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Rate Multiplier");
        ImGui::SameLine();
        UIHelpers::HelpMarker("Multiplier for rate adjustment");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float fire_mult = profile.fire_rate_multiplier;
        if (ImGui::SliderFloat("##fire_mult", &fire_mult, 0.1f, 5.0f, "%.2f")) {
            profile.fire_rate_multiplier = fire_mult;
            SAVE_PROFILE();
        }

        // Interval
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Interval");
        ImGui::SameLine();
        UIHelpers::HelpMarker("Time interval between stabilization ticks");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float interval_ms = profile.interval_ms;
        if (ImGui::SliderFloat("##interval_ms", &interval_ms, 1.0f, 100.0f, "%.1f ms")) {
            profile.interval_ms = interval_ms;
            SAVE_PROFILE();
        }

        ImGui::EndTable();
    }

    UIHelpers::EndCard();
}

static void draw_timing_settings()
{
    auto& ctx = AppContext::getInstance();
    auto& profiles = ctx.config.profile().input_profiles;
    int active_idx = ctx.config.profile().active_input_profile_index;

    if (profiles.empty() || active_idx < 0 || active_idx >= static_cast<int>(profiles.size())) {
        return;
    }

    InputProfile& profile = profiles[active_idx];

    UIHelpers::BeginCard("Timing Settings");

    UIHelpers::BeautifulText("Control when stabilization starts and stops.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();

    if (ImGui::BeginTable("##timing_table", 2, ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 130.0f);
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);

        // Start Delay
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Start Delay");
        ImGui::SameLine();
        UIHelpers::HelpMarker("Delay before starting stabilization");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        int start_delay = profile.start_delay_ms;
        if (ImGui::SliderInt("##start_delay", &start_delay, 0, 500, "%d ms")) {
            profile.start_delay_ms = start_delay;
            SAVE_PROFILE();
        }

        // End Delay
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("End Delay");
        ImGui::SameLine();
        UIHelpers::HelpMarker("Time to continue stabilization after releasing");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        int end_delay = profile.end_delay_ms;
        if (ImGui::SliderInt("##end_delay", &end_delay, 0, 500, "%d ms")) {
            profile.end_delay_ms = end_delay;
            SAVE_PROFILE();
        }

        ImGui::EndTable();
    }

    UIHelpers::EndCard();
}

static void draw_stabilizer_status()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Stabilizer");

    UIHelpers::BeautifulText("Input stabilization using profile settings.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();

    // Current key display
    std::string key_display;
    if (ctx.config.global().button_stabilizer.empty() || ctx.config.global().button_stabilizer[0] == "None") {
        key_display = "None";
    } else {
        for (size_t i = 0; i < ctx.config.global().button_stabilizer.size(); ++i) {
            if (i > 0) key_display += " + ";
            key_display += ctx.config.global().button_stabilizer[i];
        }
    }
    ImGui::Text("Hotkey: %s", key_display.c_str());
    ImGui::SameLine();
    UIHelpers::HelpMarker("Configure in Buttons tab");
    UIHelpers::CompactSpacer();

    // Status indicator
    bool is_active = ctx.stabilizer_active.load();
    if (is_active) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Status: ACTIVE");
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Status: Inactive");
    }
    UIHelpers::CompactSpacer();

    // Show current effective settings
    auto* profile = ctx.config.getCurrentInputProfile();
    if (profile) {
        float strength = profile->base_strength;
        int scope = ctx.config.profile().active_scope_magnification;

        switch (scope) {
            case 1: strength *= profile->scope_mult_1x; break;
            case 2: strength *= profile->scope_mult_2x; break;
            case 3: strength *= profile->scope_mult_3x; break;
            case 4: strength *= profile->scope_mult_4x; break;
            case 6: strength *= profile->scope_mult_6x; break;
            case 8: strength *= profile->scope_mult_8x; break;
            default: strength *= profile->scope_mult_1x; break;
        }
        strength *= profile->fire_rate_multiplier;

        ImGui::TextColored(UIHelpers::GetAccentColor(0.9f), "Current Profile: %s", profile->profile_name.c_str());
        ImGui::Text("Effective Strength: %.1f px", strength);
        ImGui::Text("Interval: %.1f ms", profile->interval_ms);
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "No input profile selected");
    }

    UIHelpers::EndCard();
}


static void draw_scope_multipliers()
{
    auto& ctx = AppContext::getInstance();
    auto& profiles = ctx.config.profile().input_profiles;
    int active_idx = ctx.config.profile().active_input_profile_index;

    if (profiles.empty() || active_idx < 0 || active_idx >= static_cast<int>(profiles.size())) {
        return;
    }

    InputProfile& profile = profiles[active_idx];

    UIHelpers::BeginCard("Scope Multipliers");

    UIHelpers::BeautifulText("Adjust stabilization based on scope magnification.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();

    // Current scope indicator
    ImGui::TextColored(UIHelpers::GetWarningColor(),
        "Current Scope: %dx", ctx.config.profile().active_scope_magnification);
    UIHelpers::CompactSpacer();

    // Scope multipliers in a clean grid
    if (ImGui::BeginTable("##scope_table", 4, ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupColumn("1x-2x");
        ImGui::TableSetupColumn("");
        ImGui::TableSetupColumn("3x-4x");
        ImGui::TableSetupColumn("");

        // Row 1: 1x and 3x
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("1x");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float mult_1x = profile.scope_mult_1x;
        if (ImGui::SliderFloat("##m1x", &mult_1x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_1x = mult_1x;
            SAVE_PROFILE();
        }

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("3x");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float mult_3x = profile.scope_mult_3x;
        if (ImGui::SliderFloat("##m3x", &mult_3x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_3x = mult_3x;
            SAVE_PROFILE();
        }

        // Row 2: 2x and 4x
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("2x");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float mult_2x = profile.scope_mult_2x;
        if (ImGui::SliderFloat("##m2x", &mult_2x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_2x = mult_2x;
            SAVE_PROFILE();
        }

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("4x");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float mult_4x = profile.scope_mult_4x;
        if (ImGui::SliderFloat("##m4x", &mult_4x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_4x = mult_4x;
            SAVE_PROFILE();
        }

        // Row 3: 6x and 8x
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("6x");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float mult_6x = profile.scope_mult_6x;
        if (ImGui::SliderFloat("##m6x", &mult_6x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_6x = mult_6x;
            SAVE_PROFILE();
        }

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("8x");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        float mult_8x = profile.scope_mult_8x;
        if (ImGui::SliderFloat("##m8x", &mult_8x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_8x = mult_8x;
            SAVE_PROFILE();
        }

        ImGui::EndTable();
    }

    UIHelpers::CompactSpacer();

    // Reset button
    if (UIHelpers::BeautifulButton("Reset All to 1.0", ImVec2(-1, 0))) {
        profile.scope_mult_1x = 1.0f;
        profile.scope_mult_2x = 1.0f;
        profile.scope_mult_3x = 1.0f;
        profile.scope_mult_4x = 1.0f;
        profile.scope_mult_6x = 1.0f;
        profile.scope_mult_8x = 1.0f;
        SAVE_PROFILE();
    }

    UIHelpers::EndCard();
}

void draw_stabilizer()
{
    draw_stabilizer_status();
    UIHelpers::Spacer();

    draw_profile_selector();
    UIHelpers::Spacer();

    draw_stabilizer_settings();
    UIHelpers::Spacer();

    draw_timing_settings();
    UIHelpers::Spacer();

    draw_scope_multipliers();
}
