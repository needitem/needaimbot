#include "draw_settings.h"
#include "../AppContext.h"
#include "ui_helpers.h"
#include "imgui.h"
#include <algorithm>

void draw_recoil()
{
    auto& ctx = AppContext::getInstance();
    auto& profiles = ctx.config.weapon_profiles;
    int& active_idx = ctx.config.active_weapon_profile_index;

    // Ensure valid index
    if (active_idx < 0 || active_idx >= static_cast<int>(profiles.size())) {
        active_idx = 0;
    }

    // Profile selector
    UIHelpers::BeginCard("Weapon Profile");

    if (!profiles.empty()) {
        // Combo box for profile selection
        const char* preview = profiles[active_idx].weapon_name.c_str();
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginCombo("Active Profile", preview)) {
            for (int i = 0; i < static_cast<int>(profiles.size()); i++) {
                bool is_selected = (active_idx == i);
                if (ImGui::Selectable(profiles[i].weapon_name.c_str(), is_selected)) {
                    active_idx = i;
                    ctx.config.current_weapon_name = profiles[i].weapon_name;
                    SAVE_PROFILE();
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SameLine();

        // Add new profile button
        if (ImGui::Button("+##add_profile")) {
            WeaponRecoilProfile new_profile("New Weapon", 3.0f, 1.0f);
            profiles.push_back(new_profile);
            active_idx = static_cast<int>(profiles.size()) - 1;
            SAVE_PROFILE();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Add new profile");

        ImGui::SameLine();

        // Delete profile button (keep at least one)
        ImGui::BeginDisabled(profiles.size() <= 1);
        if (ImGui::Button("-##del_profile")) {
            profiles.erase(profiles.begin() + active_idx);
            if (active_idx >= static_cast<int>(profiles.size())) {
                active_idx = static_cast<int>(profiles.size()) - 1;
            }
            SAVE_PROFILE();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete current profile");
        ImGui::EndDisabled();
    }

    UIHelpers::EndCard();

    if (profiles.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "No weapon profiles. Click + to add one.");
        return;
    }

    WeaponRecoilProfile& profile = profiles[active_idx];

    UIHelpers::CompactSpacer();

    // Profile name edit
    UIHelpers::BeginCard("Profile Settings");

    char name_buf[64];
    strncpy(name_buf, profile.weapon_name.c_str(), sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::InputText("Weapon Name", name_buf, sizeof(name_buf))) {
        profile.weapon_name = name_buf;
        ctx.config.current_weapon_name = name_buf;
        SAVE_PROFILE();
    }

    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    // Recoil strength settings
    UIHelpers::BeginCard("Recoil Compensation");

    float base_strength = profile.base_strength;
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::SliderFloat("Base Strength", &base_strength, 0.0f, 20.0f, "%.1f")) {
        profile.base_strength = base_strength;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    UIHelpers::InfoTooltip("Base recoil compensation strength (pixels per tick)");

    float fire_mult = profile.fire_rate_multiplier;
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::SliderFloat("Fire Rate Mult", &fire_mult, 0.1f, 5.0f, "%.2f")) {
        profile.fire_rate_multiplier = fire_mult;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    UIHelpers::InfoTooltip("Multiplier for fire rate adjustment");

    float recoil_ms = profile.recoil_ms;
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::SliderFloat("Recoil Interval (ms)", &recoil_ms, 1.0f, 100.0f, "%.1f")) {
        profile.recoil_ms = recoil_ms;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    UIHelpers::InfoTooltip("Time interval between recoil compensation ticks");

    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    // Timing settings
    UIHelpers::BeginCard("Timing");

    int start_delay = profile.start_delay_ms;
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("Start Delay (ms)", &start_delay)) {
        if (start_delay < 0) start_delay = 0;
        profile.start_delay_ms = start_delay;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    UIHelpers::InfoTooltip("Delay before starting recoil compensation after firing");

    int end_delay = profile.end_delay_ms;
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("End Delay (ms)", &end_delay)) {
        if (end_delay < 0) end_delay = 0;
        profile.end_delay_ms = end_delay;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    UIHelpers::InfoTooltip("Time to continue recoil compensation after releasing trigger");

    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    // Scope multipliers
    UIHelpers::BeginCard("Scope Multipliers");

    ImGui::Text("Adjust recoil compensation based on scope magnification:");
    ImGui::Spacing();

    ImGui::PushItemWidth(100.0f);

    if (ImGui::BeginTable("ScopeMultTable", 3, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Scope", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Multiplier", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);

        // 1x
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("1x");
        ImGui::TableNextColumn();
        float mult_1x = profile.scope_mult_1x;
        if (ImGui::SliderFloat("##scope1x", &mult_1x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_1x = mult_1x;
            SAVE_PROFILE();
        }

        // 2x
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("2x");
        ImGui::TableNextColumn();
        float mult_2x = profile.scope_mult_2x;
        if (ImGui::SliderFloat("##scope2x", &mult_2x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_2x = mult_2x;
            SAVE_PROFILE();
        }

        // 3x
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("3x");
        ImGui::TableNextColumn();
        float mult_3x = profile.scope_mult_3x;
        if (ImGui::SliderFloat("##scope3x", &mult_3x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_3x = mult_3x;
            SAVE_PROFILE();
        }

        // 4x
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("4x");
        ImGui::TableNextColumn();
        float mult_4x = profile.scope_mult_4x;
        if (ImGui::SliderFloat("##scope4x", &mult_4x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_4x = mult_4x;
            SAVE_PROFILE();
        }

        // 6x
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("6x");
        ImGui::TableNextColumn();
        float mult_6x = profile.scope_mult_6x;
        if (ImGui::SliderFloat("##scope6x", &mult_6x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_6x = mult_6x;
            SAVE_PROFILE();
        }

        // 8x
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("8x");
        ImGui::TableNextColumn();
        float mult_8x = profile.scope_mult_8x;
        if (ImGui::SliderFloat("##scope8x", &mult_8x, 0.1f, 3.0f, "%.2f")) {
            profile.scope_mult_8x = mult_8x;
            SAVE_PROFILE();
        }

        ImGui::EndTable();
    }

    ImGui::PopItemWidth();

    // Current scope indicator
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
        "Current Scope: %dx", ctx.config.active_scope_magnification);

    UIHelpers::EndCard();
}
