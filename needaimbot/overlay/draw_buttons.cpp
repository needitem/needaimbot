#include "../core/windows_headers.h"

#include "needaimbot.h"
#include "overlay.h"
#include "AppContext.h"
#include "ui_helpers.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include "../keyboard/keycodes.h"

// State for key detection
static int g_detecting_key_index = -1;
static std::string g_detecting_section;
static std::vector<std::string>* g_detecting_hotkeys = nullptr;

// Function to detect pressed key
static std::string detect_pressed_key()
{
    // Check all keyboard keys
    for (int vk = 0x01; vk <= 0xFE; vk++)
    {
        if (GetAsyncKeyState(vk) & 0x8000)
        {
            // Find the key name for this virtual key code
            for (const auto& pair : KeyCodes::key_code_map)
            {
                if (pair.second == vk)
                {
                    return pair.first;
                }
            }
        }
    }
    return "";
}

// Unified hotkey section rendering - handles both simple and card-wrapped layouts
static void draw_hotkey_section(const char* title, std::vector<std::string>& hotkeys, const char* add_id,
                                const char* description = nullptr, bool use_card = false, bool allow_empty = false)
{
    if (use_card) {
        UIHelpers::BeginCard(title);
        if (description) {
            UIHelpers::BeautifulText(description, UIHelpers::GetAccentColor(0.8f));
            UIHelpers::CompactSpacer();
        }
    }

    for (size_t i = 0; i < hotkeys.size(); )
    {
        std::string& current_key_name = hotkeys[i];
        auto it = std::find(key_names.begin(), key_names.end(), current_key_name);
        int current_index = (it != key_names.end()) ? static_cast<int>(std::distance(key_names.begin(), it)) : 0;

        std::string unique_id = std::string(add_id) + "_" + std::to_string(i);
        ImGui::PushID(unique_id.c_str());

        float remove_button_width = use_card ? 60.0f : 80.0f;
        float detect_button_width = use_card ? 80.0f : 120.0f;
        float available_width = ImGui::GetContentRegionAvail().x;
        float combo_width = available_width - remove_button_width - detect_button_width - ImGui::GetStyle().ItemSpacing.x * 2;

        bool is_detecting = (g_detecting_key_index == static_cast<int>(i) && g_detecting_section == add_id && g_detecting_hotkeys == &hotkeys);

        if (is_detecting)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.5f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.6f, 0.6f, 1.0f));

            if (ImGui::Button("Press any key...", ImVec2(combo_width + detect_button_width + ImGui::GetStyle().ItemSpacing.x, 0)))
            {
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
            }
            ImGui::PopStyleColor(3);

            std::string detected_key = detect_pressed_key();
            if (!detected_key.empty() && detected_key != "LeftMouseButton")
            {
                current_key_name = detected_key;
                SAVE_PROFILE();
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
            }
        }
        else
        {
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
                SAVE_PROFILE();
            }
            ImGui::PopStyleColor(6);

            ImGui::SameLine();
            if (UIHelpers::BeautifulButton(("Detect##" + unique_id).c_str(), ImVec2(detect_button_width, 0)))
            {
                g_detecting_key_index = static_cast<int>(i);
                g_detecting_section = add_id;
                g_detecting_hotkeys = &hotkeys;
            }
        }

        ImGui::SameLine();
        if (UIHelpers::BeautifulButton(("Remove##" + unique_id).c_str(), ImVec2(remove_button_width, 0)))
        {
            if (!allow_empty && hotkeys.size() <= 1)
            {
                hotkeys[0] = "None";
                SAVE_PROFILE();
                ImGui::PopID();
                continue;
            }
            hotkeys.erase(hotkeys.begin() + i);
            SAVE_PROFILE();
            ImGui::PopID();
            continue;
        }

        ImGui::PopID();
        ++i;
    }

    UIHelpers::CompactSpacer();
    if (UIHelpers::BeautifulButton(("Add Key##" + std::string(add_id)).c_str(), ImVec2(use_card ? -FLT_MIN : -1, 0)))
    {
        hotkeys.push_back("None");
        SAVE_PROFILE();
    }

    if (use_card) {
        UIHelpers::EndCard();
    }
}

static void draw_aiming_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Aiming Controls");

    UIHelpers::SettingsSubHeader("Targeting Controls");
    UIHelpers::BeautifulText("Configure keys for aimbot activation", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    draw_hotkey_section("Aimbot Activation Keys", ctx.config.button_targeting, "targeting_keys");

    UIHelpers::Spacer(12.0f);

    UIHelpers::SettingsSubHeader("Auto Shooting");
    UIHelpers::BeautifulText("Automatically shoot when targeting enemies", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    draw_hotkey_section("Auto Shoot Keys", ctx.config.button_auto_shoot, "auto_shoot_keys");

    UIHelpers::Spacer(12.0f);

    UIHelpers::SettingsSubHeader("Movement Restrictions");
    UIHelpers::BeautifulText("Control when aimbot should avoid certain movements", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    draw_hotkey_section("Disable Upward Aim Keys", ctx.config.button_disable_upward_aim, "disable_upward_keys");

    UIHelpers::Spacer(12.0f);

    UIHelpers::SettingsSubHeader("Single Shot Trigger");
    UIHelpers::BeautifulText("Perform one capture and one mouse move per keypress", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    draw_hotkey_section("Single Shot Keys", ctx.config.button_single_shot, "single_shot_keys");

    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    UIHelpers::BeginCard("Triggerbot Configuration");
    UIHelpers::BeautifulText("Configure the screen area where triggerbot activates", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);

    ImGui::Text("Area Size Multiplier");
    if (UIHelpers::EnhancedSliderFloat("##triggerbot_area", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f",
                                      "Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area\nLarger value = smaller area\n(1.0 = default area)")) {
        SAVE_PROFILE();
    }
    UIHelpers::EndCard();
}

void draw_buttons()
{
    auto& ctx = AppContext::getInstance();

    draw_hotkey_section("Aimbot Activation Keys", ctx.config.button_targeting, "targeting",
                       "Keys that activate the aimbot when held", true);
    UIHelpers::Spacer();

    draw_hotkey_section("Exit Keys", ctx.config.button_exit, "exit",
                       "Keys that completely exit the application", true);
    UIHelpers::Spacer();

    draw_hotkey_section("Pause Keys", ctx.config.button_pause, "pause",
                       "Keys that temporarily pause the aimbot", true);
    UIHelpers::Spacer();

    draw_hotkey_section("Reload Config Keys", ctx.config.button_reload_config, "reload_config",
                       "Keys that reload the configuration file", true);
    UIHelpers::Spacer();

    draw_hotkey_section("Overlay Toggle Keys", ctx.config.button_open_overlay, "overlay",
                       "Keys that show/hide this overlay", true, true);
    UIHelpers::Spacer();

    draw_aiming_settings();
}
