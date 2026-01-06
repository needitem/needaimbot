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
static bool g_detecting_combo = false;  // true if detecting combo (multiple keys)
static std::vector<std::string> g_combo_keys;  // keys detected so far in combo
static float g_combo_start_time = 0.0f;  // time when combo detection started
static bool g_combo_had_keys = false;  // true if any keys were detected during combo mode
static bool g_waiting_for_release = false;  // wait for initial click to be released
static float g_last_key_added_time = 0.0f;  // time when last key was added to combo

// Function to detect all currently pressed keys
static std::vector<std::string> detect_all_pressed_keys()
{
    std::vector<std::string> pressed;
    for (int vk = 0x01; vk <= 0xFE; vk++)
    {
        if (GetAsyncKeyState(vk) & 0x8000)
        {
            for (const auto& pair : KeyCodes::key_code_map)
            {
                if (pair.second == vk)
                {
                    pressed.push_back(pair.first);
                    break;
                }
            }
        }
    }
    return pressed;
}

// Build combo string from keys (e.g., "LeftShift+A")
static std::string build_combo_string(const std::vector<std::string>& keys)
{
    std::string result;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) result += "+";
        result += keys[i];
    }
    return result;
}

// Compact inline hotkey row - shows key as tag/badge with detect and remove buttons
static void draw_hotkey_row_compact(std::vector<std::string>& hotkeys, const char* section_id, bool allow_empty = false)
{
    const float item_height = 24.0f;
    const float key_badge_min_width = 80.0f;
    const float button_width = 22.0f;
    const float spacing = 4.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 3.0f));

    float available_width = ImGui::GetContentRegionAvail().x;
    float current_x = 0.0f;

    for (size_t i = 0; i < hotkeys.size(); )
    {
        std::string& current_key_name = hotkeys[i];
        std::string unique_id = std::string(section_id) + "_" + std::to_string(i);
        ImGui::PushID(unique_id.c_str());

        bool is_detecting = (g_detecting_key_index == static_cast<int>(i) &&
                            g_detecting_section == section_id &&
                            g_detecting_hotkeys == &hotkeys);

        // Calculate item width for this key (key badge + edit button + remove button)
        float key_text_width = ImGui::CalcTextSize(current_key_name.c_str()).x + 12.0f;
        float key_width = (std::max)(key_badge_min_width, key_text_width);
        float total_item_width = key_width + button_width * 3 + spacing * 3;  // 3 buttons: key, edit, remove

        // Check if we need to wrap to next line
        if (current_x + total_item_width > available_width && current_x > 0)
        {
            current_x = 0.0f;
        }
        else if (current_x > 0)
        {
            ImGui::SameLine();
        }

        if (is_detecting)
        {
            // Detecting state - show pulsing button
            float pulse = (sinf((float)ImGui::GetTime() * 4.0f) + 1.0f) * 0.5f;
            ImVec4 detect_color = ImVec4(0.3f + pulse * 0.2f, 0.6f, 0.9f, 1.0f);  // Blue for detection

            ImGui::PushStyleColor(ImGuiCol_Button, detect_color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, detect_color);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, detect_color);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

            std::string button_text;
            float current_time = (float)ImGui::GetTime();
            if (g_waiting_for_release) {
                button_text = "Release...";
            } else if (g_combo_keys.empty()) {
                button_text = "Press key...";
            } else {
                // Show countdown to auto-save (1s after last key added)
                float time_since_last = current_time - g_last_key_added_time;
                float remaining = 1.0f - time_since_last;
                if (remaining < 0) remaining = 0;
                char countdown[32];
                snprintf(countdown, sizeof(countdown), " (%.1fs)", remaining);
                button_text = build_combo_string(g_combo_keys) + countdown;
            }

            if (ImGui::Button(button_text.c_str(), ImVec2(total_item_width + 40.0f, item_height)))
            {
                // Cancel detection
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
                g_detecting_combo = false;
                g_combo_keys.clear();
                g_combo_had_keys = false;
                g_waiting_for_release = false;
            }
            ImGui::PopStyleColor(4);

            // Key/combo detection logic
            auto pressed = detect_all_pressed_keys();

            if (g_waiting_for_release) {
                // Wait for user to release the initial left-click that started detection
                bool left_mouse_pressed = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
                if (!left_mouse_pressed) {
                    // Left mouse released - now start actual detection
                    g_waiting_for_release = false;
                    g_combo_start_time = current_time;
                    g_last_key_added_time = 0.0f;
                }
            }
            else {
                // Actual detection phase - add new keys to combo instantly
                for (const auto& key : pressed) {
                    // Check if this key is already in the combo
                    bool already_in_combo = false;
                    for (const auto& existing : g_combo_keys) {
                        if (existing == key) {
                            already_in_combo = true;
                            break;
                        }
                    }
                    // Add new key to combo
                    if (!already_in_combo) {
                        g_combo_keys.push_back(key);
                        g_last_key_added_time = current_time;
                        g_combo_had_keys = true;
                    }
                }

                // Auto-save 1 second after last key was added
                if (g_combo_had_keys && g_last_key_added_time > 0.0f &&
                    (current_time - g_last_key_added_time) >= 1.0f)
                {
                    // Timeout - save the combo
                    current_key_name = build_combo_string(g_combo_keys);
                    SAVE_PROFILE();
                    g_detecting_key_index = -1;
                    g_detecting_section = "";
                    g_detecting_hotkeys = nullptr;
                    g_detecting_combo = false;
                    g_combo_keys.clear();
                    g_combo_had_keys = false;
                    g_waiting_for_release = false;
                    g_last_key_added_time = 0.0f;
                }
            }
        }
        else
        {
            // Key badge - check if it's a combo (contains +)
            bool is_none = (current_key_name == "None");
            bool is_combo = (current_key_name.find('+') != std::string::npos);
            ImVec4 badge_color = is_none ?
                ImVec4(0.3f, 0.3f, 0.35f, 0.9f) :
                (is_combo ? ImVec4(0.2f, 0.5f, 0.7f, 0.9f) : UIHelpers::GetAccentColor(0.85f));
            ImVec4 text_color = is_none ?
                ImVec4(0.6f, 0.6f, 0.6f, 1.0f) :
                ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

            ImGui::PushStyleColor(ImGuiCol_Button, badge_color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(badge_color.x * 1.1f, badge_color.y * 1.1f, badge_color.z * 1.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(badge_color.x * 0.9f, badge_color.y * 0.9f, badge_color.z * 0.9f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, text_color);

            // Click on key badge to start detection (supports both single key and combos)
            if (ImGui::Button(current_key_name.c_str(), ImVec2(key_width, item_height)))
            {
                g_detecting_key_index = static_cast<int>(i);
                g_detecting_section = section_id;
                g_detecting_hotkeys = &hotkeys;
                g_detecting_combo = true;
                g_combo_keys.clear();
                g_combo_had_keys = false;
                g_waiting_for_release = true;  // Wait for left-click to be released first
                g_combo_start_time = (float)ImGui::GetTime();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Click to detect key/combo\nPress keys to add, auto-saves after 1s");
            ImGui::PopStyleColor(4);

            // Combo edit button (pencil icon)
            ImGui::SameLine(0, 2.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.4f, 0.5f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.5f, 0.6f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.3f, 0.4f, 1.0f));
            if (ImGui::Button(("E##edit_" + unique_id).c_str(), ImVec2(button_width, item_height)))
            {
                ImGui::OpenPopup(("combo_edit_popup_" + unique_id).c_str());
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Edit key combo (e.g., LeftMouseButton+RightMouseButton)");
            ImGui::PopStyleColor(3);

            // Combo edit popup
            if (ImGui::BeginPopup(("combo_edit_popup_" + unique_id).c_str()))
            {
                ImGui::Text("Enter key combo:");
                ImGui::SameLine();
                UIHelpers::HelpMarker("Use + to combine keys\nExample: LeftMouseButton+RightMouseButton");

                static char combo_buf[256] = "";
                if (ImGui::IsWindowAppearing()) {
                    strncpy(combo_buf, current_key_name.c_str(), sizeof(combo_buf) - 1);
                    combo_buf[sizeof(combo_buf) - 1] = '\0';
                }

                ImGui::SetNextItemWidth(250.0f);
                if (ImGui::InputText("##combo_input", combo_buf, sizeof(combo_buf), ImGuiInputTextFlags_EnterReturnsTrue))
                {
                    current_key_name = combo_buf;
                    SAVE_PROFILE();
                    ImGui::CloseCurrentPopup();
                }

                ImGui::SameLine();
                if (ImGui::Button("OK"))
                {
                    current_key_name = combo_buf;
                    SAVE_PROFILE();
                    ImGui::CloseCurrentPopup();
                }

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Common keys:");
                if (ImGui::Selectable("LeftMouseButton")) { strncpy(combo_buf, "LeftMouseButton", sizeof(combo_buf)); }
                if (ImGui::Selectable("RightMouseButton")) { strncpy(combo_buf, "RightMouseButton", sizeof(combo_buf)); }
                if (ImGui::Selectable("LeftMouseButton+RightMouseButton")) { strncpy(combo_buf, "LeftMouseButton+RightMouseButton", sizeof(combo_buf)); }
                if (ImGui::Selectable("LeftShift+LeftMouseButton")) { strncpy(combo_buf, "LeftShift+LeftMouseButton", sizeof(combo_buf)); }
                if (ImGui::Selectable("LeftControl+LeftMouseButton")) { strncpy(combo_buf, "LeftControl+LeftMouseButton", sizeof(combo_buf)); }

                ImGui::EndPopup();
            }

            // Remove button (X)
            ImGui::SameLine(0, spacing);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.2f, 0.2f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));

            if (ImGui::Button(("x##rm_" + unique_id).c_str(), ImVec2(button_width, item_height)))
            {
                if (!allow_empty && hotkeys.size() <= 1)
                {
                    hotkeys[0] = "None";
                }
                else
                {
                    hotkeys.erase(hotkeys.begin() + i);
                }
                SAVE_PROFILE();
                ImGui::PopStyleColor(3);
                ImGui::PopID();
                ImGui::PopStyleVar(2);
                // Restart the function to handle the removed item
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 3.0f));
                current_x = 0.0f;
                continue;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Remove this key");
            ImGui::PopStyleColor(3);
        }

        current_x += total_item_width + spacing;
        ImGui::PopID();
        ++i;
    }

    // Add key button (+ icon)
    if (current_x > 0 && current_x + 60.0f <= available_width)
    {
        ImGui::SameLine();
    }

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.2f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));

    if (ImGui::Button(("+##add_" + std::string(section_id)).c_str(), ImVec2(button_width, item_height)))
    {
        hotkeys.push_back("None");
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Add another key");
    ImGui::PopStyleColor(3);

    ImGui::PopStyleVar(2);
}

// Hotkey table row - single row in the hotkeys table
static void draw_hotkey_table_row(const char* label, std::vector<std::string>& hotkeys,
                                   const char* section_id, const char* tooltip = nullptr,
                                   bool allow_empty = false)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    // Label column
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    if (tooltip && ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("%s", tooltip);
    }

    ImGui::TableNextColumn();

    // Keys column
    draw_hotkey_row_compact(hotkeys, section_id, allow_empty);
}

static void draw_aiming_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Aiming Hotkeys");

    if (ImGui::BeginTable("##aiming_hotkeys_table", 2,
        ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableSetupColumn("Keys", ImGuiTableColumnFlags_WidthStretch);

        draw_hotkey_table_row("Aimbot Activation", ctx.config.button_targeting, "targeting_keys",
                             "Keys that activate the aimbot when held");

        draw_hotkey_table_row("Auto Shoot", ctx.config.button_auto_shoot, "auto_shoot_keys",
                             "Automatically shoot when targeting enemies");

        draw_hotkey_table_row("Disable Upward Aim", ctx.config.button_disable_upward_aim, "disable_upward_keys",
                             "Prevent aimbot from aiming upward while held");

        draw_hotkey_table_row("Single Shot", ctx.config.button_single_shot, "single_shot_keys",
                             "One capture and one mouse move per keypress");

        draw_hotkey_table_row("No Recoil", ctx.config.button_norecoil, "norecoil_keys",
                             "Compensate recoil using weapon profile settings while held");

        ImGui::EndTable();
    }

    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    UIHelpers::BeginCard("Triggerbot Area");
    ImGui::Text("Area Size Multiplier");
    if (UIHelpers::EnhancedSliderFloat("##triggerbot_area", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f",
                                      "Central screen area where Triggerbot activates.\nSmaller = larger area, Larger = smaller area")) {
        SAVE_PROFILE();
    }
    UIHelpers::EndCard();
}

void draw_buttons()
{
    auto& ctx = AppContext::getInstance();

    // System Hotkeys Card
    UIHelpers::BeginCard("System Hotkeys");

    if (ImGui::BeginTable("##system_hotkeys_table", 2,
        ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Keys", ImGuiTableColumnFlags_WidthStretch);

        draw_hotkey_table_row("Targeting", ctx.config.button_targeting, "targeting",
                             "Keys that activate the aimbot when held");

        draw_hotkey_table_row("Exit App", ctx.config.button_exit, "exit",
                             "Keys that completely exit the application");

        draw_hotkey_table_row("Pause", ctx.config.button_pause, "pause",
                             "Keys that temporarily pause the aimbot");

        draw_hotkey_table_row("Reload Config", ctx.config.button_reload_config, "reload_config",
                             "Keys that reload the configuration file");

        draw_hotkey_table_row("Toggle Overlay", ctx.config.button_open_overlay, "overlay",
                             "Keys that show/hide this overlay", true);

        ImGui::EndTable();
    }

    UIHelpers::EndCard();

    UIHelpers::Spacer();

    // Aiming Settings
    draw_aiming_settings();
}
