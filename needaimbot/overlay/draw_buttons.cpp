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

static void draw_hotkey_section(const char* title, std::vector<std::string>& hotkeys, const char* add_id)
{
    // Don't use BeginChild here - it can cause spacing issues
    
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
        
        // Use unique ID combining section name and index
        std::string unique_id = std::string(add_id) + "_" + std::to_string(i);
        ImGui::PushID(unique_id.c_str());
        
        // Calculate proper button width
        float remove_button_width = 80.0f;
        float detect_button_width = 120.0f;
        float available_width = ImGui::GetContentRegionAvail().x;
        float combo_width = available_width - remove_button_width - detect_button_width - ImGui::GetStyle().ItemSpacing.x * 2;
        
        // Check if we're detecting for this key
        bool is_detecting = (g_detecting_key_index == static_cast<int>(i) && g_detecting_section == add_id && g_detecting_hotkeys == &hotkeys);
        
        if (is_detecting)
        {
            // Show "Press any key..." button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.5f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.6f, 0.6f, 1.0f));
            
            if (ImGui::Button("Press any key...", ImVec2(combo_width + detect_button_width + ImGui::GetStyle().ItemSpacing.x, 0)))
            {
                // Cancel detection if clicked
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
            }
            ImGui::PopStyleColor(3);
            
            // Check for pressed key
            std::string detected_key = detect_pressed_key();
            if (!detected_key.empty() && detected_key != "LeftMouseButton") // Ignore left click which would cancel
            {
                current_key_name = detected_key;
                AppContext::getInstance().config.saveConfig();
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
            }
        }
        else
        {
            // Enhanced key selector with better styling and wider width
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
                AppContext::getInstance().config.saveConfig();
            }
            ImGui::PopStyleColor(6);
            
            ImGui::SameLine();
            std::string detect_button_label = "Detect##" + unique_id;
            if (UIHelpers::BeautifulButton(detect_button_label.c_str(), ImVec2(detect_button_width, 0)))
            {
                g_detecting_key_index = static_cast<int>(i);
                g_detecting_section = add_id;
                g_detecting_hotkeys = &hotkeys;
            }
        }
        
        ImGui::SameLine();
        std::string remove_button_label = "Remove##" + unique_id;
        if (UIHelpers::BeautifulButton(remove_button_label.c_str(), ImVec2(remove_button_width, 0)))
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
    
    UIHelpers::CompactSpacer();
    std::string add_button_label = "Add Key##" + std::string(add_id);
    if (UIHelpers::BeautifulButton(add_button_label.c_str(), ImVec2(-1, 0)))
    {
        hotkeys.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
}

static void draw_aiming_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Aiming Controls");
    
    // Targeting Section
    UIHelpers::SettingsSubHeader("Targeting Controls");
    UIHelpers::BeautifulText("Configure keys for aimbot activation", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    
    draw_hotkey_section("Aimbot Activation Keys", ctx.config.button_targeting, "targeting_keys");
    
    UIHelpers::Spacer(12.0f);
    
    // Auto Shoot Section
    UIHelpers::SettingsSubHeader("Auto Shooting");
    UIHelpers::BeautifulText("Automatically shoot when targeting enemies", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    
    draw_hotkey_section("Auto Shoot Keys", ctx.config.button_auto_shoot, "auto_shoot_keys");
    
    UIHelpers::Spacer(12.0f);
    
    // Movement Restrictions
    UIHelpers::SettingsSubHeader("Movement Restrictions");
    UIHelpers::BeautifulText("Control when aimbot should avoid certain movements", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    
    draw_hotkey_section("Disable Upward Aim Keys", ctx.config.button_disable_upward_aim, "disable_upward_keys");
    
    UIHelpers::EndCard();
    
    UIHelpers::CompactSpacer();
    
    // Separate card for Triggerbot to give more space
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

static void draw_button_section(const char* title, const char* description, std::vector<std::string>& button_list, const char* add_id, const char* remove_id, bool allow_empty = false)
{
    UIHelpers::BeginCard(title);
    
    if (description) {
        UIHelpers::BeautifulText(description, UIHelpers::GetAccentColor(0.8f));
        UIHelpers::CompactSpacer();
    }
    
    for (size_t i = 0; i < button_list.size(); )
    {
        std::string& current_key_name = button_list[i];
        
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
        
        float remove_button_width = 60.0f;
        float detect_button_width = 80.0f;
        float available_width = ImGui::GetContentRegionAvail().x;
        float combo_width = available_width - remove_button_width - detect_button_width - ImGui::GetStyle().ItemSpacing.x * 2;
        
        // Check if we're detecting for this key
        bool is_detecting = (g_detecting_key_index == static_cast<int>(i) && g_detecting_section == add_id && g_detecting_hotkeys == &button_list);
        
        if (is_detecting)
        {
            // Show "Press any key..." button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.5f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.6f, 0.6f, 1.0f));
            
            if (ImGui::Button("Press any key...", ImVec2(combo_width + detect_button_width + ImGui::GetStyle().ItemSpacing.x, 0)))
            {
                // Cancel detection if clicked
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
            }
            ImGui::PopStyleColor(3);
            
            // Check for pressed key
            std::string detected_key = detect_pressed_key();
            if (!detected_key.empty() && detected_key != "LeftMouseButton") // Ignore left click which would cancel
            {
                current_key_name = detected_key;
                AppContext::getInstance().config.saveConfig();
                g_detecting_key_index = -1;
                g_detecting_section = "";
                g_detecting_hotkeys = nullptr;
            }
        }
        else
        {
            ImGui::SetNextItemWidth(combo_width);
            
            // Enhanced combo box styling
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Button, UIHelpers::GetAccentColor(0.7f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, UIHelpers::GetAccentColor(0.8f));
            ImGui::PushStyleColor(ImGuiCol_Header, UIHelpers::GetAccentColor(0.7f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, UIHelpers::GetAccentColor(0.8f));
            
            std::string combo_label = "##" + std::string(title) + std::to_string(i);
            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                current_key_name = key_names[current_index];
                AppContext::getInstance().config.saveConfig();
            }
            ImGui::PopStyleColor(6);
            
            ImGui::SameLine();
            std::string detect_button_label = "Detect##" + std::string(remove_id) + std::to_string(i);
            if (UIHelpers::BeautifulButton(detect_button_label.c_str(), ImVec2(detect_button_width, 0)))
            {
                g_detecting_key_index = static_cast<int>(i);
                g_detecting_section = add_id;
                g_detecting_hotkeys = &button_list;
            }
        }
        
        ImGui::SameLine();
        std::string remove_button_label = "Remove##" + std::string(remove_id) + std::to_string(i);
        if (UIHelpers::BeautifulButton(remove_button_label.c_str(), ImVec2(remove_button_width, 0)))
        {
            if (!allow_empty && button_list.size() <= 1)
            {
                button_list[0] = std::string("None");
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            }
            else
            {
                button_list.erase(button_list.begin() + i);
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            }
        }
        
        ImGui::PopID();
        ++i;
    }
    
    UIHelpers::CompactSpacer();
    std::string add_button_label = "Add Key##" + std::string(add_id);
    if (UIHelpers::BeautifulButton(add_button_label.c_str(), ImVec2(-FLT_MIN, 0)))
    {
        button_list.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
    
    UIHelpers::EndCard();
}

void draw_buttons()
{
    auto& ctx = AppContext::getInstance();
    
    draw_button_section("Aimbot Activation Keys", "Keys that activate the aimbot when held", 
                       ctx.config.button_targeting, "targeting", "button_targeting");
    
    UIHelpers::Spacer();
    
    draw_button_section("Exit Keys", "Keys that completely exit the application", 
                       ctx.config.button_exit, "exit", "button_exit");
    
    UIHelpers::Spacer();
    
    draw_button_section("Pause Keys", "Keys that temporarily pause the aimbot", 
                       ctx.config.button_pause, "pause", "button_pause");
    
    UIHelpers::Spacer();
    
    draw_button_section("Reload Config Keys", "Keys that reload the configuration file", 
                       ctx.config.button_reload_config, "reload_config", "button_reload_config");
    
    UIHelpers::Spacer();
    
    draw_button_section("Overlay Toggle Keys", "Keys that show/hide this overlay", 
                       ctx.config.button_open_overlay, "overlay", "button_open_overlay", true);
    
    UIHelpers::Spacer();
    
    draw_aiming_settings();
}
