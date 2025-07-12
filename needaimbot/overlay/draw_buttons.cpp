#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "AppContext.h"
#include "ui_helpers.h"

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
        
        float button_width = 70.0f;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button_width - ImGui::GetStyle().ItemSpacing.x);
        
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
        std::string remove_button_label = "Remove##" + std::string(remove_id) + std::to_string(i);
        if (UIHelpers::BeautifulButton(remove_button_label.c_str(), ImVec2(button_width, 0)))
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
    if (UIHelpers::BeautifulButton(add_button_label.c_str(), ImVec2(-1, 0)))
    {
        button_list.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
    
    UIHelpers::EndCard();
}

void draw_buttons()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.65f);
    
    // Left column - Button configurations
    draw_button_section("Aimbot Activation Keys", "Keys that activate the aimbot when held", 
                       ctx.config.button_targeting, "targeting", "button_targeting");
    
    UIHelpers::CompactSpacer();
    
    draw_button_section("Exit Keys", "Keys that completely exit the application", 
                       ctx.config.button_exit, "exit", "button_exit");
    
    UIHelpers::CompactSpacer();
    
    draw_button_section("Pause Keys", "Keys that temporarily pause the aimbot", 
                       ctx.config.button_pause, "pause", "button_pause");
    
    UIHelpers::CompactSpacer();
    
    draw_button_section("Reload Config Keys", "Keys that reload the configuration file", 
                       ctx.config.button_reload_config, "reload_config", "button_reload_config");
    
    UIHelpers::CompactSpacer();
    
    draw_button_section("Overlay Toggle Keys", "Keys that show/hide this overlay", 
                       ctx.config.button_open_overlay, "overlay", "button_open_overlay", true);
    
    UIHelpers::NextColumn();
    
    // Right column - Information panel
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("Mouse Controls (Fixed)", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Left Click: Auto-Shooting (when aiming)");
    ImGui::BulletText("Right Click: Zoom/Scope (independent of aimbot)");
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulText("Key Binding Tips", UIHelpers::GetWarningColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Use easy-to-reach keys for aimbot activation");
    ImGui::BulletText("Avoid conflicts with game controls");
    ImGui::BulletText("Test key combinations in a safe environment");
    ImGui::BulletText("Multiple keys can be assigned to the same function");
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulText("Status", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    UIHelpers::StatusIndicator("Aimbot Keys", !ctx.config.button_targeting.empty() && ctx.config.button_targeting[0] != "None");
    UIHelpers::StatusIndicator("Exit Keys", !ctx.config.button_exit.empty() && ctx.config.button_exit[0] != "None");
    UIHelpers::StatusIndicator("Pause Keys", !ctx.config.button_pause.empty() && ctx.config.button_pause[0] != "None");
    
    UIHelpers::EndInfoPanel();
    
    UIHelpers::EndTwoColumnLayout();
}
