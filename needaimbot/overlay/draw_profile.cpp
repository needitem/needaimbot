#include "AppContext.h"
#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "needaimbot.h"
#include "overlay.h"
#include "ui_helpers.h"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>


inline std::vector<const char*> getProfileCstrs(const std::vector<std::string>& profiles) {
    std::vector<const char*> cstrs;
    cstrs.reserve(profiles.size());
    for(const auto& s : profiles)
        cstrs.push_back(s.c_str());
    return cstrs;
}


void draw_profile()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.55f);
    
    // Left column - Profile list
    UIHelpers::BeginCard("Profile Management");

    static std::vector<std::string> profile_list = ctx.config.listProfiles();
    static std::vector<const char*> profile_list_cstrs = getProfileCstrs(profile_list);
    static int selected_profile_index = -1;
    static char new_profile_name[128] = "";
    static char profile_description[256] = "";
    static std::string status_message = "";
    static std::chrono::steady_clock::time_point status_time;
    static bool show_confirm_delete = false;
    static std::string current_profile_name = "Default";

    auto refresh_profiles = [&](){
        profile_list = ctx.config.listProfiles();
        profile_list_cstrs = getProfileCstrs(profile_list);
        if (selected_profile_index >= (int)profile_list.size()) {
             selected_profile_index = profile_list.empty() ? -1 : 0;
        }
         if (profile_list.empty()) {
             selected_profile_index = -1;
        }
    };

    if (selected_profile_index == -1 && !profile_list.empty()) {
        selected_profile_index = 0;
    }

    UIHelpers::BeautifulText("Available Profiles", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    // Show current profile with better styling
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 4));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.10f, 0.12f, 0.5f));
    ImGui::BeginChild("CurrentProfile", ImVec2(-1, 30), true);
    ImGui::PushStyleColor(ImGuiCol_Text, UIHelpers::GetAccentColor());
    ImGui::Text("Current: %s", current_profile_name.c_str());
    ImGui::PopStyleColor();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    
    UIHelpers::CompactSpacer();
    
    if (ImGui::BeginListBox("##ProfilesList", ImVec2(-FLT_MIN, 8 * ImGui::GetTextLineHeightWithSpacing())))
    {
        for (int n = 0; n < profile_list_cstrs.size(); n++)
        {
            const bool is_selected = (selected_profile_index == n);
            const bool is_current = (profile_list[n] == current_profile_name);
            
            if (is_current) {
                ImGui::PushStyleColor(ImGuiCol_Text, UIHelpers::GetAccentColor(0.8f));
            }
            
            char label[256];
            if (is_current) {
                snprintf(label, sizeof(label), "[*] %s", profile_list_cstrs[n]);
            } else {
                snprintf(label, sizeof(label), "    %s", profile_list_cstrs[n]);
            }
            
            if (ImGui::Selectable(label, is_selected))
            {
                selected_profile_index = n;
                // Don't auto-load, just select
            }
            
            if (is_current) {
                ImGui::PopStyleColor();
            }
            
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    UIHelpers::Spacer();
    
    // Action buttons
    bool profile_selected = (selected_profile_index >= 0 && selected_profile_index < profile_list.size());
    
    if (!profile_selected) { ImGui::BeginDisabled(); }
    if (UIHelpers::EnhancedButton("Load Selected", ImVec2(-1, 0), "Load settings from the selected profile"))
    {
        std::string profileToLoad = profile_list[selected_profile_index];
        if (ctx.config.loadProfile(profileToLoad))
        {
            status_message = "[OK] Loaded: " + profileToLoad;
            current_profile_name = profileToLoad;
            status_time = std::chrono::steady_clock::now();
        } else {
            status_message = "[ERROR] Failed to load: " + profileToLoad;
            status_time = std::chrono::steady_clock::now();
            refresh_profiles();
        }
    }
    if (!profile_selected) { ImGui::EndDisabled(); }
    
    UIHelpers::CompactSpacer();
    
    // Delete with confirmation
    if (show_confirm_delete && profile_selected)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, UIHelpers::GetWarningColor());
        ImGui::Text("Delete '%s'?", profile_list[selected_profile_index].c_str());
        ImGui::PopStyleColor();
        
        if (UIHelpers::BeautifulButton("Confirm Delete", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0)))
        {
            std::string profileToDelete = profile_list[selected_profile_index];
            if (ctx.config.deleteProfile(profileToDelete)) {
                status_message = "[OK] Deleted: " + profileToDelete;
                refresh_profiles();
            } else {
                status_message = "[ERROR] Failed to delete: " + profileToDelete;
            }
            status_time = std::chrono::steady_clock::now();
            show_confirm_delete = false;
        }
        ImGui::SameLine();
        if (UIHelpers::BeautifulButton("Cancel", ImVec2(-1, 0)))
        {
            show_confirm_delete = false;
        }
    }
    else
    {
        if (!profile_selected) { ImGui::BeginDisabled(); }
        if (UIHelpers::EnhancedButton("Delete Selected", ImVec2(-1, 0), "Delete the selected profile"))
        {
            show_confirm_delete = true;
        }
        if (!profile_selected) { ImGui::EndDisabled(); }
    }
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulButton("Refresh List", ImVec2(-1, 0)))
    {
        refresh_profiles();
        status_message = "[OK] Profile list refreshed";
        status_time = std::chrono::steady_clock::now();
    }

    UIHelpers::EndCard();
    
    UIHelpers::NextColumn();
    
    // Right column - Save profile
    UIHelpers::BeginCard("Save Current Settings");

    UIHelpers::BeautifulText("Profile Name");
    ImGui::PushItemWidth(-1);
    ImGui::InputTextWithHint("##NewProfileName", "Enter new profile name", new_profile_name, IM_ARRAYSIZE(new_profile_name));
    ImGui::PopItemWidth();
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulText("Description (Optional)");
    ImGui::PushItemWidth(-1);
    ImGui::InputTextMultiline("##ProfileDesc", profile_description, IM_ARRAYSIZE(profile_description), ImVec2(-1, 50));
    ImGui::PopItemWidth();
    
    UIHelpers::Spacer();
    
    if (UIHelpers::EnhancedButton("Save As New Profile", ImVec2(-1, 0), "Save current settings as a new profile"))
    {
        std::string name = new_profile_name;
        bool is_overwriting = false;
        if (!name.empty())
        {
            if (ctx.config.saveProfile(name))
            {
                status_message = "[OK] Saved: " + name;
                current_profile_name = name;
                refresh_profiles();
                for(size_t i = 0; i < profile_list.size(); ++i) {
                    if (profile_list[i] == name) {
                        selected_profile_index = static_cast<int>(i);
                        break;
                    }
                }
                new_profile_name[0] = '\0';
                profile_description[0] = '\0';
                status_time = std::chrono::steady_clock::now();
            } else {
                status_message = "[ERROR] Failed to save: " + name;
                status_time = std::chrono::steady_clock::now();
            }
        } else {
            status_message = "[WARNING] Enter a profile name";
            status_time = std::chrono::steady_clock::now();
        }
    }
    
    UIHelpers::CompactSpacer();
    
    // Overwrite existing profile
    if (profile_selected)
    {
        UIHelpers::BeautifulSeparator("Or");
        UIHelpers::CompactSpacer();
        
        if (UIHelpers::EnhancedButton("Overwrite Selected", ImVec2(-1, 0), "Overwrite the selected profile with current settings"))
        {
            std::string name = profile_list[selected_profile_index];
            if (ctx.config.saveProfile(name))
            {
                status_message = "[OK] Overwrote: " + name;
                current_profile_name = name;
                status_time = std::chrono::steady_clock::now();
            } else {
                status_message = "[ERROR] Failed to overwrite: " + name;
                status_time = std::chrono::steady_clock::now();
            }
        }
    }
    
    UIHelpers::Spacer();
    UIHelpers::BeautifulSeparator("Quick Actions");
    UIHelpers::CompactSpacer();
    
    // Reset to default button
    if (UIHelpers::EnhancedButton("Reset to Default", ImVec2(-1, 0), "Reset all settings to default values"))
    {
        ctx.config.resetConfig();
        status_message = "[OK] Reset to default settings";
        current_profile_name = "Default";
        status_time = std::chrono::steady_clock::now();
    }
    
    UIHelpers::EndCard();
    
    // Status message with fade effect
    if (!status_message.empty()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - status_time).count();
        
        if (elapsed < 5) { // Show for 5 seconds
            float alpha = 1.0f;
            if (elapsed > 3) { // Start fading after 3 seconds
                alpha = 1.0f - ((elapsed - 3) / 2.0f);
            }
            
            UIHelpers::BeginCard(nullptr);
            ImVec4 color = (status_message.find("Loaded") != std::string::npos || 
                           status_message.find("Saved") != std::string::npos ||
                           status_message.find("Overwrote") != std::string::npos ||
                           status_message.find("Deleted") != std::string::npos ||
                           status_message.find("Reset") != std::string::npos ||
                           status_message.find("Refreshed") != std::string::npos) ? UIHelpers::GetSuccessColor(alpha) : 
                          (status_message.find("Failed") != std::string::npos ||
                           status_message.find("Error") != std::string::npos) ? UIHelpers::GetErrorColor(alpha) : 
                          UIHelpers::GetWarningColor(alpha);
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::TextWrapped("%s", status_message.c_str());
            ImGui::PopStyleColor();
            UIHelpers::EndCard();
        } else {
            status_message.clear();
        }
    }
    
    UIHelpers::EndTwoColumnLayout();
}

