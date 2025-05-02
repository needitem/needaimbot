#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"

#include <vector>
#include <string>
#include <iostream>


inline std::vector<const char*> getProfileCstrs(const std::vector<std::string>& profiles) {
    std::vector<const char*> cstrs;
    cstrs.reserve(profiles.size());
    for(const auto& s : profiles)
        cstrs.push_back(s.c_str());
    return cstrs;
}


void draw_profile()
{
    ImGui::SeparatorText("Available Profiles");
    ImGui::Spacing();

    static std::vector<std::string> profile_list = config.listProfiles();
    static std::vector<const char*> profile_list_cstrs = getProfileCstrs(profile_list);
    static int selected_profile_index = -1;
    static char new_profile_name[128] = "";
    static std::string status_message = "";

    auto refresh_profiles = [&](){
        profile_list = config.listProfiles();
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

    if (ImGui::BeginListBox("##ProfilesList", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing())))
    {
        for (int n = 0; n < profile_list_cstrs.size(); n++)
        {
            const bool is_selected = (selected_profile_index == n);
            if (ImGui::Selectable(profile_list_cstrs[n], is_selected))
            {
                selected_profile_index = n;
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    ImGui::Spacing();
    {
        bool profile_selected = (selected_profile_index >= 0 && selected_profile_index < profile_list.size());
        if (!profile_selected) { ImGui::BeginDisabled(); }

        if (ImGui::Button("Load Selected"))
        {
            std::string profileToLoad = profile_list[selected_profile_index];
            if (config.loadProfile(profileToLoad))
            {
                status_message = "Loaded profile: " + profileToLoad;
            } else {
                status_message = "Error loading profile: " + profileToLoad;
                refresh_profiles();
            }
        }
        if (ImGui::IsItemHovered() && profile_selected) { ImGui::SetTooltip("Load settings from the selected profile."); }

        ImGui::SameLine();

        if (ImGui::Button("Delete Selected")) {
            std::string profileToDelete = profile_list[selected_profile_index];
             if (config.deleteProfile(profileToDelete)) {
                 status_message = "Deleted profile: " + profileToDelete;
                 refresh_profiles();
             } else {
                 status_message = "Error deleting profile: " + profileToDelete;
                 refresh_profiles();
             }
        }
        if (ImGui::IsItemHovered() && profile_selected) { ImGui::SetTooltip("Delete the selected profile file."); }

        if (!profile_selected) { ImGui::EndDisabled(); }
    }

    ImGui::SameLine();

    if (ImGui::Button("Refresh List"))
    {
        refresh_profiles();
        status_message = "Refreshed profile list.";
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Rescan the profiles directory for changes."); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Save Current Settings");
    ImGui::Spacing();

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
    ImGui::InputTextWithHint("##NewProfileName", "Enter new profile name or select to overwrite", new_profile_name, IM_ARRAYSIZE(new_profile_name));
    ImGui::PopItemWidth();

    ImGui::SameLine();
    if (ImGui::Button("Save##SaveProfileBtn"))
    {
        std::string name = new_profile_name;
        bool is_overwriting = false;
        if (!name.empty())
        {
             if (config.saveProfile(name))
            {
                status_message = "Saved profile: " + name;
                refresh_profiles();
                for(size_t i = 0; i < profile_list.size(); ++i) {
                     if (profile_list[i] == name) {
                         selected_profile_index = static_cast<int>(i);
                         break;
                     }
                }
                new_profile_name[0] = '\0';
            } else {
                 status_message = "Error saving profile: " + name;
            }
        } else if (selected_profile_index >= 0 && selected_profile_index < profile_list.size()) {
             name = profile_list[selected_profile_index];
              if (config.saveProfile(name))
             {
                is_overwriting = true;
                status_message = "Overwrote profile: " + name;
             } else {
                  status_message = "Error overwriting profile: " + name;
             }
        } else {
             status_message = "Enter a new name or select a profile to overwrite.";
        }
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Save the current settings. Either enter a new name or select an existing profile from the list to overwrite it."); }

    ImGui::Spacing();

    if (!status_message.empty()) {
        ImGui::TextWrapped("Status: %s", status_message.c_str());
    }
    ImGui::Spacing();
}
