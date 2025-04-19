#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "include/other_tools.h"
#include <vector>
#include <string>

// Helper function to convert vector<string> to vector<const char*>
inline std::vector<const char*> getProfileCstrs(const std::vector<std::string>& profiles) {
    std::vector<const char*> cstrs;
    cstrs.reserve(profiles.size());
    for(const auto& s : profiles)
        cstrs.push_back(s.c_str());
    return cstrs;
}

void draw_debug()
{
    ImGui::Checkbox("Show Window", &config.show_window);
    ImGui::Checkbox("Show FPS", &config.show_fps);
    ImGui::SliderInt("Window Size", &config.window_size, 10, 350);

    ImGui::Separator();
    ImGui::Text("Screenshot Buttons");

    for (size_t i = 0; i < config.screenshot_button.size(); )
    {
        std::string& current_key_name = config.screenshot_button[i];

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

        std::string combo_label = "Screenshot Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig("config.ini");
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_screenshot" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.screenshot_button.size() <= 1)
            {
                config.screenshot_button[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.screenshot_button.erase(config.screenshot_button.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##button_screenshot"))
    {
        config.screenshot_button.push_back("None");
        config.saveConfig();
    }

    ImGui::InputInt("Screenshot delay", &config.screenshot_delay, 50, 500);
    ImGui::Checkbox("Always On Top", &config.always_on_top);
    ImGui::Checkbox("Verbose console output", &config.verbose);

    // Profile Management UI Section
    ImGui::Separator();
    ImGui::Text("Config Profiles");

    // Get the list of profiles dynamically
    static std::vector<std::string> profile_names = config.listProfiles();
    static std::vector<const char*> profile_cstrs = getProfileCstrs(profile_names); // C-style strings for ImGui Combo
    static int current_profile_idx = -1; // Initialize to -1 (no selection)

    // Refresh profiles if needed (e.g., after save/delete)
    auto refresh_profiles = [&](){
        profile_names = config.listProfiles();
        profile_cstrs = getProfileCstrs(profile_names);
        // Try to maintain selection if possible, otherwise reset
        if (current_profile_idx >= (int)profile_names.size()) {
             current_profile_idx = profile_names.empty() ? -1 : 0;
        }
    };

    // Find the initial index if a profile was loaded previously or exists
    // This part might need adjustment based on how initial load is handled
    if (current_profile_idx == -1 && !profile_names.empty()) {
        current_profile_idx = 0; // Default to first profile if none selected
    }

    const char* combo_preview_value = (current_profile_idx >= 0 && current_profile_idx < (int)profile_cstrs.size()) ? profile_cstrs[current_profile_idx] : "Select Profile...";

    if (ImGui::BeginCombo("Select Profile", combo_preview_value))
    {
        for (int n = 0; n < (int)profile_cstrs.size(); n++)
        {
            const bool is_selected = (current_profile_idx == n);
            if (ImGui::Selectable(profile_cstrs[n], is_selected))
                current_profile_idx = n;

            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    static char new_profile_name[128] = "";
    ImGui::InputText("New Profile Name", new_profile_name, IM_ARRAYSIZE(new_profile_name));

    if (ImGui::Button("Save Profile")) {
        std::string profileToSave;
        if (strlen(new_profile_name) > 0) {
             profileToSave = std::string(new_profile_name);
             // Basic validation: avoid problematic characters if needed
             // profileToSave = sanitize_filename(profileToSave);
             config.saveProfile(profileToSave);
             refresh_profiles();
             // Select the newly saved profile
             for(size_t i = 0; i < profile_names.size(); ++i) {
                 if (profile_names[i] == profileToSave) {
                     current_profile_idx = static_cast<int>(i);
                     break;
                 }
             }
             new_profile_name[0] = '\0'; // Clear the input field
        } else if (current_profile_idx >= 0 && current_profile_idx < (int)profile_names.size()) {
             profileToSave = profile_names[current_profile_idx];
             config.saveProfile(profileToSave); // Overwrite selected profile
             // No need to refresh list or change index here
             std::cout << "[UI] Profile '" << profileToSave << "' overwritten." << std::endl;
        } else {
            std::cerr << "[UI] Cannot save: No profile selected and no new name entered." << std::endl;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Profile")) {
        if (current_profile_idx >= 0 && current_profile_idx < (int)profile_names.size()) {
            std::string profileToLoad = profile_names[current_profile_idx];
            if (config.loadProfile(profileToLoad)) {
                 std::cout << "[UI] Profile '" << profileToLoad << "' loaded successfully." << std::endl;
                 // UI should update automatically as config object is modified
            } else {
                 std::cerr << "[UI] Failed to load profile '" << profileToLoad << "'." << std::endl;
                 // Maybe refresh list in case the file was deleted externally
                 refresh_profiles();
            }
        } else {
            std::cerr << "[UI] Cannot load: No profile selected." << std::endl;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete Profile")) {
         if (current_profile_idx >= 0 && current_profile_idx < (int)profile_names.size()) {
            std::string profileToDelete = profile_names[current_profile_idx];
             if (config.deleteProfile(profileToDelete)) {
                 std::cout << "[UI] Profile '" << profileToDelete << "' deleted successfully." << std::endl;
                 refresh_profiles(); // Refresh list and reset index
             } else {
                 std::cerr << "[UI] Failed to delete profile '" << profileToDelete << "'." << std::endl;
                 // Refresh list in case of external changes or permissions issue
                 refresh_profiles();
             }
        } else {
             std::cerr << "[UI] Cannot delete: No profile selected." << std::endl;
        }
    }
    // End Profile Management UI Section

    ImGui::Separator();

    ImGui::Text("Test functions");
    if (ImGui::Button("Free terminal"))
    {
        HideConsole();
    }
    ImGui::SameLine();
    if (ImGui::Button("Restore terminal"))
    {
        ShowConsole();
    }
}