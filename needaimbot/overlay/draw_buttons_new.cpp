#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "AppContext.h"
#include "ui_helpers_new.h"
#include "overlay.h"

extern std::vector<std::string> key_names;
extern std::vector<const char*> key_names_cstrs;

static void draw_key_list(const char* id, std::vector<std::string>& keys)
{
    for (size_t i = 0; i < keys.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        
        int current = 0;
        for (size_t k = 0; k < key_names.size(); ++k) {
            if (key_names[k] == keys[i]) {
                current = static_cast<int>(k);
                break;
            }
        }
        
        float width = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth(width - 55);
        
        if (ImGui::Combo("##key", &current, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size()))) {
            keys[i] = key_names[current];
            AppContext::getInstance().config.saveConfig();
        }
        
        ImGui::SameLine();
        std::string x_label = std::string("X##") + id + std::to_string(i);
        if (UI::Button(x_label.c_str(), 45)) {
            if (keys.size() > 1) {
                keys.erase(keys.begin() + i);
            } else {
                keys[0] = "None";
            }
            AppContext::getInstance().config.saveConfig();
            ImGui::PopID();
            continue;
        }
        
        ImGui::PopID();
    }
    
    std::string add_label = std::string("+##") + id;
    if (UI::Button(add_label.c_str(), -1)) {
        keys.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
}

void draw_buttons_new()
{
    auto& ctx = AppContext::getInstance();
    
    UI::Space();
    
    UI::BeginColumns(0.5f);
    
    // Left Column
    UI::Section("Exit");
    draw_key_list("exit", ctx.config.button_exit);
    UI::Tip("Completely exit the application");
    
    UI::Space();
    UI::Space();
    
    UI::Section("Pause");
    draw_key_list("pause", ctx.config.button_pause);
    UI::Tip("Temporarily disable the aimbot");
    
    UI::Space();
    UI::Space();
    
    UI::Section("Reload Config");
    draw_key_list("reload", ctx.config.button_reload_config);
    UI::Tip("Reload configuration from file");
    
    UI::NextColumn();
    
    // Right Column
    UI::Section("Show Overlay");
    draw_key_list("overlay", ctx.config.button_open_overlay);
    UI::Tip("Toggle this settings window");
    
    UI::Space();
    UI::Space();
    
    UI::Section("Screenshot");
    draw_key_list("screenshot", ctx.config.screenshot_button);
    UI::Tip("Take a debug screenshot");
    
    if (!ctx.config.screenshot_button.empty() && ctx.config.screenshot_button[0] != "None") {
        UI::SmallSpace();
        
        float delay = static_cast<float>(ctx.config.screenshot_delay);
        if (UI::Slider("Delay##screenshot", &delay, 0.0f, 5000.0f, "%.0f ms")) {
            ctx.config.screenshot_delay = static_cast<int>(delay);
            ctx.config.saveConfig();
        }
    }
    
    UI::EndColumns();
    
    UI::Space();
    UI::Space();
    
    // Info panel
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.10f, 0.12f, 0.9f));
    ImGui::BeginChild("##info", ImVec2(0, 0), true);
    
    UI::Section("Key Binding Tips");
    
    ImGui::BulletText("You can assign multiple keys to each action");
    ImGui::BulletText("Set to 'None' to disable an action");
    ImGui::BulletText("Some keys may conflict with game controls");
    ImGui::BulletText("Mouse buttons are also supported");
    
    ImGui::EndChild();
    ImGui::PopStyleColor();
    
    UI::Space();
}