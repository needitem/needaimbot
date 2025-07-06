#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "AppContext.h"

void draw_buttons()
{
    auto& ctx = AppContext::getInstance();
    ImGui::Text("Aimbot Activation Key");

    for (size_t i = 0; i < ctx.config.button_targeting.size(); )
    {
        std::string& current_key_name = ctx.config.button_targeting[i];

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

        std::string combo_label = "Aimbot Key " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            ctx.config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_targeting" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (ctx.config.button_targeting.size() <= 1)
            {
                ctx.config.button_targeting[0] = std::string("None");
                ctx.config.saveConfig();
                continue;
            }
            else
            {
                ctx.config.button_targeting.erase(ctx.config.button_targeting.begin() + i);
                ctx.config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add Aimbot Key##targeting"))
    {
        ctx.config.button_targeting.push_back("None");
        ctx.config.saveConfig();
    }

    ImGui::Separator();
    ImGui::Text("Mouse Controls (Fixed):");
    ImGui::BulletText("Left Click: Auto-Shooting (when aiming)");
    ImGui::BulletText("Right Click: Zoom/Scope (independent of aimbot)");


    ImGui::Separator();

    ImGui::Text("Exit Buttons");

    for (size_t i = 0; i < ctx.config.button_exit.size(); )
    {
        std::string& current_key_name = ctx.config.button_exit[i];

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

        std::string combo_label = "Exit Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            ctx.config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_exit" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (ctx.config.button_exit.size() <= 1)
            {
                ctx.config.button_exit[0] = std::string("None");
                ctx.config.saveConfig();
                continue;
            }
            else
            {
                ctx.config.button_exit.erase(ctx.config.button_exit.begin() + i);
                ctx.config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##exit"))
    {
        ctx.config.button_exit.push_back("None");
        ctx.config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Pause Buttons");

    for (size_t i = 0; i < ctx.config.button_pause.size(); )
    {
        std::string& current_key_name = ctx.config.button_pause[i];

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

        std::string combo_label = "Pause Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            ctx.config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_pause" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (ctx.config.button_pause.size() <= 1)
            {
                ctx.config.button_pause[0] = std::string("None");
                ctx.config.saveConfig();
                continue;
            }
            else
            {
                ctx.config.button_pause.erase(ctx.config.button_pause.begin() + i);
                ctx.config.saveConfig();
                continue;
            }
        }
        ++i;
    }

    if (ImGui::Button("Add button##pause"))
    {
        ctx.config.button_pause.push_back("None");
        ctx.config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Reload Config Buttons");

    for (size_t i = 0; i < ctx.config.button_reload_config.size(); )
    {
        std::string& current_key_name = ctx.config.button_reload_config[i];

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

        std::string combo_label = "Reload Config Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            ctx.config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_reload_config" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (ctx.config.button_reload_config.size() <= 1)
            {
                ctx.config.button_reload_config[0] = std::string("None");
                ctx.config.saveConfig();
                continue;
            }
            else
            {
                ctx.config.button_reload_config.erase(ctx.config.button_reload_config.begin() + i);
                ctx.config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##reload_config"))
    {
        ctx.config.button_reload_config.push_back("None");
        ctx.config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Overlay Buttons");

    for (size_t i = 0; i < ctx.config.button_open_overlay.size(); )
    {
        std::string& current_key_name = ctx.config.button_open_overlay[i];

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

        std::string combo_label = "Overlay Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            ctx.config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_open_overlay" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            ctx.config.button_open_overlay.erase(ctx.config.button_open_overlay.begin() + i);
            ctx.config.saveConfig();
            continue;
        }

        ++i;
    }

    if (ImGui::Button("Add button##overlay"))
    {
        ctx.config.button_open_overlay.push_back("None");
        ctx.config.saveConfig();
    }
}
