#include "../core/windows_headers.h"

#include <iostream>
#include "d3d11.h"
#include "../imgui/imgui.h"

#include "AppContext.h"
#include "overlay.h"
#include "draw_settings.h"
#include "needaimbot.h"
#include "other_tools.h"
#include "memory_images.h"

ID3D11ShaderResourceView* bodyTexture = nullptr;
ImVec2 bodyImageSize;

void load_body_texture()
{
    int image_width = 0;
    int image_height = 0;

    // Concatenate all base64 chunks
    std::string body_image;
    body_image.reserve(strlen(bodyImageBase64_1) + strlen(bodyImageBase64_2) + strlen(bodyImageBase64_3));

    body_image += std::string(bodyImageBase64_1);
    body_image += std::string(bodyImageBase64_2);
    body_image += std::string(bodyImageBase64_3);

    std::cerr << "Total base64 string length: " << body_image.length() << std::endl;

    bool ret = LoadTextureFromMemory(body_image, g_pd3dDevice, &bodyTexture, &image_width, &image_height);
    if (ret)
    {
        bodyImageSize = ImVec2((float)image_width, (float)image_height);
        std::cerr << "Body texture loaded successfully: " << image_width << "x" << image_height << std::endl;
    }
    else
    {
        std::cerr << "Failed to load body texture" << std::endl;
    }
}

void release_body_texture()
{
    if (bodyTexture)
    {
        bodyTexture->Release();
        bodyTexture = nullptr;
    }
}

void draw_target()
{
    auto& ctx = AppContext::getInstance();

    // Status indicator
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    if (ctx.detection_paused.load()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        ImGui::Text("[PAUSED]");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("Aimbot is paused - Press %s to resume",
                    ctx.config.button_pause.empty() ? "F3" : ctx.config.button_pause[0].c_str());
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
        ImGui::Text("[ACTIVE]");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("Aimbot is ready");
    }
    ImGui::PopFont();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Main toggle
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
    bool auto_aim = ctx.config.auto_aim;
    if (ImGui::Checkbox("Enable Auto Aim", &auto_aim)) {
        ctx.config.auto_aim = auto_aim;
        SAVE_PROFILE();
    }
    ImGui::PopStyleVar();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enable automatic aim assistance to track detected targets");
    }

    ImGui::Spacing();

    // Auto shoot
    if (ImGui::Checkbox("Enable Auto Shoot", &ctx.config.auto_shoot)) { SAVE_PROFILE(); }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Automatically fire when a target is locked");
    }

    ImGui::Spacing();
    if (ImGui::Checkbox("Disable Upward Aim", &ctx.config.ignore_up_aim)) { SAVE_PROFILE(); }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Prevent aim from moving upward - useful for ground-only combat");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Target selection info
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Target Selection");
    ImGui::TextWrapped("Always targets the enemy closest to your crosshair");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Quick tips
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Quick Tips");
    ImGui::BulletText("Use Mouse tab to adjust aim speed and smoothness");
    ImGui::BulletText("Use Aim Offset tab to fine-tune aim position");
    ImGui::BulletText("Use Recoil tab to control weapon spray patterns");
    ImGui::BulletText("Use Detection tab to adjust AI confidence threshold");
}

 
