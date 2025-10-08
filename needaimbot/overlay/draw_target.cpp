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

    // ═══════════════════════════════════════════════════════════
    // STATUS INDICATOR
    // ═══════════════════════════════════════════════════════════

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Use default font for status

    if (ctx.detection_paused.load()) {
        // Paused state - Red indicator
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        ImGui::Text("[PAUSED] AIMBOT PAUSED");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled(" - Press %s to resume",
                    ctx.config.button_pause.empty() ? "F3" : ctx.config.button_pause[0].c_str());
    } else {
        // Active state - Green indicator
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
        ImGui::Text("[ACTIVE] AIMBOT READY");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled(" - Tracking enabled");
    }

    ImGui::PopFont();

    ImGui::Separator();
    ImGui::Spacing();

    // ═══════════════════════════════════════════════════════════
    // MAIN TOGGLE
    // ═══════════════════════════════════════════════════════════

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
    bool auto_aim = ctx.config.auto_aim;
    if (ImGui::Checkbox("Enable Auto Aim", &auto_aim)) {
        ctx.config.auto_aim = auto_aim;
        SAVE_PROFILE();
    }
    ImGui::PopStyleVar();

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Enable automatic aim assistance");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                          "When enabled, the cursor will automatically track detected targets");
        ImGui::EndTooltip();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ═══════════════════════════════════════════════════════════
    // TARGET SELECTION SETTINGS
    // ═══════════════════════════════════════════════════════════

    ImGui::Text("Target Selection: Always closest to crosshair");
}

 
