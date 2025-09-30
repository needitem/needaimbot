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

    std::string body_image = std::string(bodyImageBase64_1);
    if (strlen(bodyImageBase64_1) > 0) {
        // Concatenate chunks if provided in multiple parts
        if (strlen(bodyImageBase64_1) > 0 && strlen(bodyImageBase64_1) < 10) {
            // no-op safeguard
        }
    }
    // Try to append optional parts if they exist
    #ifdef bodyImageBase64_2
    body_image += std::string(bodyImageBase64_2);
    #endif
    #ifdef bodyImageBase64_3
    body_image += std::string(bodyImageBase64_3);
    #endif

    bool ret = LoadTextureFromMemory(body_image, g_pd3dDevice, &bodyTexture, &image_width, &image_height);
    if (ret)
    {
        bodyImageSize = ImVec2((float)image_width, (float)image_height);
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
    
    // Display pause status prominently at the top
    if (ctx.detection_paused.load()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f)); // Red color
        ImGui::Text("AIMBOT PAUSED (Press %s to resume)", 
                    ctx.config.button_pause.empty() ? "F3" : ctx.config.button_pause[0].c_str());
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f)); // Green color
        ImGui::Text("AIMBOT ACTIVE");
        ImGui::PopStyleColor();
    }
    
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::Checkbox("Auto Aim", &ctx.config.auto_aim)) {
        SAVE_PROFILE();
    }

    // Target selection controls
    ImGui::Separator();
    ImGui::Text("Target Selection");
    if (ImGui::SliderFloat("Sticky Threshold", &ctx.config.sticky_target_threshold, 0.0f, 1.0f, "%.2f")) {
        if (ctx.config.sticky_target_threshold < 0.0f) ctx.config.sticky_target_threshold = 0.0f;
        if (ctx.config.sticky_target_threshold > 1.0f) ctx.config.sticky_target_threshold = 1.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("How much better a new target must be to switch.\n0.00 = always switch to closest, 0.30~0.50 = moderate stickiness.");
    }    
}

 
