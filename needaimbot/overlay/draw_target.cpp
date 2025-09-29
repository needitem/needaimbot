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
    
    // Target Lock Settings
    ImGui::Separator();
    ImGui::Text("Target Lock Settings");
    
    if (ImGui::Checkbox("Enable Target Lock", &ctx.config.enable_target_lock)) {
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("When enabled, locks onto a target and tracks it by ID until lost");
    }
    
    if (ctx.config.enable_target_lock) {
        // Show lock status
        // TODO: Implement target lock status for TensorRT integration
        {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "🔓 NO LOCK");
        }
    }

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

void load_body_texture()
{
    int image_width = 0;
    int image_height = 0;

    std::string body_image = std::string(bodyImageBase64_1) + std::string(bodyImageBase64_2) + std::string(bodyImageBase64_3);

    bool ret = LoadTextureFromMemory(body_image, g_pd3dDevice, &bodyTexture, &image_width, &image_height);
    if (!ret)
    {
        std::cerr << "[Overlay] Can't load image!" << std::endl;
    }
    else
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
