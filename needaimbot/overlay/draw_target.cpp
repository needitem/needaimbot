#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include "d3d11.h"
#include "imgui/imgui.h"

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
    if (ctx.detectionPaused.load()) {
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
    ImGui::Checkbox("Auto Aim", &ctx.config.auto_aim);

    // Target selection is now fixed to closest target only
    // These settings are hidden but set to optimal values for closest target selection
    ctx.config.distance_weight = 1.0f;  // Maximum distance priority
    ctx.config.confidence_weight = 0.0f;  // Ignore confidence
    ctx.config.sticky_target_threshold = 0.0f;  // Always switch to closest
    
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
