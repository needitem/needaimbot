#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "d3d11.h"
#include "imgui/imgui.h"

#include "overlay.h"
#include "draw_settings.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "memory_images.h"

ID3D11ShaderResourceView* bodyTexture = nullptr;
ImVec2 bodyImageSize;

void draw_target()
{
    ImGui::Checkbox("Disable Headshot", &config.disable_headshot);

    ImGui::Separator();

    // Instructions for keyboard controls
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Arrow keys: Adjust body offset");
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Shift+Arrow keys: Adjust head offset");

    // Offset step control
    if (ImGui::InputFloat("Offset Step", &config.offset_step, 0.0f, 0.0f, "%.3f"))
    {
        config.offset_step = std::max(0.001f, std::min(config.offset_step, 0.1f));
        config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Step size for adjusting offset values with arrow keys (0.001 - 0.1)");
    }

    // Body offset slider
    ImGui::SliderFloat("Approximate Body Y Offset", &config.body_y_offset, 0.0f, 1.0f, "%.2f");
    
    // Head offset slider
    ImGui::SliderFloat("Approximate Head Y Offset", &config.head_y_offset, 0.0f, 1.0f, "%.2f");
    
    if (bodyTexture)
    {
        ImGui::Image((void*)bodyTexture, bodyImageSize);

        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();

        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw body offset line (Simplified calculation)
        float body_line_y = image_pos.y + config.body_y_offset * image_size.y;
        ImVec2 body_line_start = ImVec2(image_pos.x, body_line_y);
        ImVec2 body_line_end = ImVec2(image_pos.x + image_size.x, body_line_y);
        draw_list->AddLine(body_line_start, body_line_end, IM_COL32(255, 0, 0, 255), 2.0f);
        
        // Draw head offset line - Visualization (Use full image height)
        // head_y_offset = 0.0 maps to the top of the image
        // head_y_offset = 1.0 maps to the bottom of the image
        float head_line_y = image_pos.y + config.head_y_offset * image_size.y;
        
        ImVec2 head_line_start = ImVec2(image_pos.x, head_line_y);
        ImVec2 head_line_end = ImVec2(image_pos.x + image_size.x, head_line_y);
        draw_list->AddLine(head_line_start, head_line_end, IM_COL32(0, 255, 0, 255), 2.0f);
        
        // Add labels for the lines
        draw_list->AddText(ImVec2(body_line_end.x + 5, body_line_y - 7), IM_COL32(255, 0, 0, 255), "Body");
        draw_list->AddText(ImVec2(head_line_end.x + 5, head_line_y - 7), IM_COL32(0, 255, 0, 255), "Head");
    }
    else
    {
        ImGui::Text("Image not found!");
    }
    ImGui::Text("Note: There is a different value for each game, as the sizes of the player models may vary.");
    ImGui::Separator();
    ImGui::Checkbox("Ignore Third Person", &config.ignore_third_person);
    ImGui::Checkbox("Shooting range targets", &config.shooting_range_targets);
    ImGui::Checkbox("Auto Aim", &config.auto_aim);

    ImGui::Separator();
    ImGui::Text("Target Stickiness");
    ImGui::Spacing();

    // Sticky Bonus Slider (Negative is better, so range -100 to 0)
    if (ImGui::SliderFloat("Sticky Bonus", &config.sticky_bonus, -100.0f, 0.0f, "%.1f"))
    {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Bonus score applied to previous target (lower is better, more sticky).");
    }

    // Sticky IoU Threshold Slider (0.0 to 1.0)
    if (ImGui::SliderFloat("Sticky IoU Threshold", &config.sticky_iou_threshold, 0.0f, 1.0f, "%.2f"))
    {
        config.sticky_iou_threshold = std::max(0.0f, std::min(config.sticky_iou_threshold, 1.0f)); // Clamp value
        config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Minimum Intersection over Union required to consider a target sticky.");
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