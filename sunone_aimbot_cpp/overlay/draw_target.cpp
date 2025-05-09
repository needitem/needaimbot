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
    ImGui::SliderFloat("Body Y Offset", &config.body_y_offset, -1.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Head Y Offset", &config.head_y_offset, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Offset Step", &config.offset_step, 0.001f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Step size for adjusting offset values with arrow keys (0.001 - 0.1)");
    }

    ImGui::Separator();

    // Instructions for keyboard controls
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Arrow keys: Adjust body offset");
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Shift+Arrow keys: Adjust head offset");

    if (bodyTexture)
    {
        ImGui::Image((void*)bodyTexture, bodyImageSize);

        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();

        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw body offset line
        float normalized_body_value = (config.body_y_offset - 1.0f) / 1.0f;
        float body_line_y = image_pos.y + (1.0f + normalized_body_value) * image_size.y;
        ImVec2 body_line_start = ImVec2(image_pos.x, body_line_y);
        ImVec2 body_line_end = ImVec2(image_pos.x + image_size.x, body_line_y);
        draw_list->AddLine(body_line_start, body_line_end, IM_COL32(255, 0, 0, 255), 2.0f);
        
        // Draw head offset line - modified calculation
        // When head_y_offset is 0.0, the line is at the top of the image
        // When head_y_offset is 1.0, the line is at the position where body_y_offset is 0.15
        float body_y_pos_at_015 = image_pos.y + (1.0f + (0.15f - 1.0f) / 1.0f) * image_size.y;
        float head_top_pos = image_pos.y;
        float head_line_y = head_top_pos + (config.head_y_offset * (body_y_pos_at_015 - head_top_pos));
        
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
    ImGui::Checkbox("Auto Aim", &config.auto_aim);

    ImGui::Separator(); // Separator for Target Locking section
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Target Locking Settings");
    ImGui::Checkbox("Enable Target Locking", &config.enable_target_locking);
    ImGui::SliderFloat("Locking IoU Threshold", &config.target_locking_iou_threshold, 0.01f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("How much a new detection must overlap with the locked target (0.01 - 1.0)");
    }
    ImGui::SliderInt("Max Lost Frames", &config.target_locking_max_lost_frames, 0, 60);
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Frames the target can be lost before lock is released (0-60)");
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