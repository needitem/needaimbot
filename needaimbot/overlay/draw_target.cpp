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

    ImGui::Text("Target Selection Behavior");
    ImGui::Spacing();

    float sticky = ctx.config.sticky_target_threshold;
    ImGui::Text("Sticky Target Threshold");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##StickyThreshold", &sticky, 0.0f, 1.0f, "%.2f")) {
        // Clamp value between 0.0 and 1.0
        if (sticky < 0.0f) sticky = 0.0f;
        if (sticky > 1.0f) sticky = 1.0f;
        ctx.config.sticky_target_threshold = sticky;
        SAVE_PROFILE();
    }

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Target Switching Sensitivity");
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f, 0.9f, 1.0f, 1.0f), "Controls how easily the aim switches between targets:");
        ImGui::BulletText("0.00 = Always switch to closest target");
        ImGui::BulletText("0.30 = Moderate stickiness (Recommended)");
        ImGui::BulletText("0.50 = High stickiness");
        ImGui::BulletText("1.00 = Never switch targets");
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                          "Higher values keep aim locked on current target longer,\neven when closer targets appear.");
        ImGui::EndTooltip();
    }

    ImGui::Spacing();

    // Visual feedback for current setting
    ImGui::ProgressBar(ctx.config.sticky_target_threshold, ImVec2(-1, 0), "");
    ImGui::SameLine(0, 10);
    if (ctx.config.sticky_target_threshold < 0.2f) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "Agile");
    } else if (ctx.config.sticky_target_threshold < 0.6f) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "Balanced");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "Sticky");
    }
}

 
