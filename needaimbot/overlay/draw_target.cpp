#include "../core/windows_headers.h"

#include <iostream>
#include <algorithm>
#include "d3d11.h"
#include "../imgui/imgui.h"

#include "AppContext.h"
#include "overlay.h"
#include "draw_settings.h"
#include "needaimbot.h"
#include "../cuda/unified_graph_pipeline.h"
#include "other_tools.h"
#include "memory_images.h"
#include "ui_helpers.h"

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

    // Status Card
    UIHelpers::BeginCard("Status");
    {
        bool is_paused = ctx.detection_paused.load();
        bool is_aiming = ctx.aiming.load();

        // Status row with indicator
        if (is_paused) {
            ImGui::PushStyleColor(ImGuiCol_Text, UIHelpers::GetErrorColor());
            ImGui::Text("PAUSED");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextDisabled("- Press %s to resume",
                        ctx.config.button_pause.empty() ? "F3" : ctx.config.button_pause[0].c_str());
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, UIHelpers::GetSuccessColor());
            ImGui::Text("ACTIVE");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            if (is_aiming) {
                ImGui::TextDisabled("- Aiming...");
            } else {
                ImGui::TextDisabled("- Ready");
            }
        }
    }
    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    // Main Controls Card
    UIHelpers::BeginCard(UIStrings::CardControls().c_str());
    {
        if (ImGui::BeginTable("##target_controls", 2, ImGuiTableFlags_NoBordersInBody)) {
            ImGui::TableSetupColumn("Toggle", ImGuiTableColumnFlags_WidthFixed, 180.0f);
            ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);

            // Auto Aim
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (UIHelpers::BeautifulToggle("Auto Aim", &ctx.config.auto_aim)) {
                SAVE_PROFILE();
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Track detected targets automatically");

            // Auto Shoot
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (UIHelpers::BeautifulToggle("Auto Shoot", &ctx.config.auto_shoot)) {
                SAVE_PROFILE();
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Fire automatically when locked on");

            // Disable Upward Aim (permanent toggle)
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (UIHelpers::BeautifulToggle("Block Upward Aim", &ctx.config.ignore_up_aim)) {
                SAVE_PROFILE();
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Never aim upward (always on)");

            ImGui::EndTable();
        }

        // Show hotkey-based disable upward aim status
        if (!ctx.config.button_disable_upward_aim.empty() &&
            ctx.config.button_disable_upward_aim[0] != "None") {
            UIHelpers::CompactSpacer();
            bool hotkey_active = ctx.disable_upward_aim.load();
            if (hotkey_active) {
                ImGui::TextColored(UIHelpers::GetWarningColor(), "Upward aim blocked (hotkey held)");
            }
        }
    }
    UIHelpers::EndCard();

    UIHelpers::CompactSpacer();

    // Target Selection Card
    UIHelpers::BeginCard("Target Selection");
    {
        UIHelpers::BeautifulText("Targets enemy closest to crosshair", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        UIHelpers::CompactSpacer();

        if (UIHelpers::EnhancedSliderFloat("Target Stickiness", &ctx.config.iou_stickiness_threshold,
                                           0.0f, 0.9f, "%.2f",
                                           "Keep tracking same target if overlap > threshold.\nHigher = less target switching")) {
            ctx.config.iou_stickiness_threshold = std::clamp(ctx.config.iou_stickiness_threshold, 0.0f, 0.99f);
            SAVE_PROFILE();
            auto* pipeline = gpa::PipelineManager::getInstance().getPipeline();
            if (pipeline) {
                pipeline->markPidConfigDirty();
                pipeline->setGraphRebuildNeeded();
            }
        }
    }
    UIHelpers::EndCard();
}

 
