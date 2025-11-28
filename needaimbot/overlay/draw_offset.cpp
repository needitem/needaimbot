#include "draw_offset.h"
#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "AppContext.h"
#include "../core/constants.h"
#include "draw_settings.h"
#include "ui_helpers.h"
#include "cuda/simple_cuda_mat.h"
#include "cuda/unified_graph_pipeline.h"
#include <d3d11.h>
#include <algorithm>
#include <vector>

extern ID3D11ShaderResourceView* g_debugSRV;
extern ID3D11Texture2D* g_debugTex;
extern float debug_scale;
extern int texW, texH;

// Functions from draw_debug.cpp
void uploadDebugFrame(const SimpleMat& frameMat);
void drawDetections(ImDrawList* draw_list, ImVec2 image_pos, float scale, const std::vector<Target>* targets_override = nullptr);

// Mutex for thread-safe D3D11 resource access (defined in draw_debug.cpp)
extern std::mutex g_debugTexMutex;

// Helper: Render compact D-pad controls for offset adjustment
static void renderOffsetDPad(const char* id_prefix, float& offset_x, float& offset_y, bool& changed_flag, float adjustment_step = 1.0f)
{
    auto& ctx = AppContext::getInstance();

    const float btn_size = 28.0f;
    const float spacing = 2.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

    // Center the D-pad
    float group_width = btn_size * 3 + spacing * 2;
    float center_offset = (ImGui::GetContentRegionAvail().x - group_width) * 0.5f;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + center_offset);

    ImGui::BeginGroup();
    {
        // Row 1: Up button centered
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + btn_size + spacing);
        if (ImGui::Button(("^##" + std::string(id_prefix) + "_up").c_str(), ImVec2(btn_size, btn_size))) {
            offset_y -= adjustment_step;
            changed_flag = true;
            ctx.crosshair_offset_changed = true;
        }

        // Row 2: Left, display, Right
        if (ImGui::Button(("<##" + std::string(id_prefix) + "_left").c_str(), ImVec2(btn_size, btn_size))) {
            offset_x -= adjustment_step;
            changed_flag = true;
            ctx.crosshair_offset_changed = true;
        }
        ImGui::SameLine();

        // Center display showing current values
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        char val_buf[16];
        snprintf(val_buf, sizeof(val_buf), "%.0f", offset_x);
        ImGui::Button(val_buf, ImVec2(btn_size, btn_size));
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::Button((">##" + std::string(id_prefix) + "_right").c_str(), ImVec2(btn_size, btn_size))) {
            offset_x += adjustment_step;
            changed_flag = true;
            ctx.crosshair_offset_changed = true;
        }

        // Row 3: Down button centered
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + btn_size + spacing);
        if (ImGui::Button(("v##" + std::string(id_prefix) + "_down").c_str(), ImVec2(btn_size, btn_size))) {
            offset_y += adjustment_step;
            changed_flag = true;
            ctx.crosshair_offset_changed = true;
        }
    }
    ImGui::EndGroup();

    ImGui::PopStyleVar(2);
}

// Helper: Render fine adjustment drag floats for offset
static void renderFineAdjustment(const char* id_prefix, float& offset_x, float& offset_y, bool& changed_flag)
{
    auto& ctx = AppContext::getInstance();

    if (ImGui::BeginTable(("##fine_" + std::string(id_prefix)).c_str(), 2, ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupColumn("X", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Y", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::DragFloat(("##x_" + std::string(id_prefix)).c_str(), &offset_x,
                            Constants::OFFSET_DRAG_SPEED, Constants::OFFSET_DRAG_MIN, Constants::OFFSET_DRAG_MAX, "X: %.1f")) {
            changed_flag = true;
            ctx.crosshair_offset_changed = true;
        }

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::DragFloat(("##y_" + std::string(id_prefix)).c_str(), &offset_y,
                            Constants::OFFSET_DRAG_SPEED, Constants::OFFSET_DRAG_MIN, Constants::OFFSET_DRAG_MAX, "Y: %.1f")) {
            changed_flag = true;
            ctx.crosshair_offset_changed = true;
        }

        ImGui::EndTable();
    }
}

void renderOffsetTab()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginSettingsSection("Target Offset Settings", "Adjust where the aimbot targets on bodies and heads");

    // Body and Head Y Offset controls
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::SliderFloat("Body Y Offset", &ctx.config.body_y_offset, -1.0f, 1.0f, "%.2f")) {
        SAVE_PROFILE();
    }
    UIHelpers::InfoTooltip("Adjusts the vertical targeting position on body targets. Positive values move the target point up, negative values move it down.");

    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::SliderFloat("Head Y Offset", &ctx.config.head_y_offset, 0.0f, 1.0f, "%.2f")) {
        SAVE_PROFILE();
    }
    UIHelpers::InfoTooltip("Adjusts the vertical targeting position on head targets. 0.0 targets the top of the head, 1.0 targets the bottom.");

    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::SliderFloat("Offset Step", &ctx.config.offset_step, 0.001f, 0.1f, "%.3f")) {
        SAVE_PROFILE();
    }
    UIHelpers::InfoTooltip("Step size for adjusting offset values with arrow keys (0.001 - 0.1)");

    ImGui::Separator();
    ImGui::Spacing();

    // Keyboard shortcut instructions
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Keyboard Shortcuts:");
    ImGui::Text("Arrow keys: Adjust body offset");
    ImGui::Text("Shift+Arrow keys: Adjust head offset");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Visual representation of offsets using loaded image
    ImGui::Text("Body/Head Offset Preview:");
    ImGui::Spacing();

    extern ID3D11ShaderResourceView* bodyTexture;
    extern ImVec2 bodyImageSize;

    if (bodyTexture && bodyImageSize.x > 0 && bodyImageSize.y > 0)
    {
        // Display the body image
        ImGui::Image((void*)bodyTexture, bodyImageSize);

        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        if (image_size.x > 0 && image_size.y > 0)
        {
            // Body offset line calculation (from SunOne's code)
            float normalized_body_value = (ctx.config.body_y_offset - 1.0f) / 1.0f;
            float body_line_y = image_pos.y + (1.0f + normalized_body_value) * image_size.y;

            // Head offset line calculation (from SunOne's code)
            float body_y_pos_at_015 = image_pos.y + (1.0f + (0.15f - 1.0f) / 1.0f) * image_size.y;
            float head_top_pos = image_pos.y;
            float head_line_y = head_top_pos + (ctx.config.head_y_offset * (body_y_pos_at_015 - head_top_pos));

            // Draw body offset line (red)
            draw_list->AddLine(
                ImVec2(image_pos.x, body_line_y),
                ImVec2(image_pos.x + image_size.x, body_line_y),
                IM_COL32(255, 0, 0, 255),
                2.0f
            );

            // Draw head offset line (green)
            draw_list->AddLine(
                ImVec2(image_pos.x, head_line_y),
                ImVec2(image_pos.x + image_size.x, head_line_y),
                IM_COL32(0, 255, 0, 255),
                2.0f
            );

            // Labels
            draw_list->AddText(
                ImVec2(image_pos.x + image_size.x + 5, body_line_y - 7),
                IM_COL32(255, 0, 0, 255),
                "Body"
            );
            draw_list->AddText(
                ImVec2(image_pos.x + image_size.x + 5, head_line_y - 7),
                IM_COL32(0, 255, 0, 255),
                "Head"
            );
        }
    }
    else
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Body image not loaded");
    }

    UIHelpers::EndSettingsSection();

    UIHelpers::CompactSpacer();

    // Capture Region Offset Settings - using BeginCard for consistency
    UIHelpers::BeginCard("Capture Region Offset");

    const float adjustment_step = 1.0f;
    bool offset_changed = false;
    bool aim_shoot_offset_changed = false;

    // Use tabs for Normal vs Aim+Shoot offsets
    if (ImGui::BeginTabBar("##offset_tabs")) {
        // Normal Offset Tab
        if (ImGui::BeginTabItem("Normal")) {
            UIHelpers::CompactSpacer();
            ImGui::TextDisabled("Capture region offset (Y: %.0f)", ctx.config.crosshair_offset_y);

            renderOffsetDPad("offset", ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y, offset_changed, adjustment_step);

            UIHelpers::CompactSpacer();
            renderFineAdjustment("offset", ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y, offset_changed);

            UIHelpers::CompactSpacer();
            if (UIHelpers::BeautifulButton("Reset to Center", ImVec2(-1, 0))) {
                ctx.config.crosshair_offset_x = 0.0f;
                ctx.config.crosshair_offset_y = 0.0f;
                offset_changed = true;
                ctx.crosshair_offset_changed = true;
            }

            ImGui::EndTabItem();
        }

        // Aim+Shoot Offset Tab
        if (ImGui::BeginTabItem("Aim+Shoot")) {
            UIHelpers::CompactSpacer();

            if (UIHelpers::BeautifulToggle("Enable", &ctx.config.enable_aim_shoot_offset,
                                           "Use different offset when aiming and shooting")) {
                SAVE_PROFILE();
            }

            if (ctx.config.enable_aim_shoot_offset) {
                UIHelpers::CompactSpacer();
                ImGui::TextDisabled("Offset when aim+shoot (Y: %.0f)", ctx.config.aim_shoot_offset_y);

                renderOffsetDPad("aim_shoot", ctx.config.aim_shoot_offset_x, ctx.config.aim_shoot_offset_y, aim_shoot_offset_changed, adjustment_step);

                UIHelpers::CompactSpacer();
                renderFineAdjustment("aim_shoot", ctx.config.aim_shoot_offset_x, ctx.config.aim_shoot_offset_y, aim_shoot_offset_changed);

                UIHelpers::CompactSpacer();
                if (UIHelpers::BeautifulButton("Copy from Normal", ImVec2(-1, 0))) {
                    ctx.config.aim_shoot_offset_x = ctx.config.crosshair_offset_x;
                    ctx.config.aim_shoot_offset_y = ctx.config.crosshair_offset_y;
                    aim_shoot_offset_changed = true;
                    ctx.crosshair_offset_changed = true;
                }
            } else {
                UIHelpers::CompactSpacer();
                ImGui::TextDisabled("Enable to use different offset");
                ImGui::TextDisabled("when both buttons are pressed");
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    if (offset_changed || aim_shoot_offset_changed) {
        SAVE_PROFILE();
    }

    UIHelpers::EndCard();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Preview Window Section
    UIHelpers::BeginSettingsSection("Live Preview", "See real-time capture with offset visualization");

    bool prev_show_window_state = ctx.config.show_window;
    if (ImGui::Checkbox("Enable Preview", &ctx.config.show_window)) {
        SAVE_PROFILE();
        
        // Clean up resources when disabling preview
        if (prev_show_window_state == true && ctx.config.show_window == false) {
            if (g_debugSRV) {
                g_debugSRV->Release();
                g_debugSRV = nullptr;
            }
            if (g_debugTex) {
                g_debugTex->Release();
                g_debugTex = nullptr;
            }
            texW = 0;
            texH = 0;
        }
    }
    UIHelpers::InfoTooltip("Shows the live capture feed with crosshair offset visualization");

    if (ctx.config.show_window) {
        ImGui::Spacing();

        SimpleMat* frameToDisplay = nullptr;
        static SimpleMat previewHostFrame;

        // Throttle preview updates to 15 FPS (every ~66ms) to reduce GPU->CPU copy overhead
        static auto lastPreviewUpdate = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPreviewUpdate);
        bool shouldUpdatePreview = (elapsed.count() >= 66);

        auto& pipelineManager = needaimbot::PipelineManager::getInstance();
        auto* pipeline = pipelineManager.getPipeline();

        std::vector<Target> previewTargets;
        Target previewBestTarget;
        bool hasPreviewBestTarget = false;

        if (ctx.hasValidTarget()) {
            try {
                previewBestTarget = ctx.getBestTarget();
                hasPreviewBestTarget = previewBestTarget.hasValidDetection();
            } catch (...) {
                hasPreviewBestTarget = false;
            }
        }

        if (pipeline && pipeline->isPreviewAvailable() && shouldUpdatePreview) {
            if (pipeline->getPreviewSnapshot(previewHostFrame) &&
                !previewHostFrame.empty() &&
                previewHostFrame.cols() > 0 && previewHostFrame.rows() > 0 &&
                previewHostFrame.cols() <= 10000 && previewHostFrame.rows() <= 10000) {
                frameToDisplay = &previewHostFrame;
                lastPreviewUpdate = now;
            }
            // No message - just show black if not ready
        } else if (!shouldUpdatePreview && !previewHostFrame.empty()) {
            // Reuse previous frame
            frameToDisplay = &previewHostFrame;
        }
        // No message for unavailable - just show black

        if (frameToDisplay) {
            try {
                uploadDebugFrame(*frameToDisplay);
            } catch (const std::exception& e) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Preview upload failed: %s", e.what());
                UIHelpers::EndSettingsSection();
                return;
            }

            previewTargets = ctx.getAllTargets();
            if (previewTargets.empty()) {
                hasPreviewBestTarget = false;
            }

            ImGui::SetNextItemWidth(150.0f);
            ImGui::SliderFloat("Preview Scale", &debug_scale, 0.1f, 3.0f, "%.1fx");

            std::lock_guard<std::mutex> lock(g_debugTexMutex);

            if (g_debugSRV && texW > 0 && texH > 0 && texW < 10000 && texH < 10000) {
                float safe_scale = debug_scale;
                if (safe_scale <= 0 || safe_scale > 10.0f) {
                    safe_scale = 1.0f;
                }

                ImVec2 image_size(texW * safe_scale, texH * safe_scale);
                if (image_size.x > 0 && image_size.y > 0 && image_size.x < 10000 && image_size.y < 10000) {
                    ImGui::Image(g_debugSRV, image_size);
                } else {
                    ImGui::TextUnformatted("Invalid image dimensions for display");
                    UIHelpers::EndSettingsSection();
                    return;
                }

                ImVec2 image_pos = ImGui::GetItemRectMin();
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                if (!draw_list) {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed to get draw list");
                    UIHelpers::EndSettingsSection();
                    return;
                }

                if (debug_scale > 0 && debug_scale < 10.0f) {
                    drawDetections(draw_list, image_pos, debug_scale, &previewTargets);

                    if (hasPreviewBestTarget && previewBestTarget.hasValidDetection()) {
                        float targetCenterX = image_pos.x +
                                              (previewBestTarget.x + previewBestTarget.width / 2.0f) * debug_scale;
                        float targetCenterY = image_pos.y +
                                              (previewBestTarget.y + previewBestTarget.height / 2.0f) * debug_scale;

                        float previewMinX = image_pos.x;
                        float previewMinY = image_pos.y;
                        float previewMaxX = image_pos.x + texW * debug_scale;
                        float previewMaxY = image_pos.y + texH * debug_scale;

                        if (targetCenterX >= previewMinX && targetCenterX <= previewMaxX &&
                            targetCenterY >= previewMinY && targetCenterY <= previewMaxY) {
                            float radiusPixels = static_cast<float>(
                                std::max(previewBestTarget.width, previewBestTarget.height)) * 0.5f * debug_scale;
                            if (radiusPixels < 6.0f) {
                                radiusPixels = 6.0f;
                            } else if (radiusPixels > 80.0f) {
                                radiusPixels = 80.0f;
                            }

                            draw_list->AddCircle(ImVec2(targetCenterX, targetCenterY), radiusPixels,
                                                 IM_COL32(0, 255, 0, 200), 0, 2.5f);
                            draw_list->AddCircleFilled(ImVec2(targetCenterX, targetCenterY), 4.0f,
                                                       IM_COL32(0, 255, 0, 255));
                            draw_list->AddLine(ImVec2(targetCenterX - radiusPixels * 0.5f, targetCenterY),
                                               ImVec2(targetCenterX + radiusPixels * 0.5f, targetCenterY),
                                               IM_COL32(0, 255, 0, 200), 2.0f);
                            draw_list->AddLine(ImVec2(targetCenterX, targetCenterY - radiusPixels * 0.5f),
                                               ImVec2(targetCenterX, targetCenterY + radiusPixels * 0.5f),
                                               IM_COL32(0, 255, 0, 200), 2.0f);

                            char selectedLabel[128];
                            snprintf(selectedLabel, sizeof(selectedLabel),
                                     "Selected Target (ID %d, %.0f%%)",
                                     previewBestTarget.classId,
                                     previewBestTarget.confidence * 100.0f);

                            draw_list->AddText(ImVec2(targetCenterX + radiusPixels + 6.0f,
                                                      targetCenterY - 12.0f),
                                               IM_COL32(0, 255, 0, 255),
                                               selectedLabel);
                        }
                    }
                }

                // Draw crosshair at center
                if (texW > 0 && texH > 0 && texW < 10000 && texH < 10000) {
                    float center_x = image_pos.x + (texW * debug_scale) / 2.0f;
                    float center_y = image_pos.y + (texH * debug_scale) / 2.0f;

                    bool is_aim_shoot_active = ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load();
                    ImU32 crosshair_color = is_aim_shoot_active ? IM_COL32(255, 128, 0, 255) : IM_COL32(255, 255, 255, 255);

                    draw_list->AddLine(ImVec2(center_x - 10, center_y), ImVec2(center_x + 10, center_y), crosshair_color, 2.0f);
                    draw_list->AddLine(ImVec2(center_x, center_y - 10), ImVec2(center_x, center_y + 10), crosshair_color, 2.0f);
                    draw_list->AddCircle(ImVec2(center_x, center_y), 3.0f, crosshair_color, 0, 2.0f);
                }
            }
        }

    }
    UIHelpers::EndSettingsSection();
}
