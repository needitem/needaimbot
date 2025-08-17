#include "draw_offset.h"
#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "AppContext.h"
#include "draw_settings.h"
#include "ui_helpers.h"
#include "cuda/simple_cuda_mat.h"
#include "../detector/detector.h"
#include <d3d11.h>

extern ID3D11ShaderResourceView* bodyTexture;
extern ImVec2 bodyImageSize;
extern ID3D11ShaderResourceView* g_debugSRV;
extern ID3D11Texture2D* g_debugTex;
extern float debug_scale;
extern int texW, texH;

// Functions from draw_debug.cpp
void uploadDebugFrame(const SimpleCudaMat& frameMat);
void drawDetections(ImDrawList* draw_list, ImVec2 image_pos, float scale);

void renderOffsetTab()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginSettingsSection("Target Offset Settings", "Adjust where the aimbot targets on bodies and heads");

    // Body and Head Y Offset controls
    if (ImGui::SliderFloat("Body Y Offset", &ctx.config.body_y_offset, -1.0f, 1.0f, "%.2f")) {
        SAVE_PROFILE();
    }
    UIHelpers::InfoTooltip("Adjusts the vertical targeting position on body targets. Positive values move the target point up, negative values move it down.");
    
    if (ImGui::SliderFloat("Head Y Offset", &ctx.config.head_y_offset, 0.0f, 1.0f, "%.2f")) {
        SAVE_PROFILE();
    }
    UIHelpers::InfoTooltip("Adjusts the vertical targeting position on head targets. 0.0 targets the top of the head, 1.0 targets the bottom.");
    
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

    // Visual representation of offsets
    ImGui::Text("Body/Head Offset Preview:");
    ImGui::Spacing();
    
    if (bodyTexture && bodyImageSize.x > 0 && bodyImageSize.y > 0)
    {
        // Scale the image to fit nicely in the UI
        float scale = 0.5f;  // Scale down to 50% for better UI fit
        ImVec2 scaledImageSize(bodyImageSize.x * scale, bodyImageSize.y * scale);
        
        // Draw the body image with offset indicators
        ImGui::Image((void*)bodyTexture, scaledImageSize);

        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Ensure we have valid image dimensions
        if (image_size.x > 0 && image_size.y > 0)
        {
            // Draw body offset line (red)
            float normalized_body_value = (ctx.config.body_y_offset - 1.0f) / 1.0f;
            float body_line_y = image_pos.y + (1.0f + normalized_body_value) * image_size.y;
            
            draw_list->AddLine(
                ImVec2(image_pos.x, body_line_y), 
                ImVec2(image_pos.x + image_size.x, body_line_y), 
                IM_COL32(255, 0, 0, 255), 
                2.0f
            );
            
            // Draw head offset line (green)
            // Head offset is calculated relative to body offset at 0.15
            float body_y_pos_at_015 = image_pos.y + (1.0f + (0.15f - 1.0f) / 1.0f) * image_size.y;
            float head_top_pos = image_pos.y;
            float head_line_y = head_top_pos + (ctx.config.head_y_offset * (body_y_pos_at_015 - head_top_pos));
            
            draw_list->AddLine(
                ImVec2(image_pos.x, head_line_y), 
                ImVec2(image_pos.x + image_size.x, head_line_y), 
                IM_COL32(0, 255, 0, 255), 
                2.0f
            );
            
            // Add text labels
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
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Preview image not available");
        ImGui::Text("Debug: bodyTexture=%p, size=%.0fx%.0f", bodyTexture, bodyImageSize.x, bodyImageSize.y);
    }

    UIHelpers::EndSettingsSection();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Capture Region Offset Settings
    UIHelpers::BeginSettingsSection("Capture Region Offset", "Adjust the capture area position (crosshair stays centered)");

    const float adjustment_step = 1.0f;
    bool offset_changed = false;
    bool aim_shoot_offset_changed = false;
    
    // Create a table for better layout
    if (ImGui::BeginTable("CrosshairOffsetTable", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupColumn("Normal Offset", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Aim+Shoot Offset", ImGuiTableColumnFlags_WidthStretch);
        
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        
        // Left column - Normal Offset
        {
            ImGui::Text("Capture Region Offset (?)");
            ImGui::SameLine();
            UIHelpers::InfoTooltip("Moves the capture region while keeping crosshair centered on screen");
            ImGui::Text("X=%.0f, Y=%.0f", ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y);
            ImGui::Spacing();
            
            // Center the D-pad controls
            float button_size = 25.0f;
            float spacing = 2.0f;
            float group_width = button_size * 3 + spacing * 2;
            float center_offset = (ImGui::GetContentRegionAvail().x - group_width) * 0.5f;
            
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + center_offset);
            ImGui::BeginGroup();
            {
                // Top button (Up)
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + button_size + spacing);
                if (ImGui::Button("UP##offset_up", ImVec2(button_size, button_size))) {
                    ctx.config.crosshair_offset_y -= adjustment_step;  // Negative to move capture up
                    offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move capture region up");
                
                // Middle row (Left, Center, Right)
                if (ImGui::Button("L##offset_left", ImVec2(button_size, button_size))) {
                    ctx.config.crosshair_offset_x -= adjustment_step;  // Negative to move capture left
                    offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move capture region left");
                
                ImGui::SameLine(0, spacing);
                ImGui::Dummy(ImVec2(button_size, button_size)); // Center space
                
                ImGui::SameLine(0, spacing);
                if (ImGui::Button("R##offset_right", ImVec2(button_size, button_size))) {
                    ctx.config.crosshair_offset_x += adjustment_step;  // Positive to move capture right
                    offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move capture region right");
                
                // Bottom button (Down)
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + button_size + spacing);
                if (ImGui::Button("DN##offset_down", ImVec2(button_size, button_size))) {
                    ctx.config.crosshair_offset_y += adjustment_step;  // Positive to move capture down
                    offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move capture region down");
            }
            ImGui::EndGroup();
            
            ImGui::Spacing();
            
            // Reset button centered
            float reset_width = 60.0f;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x - reset_width) * 0.5f);
            if (ImGui::Button("Reset##offset_reset", ImVec2(reset_width, 25))) {
                ctx.config.crosshair_offset_x = 0.0f;
                ctx.config.crosshair_offset_y = 0.0f;
                offset_changed = true;
                ctx.getCaptureState().markCrosshairOffsetChanged();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset capture region to center");
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Fine adjustment controls
            ImGui::Text("Fine Adjust:");
            ImGui::PushItemWidth(100);
            ImGui::SetNextItemWidth(60);
            ImGui::Text("X"); ImGui::SameLine();
            if (ImGui::DragFloat("##offset_x_fine", &ctx.config.crosshair_offset_x, 0.1f, -100.0f, 100.0f, "%.1f")) {
                offset_changed = true;
                ctx.getCaptureState().markCrosshairOffsetChanged();
            }
            ImGui::SetNextItemWidth(60);
            ImGui::Text("Y"); ImGui::SameLine();
            if (ImGui::DragFloat("##offset_y_fine", &ctx.config.crosshair_offset_y, 0.1f, -100.0f, 100.0f, "%.1f")) {
                offset_changed = true;
                ctx.getCaptureState().markCrosshairOffsetChanged();
            }
            ImGui::PopItemWidth();
        }
        
        ImGui::TableNextColumn();
        
        // Right column - Aim+Shoot Offset
        {
            if (ImGui::Checkbox("Enable Aim+Shoot Offset (?)", &ctx.config.enable_aim_shoot_offset)) {
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            UIHelpers::InfoTooltip("Use different offset when both aim and shoot buttons are pressed");
            
            if (ctx.config.enable_aim_shoot_offset) {
                ImGui::Text("X=%.0f, Y=%.0f", ctx.config.aim_shoot_offset_x, ctx.config.aim_shoot_offset_y);
                ImGui::Spacing();
                
                // Center the D-pad controls
                float button_size = 25.0f;
                float spacing = 2.0f;
                float group_width = button_size * 3 + spacing * 2;
                float center_offset = (ImGui::GetContentRegionAvail().x - group_width) * 0.5f;
                
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + center_offset);
                ImGui::BeginGroup();
                {
                    // Top button (Up)
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + button_size + spacing);
                    if (ImGui::Button("UP##aim_shoot_up", ImVec2(button_size, button_size))) {
                        ctx.config.aim_shoot_offset_y -= adjustment_step;  // Negative to move capture up
                        aim_shoot_offset_changed = true;
                        ctx.getCaptureState().markCrosshairOffsetChanged();
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move aim+shoot capture region up");
                    
                    // Middle row (Left, Center, Right)
                    if (ImGui::Button("L##aim_shoot_left", ImVec2(button_size, button_size))) {
                        ctx.config.aim_shoot_offset_x -= adjustment_step;  // Negative to move capture left
                        aim_shoot_offset_changed = true;
                        ctx.getCaptureState().markCrosshairOffsetChanged();
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move aim+shoot capture region left");
                    
                    ImGui::SameLine(0, spacing);
                    ImGui::Dummy(ImVec2(button_size, button_size)); // Center space
                    
                    ImGui::SameLine(0, spacing);
                    if (ImGui::Button("R##aim_shoot_right", ImVec2(button_size, button_size))) {
                        ctx.config.aim_shoot_offset_x += adjustment_step;  // Positive to move capture right
                        aim_shoot_offset_changed = true;
                        ctx.getCaptureState().markCrosshairOffsetChanged();
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move aim+shoot capture region right");
                    
                    // Bottom button (Down)
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + button_size + spacing);
                    if (ImGui::Button("DN##aim_shoot_down", ImVec2(button_size, button_size))) {
                        ctx.config.aim_shoot_offset_y += adjustment_step;  // Positive to move capture down
                        aim_shoot_offset_changed = true;
                        ctx.getCaptureState().markCrosshairOffsetChanged();
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move aim+shoot capture region down");
                }
                ImGui::EndGroup();
                
                ImGui::Spacing();
                
                // Copy Normal button centered
                float copy_width = 85.0f;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x - copy_width) * 0.5f);
                if (ImGui::Button("Copy Normal##copy", ImVec2(copy_width, 25))) {
                    ctx.config.aim_shoot_offset_x = ctx.config.crosshair_offset_x;
                    ctx.config.aim_shoot_offset_y = ctx.config.crosshair_offset_y;
                    aim_shoot_offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copy offset values from normal crosshair");
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                // Fine adjustment controls
                ImGui::Text("Fine Adjust:");
                ImGui::PushItemWidth(100);
                ImGui::SetNextItemWidth(60);
                ImGui::Text("X"); ImGui::SameLine();
                if (ImGui::DragFloat("##aim_shoot_x_fine", &ctx.config.aim_shoot_offset_x, 0.1f, -100.0f, 100.0f, "%.1f")) {
                    aim_shoot_offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                ImGui::SetNextItemWidth(60);
                ImGui::Text("Y"); ImGui::SameLine();
                if (ImGui::DragFloat("##aim_shoot_y_fine", &ctx.config.aim_shoot_offset_y, 0.1f, -100.0f, 100.0f, "%.1f")) {
                    aim_shoot_offset_changed = true;
                    ctx.getCaptureState().markCrosshairOffsetChanged();
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Aim+Shoot offset is disabled");
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Enable to use different offset");
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "when both buttons are pressed");
            }
        }
        
        ImGui::EndTable();
    }
    
    // Save config when offset changes
    if (offset_changed || aim_shoot_offset_changed) {
        SAVE_PROFILE();
    }

    UIHelpers::EndSettingsSection();

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
        
        // Safely check and upload frame with defensive programming
        try {
            // Synchronize CUDA before checking frame
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "CUDA sync error: %s", cudaGetErrorString(sync_err));
                UIHelpers::EndSettingsSection();
                return;
            }
            
            // Get current frame from detector and upload it
            if (ctx.detector && !ctx.detector->currentFrame.empty()) {
                uploadDebugFrame(ctx.detector->currentFrame);
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Waiting for frame...");
            }
        } catch (const std::exception& e) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error uploading frame: %s", e.what());
            UIHelpers::EndSettingsSection();
            return;
        } catch (...) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Unknown error uploading frame");
            UIHelpers::EndSettingsSection();
            return;
        }
            
            ImGui::SliderFloat("Preview Scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
            
            if (g_debugSRV && texW > 0 && texH > 0 && texW < 10000 && texH < 10000) {
                // Validate scale
                float safe_scale = debug_scale;
                if (safe_scale <= 0 || safe_scale > 10.0f) safe_scale = 1.0f;
                
                ImVec2 image_size(texW * safe_scale, texH * safe_scale);
                
                // Safety check for ImGui rendering
                if (image_size.x > 0 && image_size.y > 0 && image_size.x < 10000 && image_size.y < 10000) {
                    ImGui::Image(g_debugSRV, image_size);
                } else {
                    ImGui::TextUnformatted("Invalid image dimensions for display");
                    UIHelpers::EndSettingsSection();
                    return;
                }
                
                ImVec2 image_pos = ImGui::GetItemRectMin();
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                
                // Validate draw list
                if (!draw_list) {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed to get draw list");
                    UIHelpers::EndSettingsSection();
                    return;
                }

                // Draw detections with safety check
                if (draw_list && debug_scale > 0 && debug_scale < 10.0f) {
                    drawDetections(draw_list, image_pos, debug_scale);
                }

                // Draw crosshair at center (no offset since capture region already includes offset)
                // Validate texture dimensions
                if (texW <= 0 || texH <= 0 || texW > 10000 || texH > 10000) {
                    // Skip drawing crosshair for invalid frame
                } else {
                    float center_x = image_pos.x + (texW * debug_scale) / 2.0f;
                    float center_y = image_pos.y + (texH * debug_scale) / 2.0f;
                
                // Determine which offset is currently active
                bool is_aim_shoot_active = ctx.config.enable_aim_shoot_offset && ctx.aiming.load() && ctx.shooting.load();
                ImU32 crosshair_color = is_aim_shoot_active ? IM_COL32(255, 128, 0, 255) : IM_COL32(255, 255, 255, 255);
                
                // Draw crosshair lines
                draw_list->AddLine(ImVec2(center_x - 10, center_y), ImVec2(center_x + 10, center_y), crosshair_color, 2.0f);
                draw_list->AddLine(ImVec2(center_x, center_y - 10), ImVec2(center_x, center_y + 10), crosshair_color, 2.0f);
                
                // Draw center circle
                draw_list->AddCircle(ImVec2(center_x, center_y), 3.0f, crosshair_color, 0, 2.0f);
                
                // Always show debug info
                char debug_info[256];
                snprintf(debug_info, sizeof(debug_info), "Debug: Aim=%d, Shoot=%d, Enabled=%d", 
                        ctx.aiming.load() ? 1 : 0, 
                        ctx.shooting.load() ? 1 : 0,
                        ctx.config.enable_aim_shoot_offset ? 1 : 0);
                draw_list->AddText(ImVec2(image_pos.x + 10, image_pos.y + 10), IM_COL32(255, 255, 0, 255), debug_info);
                
                // Show current offset info
                if (ctx.config.enable_aim_shoot_offset) {
                    const char* offset_text = is_aim_shoot_active ? "Aim+Shoot Offset Active" : "Normal Offset Active";
                    ImU32 text_color = is_aim_shoot_active ? IM_COL32(255, 128, 0, 255) : IM_COL32(255, 255, 255, 255);
                    
                    // Display current offset status
                    draw_list->AddText(ImVec2(image_pos.x + 10, image_pos.y + 30), text_color, offset_text);
                    
                    // Show offset values
                    char offset_info[256];
                    if (is_aim_shoot_active) {
                        snprintf(offset_info, sizeof(offset_info), "Offset: X=%.0f, Y=%.0f", 
                                ctx.config.aim_shoot_offset_x, ctx.config.aim_shoot_offset_y);
                    } else {
                        snprintf(offset_info, sizeof(offset_info), "Offset: X=%.0f, Y=%.0f", 
                                ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y);
                    }
                    draw_list->AddText(ImVec2(image_pos.x + 10, image_pos.y + 50), text_color, offset_info);
                } else {
                    // Show normal offset when aim+shoot is disabled
                    draw_list->AddText(ImVec2(image_pos.x + 10, image_pos.y + 30), IM_COL32(255, 255, 255, 255), "Normal Offset Active");
                    char offset_info[256];
                    snprintf(offset_info, sizeof(offset_info), "Offset: X=%.0f, Y=%.0f", 
                            ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y);
                    draw_list->AddText(ImVec2(image_pos.x + 10, image_pos.y + 50), IM_COL32(255, 255, 255, 255), offset_info);
                }
                }  // Close the else block for texture validation
            } else {
                ImGui::TextUnformatted("Preview texture unavailable for display.");
            }
    }

    UIHelpers::EndSettingsSection();
}