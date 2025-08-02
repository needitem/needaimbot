#include "draw_offset.h"
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "AppContext.h"
#include "draw_settings.h"
#include "ui_helpers.h"
#include "capture/frame_buffer_pool.h"
#include "cuda/simple_cuda_mat.h"
#include <d3d11.h>

extern ID3D11ShaderResourceView* bodyTexture;
extern ImVec2 bodyImageSize;
extern SimpleCudaMat latestFrameGpu;
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
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

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
    if (bodyTexture && bodyImageSize.x > 0 && bodyImageSize.y > 0)
    {
        ImGui::Text("Offset Preview:");
        ImGui::Spacing();
        
        // Draw the body image with offset indicators
        ImGui::Image((void*)bodyTexture, bodyImageSize);

        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();

        // Ensure we have valid image dimensions
        if (image_size.x > 0 && image_size.y > 0)
        {
            // Draw body offset line (red)
            // Body offset ranges from -1.0 to 1.0, where 0.15 is the default center
            float body_y_normalized = (ctx.config.body_y_offset + 1.0f) / 2.0f; // Convert from [-1,1] to [0,1]
            float body_line_y = image_pos.y + (1.0f - body_y_normalized) * image_size.y; // Invert Y because ImGui Y goes down
            
            draw_list->AddLine(
                ImVec2(image_pos.x, body_line_y), 
                ImVec2(image_pos.x + image_size.x, body_line_y), 
                IM_COL32(255, 0, 0, 255), 
                3.0f
            );
            
            // Draw head offset line (green)
            // Head offset ranges from 0.0 to 1.0, where 0.05 is the default
            float head_y_normalized = ctx.config.head_y_offset;
            float head_line_y = image_pos.y + head_y_normalized * image_size.y;
            
            draw_list->AddLine(
                ImVec2(image_pos.x, head_line_y), 
                ImVec2(image_pos.x + image_size.x, head_line_y), 
                IM_COL32(0, 255, 0, 255), 
                3.0f
            );
            
            // Add text labels
            draw_list->AddText(
                ImVec2(image_pos.x + image_size.x + 10, body_line_y - 8), 
                IM_COL32(255, 0, 0, 255), 
                "Body"
            );
            draw_list->AddText(
                ImVec2(image_pos.x + image_size.x + 10, head_line_y - 8), 
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

    // Crosshair Offset Settings (separate section)
    UIHelpers::BeginSettingsSection("Crosshair Offset", "Fine-tune the crosshair position for your display");

    ImGui::Text("Current Offset: X=%.1f, Y=%.1f", ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y);
    ImGui::Spacing();
    
    const float adjustment_step = 1.0f;
    bool offset_changed = false;
    
    // Directional adjustment buttons in cross formation
    ImGui::BeginGroup();
    {
        // Top button (Up - increase Y)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 30.0f);
        if (ImGui::Button("UP##offset_up", ImVec2(30, 30))) {
            ctx.config.crosshair_offset_y += adjustment_step;
            offset_changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move crosshair up");
        
        // Middle row (Left and Right)
        if (ImGui::Button("L##offset_left", ImVec2(30, 30))) {
            ctx.config.crosshair_offset_x += adjustment_step;
            offset_changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move crosshair left");
        
        ImGui::SameLine();
        if (ImGui::Button("R##offset_right", ImVec2(30, 30))) {
            ctx.config.crosshair_offset_x -= adjustment_step;
            offset_changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move crosshair right");
        
        // Bottom button (Down - decrease Y)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 30.0f);
        if (ImGui::Button("DN##offset_down", ImVec2(30, 30))) {
            ctx.config.crosshair_offset_y -= adjustment_step;
            offset_changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move crosshair down");
    }
    ImGui::EndGroup();
    
    ImGui::SameLine();
    ImGui::BeginGroup();
    {
        // Reset button
        if (ImGui::Button("Reset##offset_reset", ImVec2(60, 30))) {
            ctx.config.crosshair_offset_x = 0.0f;
            ctx.config.crosshair_offset_y = 0.0f;
            offset_changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset crosshair offset to center");
        
        // Fine adjustment controls
        ImGui::Spacing();
        ImGui::Text("Fine Adjust:");
        ImGui::PushItemWidth(80);
        if (ImGui::DragFloat("X##offset_x_fine", &ctx.config.crosshair_offset_x, 0.1f, -100.0f, 100.0f, "%.1f")) {
            offset_changed = true;
        }
        if (ImGui::DragFloat("Y##offset_y_fine", &ctx.config.crosshair_offset_y, 0.1f, -100.0f, 100.0f, "%.1f")) {
            offset_changed = true;
        }
        ImGui::PopItemWidth();
    }
    ImGui::EndGroup();
    
    // Save config when offset changes
    if (offset_changed) {
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
        
        if (!latestFrameGpu.empty()) {
            // Upload GPU frame directly
            uploadDebugFrame(latestFrameGpu);
            
            ImGui::SliderFloat("Preview Scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
            
            if (g_debugSRV && texW > 0 && texH > 0) {
                ImVec2 image_size(texW * debug_scale, texH * debug_scale);
                ImGui::Image(g_debugSRV, image_size);
                
                ImVec2 image_pos = ImGui::GetItemRectMin();
                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                // Draw detections
                drawDetections(draw_list, image_pos, debug_scale);

                // Draw center crosshair with offset
                float center_x = image_pos.x + (texW * debug_scale) / 2.0f + (ctx.config.crosshair_offset_x * debug_scale);
                float center_y = image_pos.y + (texH * debug_scale) / 2.0f + (ctx.config.crosshair_offset_y * debug_scale);
                ImU32 crosshair_color = IM_COL32(255, 255, 255, 255);
                
                // Draw crosshair lines
                draw_list->AddLine(ImVec2(center_x - 10, center_y), ImVec2(center_x + 10, center_y), crosshair_color, 2.0f);
                draw_list->AddLine(ImVec2(center_x, center_y - 10), ImVec2(center_x, center_y + 10), crosshair_color, 2.0f);
                
                // Draw center circle
                draw_list->AddCircle(ImVec2(center_x, center_y), 3.0f, crosshair_color, 0, 2.0f);
            } else {
                ImGui::TextUnformatted("Preview texture unavailable for display.");
            }
        } else {
            ImGui::TextUnformatted("Preview frame unavailable. Make sure capture is running.");
        }
    }

    UIHelpers::EndSettingsSection();
}