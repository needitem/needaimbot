#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "AppContext.h"
#include "ui_helpers_new.h"
#include "needaimbot.h"
#include "other_tools.h"
#include <d3d11.h>

extern ID3D11ShaderResourceView* bodyTexture;
extern ImVec2 bodyImageSize;

void draw_target_new()
{
    auto& ctx = AppContext::getInstance();
    
    UI::Space();
    
    // Offset Controls
    UI::Section("Target Offset");
    
    if (UI::Slider("Body Offset##target", &ctx.config.body_y_offset, -1.0f, 1.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    
    if (UI::Slider("Head Offset##target", &ctx.config.head_y_offset, 0.0f, 1.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    
    if (UI::Slider("Step Size##target", &ctx.config.offset_step, 0.001f, 0.1f, "%.3f")) {
        ctx.config.saveConfig();
    }
    UI::Tip("Adjustment step for arrow keys");
    
    UI::SmallSpace();
    ImGui::TextColored(UI::WarningColor(), "Arrow: Body | Shift+Arrow: Head");
    
    UI::Space();
    UI::Space();
    
    // Visual Guide
    if (bodyTexture) {
        UI::Section("Visual Guide");
        
        // Scale image to fit
        float max_height = 200.0f;
        float scale = max_height / bodyImageSize.y;
        ImVec2 size = ImVec2(bodyImageSize.x * scale, bodyImageSize.y * scale);
        
        // Center the image
        float avail_width = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX((avail_width - size.x) * 0.5f);
        
        ImGui::Image((void*)bodyTexture, size);
        
        // Draw offset lines
        ImVec2 pos = ImGui::GetItemRectMin();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        // Body line (red)
        float body_y = pos.y + (1.0f + (ctx.config.body_y_offset - 1.0f) / 1.0f) * size.y;
        draw_list->AddLine(ImVec2(pos.x, body_y), ImVec2(pos.x + size.x, body_y), 
                          IM_COL32(255, 0, 0, 255), 2.0f);
        draw_list->AddText(ImVec2(pos.x + size.x + 5, body_y - 7), 
                          IM_COL32(255, 0, 0, 255), "Body");
        
        // Head line (green)
        float body_015 = pos.y + (1.0f + (0.15f - 1.0f) / 1.0f) * size.y;
        float head_y = pos.y + (ctx.config.head_y_offset * (body_015 - pos.y));
        draw_list->AddLine(ImVec2(pos.x, head_y), ImVec2(pos.x + size.x, head_y), 
                          IM_COL32(0, 255, 0, 255), 2.0f);
        draw_list->AddText(ImVec2(pos.x + size.x + 5, head_y - 7), 
                          IM_COL32(0, 255, 0, 255), "Head");
    } else {
        UI::Warning("Image not loaded!");
    }
    
    UI::Space();
    UI::Space();
    
    // Features
    UI::Section("Features");
    
    if (UI::Toggle("Auto Aim##target", &ctx.config.auto_aim)) {
        ctx.config.saveConfig();
    }
    
    UI::Space();
    UI::Space();
    
    // Target Locking
    UI::Section("Target Locking");
    
    if (UI::Toggle("Enable Locking##target", &ctx.config.enable_target_locking)) {
        ctx.config.saveConfig();
    }
    
    if (ctx.config.enable_target_locking) {
        UI::SmallSpace();
        
        if (UI::Slider("IoU Threshold##locking", &ctx.config.target_locking_iou_threshold, 0.01f, 1.0f, "%.2f")) {
            ctx.config.saveConfig();
        }
        UI::Tip("Overlap required to maintain lock");
        
        float lost_frames = static_cast<float>(ctx.config.target_locking_max_lost_frames);
        if (UI::Slider("Lost Frames##locking", &lost_frames, 0.0f, 60.0f, "%.0f")) {
            ctx.config.target_locking_max_lost_frames = static_cast<int>(lost_frames);
            ctx.config.saveConfig();
        }
        UI::Tip("Frames before releasing lock");
    }
    
    UI::Space();
}