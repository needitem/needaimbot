#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "../imgui/imgui.h"
#include "../imgui/imgui_internal.h"
#include "AppContext.h"
#include "config/config.h"
#include "needaimbot.h"
#include "ui_helpers.h"
#include "draw_settings.h"

void HelpMarkerColorFilter(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void draw_color_filter_settings()
{
    auto& ctx = AppContext::getInstance();
    
    ImGui::SeparatorText("Color Filter Settings");
    ImGui::Spacing();

    bool changed = false;

    // Enable RGB color filter
    changed |= ImGui::Checkbox("Enable RGB Color Filter", &ctx.config.enable_color_filter);
    ImGui::SameLine();
    HelpMarkerColorFilter("Filter targets based on their RGB color values.");
    
    if (!ctx.config.enable_color_filter) {
        ImGui::BeginDisabled();
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // RGB Filter Controls
    ImGui::Text("RGB Color Range:");
    
    // Red channel
    ImGui::PushItemWidth(200);
    ImGui::Text("Red Channel:");
    ImGui::SameLine(120);
    changed |= ImGui::DragIntRange2("##RedRange", &ctx.config.rgb_min_r, &ctx.config.rgb_max_r, 1, 0, 255, "Min: %d", "Max: %d");
    
    // Green channel
    ImGui::Text("Green Channel:");
    ImGui::SameLine(120);
    changed |= ImGui::DragIntRange2("##GreenRange", &ctx.config.rgb_min_g, &ctx.config.rgb_max_g, 1, 0, 255, "Min: %d", "Max: %d");
    
    // Blue channel
    ImGui::Text("Blue Channel:");
    ImGui::SameLine(120);
    changed |= ImGui::DragIntRange2("##BlueRange", &ctx.config.rgb_min_b, &ctx.config.rgb_max_b, 1, 0, 255, "Min: %d", "Max: %d");
    ImGui::PopItemWidth();
    
    // Color preview
    ImGui::Spacing();
    ImGui::Text("Color Preview:");
    ImVec4 minColor(ctx.config.rgb_min_r / 255.0f, ctx.config.rgb_min_g / 255.0f, ctx.config.rgb_min_b / 255.0f, 1.0f);
    ImVec4 maxColor(ctx.config.rgb_max_r / 255.0f, ctx.config.rgb_max_g / 255.0f, ctx.config.rgb_max_b / 255.0f, 1.0f);
    
    ImGui::SameLine();
    ImGui::ColorButton("Min##preview", minColor, ImGuiColorEditFlags_NoTooltip, ImVec2(30, 20));
    ImGui::SameLine();
    ImGui::Text("-");
    ImGui::SameLine();
    ImGui::ColorButton("Max##preview", maxColor, ImGuiColorEditFlags_NoTooltip, ImVec2(30, 20));
    
    // Preset colors
    ImGui::Spacing();
    ImGui::Text("Presets:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Red")) {
        ctx.config.rgb_min_r = 180; ctx.config.rgb_max_r = 255;
        ctx.config.rgb_min_g = 0;   ctx.config.rgb_max_g = 80;
        ctx.config.rgb_min_b = 0;   ctx.config.rgb_max_b = 80;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Blue")) {
        ctx.config.rgb_min_r = 0;   ctx.config.rgb_max_r = 80;
        ctx.config.rgb_min_g = 0;   ctx.config.rgb_max_g = 80;
        ctx.config.rgb_min_b = 180; ctx.config.rgb_max_b = 255;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Green")) {
        ctx.config.rgb_min_r = 0;   ctx.config.rgb_max_r = 80;
        ctx.config.rgb_min_g = 180; ctx.config.rgb_max_g = 255;
        ctx.config.rgb_min_b = 0;   ctx.config.rgb_max_b = 80;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Yellow")) {
        ctx.config.rgb_min_r = 180; ctx.config.rgb_max_r = 255;
        ctx.config.rgb_min_g = 180; ctx.config.rgb_max_g = 255;
        ctx.config.rgb_min_b = 0;   ctx.config.rgb_max_b = 80;
        changed = true;
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Common settings
    ImGui::Text("Filter Settings:");
    
    changed |= ImGui::SliderInt("Min Pixels", &ctx.config.min_color_pixels, 1, 100);
    ImGui::SameLine();
    HelpMarkerColorFilter("Minimum number of matching pixels required in target bounding box");
    
    changed |= ImGui::Checkbox("Remove Matches", &ctx.config.remove_color_matches);
    ImGui::SameLine();
    HelpMarkerColorFilter("If enabled, removes targets that match the color. If disabled, keeps only matching targets.");
    
    if (!ctx.config.enable_color_filter) {
        ImGui::EndDisabled();
    }
    
    // Save configuration if changed
    if (changed) {
        SAVE_PROFILE();
    }
}