#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <imgui.h>
#include "imgui/imgui_internal.h"

#include "AppContext.h"
#include "config/config.h" 
#include "needaimbot.h"


void HelpMarker(const char* desc)
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





void draw_hsv_filter_settings()
{
    auto& ctx = AppContext::getInstance();
    
    ImGui::SeparatorText("HSV Color Filter Settings");
    ImGui::Spacing();

    bool changed = false;

    
    changed |= ImGui::Checkbox("Enable HSV Filter", &ctx.config.enable_hsv_filter);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Enable or disable HSV color filtering."); }
    ImGui::Spacing();

    
    ImGui::BeginDisabled(!ctx.config.enable_hsv_filter);

    
    ImGui::Text("Lower HSV Bounds:");
    changed |= ImGui::SliderInt("Lower H ##hsv", &ctx.config.hsv_lower_h, 0, 179);
    ImGui::SameLine(); HelpMarker("Hue (0-179)");
    changed |= ImGui::SliderInt("Lower S ##hsv", &ctx.config.hsv_lower_s, 0, 255);
    ImGui::SameLine(); HelpMarker("Saturation (0-255)");
    changed |= ImGui::SliderInt("Lower V ##hsv", &ctx.config.hsv_lower_v, 0, 255);
    ImGui::SameLine(); HelpMarker("Value (0-255)");

    ImGui::Spacing();
    
    ImGui::Text("Upper HSV Bounds:");
    changed |= ImGui::SliderInt("Upper H ##hsv", &ctx.config.hsv_upper_h, 0, 179);
    ImGui::SameLine(); HelpMarker("Hue (0-179)");
    changed |= ImGui::SliderInt("Upper S ##hsv", &ctx.config.hsv_upper_s, 0, 255);
    ImGui::SameLine(); HelpMarker("Saturation (0-255)");
    changed |= ImGui::SliderInt("Upper V ##hsv", &ctx.config.hsv_upper_v, 0, 255);
    ImGui::SameLine(); HelpMarker("Value (0-255)");

    ImGui::Spacing();
    
    changed |= ImGui::SliderInt("Min Pixels ##hsv", &ctx.config.min_hsv_pixels, 1, 1000);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Minimum number of pixels within the bounding box that must match the HSV range."); }

    
    changed |= ImGui::Checkbox("Remove boxes matching HSV filter", &ctx.config.remove_hsv_matches);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("By default, boxes with min_hsv_pixels matching HSV are kept; enable to remove them instead."); }

    ImGui::Spacing();
    
    if (ctx.config.hsv_lower_h > ctx.config.hsv_upper_h || ctx.config.hsv_lower_s > ctx.config.hsv_upper_s || ctx.config.hsv_lower_v > ctx.config.hsv_upper_v) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Warning: Lower bound is greater than upper bound for one or more HSV components.");
    }

    
    ImGui::Spacing();
    ImGui::Text("Preview Colors:");
    float lh = ctx.config.hsv_lower_h / 179.0f;
    float ls = ctx.config.hsv_lower_s / 255.0f;
    float lv = ctx.config.hsv_lower_v / 255.0f;
    float uh = ctx.config.hsv_upper_h / 179.0f;
    float us = ctx.config.hsv_upper_s / 255.0f;
    float uv = ctx.config.hsv_upper_v / 255.0f;
    float lr, lg, lb;
    float ur, ug, ub;
    ImGui::ColorConvertHSVtoRGB(lh, ls, lv, lr, lg, lb);
    ImGui::ColorConvertHSVtoRGB(uh, us, uv, ur, ug, ub);
    ImGui::SameLine(); ImGui::ColorButton("Lower Color##hsv", ImVec4(lr, lg, lb, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(30, 30));
    ImGui::SameLine(); ImGui::ColorButton("Upper Color##hsv", ImVec4(ur, ug, ub, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(30, 30));

    ImGui::EndDisabled();

    if (changed) {
        ctx.config.saveConfig();
    }
} 
