#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <imgui.h>
#include "imgui/imgui_internal.h"

#include "config.h" // Access to global config object
#include "needaimbot.h"

// Helper function from ImGui demo to show tooltips
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

// Ensure this extern declaration matches the one in config.h/config.cpp
// extern Config config; // Already available via include "config.h" if setup correctly
// extern std::atomic<bool> hsv_filter_settings_changed; // Already available via "needaimbot.h"

void draw_hsv_filter_settings()
{
    ImGui::SeparatorText("HSV Color Filter Settings");
    ImGui::Spacing();

    bool changed = false;

    // Enable toggle
    changed |= ImGui::Checkbox("Enable HSV Filter", &config.enable_hsv_filter);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Enable or disable HSV color filtering."); }
    ImGui::Spacing();

    // Disable following controls when filter is off
    ImGui::BeginDisabled(!config.enable_hsv_filter);

    // Lower bounds
    ImGui::Text("Lower HSV Bounds:");
    changed |= ImGui::SliderInt("Lower H ##hsv", &config.hsv_lower_h, 0, 179);
    ImGui::SameLine(); HelpMarker("Hue (0-179)");
    changed |= ImGui::SliderInt("Lower S ##hsv", &config.hsv_lower_s, 0, 255);
    ImGui::SameLine(); HelpMarker("Saturation (0-255)");
    changed |= ImGui::SliderInt("Lower V ##hsv", &config.hsv_lower_v, 0, 255);
    ImGui::SameLine(); HelpMarker("Value (0-255)");

    ImGui::Spacing();
    // Upper bounds
    ImGui::Text("Upper HSV Bounds:");
    changed |= ImGui::SliderInt("Upper H ##hsv", &config.hsv_upper_h, 0, 179);
    ImGui::SameLine(); HelpMarker("Hue (0-179)");
    changed |= ImGui::SliderInt("Upper S ##hsv", &config.hsv_upper_s, 0, 255);
    ImGui::SameLine(); HelpMarker("Saturation (0-255)");
    changed |= ImGui::SliderInt("Upper V ##hsv", &config.hsv_upper_v, 0, 255);
    ImGui::SameLine(); HelpMarker("Value (0-255)");

    ImGui::Spacing();
    // Minimum pixels
    changed |= ImGui::SliderInt("Min Pixels ##hsv", &config.min_hsv_pixels, 1, 1000);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Minimum number of pixels within the bounding box that must match the HSV range."); }

    // Filter mode: keep matching or remove matching
    changed |= ImGui::Checkbox("Remove boxes matching HSV filter", &config.remove_hsv_matches);
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("By default, boxes with min_hsv_pixels matching HSV are kept; enable to remove them instead."); }

    ImGui::Spacing();
    // Warning if bounds are invalid
    if (config.hsv_lower_h > config.hsv_upper_h || config.hsv_lower_s > config.hsv_upper_s || config.hsv_lower_v > config.hsv_upper_v) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Warning: Lower bound is greater than upper bound for one or more HSV components.");
    }

    // Color preview swatches
    ImGui::Spacing();
    ImGui::Text("Preview Colors:");
    float lh = config.hsv_lower_h / 179.0f;
    float ls = config.hsv_lower_s / 255.0f;
    float lv = config.hsv_lower_v / 255.0f;
    float uh = config.hsv_upper_h / 179.0f;
    float us = config.hsv_upper_s / 255.0f;
    float uv = config.hsv_upper_v / 255.0f;
    float lr, lg, lb;
    float ur, ug, ub;
    ImGui::ColorConvertHSVtoRGB(lh, ls, lv, lr, lg, lb);
    ImGui::ColorConvertHSVtoRGB(uh, us, uv, ur, ug, ub);
    ImGui::SameLine(); ImGui::ColorButton("Lower Color##hsv", ImVec4(lr, lg, lb, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(30, 30));
    ImGui::SameLine(); ImGui::ColorButton("Upper Color##hsv", ImVec4(ur, ug, ub, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(30, 30));

    ImGui::EndDisabled();

    if (changed) {
        config.saveConfig();
    }
} 