#include "draw_settings.h"
#include "../AppContext.h"
#include "ui_helpers.h"
#include "imgui.h"

void draw_color_filter()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginSettingsSection("Color Filter Settings", "Filter detections by color range");

    bool enabled = ctx.config.color_filter_enabled;
    if (ImGui::Checkbox("Enable Color Filter", &enabled)) {
        ctx.config.color_filter_enabled = enabled;
        SAVE_PROFILE();
    }

    ImGui::Spacing();

    const char* modes[] = { "RGB", "HSV" };
    int mode = ctx.config.color_filter_mode;
    if (ImGui::Combo("Color Mode", &mode, modes, IM_ARRAYSIZE(modes))) {
        ctx.config.color_filter_mode = mode;
        SAVE_PROFILE();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ctx.config.color_filter_mode == 0) {
        // RGB Mode
        ImGui::Text("RGB Range:");

        int r_min = ctx.config.color_filter_r_min;
        int r_max = ctx.config.color_filter_r_max;
        int g_min = ctx.config.color_filter_g_min;
        int g_max = ctx.config.color_filter_g_max;
        int b_min = ctx.config.color_filter_b_min;
        int b_max = ctx.config.color_filter_b_max;

        if (ImGui::SliderInt("R Min", &r_min, 0, 255)) {
            ctx.config.color_filter_r_min = r_min;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("R Max", &r_max, 0, 255)) {
            ctx.config.color_filter_r_max = r_max;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("G Min", &g_min, 0, 255)) {
            ctx.config.color_filter_g_min = g_min;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("G Max", &g_max, 0, 255)) {
            ctx.config.color_filter_g_max = g_max;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("B Min", &b_min, 0, 255)) {
            ctx.config.color_filter_b_min = b_min;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("B Max", &b_max, 0, 255)) {
            ctx.config.color_filter_b_max = b_max;
            SAVE_PROFILE();
        }

        // Color preview
        ImGui::Spacing();
        ImVec4 col_min(r_min / 255.0f, g_min / 255.0f, b_min / 255.0f, 1.0f);
        ImVec4 col_max(r_max / 255.0f, g_max / 255.0f, b_max / 255.0f, 1.0f);
        ImGui::ColorButton("Min Color", col_min, 0, ImVec2(50, 20));
        ImGui::SameLine();
        ImGui::Text(" to ");
        ImGui::SameLine();
        ImGui::ColorButton("Max Color", col_max, 0, ImVec2(50, 20));

    } else {
        // HSV Mode
        ImGui::Text("HSV Range:");

        int h_min = ctx.config.color_filter_h_min;
        int h_max = ctx.config.color_filter_h_max;
        int s_min = ctx.config.color_filter_s_min;
        int s_max = ctx.config.color_filter_s_max;
        int v_min = ctx.config.color_filter_v_min;
        int v_max = ctx.config.color_filter_v_max;

        if (ImGui::SliderInt("H Min", &h_min, 0, 179)) {
            ctx.config.color_filter_h_min = h_min;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("H Max", &h_max, 0, 179)) {
            ctx.config.color_filter_h_max = h_max;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("S Min", &s_min, 0, 255)) {
            ctx.config.color_filter_s_min = s_min;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("S Max", &s_max, 0, 255)) {
            ctx.config.color_filter_s_max = s_max;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("V Min", &v_min, 0, 255)) {
            ctx.config.color_filter_v_min = v_min;
            SAVE_PROFILE();
        }
        if (ImGui::SliderInt("V Max", &v_max, 0, 255)) {
            ctx.config.color_filter_v_max = v_max;
            SAVE_PROFILE();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Pixel Size Filter:");
    int min_pixels = ctx.config.color_filter_min_pixels;
    int max_pixels = ctx.config.color_filter_max_pixels;

    if (ImGui::SliderInt("Min Pixels", &min_pixels, 0, 10000)) {
        ctx.config.color_filter_min_pixels = min_pixels;
        SAVE_PROFILE();
    }
    if (ImGui::SliderInt("Max Pixels", &max_pixels, 0, 100000)) {
        ctx.config.color_filter_max_pixels = max_pixels;
        SAVE_PROFILE();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Presets
    ImGui::Text("Quick Presets:");
    if (ImGui::Button("Red")) {
        ctx.config.color_filter_mode = 0;
        ctx.config.color_filter_r_min = 150; ctx.config.color_filter_r_max = 255;
        ctx.config.color_filter_g_min = 0; ctx.config.color_filter_g_max = 100;
        ctx.config.color_filter_b_min = 0; ctx.config.color_filter_b_max = 100;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    if (ImGui::Button("Green")) {
        ctx.config.color_filter_mode = 0;
        ctx.config.color_filter_r_min = 0; ctx.config.color_filter_r_max = 100;
        ctx.config.color_filter_g_min = 150; ctx.config.color_filter_g_max = 255;
        ctx.config.color_filter_b_min = 0; ctx.config.color_filter_b_max = 100;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    if (ImGui::Button("Blue")) {
        ctx.config.color_filter_mode = 0;
        ctx.config.color_filter_r_min = 0; ctx.config.color_filter_r_max = 100;
        ctx.config.color_filter_g_min = 0; ctx.config.color_filter_g_max = 100;
        ctx.config.color_filter_b_min = 150; ctx.config.color_filter_b_max = 255;
        SAVE_PROFILE();
    }
    ImGui::SameLine();
    if (ImGui::Button("Yellow")) {
        ctx.config.color_filter_mode = 0;
        ctx.config.color_filter_r_min = 200; ctx.config.color_filter_r_max = 255;
        ctx.config.color_filter_g_min = 200; ctx.config.color_filter_g_max = 255;
        ctx.config.color_filter_b_min = 0; ctx.config.color_filter_b_max = 100;
        SAVE_PROFILE();
    }

    UIHelpers::EndSettingsSection();
}
