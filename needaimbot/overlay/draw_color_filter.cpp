#include "draw_settings.h"
#include "../AppContext.h"
#include "ui_helpers.h"
#include "imgui.h"
#include "cuda/simple_cuda_mat.h"
#include "cuda/unified_graph_pipeline.h"
#include <d3d11.h>
#include <mutex>
#include <chrono>
#include <algorithm>

extern ID3D11ShaderResourceView* g_debugSRV;
extern ID3D11Texture2D* g_debugTex;
extern float debug_scale;
extern int texW, texH;

void uploadDebugFrame(const SimpleMat& frameMat);
extern std::mutex g_debugTexMutex;

// Helper: RGB to HSV conversion
static void rgbToHsv(int r, int g, int b, int& h, int& s, int& v) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    float maxVal = std::max({rf, gf, bf});
    float minVal = std::min({rf, gf, bf});
    float delta = maxVal - minVal;

    // Value
    v = static_cast<int>(maxVal * 255);

    // Saturation
    if (maxVal > 0) {
        s = static_cast<int>((delta / maxVal) * 255);
    } else {
        s = 0;
    }

    // Hue
    if (delta < 0.00001f) {
        h = 0;
    } else if (maxVal == rf) {
        h = static_cast<int>(60.0f * fmodf((gf - bf) / delta, 6.0f));
    } else if (maxVal == gf) {
        h = static_cast<int>(60.0f * ((bf - rf) / delta + 2.0f));
    } else {
        h = static_cast<int>(60.0f * ((rf - gf) / delta + 4.0f));
    }

    if (h < 0) h += 360;
    h = h / 2; // Convert to 0-179 range (OpenCV style)
}

// Helper: HSV to RGB conversion (h: 0-179, s: 0-255, v: 0-255)
static void hsvToRgb(int h, int s, int v, int& r, int& g, int& b) {
    float hf = (h * 2.0f) / 360.0f;  // Convert from 0-179 to 0-1
    float sf = s / 255.0f;
    float vf = v / 255.0f;

    if (sf <= 0.0f) {
        r = g = b = static_cast<int>(vf * 255);
        return;
    }

    float hh = hf * 6.0f;
    if (hh >= 6.0f) hh = 0.0f;
    int i = static_cast<int>(hh);
    float ff = hh - i;
    float p = vf * (1.0f - sf);
    float q = vf * (1.0f - sf * ff);
    float t = vf * (1.0f - sf * (1.0f - ff));

    float rf, gf, bf;
    switch (i) {
        case 0: rf = vf; gf = t; bf = p; break;
        case 1: rf = q; gf = vf; bf = p; break;
        case 2: rf = p; gf = vf; bf = t; break;
        case 3: rf = p; gf = q; bf = vf; break;
        case 4: rf = t; gf = p; bf = vf; break;
        default: rf = vf; gf = p; bf = q; break;
    }

    r = static_cast<int>(rf * 255);
    g = static_cast<int>(gf * 255);
    b = static_cast<int>(bf * 255);
}

// Draw HSV hue bar with range indicators
static void drawHueBarWithRange(int h_min, int h_max, float width, float height) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    // Draw hue gradient bar
    int segments = 180;
    float segmentWidth = width / segments;
    for (int i = 0; i < segments; i++) {
        int r, g, b;
        hsvToRgb(i, 255, 255, r, g, b);
        ImU32 color = IM_COL32(r, g, b, 255);

        float x1 = pos.x + i * segmentWidth;
        float x2 = pos.x + (i + 1) * segmentWidth;
        draw_list->AddRectFilled(ImVec2(x1, pos.y), ImVec2(x2, pos.y + height), color);
    }

    // Draw range overlay (darken outside range)
    bool wraparound = h_min > h_max;
    if (!wraparound) {
        // Normal range: darken outside [h_min, h_max]
        float x_min = pos.x + (h_min / 179.0f) * width;
        float x_max = pos.x + (h_max / 179.0f) * width;

        // Darken left side
        if (h_min > 0) {
            draw_list->AddRectFilled(ImVec2(pos.x, pos.y), ImVec2(x_min, pos.y + height), IM_COL32(0, 0, 0, 180));
        }
        // Darken right side
        if (h_max < 179) {
            draw_list->AddRectFilled(ImVec2(x_max, pos.y), ImVec2(pos.x + width, pos.y + height), IM_COL32(0, 0, 0, 180));
        }

        // Draw range markers
        draw_list->AddLine(ImVec2(x_min, pos.y - 2), ImVec2(x_min, pos.y + height + 2), IM_COL32(255, 255, 255, 255), 2.0f);
        draw_list->AddLine(ImVec2(x_max, pos.y - 2), ImVec2(x_max, pos.y + height + 2), IM_COL32(255, 255, 255, 255), 2.0f);
    } else {
        // Wraparound range (e.g., red: h_min=170, h_max=10)
        float x_min = pos.x + (h_min / 179.0f) * width;
        float x_max = pos.x + (h_max / 179.0f) * width;

        // Darken middle section (between h_max and h_min)
        draw_list->AddRectFilled(ImVec2(x_max, pos.y), ImVec2(x_min, pos.y + height), IM_COL32(0, 0, 0, 180));

        // Draw range markers
        draw_list->AddLine(ImVec2(x_min, pos.y - 2), ImVec2(x_min, pos.y + height + 2), IM_COL32(255, 255, 255, 255), 2.0f);
        draw_list->AddLine(ImVec2(x_max, pos.y - 2), ImVec2(x_max, pos.y + height + 2), IM_COL32(255, 255, 255, 255), 2.0f);
    }

    // Draw border
    draw_list->AddRect(ImVec2(pos.x, pos.y), ImVec2(pos.x + width, pos.y + height), IM_COL32(128, 128, 128, 255));

    ImGui::Dummy(ImVec2(width, height + 4));
}

// Draw saturation/value gradient preview
static void drawSVPreview(int h_min, int h_max, int s_min, int s_max, int v_min, int v_max, float width, float height) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    // Use middle hue for preview
    int h_mid = (h_min <= h_max) ? (h_min + h_max) / 2 : ((h_min + h_max + 180) / 2) % 180;

    // Draw S-V gradient grid
    int gridX = 16;
    int gridY = 16;
    float cellW = width / gridX;
    float cellH = height / gridY;

    for (int sy = 0; sy < gridY; sy++) {
        for (int sx = 0; sx < gridX; sx++) {
            int s = (sx * 255) / (gridX - 1);
            int v = ((gridY - 1 - sy) * 255) / (gridY - 1);

            int r, g, b;
            hsvToRgb(h_mid, s, v, r, g, b);

            // Check if in range
            bool inRange = (s >= s_min && s <= s_max && v >= v_min && v <= v_max);

            ImU32 color = IM_COL32(r, g, b, inRange ? 255 : 60);
            float x1 = pos.x + sx * cellW;
            float y1 = pos.y + sy * cellH;
            draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x1 + cellW, y1 + cellH), color);
        }
    }

    // Draw range box
    float x1 = pos.x + (s_min / 255.0f) * width;
    float x2 = pos.x + (s_max / 255.0f) * width;
    float y1 = pos.y + ((255 - v_max) / 255.0f) * height;
    float y2 = pos.y + ((255 - v_min) / 255.0f) * height;
    draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(255, 255, 255, 255), 0.0f, 0, 2.0f);

    // Draw border
    draw_list->AddRect(ImVec2(pos.x, pos.y), ImVec2(pos.x + width, pos.y + height), IM_COL32(128, 128, 128, 255));

    ImGui::Dummy(ImVec2(width, height + 4));
}

// Draw RGB cube slice preview
static void drawRGBPreview(int r_min, int r_max, int g_min, int g_max, int b_min, int b_max, float width, float height) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    // Draw R-G gradient at middle B value
    int b_mid = (b_min + b_max) / 2;
    int gridX = 16;
    int gridY = 16;
    float cellW = width / gridX;
    float cellH = height / gridY;

    for (int gy = 0; gy < gridY; gy++) {
        for (int gx = 0; gx < gridX; gx++) {
            int r = (gx * 255) / (gridX - 1);
            int g = ((gridY - 1 - gy) * 255) / (gridY - 1);

            bool inRange = (r >= r_min && r <= r_max && g >= g_min && g <= g_max);

            ImU32 color = IM_COL32(r, g, b_mid, inRange ? 255 : 60);
            float x1 = pos.x + gx * cellW;
            float y1 = pos.y + gy * cellH;
            draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x1 + cellW, y1 + cellH), color);
        }
    }

    // Draw range box
    float x1 = pos.x + (r_min / 255.0f) * width;
    float x2 = pos.x + (r_max / 255.0f) * width;
    float y1 = pos.y + ((255 - g_max) / 255.0f) * height;
    float y2 = pos.y + ((255 - g_min) / 255.0f) * height;
    draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(255, 255, 255, 255), 0.0f, 0, 2.0f);

    // Draw border and labels
    draw_list->AddRect(ImVec2(pos.x, pos.y), ImVec2(pos.x + width, pos.y + height), IM_COL32(128, 128, 128, 255));

    ImGui::Dummy(ImVec2(width, height + 4));
}

// Apply color filter mask to frame (supports 3 or 4 channel images)
// Note: Preview frame is RGBA (converted from BGRA), so R=0, G=1, B=2
static void applyColorFilterMask(SimpleMat& frame, const Config& cfg) {
    if (frame.empty()) return;

    int channels = frame.channels();
    if (channels != 3 && channels != 4) return;

    int width = frame.cols();
    int height = frame.rows();
    unsigned char* data = frame.data();
    int stride = frame.step();

    for (int y = 0; y < height; y++) {
        unsigned char* row = data + y * stride;
        for (int x = 0; x < width; x++) {
            // Preview is RGBA format (converted from BGRA by cuda_bgra2rgba)
            int r = row[x * channels + 0];
            int g = row[x * channels + 1];
            int b = row[x * channels + 2];

            bool matches = false;

            if (cfg.color_filter_mode == 0) {
                // RGB mode
                matches = (r >= cfg.color_filter_r_min && r <= cfg.color_filter_r_max &&
                          g >= cfg.color_filter_g_min && g <= cfg.color_filter_g_max &&
                          b >= cfg.color_filter_b_min && b <= cfg.color_filter_b_max);
            } else {
                // HSV mode
                int h, s, v;
                rgbToHsv(r, g, b, h, s, v);

                // Handle hue wraparound
                bool hueMatch;
                if (cfg.color_filter_h_min <= cfg.color_filter_h_max) {
                    hueMatch = (h >= cfg.color_filter_h_min && h <= cfg.color_filter_h_max);
                } else {
                    // Wraparound case (e.g., red: 170-10)
                    hueMatch = (h >= cfg.color_filter_h_min || h <= cfg.color_filter_h_max);
                }

                matches = hueMatch &&
                         (s >= cfg.color_filter_s_min && s <= cfg.color_filter_s_max) &&
                         (v >= cfg.color_filter_v_min && v <= cfg.color_filter_v_max);
            }

            // Keep original if matches, darken if not (using configurable opacity)
            if (!matches) {
                float opacity = cfg.color_filter_mask_opacity;
                row[x * channels + 0] = static_cast<unsigned char>(r * opacity);
                row[x * channels + 1] = static_cast<unsigned char>(g * opacity);
                row[x * channels + 2] = static_cast<unsigned char>(b * opacity);
            }
            // Matching pixels keep their original color
        }
    }
}

void draw_color_filter()
{
    auto& ctx = AppContext::getInstance();

    // Enable Preview checkbox at top
    bool prev_show_window_state = ctx.config.show_window;
    if (ImGui::Checkbox("Enable Preview", &ctx.config.show_window)) {
        SAVE_PROFILE();

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
    ImGui::SameLine();
    UIHelpers::InfoTooltip("Shows the live capture feed (shared with Aim Offset tab)");

    ImGui::Spacing();

    // Two-column layout: Left = Preview, Right = Settings
    if (ImGui::BeginTable("ColorFilterLayout", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed, 350.0f);
        ImGui::TableSetupColumn("Settings", ImGuiTableColumnFlags_WidthFixed, 300.0f);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        // Left column - Preview Window
        {
            if (ctx.config.show_window) {
                SimpleMat* frameToDisplay = nullptr;
                static SimpleMat previewHostFrame;
                static SimpleMat maskedFrame;

                static auto lastPreviewUpdate = std::chrono::high_resolution_clock::now();
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPreviewUpdate);
                bool shouldUpdatePreview = (elapsed.count() >= 66);

                auto& pipelineManager = needaimbot::PipelineManager::getInstance();
                auto* pipeline = pipelineManager.getPipeline();

                // Get new frame if available
                if (pipeline && pipeline->isPreviewAvailable() && shouldUpdatePreview) {
                    if (pipeline->getPreviewSnapshot(previewHostFrame) &&
                        !previewHostFrame.empty() &&
                        previewHostFrame.cols() > 0 && previewHostFrame.rows() > 0 &&
                        previewHostFrame.cols() <= 10000 && previewHostFrame.rows() <= 10000) {
                        lastPreviewUpdate = now;
                    }
                }

                // Always apply mask to current frame (allows real-time slider updates)
                if (!previewHostFrame.empty()) {
                    if (ctx.config.color_filter_enabled) {
                        maskedFrame = previewHostFrame.clone();
                        applyColorFilterMask(maskedFrame, ctx.config);
                        frameToDisplay = &maskedFrame;
                    } else {
                        frameToDisplay = &previewHostFrame;
                    }
                }

                if (frameToDisplay) {
                    try {
                        uploadDebugFrame(*frameToDisplay);
                    } catch (const std::exception& e) {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Upload failed");
                        ImGui::EndTable();
                        return;
                    }

                    ImGui::SetNextItemWidth(120.0f);
                    ImGui::SliderFloat("Scale##cf", &debug_scale, 0.1f, 2.0f, "%.1fx");

                    std::lock_guard<std::mutex> lock(g_debugTexMutex);

                    if (g_debugSRV && texW > 0 && texH > 0 && texW < 10000 && texH < 10000) {
                        float safe_scale = debug_scale;
                        if (safe_scale <= 0 || safe_scale > 10.0f) safe_scale = 1.0f;

                        ImVec2 image_size(texW * safe_scale, texH * safe_scale);
                        if (image_size.x > 0 && image_size.y > 0 && image_size.x < 10000 && image_size.y < 10000) {
                            ImGui::Image(g_debugSRV, image_size);

                            ImVec2 image_pos = ImGui::GetItemRectMin();
                            ImDrawList* draw_list = ImGui::GetWindowDrawList();
                            if (draw_list && texW > 0 && texH > 0) {
                                float center_x = image_pos.x + (texW * debug_scale) / 2.0f;
                                float center_y = image_pos.y + (texH * debug_scale) / 2.0f;
                                ImU32 crosshair_color = ctx.config.color_filter_enabled ?
                                    IM_COL32(255, 0, 0, 255) : IM_COL32(255, 255, 255, 255);
                                draw_list->AddLine(ImVec2(center_x - 10, center_y), ImVec2(center_x + 10, center_y), crosshair_color, 2.0f);
                                draw_list->AddLine(ImVec2(center_x, center_y - 10), ImVec2(center_x, center_y + 10), crosshair_color, 2.0f);
                                draw_list->AddCircle(ImVec2(center_x, center_y), 3.0f, crosshair_color, 0, 2.0f);
                            }
                        }
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Waiting for capture...");
                }
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Preview disabled");
            }
        }

        ImGui::TableNextColumn();

        // Right column - Color Filter Settings
        {
            bool enabled = ctx.config.color_filter_enabled;
            if (ImGui::Checkbox("Enable Color Filter", &enabled)) {
                ctx.config.color_filter_enabled = enabled;
                SAVE_PROFILE();
            }

            ImGui::Spacing();

            const char* modes[] = { "RGB", "HSV" };
            int mode = ctx.config.color_filter_mode;
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("Mode", &mode, modes, IM_ARRAYSIZE(modes))) {
                ctx.config.color_filter_mode = mode;
                SAVE_PROFILE();
            }

            // Mask opacity slider
            float opacity = ctx.config.color_filter_mask_opacity;
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::SliderFloat("Non-match Opacity", &opacity, 0.0f, 1.0f, "%.2f")) {
                ctx.config.color_filter_mask_opacity = opacity;
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            UIHelpers::InfoTooltip("0 = full black, 1 = original color");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ctx.config.color_filter_mode == 0) {
                // RGB Mode
                int r_min = ctx.config.color_filter_r_min;
                int r_max = ctx.config.color_filter_r_max;
                int g_min = ctx.config.color_filter_g_min;
                int g_max = ctx.config.color_filter_g_max;
                int b_min = ctx.config.color_filter_b_min;
                int b_max = ctx.config.color_filter_b_max;

                ImGui::PushItemWidth(180.0f);

                // Red channel
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Red:");
                if (ImGui::SliderInt("##rmin", &r_min, 0, 255, "Min: %d")) {
                    ctx.config.color_filter_r_min = r_min;
                    SAVE_PROFILE();
                }
                ImGui::SameLine();
                if (ImGui::SliderInt("##rmax", &r_max, 0, 255, "Max: %d")) {
                    ctx.config.color_filter_r_max = r_max;
                    SAVE_PROFILE();
                }

                // Green channel
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Green:");
                if (ImGui::SliderInt("##gmin", &g_min, 0, 255, "Min: %d")) {
                    ctx.config.color_filter_g_min = g_min;
                    SAVE_PROFILE();
                }
                ImGui::SameLine();
                if (ImGui::SliderInt("##gmax", &g_max, 0, 255, "Max: %d")) {
                    ctx.config.color_filter_g_max = g_max;
                    SAVE_PROFILE();
                }

                // Blue channel
                ImGui::TextColored(ImVec4(0.3f, 0.3f, 1.0f, 1.0f), "Blue:");
                if (ImGui::SliderInt("##bmin", &b_min, 0, 255, "Min: %d")) {
                    ctx.config.color_filter_b_min = b_min;
                    SAVE_PROFILE();
                }
                ImGui::SameLine();
                if (ImGui::SliderInt("##bmax", &b_max, 0, 255, "Max: %d")) {
                    ctx.config.color_filter_b_max = b_max;
                    SAVE_PROFILE();
                }

                ImGui::PopItemWidth();

                // Color preview boxes
                ImGui::Spacing();
                ImVec4 col_min(r_min / 255.0f, g_min / 255.0f, b_min / 255.0f, 1.0f);
                ImVec4 col_max(r_max / 255.0f, g_max / 255.0f, b_max / 255.0f, 1.0f);
                ImGui::Text("Range: ");
                ImGui::SameLine();
                ImGui::ColorButton("##colmin", col_min, 0, ImVec2(40, 20));
                ImGui::SameLine();
                ImGui::Text("->");
                ImGui::SameLine();
                ImGui::ColorButton("##colmax", col_max, 0, ImVec2(40, 20));

                // RGB palette preview
                ImGui::Spacing();
                ImGui::Text("R-G Preview (at B=%d):", (b_min + b_max) / 2);
                drawRGBPreview(r_min, r_max, g_min, g_max, b_min, b_max, 200.0f, 100.0f);

            } else {
                // HSV Mode
                int h_min = ctx.config.color_filter_h_min;
                int h_max = ctx.config.color_filter_h_max;
                int s_min = ctx.config.color_filter_s_min;
                int s_max = ctx.config.color_filter_s_max;
                int v_min = ctx.config.color_filter_v_min;
                int v_max = ctx.config.color_filter_v_max;

                ImGui::PushItemWidth(180.0f);

                ImGui::Text("Hue (0-179):");
                if (ImGui::SliderInt("##hmin", &h_min, 0, 179, "Min: %d")) {
                    ctx.config.color_filter_h_min = h_min;
                    SAVE_PROFILE();
                }
                ImGui::SameLine();
                if (ImGui::SliderInt("##hmax", &h_max, 0, 179, "Max: %d")) {
                    ctx.config.color_filter_h_max = h_max;
                    SAVE_PROFILE();
                }

                ImGui::Text("Saturation:");
                if (ImGui::SliderInt("##smin", &s_min, 0, 255, "Min: %d")) {
                    ctx.config.color_filter_s_min = s_min;
                    SAVE_PROFILE();
                }
                ImGui::SameLine();
                if (ImGui::SliderInt("##smax", &s_max, 0, 255, "Max: %d")) {
                    ctx.config.color_filter_s_max = s_max;
                    SAVE_PROFILE();
                }

                ImGui::Text("Value:");
                if (ImGui::SliderInt("##vmin", &v_min, 0, 255, "Min: %d")) {
                    ctx.config.color_filter_v_min = v_min;
                    SAVE_PROFILE();
                }
                ImGui::SameLine();
                if (ImGui::SliderInt("##vmax", &v_max, 0, 255, "Max: %d")) {
                    ctx.config.color_filter_v_max = v_max;
                    SAVE_PROFILE();
                }

                ImGui::PopItemWidth();

                // HSV palette preview
                ImGui::Spacing();
                ImGui::Text("Hue Range:");
                drawHueBarWithRange(h_min, h_max, 250.0f, 20.0f);

                ImGui::Text("S-V Preview:");
                drawSVPreview(h_min, h_max, s_min, s_max, v_min, v_max, 150.0f, 100.0f);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Target Color Filter (filtering by color match within bbox)
            bool target_filter_enabled = ctx.config.color_filter_target_enabled;
            if (ImGui::Checkbox("Enable Target Color Filter", &target_filter_enabled)) {
                ctx.config.color_filter_target_enabled = target_filter_enabled;
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            UIHelpers::InfoTooltip("Filter targets by how much of their bbox matches the color filter");

            if (target_filter_enabled) {
                // Simplified inline UI: [value] [px/%] [이상만/이하만/범위] 필터
                int target_mode = ctx.config.color_filter_target_mode;
                int comparison = ctx.config.color_filter_comparison;

                if (target_mode == 0) {
                    // Ratio mode - single line UI
                    float min_ratio = ctx.config.color_filter_min_ratio * 100.0f;
                    float max_ratio = ctx.config.color_filter_max_ratio * 100.0f;

                    if (comparison == 2) {
                        // Between mode
                        ImGui::SetNextItemWidth(60.0f);
                        if (ImGui::InputFloat("##min_ratio", &min_ratio, 0, 0, "%.1f")) {
                            if (min_ratio < 0.0f) min_ratio = 0.0f;
                            if (min_ratio > 100.0f) min_ratio = 100.0f;
                            if (min_ratio > max_ratio) min_ratio = max_ratio;
                            ctx.config.color_filter_min_ratio = min_ratio / 100.0f;
                            SAVE_PROFILE();
                        }
                        ImGui::SameLine();
                        ImGui::Text("~");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60.0f);
                        if (ImGui::InputFloat("##max_ratio", &max_ratio, 0, 0, "%.1f")) {
                            if (max_ratio < 0.0f) max_ratio = 0.0f;
                            if (max_ratio > 100.0f) max_ratio = 100.0f;
                            if (max_ratio < min_ratio) max_ratio = min_ratio;
                            ctx.config.color_filter_max_ratio = max_ratio / 100.0f;
                            SAVE_PROFILE();
                        }
                        ImGui::SameLine();
                        ImGui::Text("%%");
                    } else {
                        // Above or Below mode
                        float& value = (comparison == 0) ? min_ratio : max_ratio;
                        ImGui::SetNextItemWidth(80.0f);
                        if (ImGui::InputFloat("##threshold_ratio", &value, 0, 0, "%.1f")) {
                            if (value < 0.0f) value = 0.0f;
                            if (value > 100.0f) value = 100.0f;
                            if (comparison == 0) {
                                ctx.config.color_filter_min_ratio = value / 100.0f;
                            } else {
                                ctx.config.color_filter_max_ratio = value / 100.0f;
                            }
                            SAVE_PROFILE();
                        }
                        ImGui::SameLine();
                        ImGui::Text("%%");
                    }
                } else {
                    // Absolute count mode - single line UI
                    int min_count = ctx.config.color_filter_min_count;
                    int max_count = ctx.config.color_filter_max_count;

                    if (comparison == 2) {
                        // Between mode
                        ImGui::SetNextItemWidth(80.0f);
                        if (ImGui::InputInt("##min_count", &min_count, 0, 0)) {
                            if (min_count < 0) min_count = 0;
                            if (min_count > max_count) min_count = max_count;
                            ctx.config.color_filter_min_count = min_count;
                            SAVE_PROFILE();
                        }
                        ImGui::SameLine();
                        ImGui::Text("~");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(80.0f);
                        if (ImGui::InputInt("##max_count", &max_count, 0, 0)) {
                            if (max_count < 0) max_count = 0;
                            if (max_count < min_count) max_count = min_count;
                            ctx.config.color_filter_max_count = max_count;
                            SAVE_PROFILE();
                        }
                        ImGui::SameLine();
                        ImGui::Text("px");
                    } else {
                        // Above or Below mode
                        int& value = (comparison == 0) ? min_count : max_count;
                        ImGui::SetNextItemWidth(100.0f);
                        if (ImGui::InputInt("##threshold_count", &value, 0, 0)) {
                            if (value < 0) value = 0;
                            if (comparison == 0) {
                                ctx.config.color_filter_min_count = value;
                            } else {
                                ctx.config.color_filter_max_count = value;
                            }
                            SAVE_PROFILE();
                        }
                        ImGui::SameLine();
                        ImGui::Text("px");
                    }
                }

                // Comparison selector on same line
                ImGui::SameLine();
                const char* comparison_labels[] = { "or more", "or less", "range" };
                ImGui::SetNextItemWidth(80.0f);
                if (ImGui::Combo("##comparison", &comparison, comparison_labels, IM_ARRAYSIZE(comparison_labels))) {
                    ctx.config.color_filter_comparison = comparison;
                    SAVE_PROFILE();
                }

                // Mode toggle (% / px)
                ImGui::SameLine();
                const char* mode_labels[] = { "%", "px" };
                ImGui::SetNextItemWidth(50.0f);
                if (ImGui::Combo("##mode", &target_mode, mode_labels, IM_ARRAYSIZE(mode_labels))) {
                    ctx.config.color_filter_target_mode = target_mode;
                    SAVE_PROFILE();
                }

                // Description text
                if (target_mode == 0) {
                    float min_r = ctx.config.color_filter_min_ratio * 100.0f;
                    float max_r = ctx.config.color_filter_max_ratio * 100.0f;
                    if (comparison == 0) {
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                            "-> Accept targets with >= %.1f%% color match", min_r);
                    } else if (comparison == 1) {
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                            "-> Accept targets with <= %.1f%% color match", max_r);
                    } else {
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                            "-> Accept targets with %.1f%% ~ %.1f%% color match", min_r, max_r);
                    }
                } else {
                    int min_c = ctx.config.color_filter_min_count;
                    int max_c = ctx.config.color_filter_max_count;
                    if (comparison == 0) {
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                            "-> Accept targets with >= %d matching pixels", min_c);
                    } else if (comparison == 1) {
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                            "-> Accept targets with <= %d matching pixels", max_c);
                    } else {
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                            "-> Accept targets with %d ~ %d matching pixels", min_c, max_c);
                    }
                }
            }

        }

        ImGui::EndTable();
    }
}
