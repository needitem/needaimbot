#include "../core/windows_headers.h"
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
#include <cmath>

// External references from draw_debug.cpp
extern ID3D11ShaderResourceView* g_debugSRV;
extern ID3D11Texture2D* g_debugTex;
extern float debug_scale;
extern int texW, texH;
extern std::mutex g_debugTexMutex;
extern ID3D11Device* g_pd3dDevice;
extern ID3D11DeviceContext* g_pd3dDeviceContext;

void uploadDebugFrame(const SimpleMat& frameMat);

// Color filter texture (separate from debug texture)
static ID3D11Texture2D* g_colorFilterTex = nullptr;
static ID3D11ShaderResourceView* g_colorFilterSRV = nullptr;
static int colorFilterTexW = 0, colorFilterTexH = 0;
static float color_filter_scale = 1.0f;

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p) do { if ((p) != nullptr) { (p)->Release(); (p) = nullptr; } } while (0)
#endif

// Helper: Convert RGB to HSV
static void RGBtoHSV(uint8_t r, uint8_t g, uint8_t b, int& h, int& s, int& v) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    float maxc = std::max({rf, gf, bf});
    float minc = std::min({rf, gf, bf});
    float diff = maxc - minc;

    v = static_cast<int>(maxc * 255);

    if (maxc == 0) {
        s = 0;
        h = 0;
        return;
    }

    s = static_cast<int>((diff / maxc) * 255);

    if (diff == 0) {
        h = 0;
    } else if (maxc == rf) {
        h = static_cast<int>(60 * fmod((gf - bf) / diff, 6.0f));
    } else if (maxc == gf) {
        h = static_cast<int>(60 * ((bf - rf) / diff + 2));
    } else {
        h = static_cast<int>(60 * ((rf - gf) / diff + 4));
    }

    if (h < 0) h += 360;
    h = h / 2;  // Convert to 0-179 range (OpenCV style)
}

// Helper: Convert HSV to RGB for color preview
static void HSVtoRGB(int h, int s, int v, uint8_t& r, uint8_t& g, uint8_t& b) {
    float hf = (h * 2) / 60.0f;  // Convert from 0-179 to 0-6
    float sf = s / 255.0f;
    float vf = v / 255.0f;

    int i = static_cast<int>(hf);
    float f = hf - i;
    float p = vf * (1 - sf);
    float q = vf * (1 - sf * f);
    float t = vf * (1 - sf * (1 - f));

    float rf, gf, bf;
    switch (i % 6) {
        case 0: rf = vf; gf = t;  bf = p;  break;
        case 1: rf = q;  gf = vf; bf = p;  break;
        case 2: rf = p;  gf = vf; bf = t;  break;
        case 3: rf = p;  gf = q;  bf = vf; break;
        case 4: rf = t;  gf = p;  bf = vf; break;
        default: rf = vf; gf = p;  bf = q;  break;
    }

    r = static_cast<uint8_t>(rf * 255);
    g = static_cast<uint8_t>(gf * 255);
    b = static_cast<uint8_t>(bf * 255);
}

// Apply color filter to frame and upload to texture
static void uploadColorFilterFrame(const SimpleMat& rgbaCpu, int mode,
    int r_min, int r_max, int g_min, int g_max, int b_min, int b_max,
    int h_min, int h_max, int s_min, int s_max, int v_min, int v_max)
{
    if (rgbaCpu.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        return;
    }

    if (rgbaCpu.cols() <= 0 || rgbaCpu.rows() <= 0 ||
        rgbaCpu.cols() > 10000 || rgbaCpu.rows() > 10000) {
        return;
    }

    int width = rgbaCpu.cols();
    int height = rgbaCpu.rows();

    // Recreate texture if size changed
    if (!g_colorFilterTex || width != colorFilterTexW || height != colorFilterTexH)
    {
        SAFE_RELEASE(g_colorFilterTex);
        SAFE_RELEASE(g_colorFilterSRV);

        colorFilterTexW = width;
        colorFilterTexH = height;

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = colorFilterTexW;
        td.Height = colorFilterTexH;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        HRESULT hr_tex = g_pd3dDevice->CreateTexture2D(&td, nullptr, &g_colorFilterTex);
        if (FAILED(hr_tex)) {
            SAFE_RELEASE(g_colorFilterTex);
            return;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = td.Format;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        HRESULT hr_srv = g_pd3dDevice->CreateShaderResourceView(g_colorFilterTex, &sd, &g_colorFilterSRV);
        if (FAILED(hr_srv)) {
            SAFE_RELEASE(g_colorFilterTex);
            SAFE_RELEASE(g_colorFilterSRV);
            return;
        }
    }

    if (!g_colorFilterTex || !g_pd3dDeviceContext) {
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms = {};
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_colorFilterTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (FAILED(hr_map) || ms.pData == nullptr) {
        return;
    }

    const uint8_t* src = rgbaCpu.data();
    size_t srcPitch = rgbaCpu.step();
    uint8_t* dst = static_cast<uint8_t*>(ms.pData);

    // Apply color filter
    for (int y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + srcPitch * y;
        uint8_t* dstRow = dst + ms.RowPitch * y;

        for (int x = 0; x < width; ++x) {
            uint8_t r = srcRow[x * 4 + 0];
            uint8_t g = srcRow[x * 4 + 1];
            uint8_t b = srcRow[x * 4 + 2];
            uint8_t a = srcRow[x * 4 + 3];

            bool passFilter = false;

            if (mode == 0) {
                // RGB mode
                passFilter = (r >= r_min && r <= r_max &&
                              g >= g_min && g <= g_max &&
                              b >= b_min && b <= b_max);
            } else {
                // HSV mode
                int h, s, v;
                RGBtoHSV(r, g, b, h, s, v);

                // Handle hue wraparound
                bool huePass;
                if (h_min <= h_max) {
                    huePass = (h >= h_min && h <= h_max);
                } else {
                    // Wraparound case (e.g., red: 170-10)
                    huePass = (h >= h_min || h <= h_max);
                }

                passFilter = (huePass &&
                              s >= s_min && s <= s_max &&
                              v >= v_min && v <= v_max);
            }

            if (passFilter) {
                // Show original color
                dstRow[x * 4 + 0] = r;
                dstRow[x * 4 + 1] = g;
                dstRow[x * 4 + 2] = b;
                dstRow[x * 4 + 3] = a;
            } else {
                // Grayscale for non-matching pixels
                uint8_t gray = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
                dstRow[x * 4 + 0] = gray;
                dstRow[x * 4 + 1] = gray;
                dstRow[x * 4 + 2] = gray;
                dstRow[x * 4 + 3] = a;
            }
        }
    }

    g_pd3dDeviceContext->Unmap(g_colorFilterTex, 0);
}

// Draw a color palette/gradient for HSV hue selection
static void DrawHuePalette(ImDrawList* draw_list, ImVec2 pos, ImVec2 size, int current_h_min, int current_h_max) {
    const int segments = 180;
    float segment_width = size.x / segments;

    for (int i = 0; i < segments; ++i) {
        uint8_t r, g, b;
        HSVtoRGB(i, 255, 255, r, g, b);

        ImVec2 p1(pos.x + i * segment_width, pos.y);
        ImVec2 p2(pos.x + (i + 1) * segment_width, pos.y + size.y);

        draw_list->AddRectFilled(p1, p2, IM_COL32(r, g, b, 255));
    }

    // Draw selection range
    float min_x = pos.x + (current_h_min / 179.0f) * size.x;
    float max_x = pos.x + (current_h_max / 179.0f) * size.x;

    if (current_h_min <= current_h_max) {
        draw_list->AddRect(ImVec2(min_x, pos.y), ImVec2(max_x, pos.y + size.y), IM_COL32(255, 255, 255, 255), 0, 0, 2.0f);
    } else {
        // Wraparound - draw two rectangles
        draw_list->AddRect(ImVec2(min_x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(255, 255, 255, 255), 0, 0, 2.0f);
        draw_list->AddRect(ImVec2(pos.x, pos.y), ImVec2(max_x, pos.y + size.y), IM_COL32(255, 255, 255, 255), 0, 0, 2.0f);
    }
}

// Draw RGB color range preview: shows From-To boxes with gradient between
static void DrawColorRangeRGB(ImDrawList* draw_list, ImVec2 pos, ImVec2 size, int r_min, int r_max, int g_min, int g_max, int b_min, int b_max) {
    float boxWidth = size.x * 0.25f;  // Each color box is 25% width
    float gradientWidth = size.x * 0.5f;  // Gradient is 50% width
    float boxHeight = size.y;

    // Left box - Min color (From)
    ImVec2 minBoxPos = pos;
    ImVec2 minBoxEnd(pos.x + boxWidth, pos.y + boxHeight);
    draw_list->AddRectFilled(minBoxPos, minBoxEnd, IM_COL32(r_min, g_min, b_min, 255));
    draw_list->AddRect(minBoxPos, minBoxEnd, IM_COL32(255, 255, 255, 255), 0, 0, 1.0f);

    // Gradient in the middle (horizontal gradient from min to max)
    ImVec2 gradStart(pos.x + boxWidth, pos.y);
    ImVec2 gradEnd(pos.x + boxWidth + gradientWidth, pos.y + boxHeight);

    const int segments = 32;
    float segmentWidth = gradientWidth / segments;
    for (int i = 0; i < segments; ++i) {
        float t = static_cast<float>(i) / (segments - 1);
        uint8_t r = static_cast<uint8_t>(r_min + t * (r_max - r_min));
        uint8_t g = static_cast<uint8_t>(g_min + t * (g_max - g_min));
        uint8_t b = static_cast<uint8_t>(b_min + t * (b_max - b_min));

        ImVec2 p1(gradStart.x + i * segmentWidth, gradStart.y);
        ImVec2 p2(gradStart.x + (i + 1) * segmentWidth, gradEnd.y);
        draw_list->AddRectFilled(p1, p2, IM_COL32(r, g, b, 255));
    }

    // Right box - Max color (To)
    ImVec2 maxBoxPos(pos.x + boxWidth + gradientWidth, pos.y);
    ImVec2 maxBoxEnd(pos.x + size.x, pos.y + boxHeight);
    draw_list->AddRectFilled(maxBoxPos, maxBoxEnd, IM_COL32(r_max, g_max, b_max, 255));
    draw_list->AddRect(maxBoxPos, maxBoxEnd, IM_COL32(255, 255, 255, 255), 0, 0, 1.0f);

    // Outer border
    draw_list->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(128, 128, 128, 255), 0, 0, 1.0f);
}

// Draw HSV color range preview: shows From-To boxes with hue gradient between
static void DrawColorRangeHSV(ImDrawList* draw_list, ImVec2 pos, ImVec2 size, int h_min, int h_max, int s_min, int s_max, int v_min, int v_max) {
    float boxWidth = size.x * 0.25f;
    float gradientWidth = size.x * 0.5f;
    float boxHeight = size.y;

    // Calculate colors
    uint8_t r_min_c, g_min_c, b_min_c;
    uint8_t r_max_c, g_max_c, b_max_c;
    int s_mid = (s_min + s_max) / 2;
    int v_mid = (v_min + v_max) / 2;

    HSVtoRGB(h_min, s_mid, v_mid, r_min_c, g_min_c, b_min_c);
    HSVtoRGB(h_max, s_mid, v_mid, r_max_c, g_max_c, b_max_c);

    // Left box - Min hue (From)
    ImVec2 minBoxPos = pos;
    ImVec2 minBoxEnd(pos.x + boxWidth, pos.y + boxHeight);
    draw_list->AddRectFilled(minBoxPos, minBoxEnd, IM_COL32(r_min_c, g_min_c, b_min_c, 255));
    draw_list->AddRect(minBoxPos, minBoxEnd, IM_COL32(255, 255, 255, 255), 0, 0, 1.0f);

    // Gradient in the middle (hue gradient)
    ImVec2 gradStart(pos.x + boxWidth, pos.y);
    ImVec2 gradEnd(pos.x + boxWidth + gradientWidth, pos.y + boxHeight);

    const int segments = 32;
    float segmentWidth = gradientWidth / segments;

    bool wraparound = (h_min > h_max);

    for (int i = 0; i < segments; ++i) {
        float t = static_cast<float>(i) / (segments - 1);
        int h;
        if (!wraparound) {
            h = h_min + static_cast<int>(t * (h_max - h_min));
        } else {
            // Wraparound: go from h_min to 179, then 0 to h_max
            int range = (180 - h_min) + h_max;
            int offset = static_cast<int>(t * range);
            h = (h_min + offset) % 180;
        }

        uint8_t r, g, b;
        HSVtoRGB(h, s_mid, v_mid, r, g, b);

        ImVec2 p1(gradStart.x + i * segmentWidth, gradStart.y);
        ImVec2 p2(gradStart.x + (i + 1) * segmentWidth, gradEnd.y);
        draw_list->AddRectFilled(p1, p2, IM_COL32(r, g, b, 255));
    }

    // Right box - Max hue (To)
    ImVec2 maxBoxPos(pos.x + boxWidth + gradientWidth, pos.y);
    ImVec2 maxBoxEnd(pos.x + size.x, pos.y + boxHeight);
    draw_list->AddRectFilled(maxBoxPos, maxBoxEnd, IM_COL32(r_max_c, g_max_c, b_max_c, 255));
    draw_list->AddRect(maxBoxPos, maxBoxEnd, IM_COL32(255, 255, 255, 255), 0, 0, 1.0f);

    // Outer border
    draw_list->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(128, 128, 128, 255), 0, 0, 1.0f);
}

void draw_color_filter()
{
    auto& ctx = AppContext::getInstance();

    ImGui::BeginChild("ColorFilterScroll", ImVec2(0, 0), false, ImGuiWindowFlags_NoNavInputs);

    // Split layout: Left = Preview, Right = Controls
    float totalWidth = ImGui::GetContentRegionAvail().x;
    float leftWidth = totalWidth * 0.55f;
    float rightWidth = totalWidth * 0.42f;

    // Left panel - Preview
    ImGui::BeginChild("ColorFilterPreview", ImVec2(leftWidth, 0), true);
    {
        ImGui::Text("Live Preview");
        ImGui::Separator();

        if (ImGui::Checkbox("Enable Preview", &ctx.config.show_window)) {
            SAVE_PROFILE();
        }

        if (ctx.config.show_window) {
            ImGui::Spacing();

            SimpleMat* frameToDisplay = nullptr;
            static SimpleMat colorFilterHostFrame;

            // Throttle preview updates
            static auto lastPreviewUpdate = std::chrono::high_resolution_clock::now();
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPreviewUpdate);
            bool shouldUpdatePreview = (elapsed.count() >= 66);

            auto& pipelineManager = needaimbot::PipelineManager::getInstance();
            auto* pipeline = pipelineManager.getPipeline();

            if (pipeline && pipeline->isPreviewAvailable() && shouldUpdatePreview) {
                if (pipeline->getPreviewSnapshot(colorFilterHostFrame) &&
                    !colorFilterHostFrame.empty() &&
                    colorFilterHostFrame.cols() > 0 && colorFilterHostFrame.rows() > 0 &&
                    colorFilterHostFrame.cols() <= 10000 && colorFilterHostFrame.rows() <= 10000) {
                    frameToDisplay = &colorFilterHostFrame;
                    lastPreviewUpdate = now;
                }
            } else if (!shouldUpdatePreview && !colorFilterHostFrame.empty()) {
                frameToDisplay = &colorFilterHostFrame;
            }

            ImGui::SliderFloat("Scale", &color_filter_scale, 0.1f, 2.0f, "%.1fx");

            if (frameToDisplay) {
                // Apply color filter if enabled
                if (ctx.config.color_filter_enabled) {
                    uploadColorFilterFrame(*frameToDisplay, ctx.config.color_filter_mode,
                        ctx.config.color_filter_r_min, ctx.config.color_filter_r_max,
                        ctx.config.color_filter_g_min, ctx.config.color_filter_g_max,
                        ctx.config.color_filter_b_min, ctx.config.color_filter_b_max,
                        ctx.config.color_filter_h_min, ctx.config.color_filter_h_max,
                        ctx.config.color_filter_s_min, ctx.config.color_filter_s_max,
                        ctx.config.color_filter_v_min, ctx.config.color_filter_v_max);
                } else {
                    uploadDebugFrame(*frameToDisplay);
                }

                ID3D11ShaderResourceView* displaySRV = ctx.config.color_filter_enabled ? g_colorFilterSRV : g_debugSRV;
                int displayW = ctx.config.color_filter_enabled ? colorFilterTexW : texW;
                int displayH = ctx.config.color_filter_enabled ? colorFilterTexH : texH;

                if (displaySRV && displayW > 0 && displayH > 0) {
                    float safe_scale = color_filter_scale;
                    if (safe_scale <= 0 || safe_scale > 10.0f) safe_scale = 1.0f;

                    ImVec2 image_size(displayW * safe_scale, displayH * safe_scale);
                    ImGui::Image(displaySRV, image_size);

                    // Draw crosshair
                    ImVec2 image_pos = ImGui::GetItemRectMin();
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    if (draw_list && displayW > 0 && displayH > 0) {
                        float center_x = image_pos.x + (displayW * safe_scale) / 2.0f;
                        float center_y = image_pos.y + (displayH * safe_scale) / 2.0f;

                        ImU32 crosshair_color = IM_COL32(255, 255, 255, 255);
                        draw_list->AddLine(ImVec2(center_x - 10, center_y), ImVec2(center_x + 10, center_y), crosshair_color, 2.0f);
                        draw_list->AddLine(ImVec2(center_x, center_y - 10), ImVec2(center_x, center_y + 10), crosshair_color, 2.0f);
                        draw_list->AddCircle(ImVec2(center_x, center_y), 3.0f, crosshair_color, 0, 2.0f);
                    }
                }
            } else {
                // Placeholder
                if (colorFilterTexW > 0 && colorFilterTexH > 0) {
                    float safe_scale = color_filter_scale;
                    if (safe_scale <= 0 || safe_scale > 10.0f) safe_scale = 1.0f;
                    ImGui::Dummy(ImVec2(colorFilterTexW * safe_scale, colorFilterTexH * safe_scale));
                } else if (texW > 0 && texH > 0) {
                    float safe_scale = color_filter_scale;
                    if (safe_scale <= 0 || safe_scale > 10.0f) safe_scale = 1.0f;
                    ImGui::Dummy(ImVec2(texW * safe_scale, texH * safe_scale));
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Waiting for preview...");
                }
            }
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel - Color Filter Controls
    ImGui::BeginChild("ColorFilterControls", ImVec2(rightWidth, 0), true);
    {
        ImGui::Text("Color Filter");
        ImGui::Separator();

        // Enable checkbox
        if (ImGui::Checkbox("Enable Filter", &ctx.config.color_filter_enabled)) {
            SAVE_PROFILE();
        }

        ImGui::Spacing();

        // Color mode selection
        const char* modes[] = { "RGB", "HSV" };
        if (ImGui::Combo("Mode", &ctx.config.color_filter_mode, modes, IM_ARRAYSIZE(modes))) {
            SAVE_PROFILE();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Color range preview box with From-To visualization
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Labels above the color boxes
        float totalWidth = ImGui::GetContentRegionAvail().x - 10;
        float boxWidth = totalWidth * 0.25f;
        float gradientWidth = totalWidth * 0.5f;

        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "From");
        ImGui::SameLine(boxWidth + gradientWidth * 0.5f - 10);
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Range");
        ImGui::SameLine(boxWidth + gradientWidth + boxWidth * 0.5f - 5);
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "To");

        ImVec2 preview_pos = ImGui::GetCursorScreenPos();
        ImVec2 preview_size(totalWidth, 40);

        if (ctx.config.color_filter_mode == 0) {
            DrawColorRangeRGB(draw_list, preview_pos, preview_size,
                ctx.config.color_filter_r_min, ctx.config.color_filter_r_max,
                ctx.config.color_filter_g_min, ctx.config.color_filter_g_max,
                ctx.config.color_filter_b_min, ctx.config.color_filter_b_max);
        } else {
            DrawColorRangeHSV(draw_list, preview_pos, preview_size,
                ctx.config.color_filter_h_min, ctx.config.color_filter_h_max,
                ctx.config.color_filter_s_min, ctx.config.color_filter_s_max,
                ctx.config.color_filter_v_min, ctx.config.color_filter_v_max);
        }
        ImGui::Dummy(preview_size);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ctx.config.color_filter_mode == 0) {
            // RGB mode controls
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Red");
            if (ImGui::DragIntRange2("##R", &ctx.config.color_filter_r_min, &ctx.config.color_filter_r_max, 1, 0, 255, "Min: %d", "Max: %d")) { SAVE_PROFILE(); }

            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Green");
            if (ImGui::DragIntRange2("##G", &ctx.config.color_filter_g_min, &ctx.config.color_filter_g_max, 1, 0, 255, "Min: %d", "Max: %d")) { SAVE_PROFILE(); }

            ImGui::TextColored(ImVec4(0.3f, 0.3f, 1.0f, 1.0f), "Blue");
            if (ImGui::DragIntRange2("##B", &ctx.config.color_filter_b_min, &ctx.config.color_filter_b_max, 1, 0, 255, "Min: %d", "Max: %d")) { SAVE_PROFILE(); }
        } else {
            // HSV mode - Hue palette
            ImVec2 palette_pos = ImGui::GetCursorScreenPos();
            ImVec2 palette_size(ImGui::GetContentRegionAvail().x - 10, 25);
            DrawHuePalette(draw_list, palette_pos, palette_size,
                ctx.config.color_filter_h_min, ctx.config.color_filter_h_max);
            ImGui::Dummy(palette_size);

            ImGui::Text("Hue (0-179)");
            if (ImGui::DragIntRange2("##H", &ctx.config.color_filter_h_min, &ctx.config.color_filter_h_max, 1, 0, 179, "Min: %d", "Max: %d")) { SAVE_PROFILE(); }
            if (ctx.config.color_filter_h_min > ctx.config.color_filter_h_max) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Wraparound (for red)");
            }

            ImGui::Spacing();
            ImGui::Text("Saturation (0-255)");
            if (ImGui::DragIntRange2("##S", &ctx.config.color_filter_s_min, &ctx.config.color_filter_s_max, 1, 0, 255, "Min: %d", "Max: %d")) { SAVE_PROFILE(); }

            ImGui::Spacing();
            ImGui::Text("Value (0-255)");
            if (ImGui::DragIntRange2("##V", &ctx.config.color_filter_v_min, &ctx.config.color_filter_v_max, 1, 0, 255, "Min: %d", "Max: %d")) { SAVE_PROFILE(); }

            // Presets
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Presets:");
            if (ImGui::Button("Red", ImVec2(50, 0))) {
                ctx.config.color_filter_h_min = 0; ctx.config.color_filter_h_max = 10;
                ctx.config.color_filter_s_min = 100; ctx.config.color_filter_s_max = 255;
                ctx.config.color_filter_v_min = 100; ctx.config.color_filter_v_max = 255;
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("Yellow", ImVec2(50, 0))) {
                ctx.config.color_filter_h_min = 20; ctx.config.color_filter_h_max = 35;
                ctx.config.color_filter_s_min = 100; ctx.config.color_filter_s_max = 255;
                ctx.config.color_filter_v_min = 100; ctx.config.color_filter_v_max = 255;
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("Green", ImVec2(50, 0))) {
                ctx.config.color_filter_h_min = 35; ctx.config.color_filter_h_max = 85;
                ctx.config.color_filter_s_min = 100; ctx.config.color_filter_s_max = 255;
                ctx.config.color_filter_v_min = 100; ctx.config.color_filter_v_max = 255;
                SAVE_PROFILE();
            }
            if (ImGui::Button("Blue", ImVec2(50, 0))) {
                ctx.config.color_filter_h_min = 100; ctx.config.color_filter_h_max = 130;
                ctx.config.color_filter_s_min = 100; ctx.config.color_filter_s_max = 255;
                ctx.config.color_filter_v_min = 100; ctx.config.color_filter_v_max = 255;
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("Purple", ImVec2(50, 0))) {
                ctx.config.color_filter_h_min = 130; ctx.config.color_filter_h_max = 160;
                ctx.config.color_filter_s_min = 100; ctx.config.color_filter_s_max = 255;
                ctx.config.color_filter_v_min = 100; ctx.config.color_filter_v_max = 255;
                SAVE_PROFILE();
            }
            ImGui::SameLine();
            if (ImGui::Button("Orange", ImVec2(50, 0))) {
                ctx.config.color_filter_h_min = 10; ctx.config.color_filter_h_max = 25;
                ctx.config.color_filter_s_min = 100; ctx.config.color_filter_s_max = 255;
                ctx.config.color_filter_v_min = 100; ctx.config.color_filter_v_max = 255;
                SAVE_PROFILE();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Pixel size filter
        ImGui::Text("Pixel Size Filter");
        ImGui::Text("Min Pixels:");
        if (ImGui::InputInt("##min_px", &ctx.config.color_filter_min_pixels, 10, 100)) {
            if (ctx.config.color_filter_min_pixels < 0) ctx.config.color_filter_min_pixels = 0;
            SAVE_PROFILE();
        }
        ImGui::Text("Max Pixels:");
        if (ImGui::InputInt("##max_px", &ctx.config.color_filter_max_pixels, 100, 1000)) {
            if (ctx.config.color_filter_max_pixels < 0) ctx.config.color_filter_max_pixels = 100000;
            SAVE_PROFILE();
        }
    }
    ImGui::EndChild();

    ImGui::EndChild();
}

// Cleanup function for color filter resources
void cleanup_color_filter_resources()
{
    SAFE_RELEASE(g_colorFilterTex);
    SAFE_RELEASE(g_colorFilterSRV);
    colorFilterTexW = 0;
    colorFilterTexH = 0;
}
