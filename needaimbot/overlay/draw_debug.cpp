#include "../AppContext.h"
#include "../core/constants.h"
#include "needaimbot.h"
#include "overlay.h"
#include "include/other_tools.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include <vector>
#include <string>
#include <d3d11.h>
// #include "../capture/capture.h" - removed, using GPU capture now
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cstring>
// OpenCV removed - using custom CUDA utilities
#include "../cuda/simple_cuda_mat.h"





#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)     \
    do {                    \
        if ((p) != nullptr) { \
            (p)->Release();   \
            (p) = nullptr;    \
        }                     \
    } while (0)
#endif


ID3D11Texture2D* g_debugTex = nullptr;
ID3D11ShaderResourceView* g_debugSRV = nullptr;
int texW = 0, texH = 0;
float debug_scale = 1.0f;

// Mutex for thread-safe D3D11 resource access
std::mutex g_debugTexMutex; 


static ID3D11Texture2D* g_colorMaskTex = nullptr;
static ID3D11ShaderResourceView* g_colorMaskSRV = nullptr;
static int colorTexW = 0, colorTexH = 0;
static float color_mask_preview_scale = 0.5f; 



static int g_crosshairH = 0, g_crosshairS = 0, g_crosshairV = 0;
static bool g_crosshairHsvValid = false;



// Upload a CPU RGBA frame into the debug texture for ImGui rendering
void uploadDebugFrame(const SimpleMat& rgbaCpu)
{
    std::lock_guard<std::mutex> lock(g_debugTexMutex);

    if (rgbaCpu.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        return;
    }

    if (rgbaCpu.cols() <= 0 || rgbaCpu.rows() <= 0 ||
        rgbaCpu.cols() > 10000 || rgbaCpu.rows() > 10000) {
        return;
    }

    if (rgbaCpu.data() == nullptr) {
        return;
    }

    if (!g_debugTex || rgbaCpu.cols() != texW || rgbaCpu.rows() != texH)
    {
        SAFE_RELEASE(g_debugTex);
        SAFE_RELEASE(g_debugSRV);

        texW = rgbaCpu.cols();  texH = rgbaCpu.rows();

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = texW;
        td.Height = texH;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        HRESULT hr_tex = g_pd3dDevice->CreateTexture2D(&td, nullptr, &g_debugTex);
        if (FAILED(hr_tex))
        {
            SAFE_RELEASE(g_debugTex);
            std::cerr << "[Debug] Failed to create debug texture! HRESULT=" << std::hex << hr_tex << std::dec << std::endl;
            return;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = td.Format;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        HRESULT hr_srv = g_pd3dDevice->CreateShaderResourceView(g_debugTex, &sd, &g_debugSRV);
        if (FAILED(hr_srv))
        {
            SAFE_RELEASE(g_debugTex);
            SAFE_RELEASE(g_debugSRV);
            std::cerr << "[Debug] Failed to create shader resource view for preview texture. HRESULT="
                      << std::hex << hr_srv << std::dec << std::endl;
            return;
        }
    }

    if (!g_debugTex || !g_pd3dDeviceContext) {
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms = {};
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_debugTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (FAILED(hr_map) || ms.pData == nullptr)
    {
        std::cerr << "[Debug] Failed to map texture: HRESULT=" << std::hex << hr_map << std::dec << std::endl;
        return;
    }

    try {
        const uint8_t* src = rgbaCpu.data();
        size_t srcPitch = rgbaCpu.step();
        size_t rowBytes = std::min<size_t>(texW * 4, ms.RowPitch);

        for (int y = 0; y < texH; ++y) {
            std::memcpy(static_cast<uint8_t*>(ms.pData) + ms.RowPitch * y,
                        src + srcPitch * y,
                        rowBytes);
        }
    } catch (...) {
        std::cerr << "[Debug] Exception during texture copy" << std::endl;
    }

    g_pd3dDeviceContext->Unmap(g_debugTex, 0);
}




static void uploadColorMaskTexture(const SimpleMat& grayMask)
{
    if (grayMask.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        return;
    }

    if (grayMask.channels() != 1) {
        // Not a single channel mask
        return;
    }

    if (!g_colorMaskTex || grayMask.cols() != colorTexW || grayMask.rows() != colorTexH)
    {
        SAFE_RELEASE(g_colorMaskTex);
        SAFE_RELEASE(g_colorMaskSRV);

        colorTexW = grayMask.cols();  colorTexH = grayMask.rows();

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = colorTexW;
        td.Height = colorTexH;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; 
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        HRESULT hr_tex = g_pd3dDevice->CreateTexture2D(&td, nullptr, &g_colorMaskTex);
        if (FAILED(hr_tex))
        {
            SAFE_RELEASE(g_colorMaskTex);
            
            return;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = td.Format;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        HRESULT hr_srv = g_pd3dDevice->CreateShaderResourceView(g_colorMaskTex, &sd, &g_colorMaskSRV);
        if (FAILED(hr_srv))
        {
            SAFE_RELEASE(g_colorMaskTex);
            SAFE_RELEASE(g_colorMaskSRV);
            
            return;
        }
    }

    // Convert grayscale mask to RGBA
    static SimpleMat rgbaMask;
    rgbaMask.create(grayMask.rows(), grayMask.cols(), 4);
    
    // Manual GRAY to RGBA conversion
    const uint8_t* src = grayMask.data();
    uint8_t* dst = rgbaMask.data();
    for (int y = 0; y < grayMask.rows(); y++) {
        for (int x = 0; x < grayMask.cols(); x++) {
            uint8_t gray_val = src[y * grayMask.step() + x];
            int dst_idx = static_cast<int>(y * rgbaMask.step() + x * 4);
            dst[dst_idx + 0] = gray_val; // R
            dst[dst_idx + 1] = gray_val; // G
            dst[dst_idx + 2] = gray_val; // B
            dst[dst_idx + 3] = 255;       // A
        }
    }

    if (rgbaMask.empty() || rgbaMask.cols() <= 0 || rgbaMask.rows() <= 0) {
        
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_colorMaskTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (SUCCEEDED(hr_map))
    {
        for (int y = 0; y < colorTexH; ++y)
            memcpy((uint8_t*)ms.pData + ms.RowPitch * y, rgbaMask.data() + y * rgbaMask.step(), colorTexW * 4); 
        g_pd3dDeviceContext->Unmap(g_colorMaskTex, 0);
    } else {
        
    }
}

void drawDetections(ImDrawList* draw_list, ImVec2 image_pos, float debug_scale) {
    auto& ctx = AppContext::getInstance();
    
    // Validate parameters
    if (!draw_list || debug_scale <= 0) return;
    
    // Get all detected targets
    std::vector<Target> all_targets = ctx.getAllTargets();
    
    // Early exit if no targets
    if (all_targets.empty()) {
        return;
    }
    
    // Get best target for highlighting
    Target best_target{};
    bool has_best_target = false;
    try {
        if (ctx.hasValidTarget()) {
            best_target = ctx.getBestTarget();
            has_best_target = true;
        }
    } catch (...) {
        has_best_target = false;
    }
    
    // Draw all targets
    for (const auto& det : all_targets) {
        // Skip invalid detections
        if (det.width <= 0 || det.height <= 0 || 
            det.confidence <= 0.0f || det.classId < 0 ||
            det.x < 0 || det.y < 0) {
            continue;
        }
        
        // Calculate screen positions
        ImVec2 p1(image_pos.x + det.x * debug_scale,
                  image_pos.y + det.y * debug_scale);
        ImVec2 p2(image_pos.x + (det.x + det.width) * debug_scale,
                  image_pos.y + (det.y + det.height) * debug_scale);

        // Check if this is the best target
        bool is_best_target = false;
        if (has_best_target) {
            if (abs(det.x - best_target.x) < 1.0f &&
                abs(det.y - best_target.y) < 1.0f &&
                abs(det.width - best_target.width) < 1.0f &&
                abs(det.height - best_target.height) < 1.0f) {
                is_best_target = true;
            }
        }

        // Color coding: Green=best target, Yellow=valid target, Red=low confidence
        ImU32 color;
        float thickness;
        if (is_best_target) {
            color = IM_COL32(0, 255, 0, 255);  // Green for best target
            thickness = 3.0f;
        } else if (det.confidence >= ctx.config.confidence_threshold) {
            color = IM_COL32(255, 255, 0, 255);  // Yellow for valid targets
            thickness = 2.0f;
        } else {
            color = IM_COL32(255, 0, 0, 255);   // Red for low confidence
            thickness = 1.0f;
        }

        draw_list->AddRect(p1, p2, color, 1.0f, 0, thickness); 
        
        // Build label
        std::string className = "Class" + std::to_string(det.classId);  // Simplified class name
        
        char conf_str[32];
        snprintf(conf_str, sizeof(conf_str), "%.0f%%", det.confidence * 100.0f);
        
        std::string label = className + " (" + conf_str + ")";
        
        
        if (is_best_target) {
            label = "[TARGET] " + label;
        }
        
        // Ensure label isn't too long
        if (label.length() > Constants::MAX_DEBUG_LABEL_LENGTH) {
            label = label.substr(0, Constants::MAX_DEBUG_LABEL_LENGTH - 3) + "...";
        }
        
        ImU32 text_color = is_best_target ? IM_COL32(0, 255, 0, 255) : 
                          (det.confidence >= ctx.config.confidence_threshold ? IM_COL32(255, 255, 0, 255) : IM_COL32(255, 0, 0, 255));
        
        // Check text position is valid
        if (p1.x >= 0 && p1.y >= 16) {
            draw_list->AddText(ImVec2(p1.x, p1.y - 16), text_color, label.c_str());
        }
    }
}








void draw_debug()
{
    auto& ctx = AppContext::getInstance();
    
    // Display pause status prominently
    if (ctx.detection_paused.load()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f)); // Red color
        ImGui::Text("AIMBOT PAUSED");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f)); // Green color
        ImGui::Text("AIMBOT ACTIVE");
        ImGui::PopStyleColor();
    }
    

    if (ImGui::Checkbox("Enable FPS Display", &ctx.config.show_fps)) { SAVE_PROFILE(); }

    ImGui::Spacing();
    ImGui::Separator(); 
    ImGui::Spacing();

    
    ImGui::SeparatorText("RGB Color Filter Debug");
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable RGB Color Filter (in Config)", &ctx.config.enable_color_filter)) {
        SAVE_PROFILE(); 
        // TODO: Implement flag update for TensorRT integration 
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Toggles the RGB color filtering logic (config.enable_color_filter)."); }

    if (ctx.config.enable_color_filter) {
        ImGui::Text("Min Color Pixels: %d", ctx.config.min_color_pixels);
        ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
        ImGui::Text(ctx.config.remove_color_matches ? "Mode: Remove if color matches" : "Mode: Keep if color matches");

        ImGui::Text("Red Range: %3d - %3d", ctx.config.rgb_min_r, ctx.config.rgb_max_r);
        ImGui::Text("Green Range: %3d - %3d", ctx.config.rgb_min_g, ctx.config.rgb_max_g);
        ImGui::Text("Blue Range: %3d - %3d", ctx.config.rgb_min_b, ctx.config.rgb_max_b);
        ImGui::Spacing();

        static bool show_color_mask_preview = false;
        ImGui::Checkbox("Show Color Mask Preview", &show_color_mask_preview);

        if (show_color_mask_preview) {
            
            SimpleCudaMat colorMaskGpu;
            // TODO: Implement color mask retrieval for TensorRT integration
            // colorMaskGpu = nullptr; // Temporarily disabled 
            if (!colorMaskGpu.empty()) {
                static SimpleMat colorMaskCpu; 
                try {
                    colorMaskCpu.create(colorMaskGpu.rows(), colorMaskGpu.cols(), colorMaskGpu.channels());
                    colorMaskGpu.download(colorMaskCpu.data(), colorMaskCpu.step()); 
                } catch (...) {
                    ImGui::Text("Error downloading Color Mask");
                    colorMaskCpu.release(); 
                }

                if (!colorMaskCpu.empty()) {
                    uploadColorMaskTexture(colorMaskCpu); 
                    if (g_colorMaskSRV && colorTexW > 0 && colorTexH > 0) {
                        ImGui::SliderFloat("Color Mask Scale", &color_mask_preview_scale, 0.1f, 2.0f, "%.1fx");
                        ImVec2 mask_img_size(colorTexW * color_mask_preview_scale, colorTexH * color_mask_preview_scale);
                        ImGui::Image(g_colorMaskSRV, mask_img_size);
                    } else {
                        ImGui::Text("Color Mask Texture not available for display.");
                    }
                } else {
                    ImGui::Text("Color Mask (CPU) is empty or download failed.");
                }
            } else {
                ImGui::Text("HSV Mask (GPU) is not available from detector or not generated.");
            }
        }
    } else {
        ImGui::TextUnformatted("HSV Filter is currently disabled in config.");
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    
    ImGui::SeparatorText("Screenshot Settings");
    ImGui::Spacing();

    CommonHelpers::drawKeyBindingList("Screenshot Button", ctx.config.screenshot_button, key_names, key_names_cstrs);

    ImGui::Spacing();
    if (ImGui::InputInt("Screenshot Delay (ms)", &ctx.config.screenshot_delay, 50, 500)) { SAVE_PROFILE(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Delay in milliseconds after pressing the button before taking the screenshot."); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Miscellaneous");
    ImGui::Spacing();

    if (ImGui::Checkbox("Always On Top", &ctx.config.always_on_top)) { SAVE_PROFILE(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Keeps the overlay window always on top of other windows."); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Console Window Control");
    ImGui::Spacing();

    if (ImGui::Button("Hide Console"))
    {
        HideConsole();
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Hides the black console window."); }
    ImGui::SameLine();
    if (ImGui::Button("Show Console"))
    {
        ShowConsole();
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Shows the black console window if it's hidden."); }
    ImGui::Spacing();
}
