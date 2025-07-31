#include "../AppContext.h"
#include "../detector/detector.h"
#include "needaimbot.h"
#include "overlay.h"
#include "include/other_tools.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include <vector>
#include <string>
#include <d3d11.h> 
#include "../capture/capture.h"
#include "../capture/capture.h" 
#include <cuda_runtime.h> 
#include "../postprocess/postProcess.h" 
#include "../cuda/color_conversion.h"
 


#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
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


static ID3D11Texture2D* g_debugTex = nullptr;
static ID3D11ShaderResourceView* g_debugSRV = nullptr;
static int texW = 0, texH = 0;
static float debug_scale = 1.0f; 


static ID3D11Texture2D* g_colorMaskTex = nullptr;
static ID3D11ShaderResourceView* g_colorMaskSRV = nullptr;
static int colorTexW = 0, colorTexH = 0;
static float color_mask_preview_scale = 0.5f; 



static int g_crosshairH = 0, g_crosshairS = 0, g_crosshairV = 0;
static bool g_crosshairHsvValid = false;



// GPU-accelerated version with direct GPU memory handling
static void uploadDebugFrame(const SimpleCudaMat& bgrGpu)
{
    if (bgrGpu.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        return;
    }

    if (!g_debugTex || bgrGpu.cols() != texW || bgrGpu.rows() != texH)
    {
        SAFE_RELEASE(g_debugTex);
        SAFE_RELEASE(g_debugSRV);

        texW = bgrGpu.cols();  texH = bgrGpu.rows();

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
            return;
        }
    }

    // GPU-based color conversion
    static SimpleCudaMat rgbaGpu;
    rgbaGpu.create(bgrGpu.rows(), bgrGpu.cols(), 4);
    
    cudaError_t err;
    if (bgrGpu.channels() == 4) {
        // BGRA to RGBA conversion
        err = cuda_bgra2rgba(bgrGpu.data(), rgbaGpu.data(),
                             bgrGpu.cols(), bgrGpu.rows(),
                             static_cast<int>(bgrGpu.step()), 
                             static_cast<int>(rgbaGpu.step()));
    } else {
        // BGR to RGBA conversion
        err = cuda_bgr2rgba(bgrGpu.data(), rgbaGpu.data(),
                            bgrGpu.cols(), bgrGpu.rows(),
                            static_cast<int>(bgrGpu.step()), 
                            static_cast<int>(rgbaGpu.step()));
    }
    
    if (err != cudaSuccess) {
        std::cerr << "[Debug] CUDA color conversion failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Download converted data from GPU
    static SimpleMat rgba;
    rgba.create(rgbaGpu.rows(), rgbaGpu.cols(), 4);
    rgbaGpu.download(rgba.data(), rgba.step());

    if (rgba.empty() || rgba.cols() <= 0 || rgba.rows() <= 0) { 
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_debugTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (SUCCEEDED(hr_map))
    {
        // Copy row by row, handling different row pitches
        size_t copy_width = texW * 4;  // Width in bytes for RGBA
        for (int y = 0; y < texH; ++y) {
            memcpy((uint8_t*)ms.pData + ms.RowPitch * y, 
                   rgba.data() + y * rgba.step(), 
                   copy_width);
        }
        g_pd3dDeviceContext->Unmap(g_debugTex, 0);
    }
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

static void drawDetections(ImDrawList* draw_list, ImVec2 image_pos, float debug_scale) {
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.detector) return;
    
    std::lock_guard<std::mutex> det_lock(ctx.detector->detectionMutex);
    
    // Check if detection results are fresh (less than 100ms old)
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastDetection = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.detector->m_lastDetectionTime).count();
    
    if (timeSinceLastDetection > 100) {
        // Detection results are too old, don't display them
        return;
    }
    
    if (ctx.detector->m_finalDetectionsCountHost > 0 && ctx.detector->m_finalDetectionsGpu.get() != nullptr)
    {
        std::vector<Detection> host_detections(ctx.detector->m_finalDetectionsCountHost);
        cudaError_t err = cudaMemcpy(host_detections.data(), ctx.detector->m_finalDetectionsGpu.get(),
                                     ctx.detector->m_finalDetectionsCountHost * sizeof(Detection),
                                     cudaMemcpyDeviceToHost);

        if (err == cudaSuccess)
        {
            for (const auto& det : host_detections)
            {
                // Skip invalid detections
                if (det.width <= 0 || det.height <= 0) {
                    continue;
                }
                
                ImVec2 p1(image_pos.x + det.x * debug_scale,
                          image_pos.y + det.y * debug_scale);
                ImVec2 p2(image_pos.x + (det.x + det.width) * debug_scale,
                          image_pos.y + (det.y + det.height) * debug_scale);

                // Check if this is the best target
                bool is_best_target = false;
                {
                    std::lock_guard<std::mutex> lock(ctx.overlay_target_mutex);
                    if (ctx.overlay_has_target.load() && 
                        det.x == ctx.overlay_target_info.x &&
                        det.y == ctx.overlay_target_info.y &&
                        det.width == ctx.overlay_target_info.width &&
                        det.height == ctx.overlay_target_info.height) {
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

                std::string className = CommonHelpers::getClassNameById(det.classId);
                std::string label = className + " (" + std::to_string(static_cast<int>(det.confidence * 100.0f)) + "%)";
                if (is_best_target) {
                    label = "[TARGET] " + label;
                }
                ImU32 text_color = is_best_target ? IM_COL32(0, 255, 0, 255) : 
                                  (det.confidence >= ctx.config.confidence_threshold ? IM_COL32(255, 255, 0, 255) : IM_COL32(255, 0, 0, 255));
                draw_list->AddText(ImVec2(p1.x, p1.y - 16), text_color, label.c_str());
            }
        }
    }
}


void draw_debug_frame()
{
    auto& ctx = AppContext::getInstance();
    
    // Only process if debug window is visible
    if (!ctx.config.show_window) {
        ImGui::TextUnformatted("Debug window disabled.");
        return;
    }
    
    // Use GPU frame directly without downloading
    extern SimpleCudaMat latestFrameGpu;
    
    if (latestFrameGpu.empty()) {
        ImGui::TextUnformatted("Debug frame unavailable.");
        return;
    }
    
    // Upload GPU frame directly
    uploadDebugFrame(latestFrameGpu);
    
    ImGui::SliderFloat("Debug scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
    ImVec2 image_size(texW * debug_scale, texH * debug_scale);
    if (g_debugSRV) {
        ImGui::Image(g_debugSRV, image_size);
    } else {
        ImGui::TextUnformatted("Debug frame: Data processed, but texture unavailable for display.");
        ImGui::TextUnformatted("Overlays skipped: Debug texture unavailable.");
        return;
    }

    ImVec2 image_pos = ImGui::GetItemRectMin();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Draw detections
    drawDetections(draw_list, image_pos, debug_scale);


    
    ImGui::SeparatorText("Crosshair Pixel HSV");
    ImGui::Spacing();
    if (g_crosshairHsvValid) {
        ImGui::Text("Crosshair H: %d, S: %d, V: %d", g_crosshairH, g_crosshairS, g_crosshairV);
    } else {
        ImGui::TextUnformatted("Crosshair HSV: N/A (Debug preview not active or frame empty)");
    }
    ImGui::Spacing();
    
    
    
}





inline std::vector<const char*> getProfileCstrs(const std::vector<std::string>& profiles) {
    std::vector<const char*> cstrs;
    cstrs.reserve(profiles.size());
    for(const auto& s : profiles)
        cstrs.push_back(s.c_str());
    return cstrs;
}

void draw_debug()
{
    auto& ctx = AppContext::getInstance();
    
    // Display pause status prominently
    if (ctx.detectionPaused.load()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f)); // Red color
        ImGui::Text("AIMBOT PAUSED");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f)); // Green color
        ImGui::Text("AIMBOT ACTIVE");
        ImGui::PopStyleColor();
    }
    
    if (ctx.config.show_fps) {
        ImGui::Text("Capture FPS: %.1f", ctx.g_current_capture_fps.load());
        ImGui::Text("Inference Time: %.2f ms", ctx.g_current_inference_time_ms.load());
        
        ImGui::Separator();
        ImGui::Spacing();
    }

    ImGui::SeparatorText("Debug Preview & Overlay"); 
    ImGui::Spacing();

    bool prev_show_window_state = ctx.config.show_window; 
    if (ImGui::Checkbox("Show Preview Window", &ctx.config.show_window)) 
    {
        
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

            
            if (g_colorMaskSRV) {
                g_colorMaskSRV->Release();
                g_colorMaskSRV = nullptr;
            }
            if (g_colorMaskTex) {
                g_colorMaskTex->Release();
                g_colorMaskTex = nullptr;
            }
            colorTexW = 0;
            colorTexH = 0;
        }
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Toggles the live debug frame preview below."); }

    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    
    
    if (ImGui::Checkbox("Enable FPS Display", &ctx.config.show_fps)) { SAVE_PROFILE(); } 

    // Crosshair offset adjustment controls
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    ImGui::Text("Crosshair Offset X=%.1f, Y=%.1f", ctx.config.crosshair_offset_x, ctx.config.crosshair_offset_y);
    
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

    ImGui::Spacing();

    if (ctx.config.show_window) 
    {
        // Use GPU frame directly
        extern SimpleCudaMat latestFrameGpu;
        
        if (latestFrameGpu.empty()) {
            ImGui::TextUnformatted("Debug frame unavailable.");
            return;
        }
        
        // Upload GPU frame directly
        uploadDebugFrame(latestFrameGpu);
        
        ImGui::SliderFloat("Debug scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
        ImVec2 image_size(texW * debug_scale, texH * debug_scale);
        if (g_debugSRV) {
            ImGui::Image(g_debugSRV, image_size);
        } else {
            ImGui::TextUnformatted("Debug frame: Data processed, but texture unavailable for display.");
            ImGui::TextUnformatted("Overlays skipped: Debug texture unavailable.");
            return;
        }

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
        
        // Debug text - moved below image
        ImGui::Separator();
        ImGui::Text("Color Legend: Green=Best Target, Yellow=Valid, Red=Low Confidence");
        
        // Draw target offset if best target exists and is valid (using synchronized overlay data)
        Detection target;
        bool hasTarget = false;
        {
            std::lock_guard<std::mutex> lock(ctx.overlay_target_mutex);
            hasTarget = ctx.overlay_has_target.load();
            if (hasTarget) {
                target = ctx.overlay_target_info;
            }
        }
        
        if (hasTarget && target.width > 0 && target.height > 0) {
            float target_center_x = image_pos.x + (target.x + target.width / 2.0f) * debug_scale;
            
            // Calculate Y offset based on head/body settings
            float y_offset;
            bool is_head = false;
            for (const auto& cs : ctx.config.class_settings) {
                if (cs.id == target.classId && cs.name == ctx.config.head_class_name && !cs.ignore) {
                    is_head = true;
                    break;
                }
            }
            y_offset = is_head ? ctx.config.head_y_offset : ctx.config.body_y_offset;
            
            float target_center_y = image_pos.y + (target.y + target.height * y_offset) * debug_scale;
            
            // Draw line from center to target
            draw_list->AddLine(ImVec2(center_x, center_y), ImVec2(target_center_x, target_center_y), 
                               IM_COL32(0, 255, 255, 255), 2.0f);
            
            // Draw target point
            draw_list->AddCircleFilled(ImVec2(target_center_x, target_center_y), 4.0f, IM_COL32(0, 255, 255, 255));
        }



        
        ImGui::SeparatorText("Crosshair Pixel HSV");
        ImGui::Spacing();
        if (g_crosshairHsvValid) {
            ImGui::Text("Crosshair H: %d, S: %d, V: %d", g_crosshairH, g_crosshairS, g_crosshairV);
        } else {
            ImGui::TextUnformatted("Crosshair HSV: N/A (Debug preview not active or frame empty)");
        }
        ImGui::Spacing();
    }

    ImGui::Spacing();
    ImGui::Separator(); 
    ImGui::Spacing();

    
    ImGui::SeparatorText("RGB Color Filter Debug");
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable RGB Color Filter (in Config)", &ctx.config.enable_color_filter)) {
        SAVE_PROFILE(); 
        if (ctx.detector) ctx.detector->m_ignore_flags_need_update = true; 
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
            if (ctx.detector) {
                try {
                    colorMaskGpu = ctx.detector->getColorMaskGpu();
                } catch (const std::exception&) {
                    // Handle any exceptions from getColorMaskGpu
                    colorMaskGpu = SimpleCudaMat();
                }
            } 
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
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    if (ImGui::Checkbox("Verbose Console Output", &ctx.config.verbose)) { SAVE_PROFILE(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Prints more detailed information to the console window for debugging."); }

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
