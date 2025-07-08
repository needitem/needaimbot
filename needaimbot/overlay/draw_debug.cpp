#include "../AppContext.h"
#include "../detector/detector.h"
#include "imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "include/other_tools.h"
#include <vector>
#include <string>
#include <d3d11.h> 
#include "capture.h"
#include "detector/capture.h" 
#include <cuda_runtime.h> 
#include "detector/postProcess.h" 
 


#include <array>
#include <atomic>
#include "opencv2/core/cuda.hpp" 





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


static ID3D11Texture2D* g_hsvMaskTex = nullptr;
static ID3D11ShaderResourceView* g_hsvMaskSRV = nullptr;
static int hsvTexW = 0, hsvTexH = 0;
static float hsv_mask_preview_scale = 0.5f; 



static int g_crosshairH = 0, g_crosshairS = 0, g_crosshairV = 0;
static bool g_crosshairHsvValid = false;



static void uploadDebugFrame(const cv::Mat& bgr)
{
    
    if (bgr.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        return;
    }

    if (!g_debugTex || bgr.cols != texW || bgr.rows != texH)
    {
        SAFE_RELEASE(g_debugTex);
        SAFE_RELEASE(g_debugSRV);

        texW = bgr.cols;  texH = bgr.rows;

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

    static cv::Mat rgba;
    try {
        cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    } catch (const cv::Exception& e) {
        
        
        return;
    }

    if (rgba.empty() || rgba.cols <= 0 || rgba.rows <= 0) { 
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_debugTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (SUCCEEDED(hr_map))
    {
        for (int y = 0; y < texH; ++y)
            memcpy((uint8_t*)ms.pData + ms.RowPitch * y, rgba.ptr(y), texW * 4);
        g_pd3dDeviceContext->Unmap(g_debugTex, 0);
    } else {
        
        
    }
}


static void uploadHsvMaskTexture(const cv::Mat& grayMask)
{
    if (grayMask.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        return;
    }

    if (grayMask.type() != CV_8UC1) {
        
        return;
    }

    if (!g_hsvMaskTex || grayMask.cols != hsvTexW || grayMask.rows != hsvTexH)
    {
        SAFE_RELEASE(g_hsvMaskTex);
        SAFE_RELEASE(g_hsvMaskSRV);

        hsvTexW = grayMask.cols;  hsvTexH = grayMask.rows;

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = hsvTexW;
        td.Height = hsvTexH;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; 
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        HRESULT hr_tex = g_pd3dDevice->CreateTexture2D(&td, nullptr, &g_hsvMaskTex);
        if (FAILED(hr_tex))
        {
            SAFE_RELEASE(g_hsvMaskTex);
            
            return;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = td.Format;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        HRESULT hr_srv = g_pd3dDevice->CreateShaderResourceView(g_hsvMaskTex, &sd, &g_hsvMaskSRV);
        if (FAILED(hr_srv))
        {
            SAFE_RELEASE(g_hsvMaskTex);
            SAFE_RELEASE(g_hsvMaskSRV);
            
            return;
        }
    }

    static cv::Mat rgbaMask; 
    try {
        cv::cvtColor(grayMask, rgbaMask, cv::COLOR_GRAY2RGBA);
    } catch (const cv::Exception& e) {
        
        return;
    }

    if (rgbaMask.empty() || rgbaMask.cols <= 0 || rgbaMask.rows <= 0) {
        
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_hsvMaskTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (SUCCEEDED(hr_map))
    {
        for (int y = 0; y < hsvTexH; ++y)
            memcpy((uint8_t*)ms.pData + ms.RowPitch * y, rgbaMask.ptr(y), hsvTexW * 4); 
        g_pd3dDeviceContext->Unmap(g_hsvMaskTex, 0);
    } else {
        
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
    
    // Get latest captured frame - cache it to avoid multiple downloads
    static cv::Mat frameCopy;
    static bool frameValid = false;
    
    extern cv::cuda::GpuMat latestFrameGpu;
    try {
        if (!latestFrameGpu.empty()) {
            latestFrameGpu.download(frameCopy);
            frameValid = true;
        } else {
            frameValid = false;
        }
    } catch (const cv::Exception&) { 
        frameCopy.release(); 
        frameValid = false;
    }
    
    if (!frameValid || frameCopy.empty()) {
        ImGui::TextUnformatted("Debug frame unavailable.");
        return;
    }
    
    uploadDebugFrame(frameCopy);
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

    
    {
        if (ctx.detector) {
            std::lock_guard<std::mutex> det_lock(ctx.detector->detectionMutex);
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
                        if (det.box.width <= 0 || det.box.height <= 0) {
                            continue;
                        }
                        
                        ImVec2 p1(image_pos.x + det.box.x * debug_scale,
                                  image_pos.y + det.box.y * debug_scale);
                        ImVec2 p2(image_pos.x + (det.box.x + det.box.width) * debug_scale,
                                  image_pos.y + (det.box.y + det.box.height) * debug_scale);

                        ImU32 color = IM_COL32(255, 0, 0, 255); 

                        draw_list->AddRect(p1, p2, color, 1.0f, 0, 1.5f); 

                        std::string className = "Unknown";
                        for(const auto& cs : ctx.config.class_settings) { 
                            if (cs.id == det.classId) {
                                className = cs.name;
                                break;
                            }
                        }
                        std::string label = className + " (" + std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
                        draw_list->AddText(ImVec2(p1.x, p1.y - 16), IM_COL32(255, 255, 0, 255), label.c_str());
                    }
                }
            }
        }
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

    // Declare these variables at the top of the function scope
    bool has_target_for_overlay = false;
    Detection target_for_overlay = {};
    
    // Acquire lock and populate variables once per frame
    {
        std::lock_guard<std::mutex> lock(ctx.overlay_target_mutex);
        has_target_for_overlay = ctx.overlay_has_target.load();
        if (has_target_for_overlay) {
            target_for_overlay = ctx.overlay_target_info;
        }
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
        
        ctx.config.saveConfig(); 

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

            
            if (g_hsvMaskSRV) {
                g_hsvMaskSRV->Release();
                g_hsvMaskSRV = nullptr;
            }
            if (g_hsvMaskTex) {
                g_hsvMaskTex->Release();
                g_hsvMaskTex = nullptr;
            }
            hsvTexW = 0;
            hsvTexH = 0;
        }
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Toggles the live debug frame preview below."); }

    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    
    
    if (ImGui::Checkbox("Enable FPS Display", &ctx.config.show_fps)) { ctx.config.saveConfig(); } 

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
        ctx.config.saveConfig();
    }

    ImGui::Spacing();

    if (ctx.config.show_window) 
    {
        // Get latest captured frame
        cv::Mat frameCopy;
        extern cv::cuda::GpuMat latestFrameGpu;
        try {
            if (!latestFrameGpu.empty()) {
                latestFrameGpu.download(frameCopy);
            }
        } catch (const cv::Exception&) { 
            frameCopy.release(); 
        }

        if (frameCopy.empty()) {
            ImGui::TextUnformatted("Debug frame unavailable.");
            return;
        }
        uploadDebugFrame(frameCopy);
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

        
        {
            if (ctx.detector) {
                std::lock_guard<std::mutex> det_lock(ctx.detector->detectionMutex);
                

                if (ctx.detector->m_finalDetectionsCountHost > 0 && ctx.detector->m_finalDetectionsGpu.get() != nullptr)
                {
                    std::vector<Detection> host_detections(ctx.detector->m_finalDetectionsCountHost);
                    cudaError_t err = cudaMemcpy(host_detections.data(), ctx.detector->m_finalDetectionsGpu.get(),
                                                 ctx.detector->m_finalDetectionsCountHost * sizeof(Detection),
                                                 cudaMemcpyDeviceToHost);

                    if (err == cudaSuccess)
                    {
                        for (size_t i = 0; i < host_detections.size(); ++i)
                        {
                            const auto& det = host_detections[i];
                            
                            // Skip invalid detections
                            if (det.box.width <= 0 || det.box.height <= 0) {
                                continue;
                            }
                            
                            ImVec2 p1(image_pos.x + det.box.x * debug_scale,
                                      image_pos.y + det.box.y * debug_scale);
                            ImVec2 p2(image_pos.x + (det.box.x + det.box.width) * debug_scale,
                                      image_pos.y + (det.box.y + det.box.height) * debug_scale);

                            // Check if this is the best target (using overlay's synchronized data)
                            bool is_best_target = false;
                            if (has_target_for_overlay && 
                                det.box.x == target_for_overlay.box.x &&
                                det.box.y == target_for_overlay.box.y &&
                                det.box.width == target_for_overlay.box.width &&
                                det.box.height == target_for_overlay.box.height) {
                                is_best_target = true;
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

                            std::string className = "Unknown";
                            for(const auto& cs : ctx.config.class_settings) { 
                                if (cs.id == det.classId) {
                                    className = cs.name;
                                    break;
                                }
                            }
                            std::string label = className + " (" + std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
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
        }

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
        if (has_target_for_overlay && 
            target_for_overlay.box.width > 0 && 
            target_for_overlay.box.height > 0) {
            auto& target = target_for_overlay;
            float target_center_x = image_pos.x + (target.box.x + target.box.width / 2.0f) * debug_scale;
            
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
            
            float target_center_y = image_pos.y + (target.box.y + target.box.height * y_offset) * debug_scale;
            
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

    
    ImGui::SeparatorText("HSV Filter Debug");
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable HSV Filter (in Config)", &ctx.config.enable_hsv_filter)) {
        ctx.config.saveConfig(); 
        if (ctx.detector) ctx.detector->m_ignore_flags_need_update = true; 
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Toggles the HSV filtering logic (config.enable_hsv_filter)."); }

    if (ctx.config.enable_hsv_filter) {
        ImGui::Text("Min HSV Pixels: %d", ctx.config.min_hsv_pixels);
        ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
        ImGui::Text(ctx.config.remove_hsv_matches ? "Mode: Remove if HSV matches" : "Mode: Keep if HSV matches");

        ImGui::Text("Lower H:%3d S:%3d V:%3d", ctx.config.hsv_lower_h, ctx.config.hsv_lower_s, ctx.config.hsv_lower_v);
        ImGui::Text("Upper H:%3d S:%3d V:%3d", ctx.config.hsv_upper_h, ctx.config.hsv_upper_s, ctx.config.hsv_upper_v);
        ImGui::Spacing();

        static bool show_hsv_mask_preview = false;
        ImGui::Checkbox("Show HSV Mask Preview", &show_hsv_mask_preview);

        if (show_hsv_mask_preview) {
            
            cv::cuda::GpuMat hsvMaskGpu;
            if (ctx.detector) {
                try {
                    hsvMaskGpu = ctx.detector->getHsvMaskGpu();
                } catch (const std::exception& e) {
                    // Handle any exceptions from getHsvMaskGpu
                    hsvMaskGpu = cv::cuda::GpuMat();
                }
            } 
            if (!hsvMaskGpu.empty()) {
                static cv::Mat hsvMaskCpu; 
                try {
                    hsvMaskGpu.download(hsvMaskCpu); 
                } catch (const cv::Exception& e) {
                    ImGui::Text("Error downloading HSV Mask: %s", e.what());
                    hsvMaskCpu.release(); 
                }

                if (!hsvMaskCpu.empty()) {
                    uploadHsvMaskTexture(hsvMaskCpu); 
                    if (g_hsvMaskSRV && hsvTexW > 0 && hsvTexH > 0) {
                        ImGui::SliderFloat("HSV Mask Scale", &hsv_mask_preview_scale, 0.1f, 2.0f, "%.1fx");
                        ImVec2 mask_img_size(hsvTexW * hsv_mask_preview_scale, hsvTexH * hsv_mask_preview_scale);
                        ImGui::Image(g_hsvMaskSRV, mask_img_size);
                    } else {
                        ImGui::Text("HSV Mask Texture not available for display.");
                    }
                } else {
                    ImGui::Text("HSV Mask (CPU) is empty or download failed.");
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

    for (size_t i = 0; i < ctx.config.screenshot_button.size(); )
    {
        std::string& current_key_name = ctx.config.screenshot_button[i];

        int current_index = -1;
        for (size_t k = 0; k < key_names.size(); ++k)
        {
            if (key_names[k] == current_key_name)
            {
                current_index = static_cast<int>(k);
                break;
            }
        }

        if (current_index == -1)
        {
            current_index = 0;
        }

        std::string combo_label = "Screenshot Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            ctx.config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_screenshot" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (ctx.config.screenshot_button.size() <= 1)
            {
                ctx.config.screenshot_button[0] = std::string("None");
                ctx.config.saveConfig();
                continue;
            }
            else
            {
                ctx.config.screenshot_button.erase(ctx.config.screenshot_button.begin() + i);
                ctx.config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    ImGui::Spacing();
    if (ImGui::InputInt("Screenshot Delay (ms)", &ctx.config.screenshot_delay, 50, 500)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Delay in milliseconds after pressing the button before taking the screenshot."); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Miscellaneous");
    ImGui::Spacing();

    if (ImGui::Checkbox("Always On Top", &ctx.config.always_on_top)) { ctx.config.saveConfig(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Keeps the overlay window always on top of other windows."); }
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    if (ImGui::Checkbox("Verbose Console Output", &ctx.config.verbose)) { ctx.config.saveConfig(); }
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
