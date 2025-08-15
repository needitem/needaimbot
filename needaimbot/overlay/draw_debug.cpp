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
// #include "../capture/capture.h" - removed, using GPU capture now 
#include <cuda_runtime.h> 
#include "../cuda/detection/postProcess.h" 
#include "../cuda/capture/color_conversion.h"
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


ID3D11Texture2D* g_debugTex = nullptr;
ID3D11ShaderResourceView* g_debugSRV = nullptr;
int texW = 0, texH = 0;
float debug_scale = 1.0f; 


static ID3D11Texture2D* g_colorMaskTex = nullptr;
static ID3D11ShaderResourceView* g_colorMaskSRV = nullptr;
static int colorTexW = 0, colorTexH = 0;
static float color_mask_preview_scale = 0.5f; 



static int g_crosshairH = 0, g_crosshairS = 0, g_crosshairV = 0;
static bool g_crosshairHsvValid = false;



// GPU-accelerated version with direct GPU memory handling
void uploadDebugFrame(const SimpleCudaMat& bgrGpu)
{
    static int uploadCount = 0;
    uploadCount++;
    
    
    
    // Comprehensive safety checks
    if (bgrGpu.empty() || !g_pd3dDevice || !g_pd3dDeviceContext) {
        if (uploadCount <= 3) {
            std::cout << "[uploadDebugFrame] Early return - empty=" << bgrGpu.empty() 
                      << " device=" << (g_pd3dDevice != nullptr)
                      << " context=" << (g_pd3dDeviceContext != nullptr) << std::endl;
        }
        return;
    }
    
    // Validate dimensions
    if (bgrGpu.cols() <= 0 || bgrGpu.rows() <= 0 || 
        bgrGpu.cols() > 10000 || bgrGpu.rows() > 10000) {
        return;
    }
    
    // Check for valid GPU data pointer
    if (bgrGpu.data() == nullptr) {
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

    // GPU-based color conversion with error handling
    static SimpleCudaMat rgbaGpu;
    
    try {
        rgbaGpu.create(bgrGpu.rows(), bgrGpu.cols(), 4);
    } catch (...) {
        std::cerr << "[Debug] Failed to create RGBA GPU buffer" << std::endl;
        return;
    }
    
    // Ensure CUDA operations are synchronized
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "[Debug] CUDA sync failed before conversion: " << cudaGetErrorString(sync_err) << std::endl;
        return;
    }
    
    cudaError_t err;
    if (bgrGpu.channels() == 4) {
        // BGRA to RGBA conversion
        err = cuda_bgra2rgba(bgrGpu.data(), rgbaGpu.data(),
                             bgrGpu.cols(), bgrGpu.rows(),
                             static_cast<int>(bgrGpu.step()), 
                             static_cast<int>(rgbaGpu.step()));
    } else if (bgrGpu.channels() == 3) {
        // BGR to RGBA conversion
        err = cuda_bgr2rgba(bgrGpu.data(), rgbaGpu.data(),
                            bgrGpu.cols(), bgrGpu.rows(),
                            static_cast<int>(bgrGpu.step()), 
                            static_cast<int>(rgbaGpu.step()));
    } else {
        std::cerr << "[Debug] Unsupported channel count: " << bgrGpu.channels() << std::endl;
        return;
    }
    
    if (err != cudaSuccess) {
        std::cerr << "[Debug] CUDA color conversion failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Sync after conversion
    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "[Debug] CUDA sync failed after conversion: " << cudaGetErrorString(sync_err) << std::endl;
        return;
    }
    
    // Download converted data from GPU with error handling
    static SimpleMat rgba;
    try {
        rgba.create(rgbaGpu.rows(), rgbaGpu.cols(), 4);
        
        // Validate CPU buffer
        if (rgba.data() == nullptr) {
            std::cerr << "[Debug] Failed to allocate CPU buffer" << std::endl;
            return;
        }
        
        rgbaGpu.download(rgba.data(), rgba.step());
        
        // Final sync to ensure download is complete
        cudaError_t download_sync = cudaDeviceSynchronize();
        if (download_sync != cudaSuccess) {
            std::cerr << "[Debug] CUDA sync failed after download: " << cudaGetErrorString(download_sync) << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Debug] Exception during download: " << e.what() << std::endl;
        return;
    } catch (...) {
        std::cerr << "[Debug] Unknown exception during download" << std::endl;
        return;
    }

    if (rgba.empty() || rgba.cols() <= 0 || rgba.rows() <= 0 || rgba.data() == nullptr) { 
        return;
    }

    D3D11_MAPPED_SUBRESOURCE ms = {};
    HRESULT hr_map = g_pd3dDeviceContext->Map(g_debugTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    if (SUCCEEDED(hr_map) && ms.pData != nullptr)
    {
        try {
            // Copy row by row, handling different row pitches
            size_t copy_width = std::min<size_t>(texW * 4, ms.RowPitch);  // Ensure we don't overrun
            uint8_t* src_data = rgba.data();
            uint8_t* dst_data = static_cast<uint8_t*>(ms.pData);
            
            if (src_data && dst_data) {
                for (int y = 0; y < texH; ++y) {
                    // Bounds check
                    if (y * rgba.step() + copy_width <= rgba.rows() * rgba.step()) {
                        memcpy(dst_data + ms.RowPitch * y, 
                               src_data + y * rgba.step(), 
                               copy_width);
                    }
                }
            }
        } catch (...) {
            std::cerr << "[Debug] Exception during texture copy" << std::endl;
        }
        g_pd3dDeviceContext->Unmap(g_debugTex, 0);
    } else if (FAILED(hr_map)) {
        std::cerr << "[Debug] Failed to map texture: HRESULT=" << std::hex << hr_map << std::dec << std::endl;
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

void drawDetections(ImDrawList* draw_list, ImVec2 image_pos, float debug_scale) {
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.detector) return;
    
    // Validate parameters
    if (!draw_list || debug_scale <= 0) return;
    
    std::lock_guard<std::mutex> det_lock(ctx.detector->detectionMutex);
    
    // Validate detection count
    if (ctx.detector->m_finalTargetsCountHost <= 0 || 
        ctx.detector->m_finalTargetsCountHost > 1000 ||  // Sanity check
        ctx.detector->m_finalTargetsGpu.get() == nullptr) {
        return;
    }
    
    std::vector<Target> host_detections(ctx.detector->m_finalTargetsCountHost);
    cudaError_t err = cudaMemcpy(host_detections.data(), ctx.detector->m_finalTargetsGpu.get(),
                                 ctx.detector->m_finalTargetsCountHost * sizeof(Target),
                                 cudaMemcpyDeviceToHost);

    if (err == cudaSuccess)
    {
        for (const auto& det : host_detections)
        {
                // Skip invalid detections - check multiple conditions
                if (det.width <= 0 || det.height <= 0 || 
                    det.confidence <= 0.0f || det.classId < 0 ||
                    det.x < 0 || det.y < 0) {
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
                
                // Draw predicted position if Kalman filter is enabled and velocity is available
                if (ctx.config.enable_kalman_filter && ctx.config.enable_tracking) {
                    // Check if velocity data is available (stored in velocity fields)
                    if (det.velocity_x != 0 || det.velocity_y != 0) {
                        // Calculate predicted position based on lookahead frames
                        float lookahead_frames = ctx.config.kalman_lookahead_time * 60.0f;
                        float pred_center_x = (det.x + det.width * 0.5f) + det.velocity_x * lookahead_frames;
                        float pred_center_y = (det.y + det.height * 0.5f) + det.velocity_y * lookahead_frames;
                        
                        // Draw predicted box in cyan (lighter color)
                        ImVec2 pred_p1(image_pos.x + (pred_center_x - det.width * 0.5f) * debug_scale,
                                      image_pos.y + (pred_center_y - det.height * 0.5f) * debug_scale);
                        ImVec2 pred_p2(image_pos.x + (pred_center_x + det.width * 0.5f) * debug_scale,
                                      image_pos.y + (pred_center_y + det.height * 0.5f) * debug_scale);
                        
                        // Draw predicted box with dashed line (approximated with multiple small lines)
                        ImU32 pred_color = IM_COL32(0, 255, 255, 180);  // Cyan with transparency
                        float dash_length = 5.0f;
                        float gap_length = 3.0f;
                        
                        // Top edge
                        float x = pred_p1.x;
                        while (x < pred_p2.x) {
                            float end_x = std::min(x + dash_length, pred_p2.x);
                            draw_list->AddLine(ImVec2(x, pred_p1.y), ImVec2(end_x, pred_p1.y), pred_color, 1.5f);
                            x += dash_length + gap_length;
                        }
                        
                        // Bottom edge
                        x = pred_p1.x;
                        while (x < pred_p2.x) {
                            float end_x = std::min(x + dash_length, pred_p2.x);
                            draw_list->AddLine(ImVec2(x, pred_p2.y), ImVec2(end_x, pred_p2.y), pred_color, 1.5f);
                            x += dash_length + gap_length;
                        }
                        
                        // Left edge
                        float y = pred_p1.y;
                        while (y < pred_p2.y) {
                            float end_y = std::min(y + dash_length, pred_p2.y);
                            draw_list->AddLine(ImVec2(pred_p1.x, y), ImVec2(pred_p1.x, end_y), pred_color, 1.5f);
                            y += dash_length + gap_length;
                        }
                        
                        // Right edge
                        y = pred_p1.y;
                        while (y < pred_p2.y) {
                            float end_y = std::min(y + dash_length, pred_p2.y);
                            draw_list->AddLine(ImVec2(pred_p2.x, y), ImVec2(pred_p2.x, end_y), pred_color, 1.5f);
                            y += dash_length + gap_length;
                        }
                        
                        // Draw velocity vector arrow
                        ImVec2 current_center(image_pos.x + (det.x + det.width * 0.5f) * debug_scale,
                                            image_pos.y + (det.y + det.height * 0.5f) * debug_scale);
                        ImVec2 predicted_center(image_pos.x + pred_center_x * debug_scale,
                                              image_pos.y + pred_center_y * debug_scale);
                        
                        // Draw arrow line
                        draw_list->AddLine(current_center, predicted_center, IM_COL32(255, 0, 255, 255), 2.0f);
                        
                        // Draw arrowhead
                        float arrow_size = 8.0f;
                        ImVec2 dir = ImVec2(predicted_center.x - current_center.x, 
                                          predicted_center.y - current_center.y);
                        float len = sqrtf(dir.x * dir.x + dir.y * dir.y);
                        if (len > 0) {
                            dir.x /= len;
                            dir.y /= len;
                            
                            ImVec2 perp(-dir.y, dir.x);
                            ImVec2 arrow_p1(predicted_center.x - dir.x * arrow_size - perp.x * arrow_size * 0.5f,
                                          predicted_center.y - dir.y * arrow_size - perp.y * arrow_size * 0.5f);
                            ImVec2 arrow_p2(predicted_center.x - dir.x * arrow_size + perp.x * arrow_size * 0.5f,
                                          predicted_center.y - dir.y * arrow_size + perp.y * arrow_size * 0.5f);
                            
                            draw_list->AddTriangleFilled(predicted_center, arrow_p1, arrow_p2, 
                                                        IM_COL32(255, 0, 255, 255));
                        }
                    }
                }

                // Use track ID from detection if available
                int track_id = det.id;  // Detection already has ID field from tracking
                
                // If tracking is disabled, ID will be -1
                if (!ctx.config.enable_tracking) {
                    track_id = -1;
                }

                // Build label safely
                std::string className = CommonHelpers::getClassNameById(det.classId);
                if (className.empty()) className = "Unknown";
                
                char conf_str[32];
                snprintf(conf_str, sizeof(conf_str), "%.0f%%", det.confidence * 100.0f);
                
                std::string label = className + " (" + conf_str + ")";
                
                if (track_id >= 0 && track_id < 10000) {  // Sanity check
                    label = "ID:" + std::to_string(track_id) + " " + label;
                }
                
                // Add velocity info if available
                if (ctx.config.enable_kalman_filter && (det.velocity_x != 0 || det.velocity_y != 0)) {
                    char vel_str[64];
                    snprintf(vel_str, sizeof(vel_str), " [V:%.1f,%.1f]", det.velocity_x, det.velocity_y);
                    label += vel_str;
                }
                
                if (is_best_target) {
                    label = "[TARGET] " + label;
                }
                
                // Ensure label isn't too long
                if (label.length() > 100) {
                    label = label.substr(0, 97) + "...";
                }
                
                ImU32 text_color = is_best_target ? IM_COL32(0, 255, 0, 255) : 
                                  (det.confidence >= ctx.config.confidence_threshold ? IM_COL32(255, 255, 0, 255) : IM_COL32(255, 0, 0, 255));
                
                // Check text position is valid
                if (p1.x >= 0 && p1.y >= 16) {
                    draw_list->AddText(ImVec2(p1.x, p1.y - 16), text_color, label.c_str());
                }
            }
    } else {
        // Log CUDA error but don't crash
        static int error_count = 0;
        if (error_count < 5) {  // Limit error logging
            std::cerr << "[drawDetections] CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
            error_count++;
        }
    }
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

    ImGui::SeparatorText("Preview Notice"); 
    ImGui::Spacing();
    
    ImGui::Text("The preview window has been moved to the Offset tab.");
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Please switch to the Offset tab to see the live preview.");
    
    ImGui::Spacing();
    
    if (ImGui::Checkbox("Enable FPS Display", &ctx.config.show_fps)) { SAVE_PROFILE(); }

    ImGui::Spacing();
    ImGui::Separator(); 
    ImGui::Spacing();

    
    ImGui::SeparatorText("RGB Color Filter Debug");
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable RGB Color Filter (in Config)", &ctx.config.enable_color_filter)) {
        SAVE_PROFILE(); 
        if (ctx.detector) ctx.detector->m_allow_flags_need_update = true; 
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
