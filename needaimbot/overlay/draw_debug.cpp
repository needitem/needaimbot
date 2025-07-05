#include "../AppContext.h"
#include "imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "include/other_tools.h"
#include <vector>
#include <string>
#include <d3d11.h> 
#include "capture.h" 
#include <cuda_runtime.h> 
#include "detector/postProcess.h" 
#include "capture/optical_flow.h" 


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
    // Simplified frame acquisition using atomic index
    cv::Mat frameCopy;
    if (ctx.config.capture_use_cuda) {
        int idx = captureGpuWriteIdx.load(std::memory_order_acquire);
        if (idx >= 0 && idx < FRAME_BUFFER_COUNT) {
            try { captureGpuBuffer[idx].download(frameCopy); } catch (const cv::Exception&) { frameCopy.release(); }
        }
    } else {
        int idx = captureCpuWriteIdx.load(std::memory_order_acquire);
        if (idx >= 0 && idx < FRAME_BUFFER_COUNT) {
            try { captureCpuBuffer[idx].copyTo(frameCopy); } catch (const cv::Exception&) { frameCopy.release(); }
        }
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
        std::lock_guard<std::mutex> det_lock(ctx.detector.detectionMutex);
        if (ctx.detector.m_finalDetectionsCountHost > 0 && ctx.detector.m_finalDetectionsGpu.get() != nullptr)
        {
            std::vector<Detection> host_detections(ctx.detector.m_finalDetectionsCountHost);
            cudaError_t err = cudaMemcpy(host_detections.data(), ctx.detector.m_finalDetectionsGpu.get(),
                                         ctx.detector.m_finalDetectionsCountHost * sizeof(Detection),
                                         cudaMemcpyDeviceToHost);

            if (err == cudaSuccess)
            {
                for (const auto& det : host_detections)
                {
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
            } else {
                
                
            }
        }
    }

    
    if (ctx.config.draw_optical_flow && ctx.opticalFlow.isFlowValidAtomic.load() && !ctx.opticalFlow.flow.empty())
    {
        cv::Mat flowCpu;
        ctx.opticalFlow.flow.download(flowCpu); 

        if (!flowCpu.empty() && flowCpu.type() == CV_32FC2) 
        {
            
            

            cv::Mat flowChannels[2];
            cv::split(flowCpu, flowChannels); 

            cv::Mat magnitude;
            cv::magnitude(flowChannels[0], flowChannels[1], magnitude);

            
            if (texW <=0 || texH <=0) return;


            float scaleX = static_cast<float>(texW) / flowCpu.cols;
            float scaleY = static_cast<float>(texH) / flowCpu.rows;
            float visualScaleX = debug_scale * scaleX;
            float visualScaleY = debug_scale * scaleY;

            int step = ctx.config.draw_optical_flow_steps > 0 ? ctx.config.draw_optical_flow_steps : 16; 
            double magThreshold = ctx.config.optical_flow_magnitudeThreshold;

            draw_list->PushClipRect(image_pos,
                                    ImVec2(image_pos.x + texW * debug_scale,
                                           image_pos.y + texH * debug_scale),
                                    true);

            for (int y = 0; y < flowCpu.rows; y += step)
            {
                for (int x = 0; x < flowCpu.cols; x += step)
                {
                    float mag = magnitude.at<float>(y, x);
                    if (mag > magThreshold)
                    {
                        const cv::Point2f& fxy = flowCpu.at<cv::Point2f>(y, x);

                        ImVec2 p1(image_pos.x + x * visualScaleX,
                                  image_pos.y + y * visualScaleY);
                        
                        ImVec2 p2 = ImVec2(p1.x + fxy.x * debug_scale * scaleX,  
                                          p1.y + fxy.y * debug_scale * scaleY); 

                        draw_list->AddLine(p1, p2, IM_COL32(0, 223, 255, 255), 1.0f); 
                        draw_list->AddCircleFilled(p1, 1.5f * debug_scale, IM_COL32(0, 223, 255, 255)); 
                    }
                }
            }
            draw_list->PopClipRect();
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
    
    
    if (config.show_fps) {
        ImGui::Text("Capture FPS: %.1f", g_current_capture_fps.load());
        ImGui::Text("Inference Time: %.2f ms", g_current_inference_time_ms.load());
        
        ImGui::Separator();
        ImGui::Spacing();
    }

    ImGui::SeparatorText("Debug Preview & Overlay"); 
    ImGui::Spacing();

    bool prev_show_window_state = config.show_window; 
    if (ImGui::Checkbox("Show Preview Window", &config.show_window)) 
    {
        
        config.saveConfig(); 

        if (prev_show_window_state == true && config.show_window == false) {
            
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
    
    
    if (ImGui::Checkbox("Enable FPS Display", &config.show_fps)) { config.saveConfig(); } 

    ImGui::Spacing();

    if (config.show_window) 
    {
        // Simplified frame acquisition using atomic index
        cv::Mat frameCopy;
        int idx = captureGpuWriteIdx.load(std::memory_order_acquire);
        if (idx >= 0 && idx < FRAME_BUFFER_COUNT) {
            try { captureGpuBuffer[idx].download(frameCopy); } catch (const cv::Exception&) { frameCopy.release(); }
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
            std::lock_guard<std::mutex> det_lock(detector.detectionMutex);
            if (detector.m_finalDetectionsCountHost > 0 && detector.m_finalDetectionsGpu != nullptr)
            {
                std::vector<Detection> host_detections(detector.m_finalDetectionsCountHost);
                cudaError_t err = cudaMemcpy(host_detections.data(), detector.m_finalDetectionsGpu,
                                             detector.m_finalDetectionsCountHost * sizeof(Detection),
                                             cudaMemcpyDeviceToHost);

                if (err == cudaSuccess)
                {
                    for (const auto& det : host_detections)
                    {
                        ImVec2 p1(image_pos.x + det.box.x * debug_scale,
                                  image_pos.y + det.box.y * debug_scale);
                        ImVec2 p2(image_pos.x + (det.box.x + det.box.width) * debug_scale,
                                  image_pos.y + (det.box.y + det.box.height) * debug_scale);

                        ImU32 color = IM_COL32(255, 0, 0, 255); 

                        
                        

                        draw_list->AddRect(p1, p2, color, 1.0f, 0, 1.5f); 

                        std::string className = "Unknown";
                        for(const auto& cs : config.class_settings) { 
                            if (cs.id == det.classId) {
                                className = cs.name;
                                break;
                            }
                        }
                        std::string label = className + " (" + std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
                        draw_list->AddText(ImVec2(p1.x, p1.y - 16), IM_COL32(255, 255, 0, 255), label.c_str());
                    }
                } else {
                    
                    
                }
            }
        }

        
        if (ctx.config.draw_optical_flow && ctx.opticalFlow.isFlowValidAtomic.load() && !ctx.opticalFlow.flow.empty())
    {
        cv::Mat flowCpu;
        ctx.opticalFlow.flow.download(flowCpu); 

        if (!flowCpu.empty() && flowCpu.type() == CV_32FC2) 
        {
            
            

            cv::Mat flowChannels[2];
            cv::split(flowCpu, flowChannels); 

            cv::Mat magnitude;
            cv::magnitude(flowChannels[0], flowChannels[1], magnitude);

            
            if (texW <=0 || texH <=0) return;


            float scaleX = static_cast<float>(texW) / flowCpu.cols;
            float scaleY = static_cast<float>(texH) / flowCpu.rows;
            float visualScaleX = debug_scale * scaleX;
            float visualScaleY = debug_scale * scaleY;

            int step = ctx.config.draw_optical_flow_steps > 0 ? ctx.config.draw_optical_flow_steps : 16; 
            double magThreshold = ctx.config.optical_flow_magnitudeThreshold;

            draw_list->PushClipRect(image_pos,
                                    ImVec2(image_pos.x + texW * debug_scale,
                                           image_pos.y + texH * debug_scale),
                                    true);

            for (int y = 0; y < flowCpu.rows; y += step)
            {
                for (int x = 0; x < flowCpu.cols; x += step)
                {
                    float mag = magnitude.at<float>(y, x);
                    if (mag > magThreshold)
                    {
                        const cv::Point2f& fxy = flowCpu.at<cv::Point2f>(y, x);

                        ImVec2 p1(image_pos.x + x * visualScaleX,
                                  image_pos.y + y * visualScaleY);
                        
                        ImVec2 p2 = ImVec2(p1.x + fxy.x * debug_scale * scaleX,  
                                          p1.y + fxy.y * debug_scale * scaleY); 

                        draw_list->AddLine(p1, p2, IM_COL32(0, 223, 255, 255), 1.0f); 
                        draw_list->AddCircleFilled(p1, 1.5f * debug_scale, IM_COL32(0, 223, 255, 255)); 
                    }
                }
            }
            draw_list->PopClipRect();
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

    ImGui::Spacing();
    ImGui::Separator(); 
    ImGui::Spacing();

    
    ImGui::SeparatorText("HSV Filter Debug");
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable HSV Filter (in Config)", &config.enable_hsv_filter)) {
        config.saveConfig(); 
        detector.m_ignore_flags_need_update = true; 
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Toggles the HSV filtering logic (config.enable_hsv_filter)."); }

    if (config.enable_hsv_filter) {
        ImGui::Text("Min HSV Pixels: %d", config.min_hsv_pixels);
        ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
        ImGui::Text(config.remove_hsv_matches ? "Mode: Remove if HSV matches" : "Mode: Keep if HSV matches");

        ImGui::Text("Lower H:%3d S:%3d V:%3d", config.hsv_lower_h, config.hsv_lower_s, config.hsv_lower_v);
        ImGui::Text("Upper H:%3d S:%3d V:%3d", config.hsv_upper_h, config.hsv_upper_s, config.hsv_upper_v);
        ImGui::Spacing();

        static bool show_hsv_mask_preview = false;
        ImGui::Checkbox("Show HSV Mask Preview", &show_hsv_mask_preview);

        if (show_hsv_mask_preview) {
            
            cv::cuda::GpuMat hsvMaskGpu = detector.getHsvMaskGpu(); 
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

    for (size_t i = 0; i < config.screenshot_button.size(); )
    {
        std::string& current_key_name = config.screenshot_button[i];

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
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_screenshot" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.screenshot_button.size() <= 1)
            {
                config.screenshot_button[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.screenshot_button.erase(config.screenshot_button.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    ImGui::Spacing();
    if (ImGui::InputInt("Screenshot Delay (ms)", &config.screenshot_delay, 50, 500)) { config.saveConfig(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Delay in milliseconds after pressing the button before taking the screenshot."); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SeparatorText("Miscellaneous");
    ImGui::Spacing();

    if (ImGui::Checkbox("Always On Top", &config.always_on_top)) { config.saveConfig(); }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Keeps the overlay window always on top of other windows."); }
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    if (ImGui::Checkbox("Verbose Console Output", &config.verbose)) { config.saveConfig(); }
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
