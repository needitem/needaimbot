#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
// #include <cstdio> // Removed for printf

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"
#include "include/other_tools.h"
#include <vector>
#include <string>
#include <d3d11.h> // For D3D11_TEXTURE2D_DESC etc.
#include "capture.h" // For latestFrameCpu, frameMutex
#include <cuda_runtime.h> // For cudaMemcpy
#include "detector/postProcess.h" // For Detection struct
#include "capture/optical_flow.h" // For OpticalFlow class (assuming g_opticalFlow)

// Added for ring buffer access
#include <array>
#include <atomic>
#include "opencv2/core/cuda.hpp" // For cv::cuda::GpuMat
// cv::Mat is likely included via other headers like needaimbot.h or opencv.hpp transitively

// Assuming FRAME_BUFFER_COUNT is defined in needaimbot.h (already included)
// Extern declarations for ring buffers and indices from capture.cpp
extern std::array<cv::cuda::GpuMat, FRAME_BUFFER_COUNT> captureGpuBuffer;
extern std::array<cv::Mat, FRAME_BUFFER_COUNT> captureCpuBuffer;
extern std::atomic<int> captureGpuWriteIdx;
extern std::atomic<int> captureCpuWriteIdx;
// extern Config config; // Already declared and used
// extern std::mutex frameMutex; // Already used, implicitly extern

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)     \
    do {                    \
        if ((p) != nullptr) { \
            (p)->Release();   \
            (p) = nullptr;    \
        }                     \
    } while (0)
#endif

// Globals for debug frame rendering
static ID3D11Texture2D* g_debugTex = nullptr;
static ID3D11ShaderResourceView* g_debugSRV = nullptr;
static int texW = 0, texH = 0;
static float debug_scale = 1.0f; // Default scale

// Assume these are declared extern if not already available through includes
extern Detector detector; // Access to the global detector object
extern OpticalFlow g_opticalFlow; // Access to the global optical flow object
extern Config config; // Already used

// Function to upload cv::Mat to a D3D11 texture
static void uploadDebugFrame(const cv::Mat& bgr)
{
    // All printf statements removed by user request
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
        // OutputDebugStringA could be used for minimal logging if desired in future
        // OutputDebugStringA(("[uploadDebugFrame] cv::Exception during cvtColor: " + std::string(e.what()) + "\\\\n").c_str());
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
        // OutputDebugStringA could be used for minimal logging if desired in future
        // OutputDebugStringA(("[uploadDebugFrame] Error: Failed to map texture! HRESULT: " + std::to_string(hr_map) + "\\\\n").c_str());
    }
}

// Function to draw the debug frame with overlays
void draw_debug_frame()
{
    // All printf statements removed by user request
    cv::Mat frameCopy;
    bool got_lock = false;
    // --- Re-enabling frame copy --- //
    {
        // printf("[DebugTab] Attempting to try_lock frameMutex...\\n"); // Optional: for debugging
        std::unique_lock<std::mutex> lock(frameMutex, std::try_to_lock); // Freeze was occurring here
        if (lock.owns_lock()) {
            // printf("[DebugTab] frameMutex try_lock succeeded.\\n"); // Optional
            got_lock = true;

            cv::Mat tempFrameFromRingBuffer;
            if (config.capture_use_cuda) {
                int read_idx_gpu = captureGpuWriteIdx.load(std::memory_order_acquire);
                if (read_idx_gpu >= 0 && read_idx_gpu < FRAME_BUFFER_COUNT && !captureGpuBuffer[read_idx_gpu].empty()) {
                    try {
                        captureGpuBuffer[read_idx_gpu].download(tempFrameFromRingBuffer);
                    } catch (const cv::Exception& e) {
                        // OutputDebugStringA(("[draw_debug_frame] cv::Exception during GpuMat download: " + std::string(e.what()) + "\\\\n").c_str());
                    }
                }
            } else {
                int read_idx_cpu = captureCpuWriteIdx.load(std::memory_order_acquire);
                 if (read_idx_cpu >= 0 && read_idx_cpu < FRAME_BUFFER_COUNT && !captureCpuBuffer[read_idx_cpu].empty()) {
                     try {
                        captureCpuBuffer[read_idx_cpu].copyTo(tempFrameFromRingBuffer);
                     } catch (const cv::Exception& e) {
                        // OutputDebugStringA(("[draw_debug_frame] cv::Exception during Mat copyTo: " + std::string(e.what()) + "\\\\n").c_str());
                     }
                }
            }

            if (!tempFrameFromRingBuffer.empty() && tempFrameFromRingBuffer.cols > 0 && tempFrameFromRingBuffer.rows > 0) {
                try {
                    tempFrameFromRingBuffer.copyTo(frameCopy);
                } catch (const cv::Exception& e) {
                    // OutputDebugStringA(("[draw_debug_frame] Error: cv::Exception during tempFrameFromRingBuffer.copyTo(frameCopy): " + std::string(e.what()) + "\\\\n").c_str());
                }
            } else {
                // frameCopy remains empty if no valid frame from ring buffer
                // OutputDebugStringA("[draw_debug_frame] Info: tempFrameFromRingBuffer is empty or invalid.\\\\n");
            }
        } else {
            // printf("[DebugTab] frameMutex try_lock FAILED.\\n"); // Optional
            // If lock fails, got_lock remains false. frameCopy remains empty.
            // We can choose to display the last rendered frame or a message.
        }
    } // Mutex (if acquired) is released here
    // --- End re-enabled frame copy --- //

    // --- Re-enabling frame display --- //
    if (!got_lock) {
        // If we couldn't get the lock, we can either show the last frame or a message.
        // For now, if there's an existing texture, show it. Otherwise, show a message.
        if (g_debugSRV && texW > 0 && texH > 0) {
            ImGui::SliderFloat("Debug scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
            // --- ImGui::Image call is currently commented out --- // Now uncommented
            ImVec2 image_size(texW * debug_scale, texH * debug_scale);
            ImGui::Image(g_debugSRV, image_size);
           // Since ImGui::Image is commented out, perhaps just indicate status:
           // ImGui::TextUnformatted("Debug frame: Displaying last available frame (mutex was busy).");
        } else {
            ImGui::TextUnformatted("Debug frame unavailable (mutex busy or no data).");
        }
        // We might not want to return entirely, to allow other debug elements to draw.
        // Depending on desired behavior, could skip just the image rendering section.
        // For now, let's allow rest of the debug UI to render.
    } else if (frameCopy.empty() || frameCopy.cols <= 0 || frameCopy.rows <= 0) {
        // printf("[draw_debug_frame] Error: frameCopy is empty or invalid before uploadDebugFrame.\\n");
        if (g_debugSRV && texW > 0 && texH > 0) {
            // ImGui::TextUnformatted("Debug frame: No new data, displaying last available frame.");
            ImGui::SliderFloat("Debug scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
            // --- ImGui::Image call is currently commented out --- // Now uncommented
            ImVec2 image_size(texW * debug_scale, texH * debug_scale);
            ImGui::Image(g_debugSRV, image_size);
        } else {
            ImGui::TextUnformatted("Debug frame: No data to display.");
        }
        // Allow rest of the debug UI to render.
    } else {
        // Got lock and valid frameCopy
        uploadDebugFrame(frameCopy);
        ImGui::SliderFloat("Debug scale", &debug_scale, 0.1f, 3.0f, "%.1fx");
        // --- ImGui::Image call is currently commented out --- // Now uncommented
        ImVec2 image_size(texW * debug_scale, texH * debug_scale);
        if (g_debugSRV) { // Check g_debugSRV before use
            ImGui::Image(g_debugSRV, image_size);
        } else {
            ImGui::TextUnformatted("Debug frame: Data processed, but texture unavailable for display.");
        }
    }

    if (!g_debugSRV) {
        ImGui::TextUnformatted("Overlays skipped: Debug texture unavailable."); // Added for clarity
        return; // If no texture, can't get image_pos or draw on it
    }

    ImVec2 image_pos = ImGui::GetItemRectMin(); // Get position of the drawn image
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // --- Draw All Detections ---
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

                    ImU32 color = IM_COL32(255, 0, 0, 255); // Default red for detections

                    // Example: Change color based on classId or confidence if needed
                    // if (det.classId == 0) color = IM_COL32(0, 255, 0, 255); // Green for class 0

                    draw_list->AddRect(p1, p2, color, 1.0f, 0, 1.5f); // Rounded corners, thickness 1.5

                    std::string className = "Unknown";
                    for(const auto& cs : config.class_settings) { // Assuming config.class_settings is available
                        if (cs.id == det.classId) {
                            className = cs.name;
                            break;
                        }
                    }
                    std::string label = className + " (" + std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
                    draw_list->AddText(ImVec2(p1.x, p1.y - 16), IM_COL32(255, 255, 0, 255), label.c_str());
                }
            } else {
                // Optionally log cudaMemcpy error
                // ImGui::Text("Error copying detections to host.");
            }
        }
    }

    // --- Draw Optical Flow ---
    if (config.draw_optical_flow && g_opticalFlow.isFlowValidAtomic.load() && !g_opticalFlow.flow.empty())
    {
        cv::Mat flowCpu;
        g_opticalFlow.flow.download(flowCpu); // Assuming g_opticalFlow.flow is a GpuMat

        if (!flowCpu.empty() && flowCpu.type() == CV_32FC2) // Ensure it's downloaded and of correct type
        {
            // cv::Mat flowFloat; // flowCpu is already float if type is CV_32FC2
            // flowCpu.convertTo(flowFloat, CV_32FC2); // No need if already CV_32FC2, adjust if type is different

            cv::Mat flowChannels[2];
            cv::split(flowCpu, flowChannels); // Use flowCpu directly

            cv::Mat magnitude;
            cv::magnitude(flowChannels[0], flowChannels[1], magnitude);

            // Ensure texW and texH are valid (from uploadDebugFrame)
            if (texW <=0 || texH <=0) return;


            float scaleX = static_cast<float>(texW) / flowCpu.cols;
            float scaleY = static_cast<float>(texH) / flowCpu.rows;
            float visualScaleX = debug_scale * scaleX;
            float visualScaleY = debug_scale * scaleY;

            int step = config.draw_optical_flow_steps > 0 ? config.draw_optical_flow_steps : 16; // Ensure step is positive
            double magThreshold = config.optical_flow_magnitudeThreshold;

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
                        // Scale the flow vector itself by debug_scale only, as its components are already in pixel units of the original flow field
                        ImVec2 p2 = ImVec2(p1.x + fxy.x * debug_scale * scaleX,  // Apply scaling to match display
                                          p1.y + fxy.y * debug_scale * scaleY); // Apply scaling to match display

                        draw_list->AddLine(p1, p2, IM_COL32(0, 223, 255, 255), 1.0f); // Cyan lines
                        draw_list->AddCircleFilled(p1, 1.5f * debug_scale, IM_COL32(0, 223, 255, 255)); // Small circle at start
                    }
                }
            }
            draw_list->PopClipRect();
        }
    }
}

// --- Optical Flow Settings UI (Moved to its own function) ---
/* Entire draw_optical_flow() function removed as per user request to refactor optical flow elsewhere
void draw_optical_flow() // Implementation for the function declared in draw_settings.h
{
    ImGui::SeparatorText("Optical Flow Settings");
    // ... (rest of the function was here)
    config_optical_flow_changed = true; // Set the global atomic flag
}
*/

// Helper function to convert vector<string> to vector<const char*>
inline std::vector<const char*> getProfileCstrs(const std::vector<std::string>& profiles) {
    std::vector<const char*> cstrs;
    cstrs.reserve(profiles.size());
    for(const auto& s : profiles)
        cstrs.push_back(s.c_str());
    return cstrs;
}

void draw_debug()
{
    // All printf statements removed by user request
    // FPS Display (Moved from potential separate window)
    if (config.show_fps) {
        ImGui::Text("Capture FPS: %.1f", g_current_capture_fps.load());
        ImGui::Text("Inference Time: %.2f ms", g_current_inference_time_ms.load());
        // Add other relevant FPS/timing info here if needed
        ImGui::Separator();
        ImGui::Spacing();
    }

    ImGui::SeparatorText("Debug Preview & Overlay"); // Changed SeparatorText for clarity
    ImGui::Spacing();

    bool prev_show_window_state = config.show_window; // Store state before checkbox
    if (ImGui::Checkbox("Show Preview Window", &config.show_window)) 
    {
        // show_window_changed.store(true); // This global flag might be used elsewhere, if so, keep it.
        config.saveConfig(); 

        if (prev_show_window_state == true && config.show_window == false) {
            // Preview window was just turned OFF. Release its D3D resources.
            // These are static globals, so releasing them here means they'll be null
            // until uploadDebugFrame recreates them if the window is shown again.
            if (g_debugSRV) {
                g_debugSRV->Release();
                g_debugSRV = nullptr;
            }
            if (g_debugTex) {
                g_debugTex->Release();
                g_debugTex = nullptr;
            }
            // Reset dimensions as well, so they are re-evaluated by uploadDebugFrame
            texW = 0; 
            texH = 0;
        }
    }
    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Toggles the live debug frame preview below."); }

    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    // The old "Show FPS" checkbox is removed as FPS is now shown above if config.show_fps is true.
    // We can add a checkbox to toggle config.show_fps itself if needed.
    if (ImGui::Checkbox("Enable FPS Display", &config.show_fps)) { config.saveConfig(); } // Toggle for the text FPS

    ImGui::Spacing();

    if (config.show_window) // Call draw_debug_frame if the window is to be shown
    {
        draw_debug_frame(); // This function draws the actual preview image and overlays
    }

    ImGui::Spacing();
    ImGui::Separator(); // Separator after the preview window or its controls
    ImGui::Spacing();

    // Screenshot Settings remain here
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