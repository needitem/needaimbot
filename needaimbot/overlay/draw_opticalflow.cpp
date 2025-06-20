#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "config.h"       
#include "needaimbot.h"   
#include "overlay.h"      





void draw_optical_flow() 
{
    ImGui::SeparatorText("Optical Flow Settings");
    ImGui::Spacing();

    bool settings_changed_locally = false;

    if (ImGui::Checkbox("Enable Optical Flow Processing", &config.enable_optical_flow)) {
        settings_changed_locally = true;
    }
    if (ImGui::IsItemHovered()) { 
        ImGui::SetTooltip("Enables/disables optical flow calculation thread.\\nChanges may require restart or re-initialization of the optical flow system."); 
    }

    if (config.enable_optical_flow) {
        ImGui::Spacing();
        ImGui::Indent();

        if (ImGui::Checkbox("Draw Optical Flow Vectors", &config.draw_optical_flow)) {
            settings_changed_locally = true;
        }
        if (ImGui::IsItemHovered()) { 
            ImGui::SetTooltip("Show calculated optical flow vectors on the debug preview window (if implemented)."); 
        }
        
        
        if (ImGui::SliderInt("Flow Vis Steps", &config.draw_optical_flow_steps, 1, 32)) {
            settings_changed_locally = true;
        }
        if (ImGui::IsItemHovered()) { 
            ImGui::SetTooltip("Density of drawn flow vectors. Higher is less dense."); 
        }

        if (ImGui::SliderFloat("Flow Mag Threshold", &config.optical_flow_magnitudeThreshold, 0.0f, 10.0f, "%.2f")) {
            settings_changed_locally = true;
        }
        if (ImGui::IsItemHovered()) { 
            ImGui::SetTooltip("Minimum magnitude for a flow vector to be drawn/considered."); 
        }
        
        ImGui::Spacing();
        ImGui::Text("OF Algorithm Parameters:");

        if (ImGui::SliderFloat("Static Frame Threshold", &config.staticFrameThreshold, 0.01f, 10.0f, "%.2f")) { 
            settings_changed_locally = true; 
        }
        if (ImGui::IsItemHovered()) { 
            ImGui::SetTooltip("Threshold for mean pixel difference to consider a frame static (may pause OF)."); 
        }

        if (ImGui::SliderFloat("FOV X (degrees)", &config.fovX, 30.0f, 150.0f, "%.1f")) { 
            settings_changed_locally = true; 
        }
        if (ImGui::IsItemHovered()) { 
            ImGui::SetTooltip("Horizontal Field of View. Used in some OF calculations."); 
        }

        if (ImGui::SliderFloat("FOV Y (degrees)", &config.fovY, 30.0f, 120.0f, "%.1f")) { 
            settings_changed_locally = true; 
        }
        if (ImGui::IsItemHovered()) { 
            ImGui::SetTooltip("Vertical Field of View. Used in some OF calculations."); 
        }
        
        
        
        
        
        

        ImGui::Unindent();
    }

    if (settings_changed_locally) {
        
        g_config_optical_flow_changed.store(true); 
        
        config.saveConfig(); 
    }
}

