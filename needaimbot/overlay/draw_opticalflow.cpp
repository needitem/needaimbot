#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "config.h"       // For config object
#include "needaimbot.h"   // For config_optical_flow_changed (assuming it's declared here or in a file included by it)
#include "overlay.h"      // Added to bring in declaration of g_config_optical_flow_changed
// Potentially include "overlay.h" if config_optical_flow_changed is there

// Ensure this global flag is declared (e.g., in needaimbot.h or overlay.h)
// extern std::atomic<bool> config_optical_flow_changed; 

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
        
        // Example parameters, adjust as per your OpticalFlow class needs
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
        
        // Add other optical flow specific parameters from your config here...
        // Example:
        // if (ImGui::SliderFloat("Parameter X", &config.optical_flow_param_x, 0.0f, 1.0f)) {
        //     settings_changed_locally = true;
        // }

        ImGui::Unindent();
    }

    if (settings_changed_locally) {
        // if (config_optical_flow_changed_global_atomic_flag != nullptr) { // No longer using a pointer
        g_config_optical_flow_changed.store(true); // Set the global atomic flag using the correct name
        // }
        config.saveConfig(); // Save configuration changes
    }
}
