#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "config.h"       
#include "needaimbot.h"   
#include "overlay.h"      
#include "ui_helpers.h"

void draw_pid_settings() 
{
    UIHelpers::PushStyleColors();
    
    UIHelpers::BeautifulSection("PID Controller Settings");
    
    ImGui::Text("PID parameters control how the aimbot tracks targets.");
    ImGui::Spacing();
    
    // X-axis PID
    UIHelpers::BeautifulSeparator("X-Axis (Horizontal)");
    
    if (UIHelpers::BeautifulSlider("Kp X", (float*)&config.kp_x, 0.1f, 10.0f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Proportional gain for X-axis. Higher values = faster response but may cause oscillation.");
    }
    
    if (UIHelpers::BeautifulSlider("Ki X", (float*)&config.ki_x, 0.0f, 2.0f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Integral gain for X-axis. Helps eliminate steady-state error.");
    }
    
    if (UIHelpers::BeautifulSlider("Kd X", (float*)&config.kd_x, 0.0f, 1.0f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Derivative gain for X-axis. Reduces overshoot and oscillation.");
    }
    
    ImGui::Spacing();
    
    // Y-axis PID
    UIHelpers::BeautifulSeparator("Y-Axis (Vertical)");
    
    if (UIHelpers::BeautifulSlider("Kp Y", (float*)&config.kp_y, 0.1f, 10.0f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Proportional gain for Y-axis. Higher values = faster response but may cause oscillation.");
    }
    
    if (UIHelpers::BeautifulSlider("Ki Y", (float*)&config.ki_y, 0.0f, 2.0f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Integral gain for Y-axis. Helps eliminate steady-state error.");
    }
    
    if (UIHelpers::BeautifulSlider("Kd Y", (float*)&config.kd_y, 0.0f, 1.0f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Derivative gain for Y-axis. Reduces overshoot and oscillation.");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Advanced Settings
    UIHelpers::BeautifulSeparator("Advanced Settings");
    
    if (UIHelpers::BeautifulToggle("Enable Adaptive PID", &config.enable_adaptive_pid, 
                                   "Uses distance-based PID adjustment for better stability at different ranges.")) {
        config.saveConfig();
    }
    
    if (UIHelpers::BeautifulSlider("Derivative Smoothing", &config.pid_derivative_smoothing, 0.0f, 0.8f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Smooths derivative calculation to reduce noise. Higher values = more smoothing but slower response to rapid changes.");
    }
    
    if (UIHelpers::BeautifulSlider("Movement Smoothing", &config.movement_smoothing, 0.0f, 0.6f, "%.3f")) {
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Smooths final mouse movement. Higher values = less jitter but may reduce responsiveness.");
    }
    
    ImGui::Spacing();
    
    // Info section
    UIHelpers::BeautifulSeparator("Information");
    
    ImGui::BeginChild("PIDInfo", ImVec2(0, 80), true, ImGuiWindowFlags_AlwaysUseWindowPadding);
    {
        UIHelpers::BeautifulText("Tuning Tips:", UIHelpers::GetAccentColor());
        ImGui::Text("• Start with Kp, then add Kd to reduce oscillation");
        ImGui::Text("• Use Ki sparingly - too much causes overshoot");
        ImGui::Text("• Enable Adaptive PID for better long-range stability");
        ImGui::Text("• Increase smoothing if you experience jitter");
    }
    ImGui::EndChild();
    
    UIHelpers::PopStyleColors();
}