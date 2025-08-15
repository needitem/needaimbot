#include "../imgui/imgui.h"
#include "../AppContext.h"
#include "ui_helpers.h"
#include <iomanip>
#include <sstream>

#define SAVE_PROFILE() ctx.config.saveProfile(ctx.config.active_profile_name)

static void draw_tracking_toggle()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("GPU Tracking System");
    
    // Main toggle with status indicator
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 8));
    bool tracking_enabled = ctx.config.enable_tracking;
    
    // Status indicator color
    ImVec4 status_color = tracking_enabled ? 
        ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : 
        ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
    
    ImGui::PushStyleColor(ImGuiCol_Text, status_color);
    ImGui::Text(tracking_enabled ? "● ACTIVE" : "● DISABLED");
    ImGui::PopStyleColor();
    
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - 150);
    
    if (ImGui::Checkbox("##EnableTracking", &ctx.config.enable_tracking)) {
        std::cout << "[UI] Tracking toggled to: " << (ctx.config.enable_tracking ? "ON" : "OFF") << std::endl;
        SAVE_PROFILE();
    }
    
    ImGui::PopStyleVar();
    
    if (tracking_enabled) {
        UIHelpers::BeautifulText("GPU-accelerated multi-target tracking with Kalman filtering", UIHelpers::GetAccentColor(0.8f));
        
        // Show info about GPU tracking
        ImGui::TextUnformatted("Tracking is processed entirely on GPU for maximum performance.");
        ImGui::TextUnformatted("All tracking parameters are optimized automatically.");
    } else {
        UIHelpers::BeautifulText("Tracking disabled - Direct detection only", UIHelpers::GetAccentColor(0.5f));
    }
    
    UIHelpers::EndCard();
}

static void draw_kalman_settings()
{
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.config.enable_tracking) {
        return;  // Don't show Kalman settings if tracking is disabled
    }
    
    UIHelpers::BeginCard("Kalman Filter Settings");
    
    // Enable Kalman filter
    if (ImGui::Checkbox("Enable Kalman Filter", &ctx.config.enable_kalman_filter)) {
        SAVE_PROFILE();
    }
    
    if (ctx.config.enable_kalman_filter) {
        UIHelpers::CompactSpacer();
        
        // Process noise
        if (UIHelpers::EnhancedSliderFloat("Process Noise", &ctx.config.kalman_process_noise, 
                                          0.1f, 10.0f, "%.1f",
                                          "Higher values = more responsive to sudden movements")) {
            ctx.config.kalman_process_noise = std::max(0.1f, std::min(10.0f, ctx.config.kalman_process_noise));
            SAVE_PROFILE();
        }
        
        // Measurement noise
        if (UIHelpers::EnhancedSliderFloat("Measurement Noise", &ctx.config.kalman_measurement_noise,
                                          0.1f, 10.0f, "%.1f",
                                          "Higher values = more smoothing, less jitter")) {
            ctx.config.kalman_measurement_noise = std::max(0.1f, std::min(10.0f, ctx.config.kalman_measurement_noise));
            SAVE_PROFILE();
        }
        
        // Time delta
        if (UIHelpers::EnhancedSliderFloat("Time Delta (ms)", &ctx.config.kalman_dt,
                                          10.0f, 100.0f, "%.0f ms",
                                          "Frame time interval for prediction")) {
            ctx.config.kalman_dt = std::max(10.0f, std::min(100.0f, ctx.config.kalman_dt)) / 1000.0f;
            SAVE_PROFILE();
        }
        
        UIHelpers::CompactSpacer();
        UIHelpers::BeautifulText("GPU-accelerated Kalman filtering for smooth tracking", UIHelpers::GetAccentColor(0.6f));
    }
    
    UIHelpers::EndCard();
}

void draw_tracker()
{
    draw_tracking_toggle();
    draw_kalman_settings();
}