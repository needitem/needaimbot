#include "../imgui/imgui.h"
#include "../AppContext.h"
#include "ui_helpers.h"
#include "../detector/detector.h"
#include "../tracking/SORTTracker.h"
#include <iomanip>
#include <sstream>

#define SAVE_PROFILE() ctx.config.saveProfile(ctx.config.active_profile_name)

static void draw_tracking_toggle()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Tracking System");
    
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
        SAVE_PROFILE();
        
        // Reset tracker when toggling
        if (ctx.detector && ctx.detector->m_sortTracker) {
            ctx.detector->m_sortTracker->reset();
        }
    }
    
    ImGui::PopStyleVar();
    
    if (tracking_enabled) {
        UIHelpers::BeautifulText("Multi-target tracking with motion prediction", UIHelpers::GetAccentColor(0.8f));
        
        // Show tracked targets count if available
        if (ctx.detector) {
            std::lock_guard<std::mutex> lock(ctx.detector->m_trackingMutex);
            if (!ctx.detector->m_trackedObjects.empty()) {
                ImGui::Text("Tracked Targets: %zu", ctx.detector->m_trackedObjects.size());
            }
        }
    } else {
        UIHelpers::BeautifulText("Enable to track targets across frames", ImVec4(0.7f, 0.7f, 0.7f, 0.6f));
    }
    
    UIHelpers::EndCard();
}

static void draw_sort_tracker_settings()
{
    auto& ctx = AppContext::getInstance();
    
    if (!ctx.config.enable_tracking) {
        UIHelpers::BeginCard("SORT Tracker Settings");
        UIHelpers::BeautifulText("Enable tracking to configure SORT parameters", ImVec4(0.7f, 0.7f, 0.7f, 0.5f));
        UIHelpers::EndCard();
        return;
    }
    
    UIHelpers::BeginCard("SORT Tracker Configuration");
    
    UIHelpers::SettingsSubHeader("Tracking Parameters");
    UIHelpers::BeautifulText("Controls how targets are tracked between frames", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::CompactSpacer();
    
    // Max Age setting (increased range for better persistence)
    if (ImGui::SliderInt("Max Age##SORT", &ctx.config.tracker_max_age, 1, 60)) {
        SAVE_PROFILE();
        
        // Update tracker in real-time
        if (ctx.detector && ctx.detector->m_sortTracker) {
            ctx.detector->m_sortTracker->setMaxAge(ctx.config.tracker_max_age);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maximum frames to maintain a track without detection\nHigher = keeps tracking longer when target is occluded\nRecommended: 30+ for stable tracking");
    }
    
    // Min Hits setting
    if (ImGui::SliderInt("Min Hits##SORT", &ctx.config.tracker_min_hits, 1, 10)) {
        SAVE_PROFILE();
        
        if (ctx.detector && ctx.detector->m_sortTracker) {
            ctx.detector->m_sortTracker->setMinHits(ctx.config.tracker_min_hits);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Minimum consecutive detections before confirming track\nHigher = more stable but slower to acquire");
    }
    
    // IOU Threshold setting (lower default for better matching)
    if (UIHelpers::EnhancedSliderFloat("IOU Threshold", &ctx.config.tracker_iou_threshold, 0.05f, 0.5f, "%.2f",
                                       "Minimum overlap for matching detections to tracks\nLower = more lenient matching\nRecommended: 0.1 or lower")) {
        SAVE_PROFILE();
        
        if (ctx.detector && ctx.detector->m_sortTracker) {
            ctx.detector->m_sortTracker->setIOUThreshold(ctx.config.tracker_iou_threshold);
        }
    }
    
    UIHelpers::Spacer(10.0f);
    UIHelpers::SettingsSubHeader("Quick Presets");
    
    if (ImGui::Button("Conservative##SORT", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3, 0))) {
        ctx.config.tracker_max_age = 10;
        ctx.config.tracker_min_hits = 3;
        ctx.config.tracker_iou_threshold = 0.3f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Strict tracking - fewer false positives");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Balanced##SORT", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2, 0))) {
        ctx.config.tracker_max_age = 20;
        ctx.config.tracker_min_hits = 2;
        ctx.config.tracker_iou_threshold = 0.15f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Balanced settings - good for most cases");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Aggressive##SORT", ImVec2(-1, 0))) {
        ctx.config.tracker_max_age = 30;
        ctx.config.tracker_min_hits = 1;
        ctx.config.tracker_iou_threshold = 0.1f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Most lenient - best ID persistence");
    }
    
    UIHelpers::EndCard();
}

static void draw_kalman_filter_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Kalman Filter");
    
    // Kalman filter toggle
    bool kalman_enabled = ctx.config.enable_kalman_filter;
    
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 8));
    
    // Status indicator
    ImVec4 kalman_color = kalman_enabled ? 
        ImVec4(0.2f, 0.6f, 0.9f, 1.0f) : 
        ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    
    ImGui::PushStyleColor(ImGuiCol_Text, kalman_color);
    ImGui::Text(kalman_enabled ? "● FILTERING" : "● DISABLED");
    ImGui::PopStyleColor();
    
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - 150);
    
    if (ImGui::Checkbox("##EnableKalman", &ctx.config.enable_kalman_filter)) {
        SAVE_PROFILE();
    }
    
    ImGui::PopStyleVar();
    
    if (!kalman_enabled) {
        UIHelpers::BeautifulText("Enable to smooth and predict target movement", ImVec4(0.7f, 0.7f, 0.7f, 0.6f));
        UIHelpers::EndCard();
        return;
    }
    
    UIHelpers::CompactSpacer();
    UIHelpers::BeautifulText("GPU-accelerated Kalman filter with CUDA graphs", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(10.0f);
    
    // CUDA Graph optimization toggle
    UIHelpers::SettingsSubHeader("GPU Optimization");
    if (ImGui::Checkbox("Use CUDA Graph", &ctx.config.kalman_use_cuda_graph)) {
        SAVE_PROFILE();
        if (ctx.config.kalman_use_cuda_graph) {
            UIHelpers::BeautifulText("Optimized execution with CUDA graphs", ImVec4(0.2f, 0.8f, 0.2f, 0.7f));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("CUDA graphs optimize kernel launches for better performance\nDisable if experiencing issues");
    }
    
    UIHelpers::Spacer(10.0f);
    UIHelpers::SettingsSubHeader("Filter Parameters");
    
    // Process Noise
    if (UIHelpers::EnhancedSliderFloat("Process Noise", 
                                       &ctx.config.kalman_process_noise, 
                                       0.1f, 10.0f, "%.1f",
                                       "Model uncertainty - how much the target can accelerate\nHigher = adapts faster to sudden movements")) {
        SAVE_PROFILE();
    }
    
    // Measurement Noise
    if (UIHelpers::EnhancedSliderFloat("Measurement Noise", 
                                       &ctx.config.kalman_measurement_noise, 
                                       1.0f, 50.0f, "%.1f",
                                       "Target uncertainty - how noisy the detections are\nHigher = more smoothing, less responsive")) {
        SAVE_PROFILE();
    }
    
    UIHelpers::Spacer(10.0f);
    UIHelpers::SettingsSubHeader("Timing");
    
    // Time Delta
    float dt_ms = ctx.config.kalman_dt * 1000.0f;
    if (UIHelpers::EnhancedSliderFloat("Time Delta", 
                                       &dt_ms, 
                                       16.0f, 100.0f, "%.1f ms",
                                       "Time between frames for prediction\n16ms = 60 FPS, 33ms = 30 FPS")) {
        ctx.config.kalman_dt = dt_ms / 1000.0f;
        SAVE_PROFILE();
    }
    
    UIHelpers::Spacer(10.0f);
    UIHelpers::SettingsSubHeader("Track Management");
    
    // Min Hits
    if (ImGui::SliderInt("Min Hits##Kalman", &ctx.config.kalman_min_hits, 1, 10)) {
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Minimum detections before track is confirmed");
    }
    
    // Max Age
    if (ImGui::SliderInt("Max Age##Kalman", &ctx.config.kalman_max_age, 1, 20)) {
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maximum frames without detection before track is removed");
    }
    
    // Visual representation of timing
    if (dt_ms > 0) {
        float fps = 1000.0f / dt_ms;
        float frames_ahead = 1.0f;  // Predicting 1 frame ahead
        
        ImGui::Text("Running at %.1f FPS (%.1f ms frame time)", fps, dt_ms);
        
        // Draw frame visualization
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size(ImGui::GetContentRegionAvail().x, 30);
        
        // Draw frame boxes
        float box_width = 20.0f;
        float spacing = 5.0f;
        int max_frames_to_show = 5;
        
        for (int i = 0; i <= std::min((int)frames_ahead, max_frames_to_show); i++) {
            float x = canvas_pos.x + i * (box_width + spacing);
            float y = canvas_pos.y;
            
            ImU32 color = (i <= frames_ahead) ? 
                IM_COL32(100, 200, 100, 255) : 
                IM_COL32(100, 100, 100, 255);
            
            if (i == 0) {
                // Current frame
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + box_width, y + 20), IM_COL32(200, 200, 100, 255));
                draw_list->AddText(ImVec2(x + 5, y + 2), IM_COL32(0, 0, 0, 255), "0");
            } else if (i <= frames_ahead) {
                // Predicted frames
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + box_width, y + 20), color);
                draw_list->AddText(ImVec2(x + 5, y + 2), IM_COL32(255, 255, 255, 255), std::to_string(i).c_str());
            }
        }
        
        ImGui::Dummy(canvas_size);
    }
    
    UIHelpers::Spacer(10.0f);
    UIHelpers::SettingsSubHeader("Kalman Presets");
    
    // Preset buttons in a grid
    if (ImGui::Button("No Prediction##Kalman", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3, 0))) {
        ctx.config.kalman_lookahead_time = 0.0f;
        ctx.config.kalman_process_noise = 1.0f;
        ctx.config.kalman_measurement_noise = 10.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Pure smoothing without prediction\nGood for stable targets");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Low Latency##Kalman", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2, 0))) {
        ctx.config.kalman_lookahead_time = 0.008f;  // Half frame at 60fps
        ctx.config.kalman_process_noise = 1.5f;
        ctx.config.kalman_measurement_noise = 12.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Predict 0.5 frames ahead\nCompensates for input lag");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Standard##Kalman", ImVec2(-1, 0))) {
        ctx.config.kalman_lookahead_time = 0.016f;  // 1 frame at 60fps
        ctx.config.kalman_process_noise = 2.0f;
        ctx.config.kalman_measurement_noise = 15.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Predict 1 frame ahead\nBalanced prediction");
    }
    
    // Second row of presets
    if (ImGui::Button("Aggressive##Kalman", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3, 0))) {
        ctx.config.kalman_lookahead_time = 0.033f;  // 2 frames at 60fps
        ctx.config.kalman_process_noise = 3.0f;
        ctx.config.kalman_measurement_noise = 20.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Predict 2 frames ahead\nFor fast-moving targets");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Ultra##Kalman", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2, 0))) {
        ctx.config.kalman_lookahead_time = 0.050f;  // 3 frames at 60fps
        ctx.config.kalman_process_noise = 4.0f;
        ctx.config.kalman_measurement_noise = 25.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Predict 3 frames ahead\nMaximum prediction");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Custom##Kalman", ImVec2(-1, 0))) {
        // Reset to allow custom configuration
        ctx.config.kalman_lookahead_time = 0.025f;
        ctx.config.kalman_process_noise = 2.5f;
        ctx.config.kalman_measurement_noise = 18.0f;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Configure your own settings");
    }
    
    UIHelpers::EndCard();
}

static void draw_tracking_statistics()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Tracking Statistics");
    
    if (!ctx.config.enable_tracking || !ctx.detector) {
        UIHelpers::BeautifulText("Enable tracking to view statistics", ImVec4(0.7f, 0.7f, 0.7f, 0.5f));
        UIHelpers::EndCard();
        return;
    }
    
    // Get tracking stats
    size_t num_tracks = 0;
    if (ctx.detector) {
        std::lock_guard<std::mutex> lock(ctx.detector->m_trackingMutex);
        num_tracks = ctx.detector->m_trackedObjects.size();
    }
    
    ImGui::Columns(2, "tracking_stats", false);
    
    UIHelpers::SettingsSubHeader("Current Status");
    ImGui::Text("Active Tracks:");
    ImGui::Text("Target FPS:");
    ImGui::Text("Tracking Latency:");
    
    ImGui::NextColumn();
    
    ImGui::Text("%zu", num_tracks);
    
    // Calculate FPS from inference time
    float inference_time = ctx.g_current_inference_time_ms.load();
    float fps = inference_time > 0 ? 1000.0f / inference_time : 0.0f;
    ImGui::Text("%.1f", fps);
    
    ImGui::Text("%.2f ms", inference_time * 0.1f); // Rough estimate of tracking overhead
    
    ImGui::Columns(1);
    
    if (num_tracks > 0 && num_tracks < 100) {  // Sanity check
        UIHelpers::Spacer(10.0f);
        UIHelpers::SettingsSubHeader("Track Details");
        
        // Show details of tracked objects
        try {
            std::lock_guard<std::mutex> lock(ctx.detector->m_trackingMutex);
            
            // Double check size after lock
            size_t safe_count = ctx.detector->m_trackedObjects.size();
            if (safe_count > 100) safe_count = 100;  // Limit display
            
            for (size_t i = 0; i < safe_count && i < 10; i++) {  // Show max 10 tracks
                const auto& track = ctx.detector->m_trackedObjects[i];
                
                // Comprehensive safety checks
                if (track.width <= 0 || track.height <= 0 ||
                    track.width > 10000 || track.height > 10000 ||
                    track.x < -10000 || track.x > 10000 ||
                    track.y < -10000 || track.y > 10000 ||
                    track.id < 0 || track.id > 100000) {
                    continue;
                }
                
                // Use safer formatting
                char track_info[256];
                snprintf(track_info, sizeof(track_info), 
                         "ID %d: Pos(%.0f, %.0f) Vel(%.1f, %.1f)",
                         track.id,
                         track.center_x, track.center_y,
                         track.velocity_x, track.velocity_y);
                
                ImGui::Text("%s", track_info);
                
                // Show confidence bar with bounds check
                float safe_confidence = track.confidence;
                if (safe_confidence < 0.0f) safe_confidence = 0.0f;
                if (safe_confidence > 1.0f) safe_confidence = 1.0f;
                
                ImGui::SameLine();
                ImGui::ProgressBar(safe_confidence, ImVec2(100, 0), "");
            }
        } catch (const std::exception& e) {
            ImGui::Text("Error displaying tracks: %s", e.what());
        } catch (...) {
            ImGui::Text("Unknown error displaying tracks");
        }
    }
    
    UIHelpers::EndCard();
}

void draw_tracker()
{
    draw_tracking_toggle();
    UIHelpers::Spacer();
    
    draw_sort_tracker_settings();
    UIHelpers::Spacer();
    
    draw_kalman_filter_settings();
    UIHelpers::Spacer();
    
    draw_tracking_statistics();
}