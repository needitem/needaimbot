#include "imgui/imgui.h"
#include "ui_helpers.h"
#include "../utils/AutoTuner.h"
#include "../AppContext.h"

void draw_tuning() {
    auto& tuner = AutoTuner::getInstance();
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Hyperparameter Auto-Tuning");
    
    UIHelpers::BeautifulText("Automatically optimize aimbot parameters for best performance", 
                           UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    if (!tuner.isTuning()) {
        UIHelpers::BeautifulText("Auto-tuning will test different parameter combinations to find optimal settings.", 
                               ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        UIHelpers::CompactSpacer();
        
        ImGui::BulletText("Make sure you're in a training environment");
        ImGui::BulletText("Tuning will take approximately 5-10 minutes");
        ImGui::BulletText("Keep engaging targets during the process");
        
        UIHelpers::CompactSpacer();
        
        if (UIHelpers::BeautifulButton("Start Auto-Tuning", ImVec2(-1, 0))) {
            tuner.initializeParameters();
            tuner.startAutoTuning();
        }
    } else {
        // Show tuning progress
        float progress = tuner.getTuningProgress();
        UIHelpers::BeautifulProgressBar(progress, ImVec2(-1, 0), 
                                      (std::to_string(static_cast<int>(progress * 100)) + "%").c_str());
        
        UIHelpers::CompactSpacer();
        
        UIHelpers::StatusIndicator("Auto-Tuning Active", true, "Testing parameter combinations...");
        
        UIHelpers::CompactSpacer();
        
        if (UIHelpers::BeautifulButton("Stop Tuning", ImVec2(-1, 0))) {
            tuner.stopAutoTuning();
        }
    }
    
    UIHelpers::EndCard();
    
    // Current parameters display
    UIHelpers::BeginCard("Current Parameters");
    
    ImGui::Columns(2, "param_columns", false);
    
    ImGui::Text("PID X-Axis");
    ImGui::NextColumn();
    ImGui::Text("Kp: %.3f, Ki: %.3f, Kd: %.3f", 
                ctx.config.kp_x, ctx.config.ki_x, ctx.config.kd_x);
    ImGui::NextColumn();
    
    ImGui::Text("PID Y-Axis");
    ImGui::NextColumn();
    ImGui::Text("Kp: %.3f, Ki: %.3f, Kd: %.3f", 
                ctx.config.kp_y, ctx.config.ki_y, ctx.config.kd_y);
    ImGui::NextColumn();
    
    
    ImGui::Columns(1);
    
    UIHelpers::EndCard();
    
    // Tips
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("Auto-Tuning Tips", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Use a consistent training scenario");
    ImGui::BulletText("Engage targets at various distances");
    ImGui::BulletText("Include both stationary and moving targets");
    ImGui::BulletText("Results are saved to tuning_results.txt");
    
    UIHelpers::EndInfoPanel();
}