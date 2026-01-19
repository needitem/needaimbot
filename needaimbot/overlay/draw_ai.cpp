#include "../core/windows_headers.h"
#include <vector>
#include <string>
#include <algorithm> 
#include <iterator> 
#include <filesystem>

#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "AppContext.h"
#include "../core/constants.h"
#include "ui_helpers.h"
#include "common_helpers.h"
#include "draw_settings.h"
#include "../cuda/unified_graph_pipeline.h"

static void draw_model_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Model & Engine Settings");
    
    // Cache model list - only refresh when needed
    static std::vector<std::string> availableModels;
    static std::vector<const char*> modelsItems;
    static bool models_initialized = false;
    static std::filesystem::file_time_type last_model_check;
    
    // Only refresh if not initialized or model folder changed
    auto current_time = std::filesystem::file_time_type::clock::now();
    bool should_refresh = !models_initialized || ctx.model_changed;
    
    // Check for model folder changes every 5 seconds
    if (!should_refresh && models_initialized) {
        auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_model_check).count();
        if (time_diff > 5) {
            should_refresh = true;
            last_model_check = current_time;
        }
    }
    
    if (should_refresh) {
        availableModels = getAvailableModels();
        modelsItems.clear();
        modelsItems.reserve(availableModels.size());
        for (const auto& modelName : availableModels) {
            modelsItems.push_back(modelName.c_str());
        }
        models_initialized = true;
        ctx.model_changed = false;
    }
    
    if (availableModels.empty())
    {
        UIHelpers::BeautifulText("No models available in the 'models' folder.", UIHelpers::GetWarningColor());
    }
    else
    {
        int currentModelIndex = 0;
        auto it = std::find(availableModels.begin(), availableModels.end(), ctx.config.profile().ai_model);

        if (it != availableModels.end())
        {
            currentModelIndex = static_cast<int>(std::distance(availableModels.begin(), it));
        }

        if (UIHelpers::EnhancedCombo("AI Model", &currentModelIndex, modelsItems.data(), static_cast<int>(modelsItems.size()), 
                                    "Select the AI model file to use for target detection. Models should be placed in the 'models' folder."))
        {
            if (ctx.config.profile().ai_model != availableModels[currentModelIndex])
            {
                ctx.config.profile().ai_model = availableModels[currentModelIndex];
                SAVE_PROFILE();
                ctx.model_changed = true;
            }
        }
    }
    
    UIHelpers::CompactSpacer();
    
    // Engine conversion info
    UIHelpers::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Engine Conversion:");
    ImGui::TextWrapped("Use: https://github.com/needitem/EngineExport");
    
    UIHelpers::EndCard();
}

static void draw_detection_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Detection Parameters");
    
    // Cache postprocess options - initialize once
    static const std::vector<std::string> postprocessOptions = { "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yolo_nms" };
    static std::vector<const char*> postprocessItems = [](){
        std::vector<const char*> items;
        items.reserve(postprocessOptions.size());
        for (const auto& option : postprocessOptions) {
            items.push_back(option.c_str());
        }
        return items;
    }();

    // Optimize: use std::find instead of manual loop
    auto it = std::find(postprocessOptions.begin(), postprocessOptions.end(), ctx.config.profile().postprocess);
    int currentPostprocessIndex = (it != postprocessOptions.end()) ? 
        static_cast<int>(std::distance(postprocessOptions.begin(), it)) : 0;

    if (UIHelpers::EnhancedCombo("Postprocess Algorithm", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size()),
                                "Select the YOLO postprocessing algorithm that matches your model version."))
    {
        ctx.config.profile().postprocess = postprocessOptions[currentPostprocessIndex];
        SAVE_PROFILE();
        ctx.model_changed = true;
    }

    UIHelpers::Spacer();

    if (UIHelpers::EnhancedSliderFloat("Confidence Threshold", &ctx.config.profile().confidence_threshold, 0.01f, 1.00f, "%.2f",
                                      "Minimum confidence score required for target detection. Higher values = fewer false positives."))
    {
        SAVE_PROFILE();
    }

    // Max detections slider with better styling
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, UIHelpers::GetAccentColor(0.9f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, UIHelpers::GetAccentColor(1.0f));
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("Max Detections", &ctx.config.profile().max_detections, 1, Constants::MAX_DETECTIONS_LIMIT, "%d detections")) {
        SAVE_PROFILE();
    }
    ImGui::PopStyleColor(4);
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Maximum number of targets to detect per frame. Higher values may impact performance.");
        ImGui::EndTooltip();
    }
    
    UIHelpers::Spacer();
    
    // NMS Settings
    if (ImGui::Checkbox("Enable NMS", &ctx.config.profile().enable_nms)) {
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Non-Maximum Suppression removes duplicate/overlapping detections.");
        ImGui::Text("Recommended for YOLO8/9/11/12. YOLO10/yolo_nms already includes NMS.");
        ImGui::EndTooltip();
    }
    
    if (ctx.config.profile().enable_nms) {
        ImGui::Indent();
        if (UIHelpers::EnhancedSliderFloat("IoU Threshold", &ctx.config.profile().nms_iou_threshold, 0.1f, 0.9f, "%.2f",
                                           "IoU threshold for suppression. Lower = more aggressive filtering."))
        {
            SAVE_PROFILE();
        }
        ImGui::Unindent();
    }
    
    UIHelpers::EndCard();
}

static void draw_class_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Target Classes");

    // Head class name input
    static char head_class_name_buffer[128];
    strncpy_s(head_class_name_buffer, sizeof(head_class_name_buffer), ctx.config.profile().head_class_name.c_str(), _TRUNCATE);
    head_class_name_buffer[sizeof(head_class_name_buffer) - 1] = '\0';

    ImGui::Text("Head Class Name");
    ImGui::SameLine();
    UIHelpers::HelpMarker("Class name treated as 'head' for head-specific aim offset");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##head_class", head_class_name_buffer, sizeof(head_class_name_buffer))) {
        // preview
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.profile().head_class_name = head_class_name_buffer;
        SAVE_PROFILE();
    }

    UIHelpers::CompactSpacer();

    // Class table with better styling
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(6, 4));
    if (ImGui::BeginTable("##class_table", 4,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
        ImVec2(0, 150))) {

        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24.0f);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < ctx.config.profile().class_settings.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ClassSetting& setting = ctx.config.profile().class_settings[i];

            ImGui::TableNextRow();

            // ID
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputInt("##id", &setting.id, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
                SAVE_PROFILE();
            }

            // Name
            ImGui::TableNextColumn();
            char name_buf[128];
            strncpy_s(name_buf, sizeof(name_buf), setting.name.c_str(), _TRUNCATE);
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputText("##name", name_buf, sizeof(name_buf))) {
                setting.name = name_buf;
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                SAVE_PROFILE();
            }

            // Allow checkbox
            ImGui::TableNextColumn();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 12);
            if (ImGui::Checkbox("##allow", &setting.allow)) {
                SAVE_PROFILE();
            }

            // Remove button
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.2f, 0.2f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("x##rm", ImVec2(20, 0))) {
                ctx.config.profile().class_settings.erase(ctx.config.profile().class_settings.begin() + i);
                SAVE_PROFILE();
                ImGui::PopStyleColor(2);
                ImGui::PopID();
                i--;
                continue;
            }
            ImGui::PopStyleColor(2);

            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();

    UIHelpers::CompactSpacer();

    // Add new class - compact row
    static int new_class_id = 0;
    static char new_class_name_buf[128] = "";
    static bool new_class_allow = true;

    if (ImGui::BeginTable("##add_class", 4, ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Allow", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Add", ImGuiTableColumnFlags_WidthFixed, 50.0f);

        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputInt("##new_id", &new_class_id, 0, 0);

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##new_name", "Class name...", new_class_name_buf, sizeof(new_class_name_buf));

        ImGui::TableNextColumn();
        ImGui::Checkbox("##new_allow", &new_class_allow);

        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.2f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.3f, 1.0f));
        if (ImGui::Button("+##add", ImVec2(-1, 0))) {
            bool id_exists = false;
            for (const auto& cs : ctx.config.profile().class_settings) {
                if (cs.id == new_class_id) {
                    id_exists = true;
                    break;
                }
            }
            std::string temp_name = new_class_name_buf;
            if (!id_exists && !temp_name.empty()) {
                ctx.config.profile().class_settings.emplace_back(new_class_id, temp_name, new_class_allow);
                SAVE_PROFILE();
                new_class_id = CommonHelpers::getNextClassId();
                new_class_name_buf[0] = '\0';
                new_class_allow = true;
            }
        }
        ImGui::PopStyleColor(2);

        ImGui::EndTable();
    }

    UIHelpers::EndCard();
}

static void draw_advanced_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("GPU Settings");

    if (ImGui::BeginTable("##gpu_settings", 2, ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupColumn("Setting", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.0f);

        // CUDA Device
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("CUDA Device ID");
        UIHelpers::HelpMarker("GPU to use for inference (requires restart)");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputInt("##cuda_device", &ctx.config.global().cuda_device_id, 0, 0)) {
            ctx.config.global().cuda_device_id = std::max(0, ctx.config.global().cuda_device_id);
            SAVE_PROFILE();
        }

        // L2 Cache
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("L2 Cache (MB)");
        UIHelpers::HelpMarker("TensorRT persistent cache.\nRTX 40: 24-72MB\nRTX 30: 4-6MB");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputInt("##l2_cache", &ctx.config.global().persistent_cache_limit_mb, 0, 0)) {
            ctx.config.global().persistent_cache_limit_mb = std::clamp(ctx.config.global().persistent_cache_limit_mb, 1, 128);
            SAVE_PROFILE();
        }

        ImGui::EndTable();
    }

    UIHelpers::CompactSpacer();

    // CUDA Graph toggle
    if (UIHelpers::BeautifulToggle("CUDA Graph Optimization", &ctx.config.global().use_cuda_graph,
                                   "Faster inference, but may not work with all models")) {
        SAVE_PROFILE();
        auto& pipelineManager = gpa::PipelineManager::getInstance();
        if (pipelineManager.getPipeline()) {
            pipelineManager.getPipeline()->setGraphRebuildNeeded();
        }
    }

    UIHelpers::EndCard();
}

void draw_ai()
{
    draw_model_settings();
    UIHelpers::Spacer();
    
    draw_detection_settings();
    UIHelpers::Spacer();
    
    draw_class_settings();
    UIHelpers::Spacer();
    
    draw_advanced_settings();
}
