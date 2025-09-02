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
    
    std::vector<std::string> availableModels = getAvailableModels();
    if (availableModels.empty())
    {
        UIHelpers::BeautifulText("No models available in the 'models' folder.", UIHelpers::GetWarningColor());
    }
    else
    {
        int currentModelIndex = 0;
        auto it = std::find(availableModels.begin(), availableModels.end(), ctx.config.ai_model);

        if (it != availableModels.end())
        {
            currentModelIndex = static_cast<int>(std::distance(availableModels.begin(), it));
        }

        std::vector<const char*> modelsItems;
        modelsItems.reserve(availableModels.size());

        for (const auto& modelName : availableModels)
        {
            modelsItems.push_back(modelName.c_str());
        }

        if (UIHelpers::EnhancedCombo("AI Model", &currentModelIndex, modelsItems.data(), static_cast<int>(modelsItems.size()), 
                                    "Select the AI model file to use for target detection. Models should be placed in the 'models' folder."))
        {
            if (ctx.config.ai_model != availableModels[currentModelIndex])
            {
                ctx.config.ai_model = availableModels[currentModelIndex];
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
    
    std::vector<std::string> postprocessOptions = { "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yolo_nms" };
    std::vector<const char*> postprocessItems;
    for (const auto& option : postprocessOptions)
    {
        postprocessItems.push_back(option.c_str());
    }

    int currentPostprocessIndex = 0;
    for (size_t i = 0; i < postprocessOptions.size(); ++i)
    {
        if (postprocessOptions[i] == ctx.config.postprocess)
        {
            currentPostprocessIndex = static_cast<int>(i);
            break;
        }
    }

    if (UIHelpers::EnhancedCombo("Postprocess Algorithm", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size()),
                                "Select the YOLO postprocessing algorithm that matches your model version."))
    {
        ctx.config.postprocess = postprocessOptions[currentPostprocessIndex];
        SAVE_PROFILE();
        ctx.model_changed = true;
    }
    
    UIHelpers::Spacer();
    
    if (UIHelpers::EnhancedSliderFloat("Confidence Threshold", &ctx.config.confidence_threshold, 0.01f, 1.00f, "%.2f", 
                                      "Minimum confidence score required for target detection. Higher values = fewer false positives."))
    {
        SAVE_PROFILE();
    }
    
    // NMS removed - no longer needed
    
    // Max detections slider with better styling
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, UIHelpers::GetAccentColor(0.9f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, UIHelpers::GetAccentColor(1.0f));
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("Max Detections", &ctx.config.max_detections, 1, Constants::MAX_DETECTIONS_LIMIT, "%d detections")) {
        SAVE_PROFILE();
    }
    ImGui::PopStyleColor(4);
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Maximum number of targets to detect per frame. Higher values may impact performance.");
        ImGui::EndTooltip();
    }
    
    UIHelpers::EndCard();
}

static void draw_class_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Class & Targeting Definitions");
    
    static char head_class_name_buffer[128];
    strncpy_s(head_class_name_buffer, sizeof(head_class_name_buffer), ctx.config.head_class_name.c_str(), _TRUNCATE);
    head_class_name_buffer[sizeof(head_class_name_buffer) - 1] = '\0';
    
    ImGui::PushItemWidth(-1);
    ImGui::InputText("##head_class", head_class_name_buffer, sizeof(head_class_name_buffer));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Head Class Name");
    UIHelpers::InfoTooltip("The name of the class that should be treated as 'Head' for specific aiming logic (e.g., head_y_offset).");
    
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.head_class_name = head_class_name_buffer;
        SAVE_PROFILE();
    }
    
    UIHelpers::CompactSpacer();
    
    if (ImGui::BeginTable("class_settings_table", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthStretch, 0.15f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 0.50f);
        ImGui::TableSetupColumn("Allow", ImGuiTableColumnFlags_WidthStretch, 0.15f);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 0.20f);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < ctx.config.class_settings.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ClassSetting& setting = ctx.config.class_settings[i];

            ImGui::TableNextRow();
            
            ImGui::TableSetColumnIndex(0);
            if (ImGui::InputInt("##ID", &setting.id, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
                SAVE_PROFILE();
            }

            ImGui::TableSetColumnIndex(1);
            char name_buf[128];
            strncpy_s(name_buf, sizeof(name_buf), setting.name.c_str(), _TRUNCATE);
            name_buf[sizeof(name_buf) - 1] = '\0';
            if (ImGui::InputText("##Name", name_buf, sizeof(name_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                setting.name = name_buf;
                SAVE_PROFILE();
            }
            if (ImGui::IsItemDeactivatedAfterEdit() && setting.name != name_buf) { 
                setting.name = name_buf;
                SAVE_PROFILE();
            }

            ImGui::TableSetColumnIndex(2);
            if (ImGui::Checkbox("##Allow", &setting.allow)) {
                SAVE_PROFILE();
                // TODO: Implement flag update for TensorRT integration
            }

            ImGui::TableSetColumnIndex(3);
            if (UIHelpers::BeautifulButton("Remove", ImVec2(-1, 0))) {
                ctx.config.class_settings.erase(ctx.config.class_settings.begin() + i);
                SAVE_PROFILE();
                // TODO: Implement flag update for TensorRT integration
                ImGui::PopID(); 
                i--; 
                continue; 
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulSeparator("Add New Class");
    
    static int new_class_id = 0; 
    static char new_class_name_buf[128] = "";
    static bool new_class_allow = true;

    ImGui::Columns(3, "new_class_columns", false);
    
    ImGui::PushItemWidth(-1);
    ImGui::InputInt("##new_id", &new_class_id);
    ImGui::PopItemWidth();
    ImGui::Text("ID");
    
    ImGui::NextColumn();
    
    ImGui::PushItemWidth(-1);
    ImGui::InputText("##new_name", new_class_name_buf, sizeof(new_class_name_buf));
    ImGui::PopItemWidth();
    ImGui::Text("Name");
    
    ImGui::NextColumn();
    
    ImGui::Checkbox("Allow", &new_class_allow);
    
    ImGui::Columns(1);
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulButton("Suggest Next ID", ImVec2(-1, 0))) {
        new_class_id = CommonHelpers::getNextClassId();
    }
    
    if (UIHelpers::BeautifulButton("Add Class", ImVec2(-1, 0))) {
        bool id_exists = false;
        for (const auto& cs : ctx.config.class_settings) {
            if (cs.id == new_class_id) {
                id_exists = true;
                break;
            }
        }
        std::string temp_name = new_class_name_buf;
        if (!id_exists && !temp_name.empty()) {
            ctx.config.class_settings.emplace_back(new_class_id, temp_name, new_class_allow);
            SAVE_PROFILE();
            // TODO: Implement flag update for TensorRT integration
            
            new_class_id = CommonHelpers::getNextClassId();
            new_class_name_buf[0] = '\0'; 
            new_class_allow = true;
        }
    }
    
    UIHelpers::EndCard();
}

static void draw_advanced_settings()
{
    auto& ctx = AppContext::getInstance();
    
    static bool advanced_open = false;
    UIHelpers::BeginCard(nullptr);
    
    if (ImGui::CollapsingHeader("Advanced Settings", &advanced_open)) {
        UIHelpers::CompactSpacer();
        
        ImGui::PushItemWidth(-1);
        if (ImGui::InputInt("##cuda_device", &ctx.config.cuda_device_id)) {
            if (ctx.config.cuda_device_id < 0) ctx.config.cuda_device_id = 0;
            SAVE_PROFILE();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Text("CUDA Device ID");
        UIHelpers::InfoTooltip("Set the CUDA device ID to use for detection (requires restart).");
        
        UIHelpers::Spacer();
        
        // GPU Performance Settings
        UIHelpers::BeautifulSeparator("GPU Performance Settings");
        
        ImGui::PushItemWidth(-1);
        if (ImGui::InputInt("##persistent_cache", &ctx.config.persistent_cache_limit_mb)) {
            if (ctx.config.persistent_cache_limit_mb < 1) ctx.config.persistent_cache_limit_mb = 1;
            if (ctx.config.persistent_cache_limit_mb > 128) ctx.config.persistent_cache_limit_mb = 128;
            SAVE_PROFILE();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Text("Persistent L2 Cache (MB)");
        UIHelpers::InfoTooltip("TensorRT persistent L2 cache size in MB. RTX 40 series: 24-72MB, RTX 30 series: 4-6MB, RTX 20 series: 4-5MB. Default: 32MB");
        
        ImGui::Spacing();
        
        if (ImGui::Checkbox("Use CUDA Graph", &ctx.config.use_cuda_graph)) {
            SAVE_PROFILE();
            // Graph 모드 변경 시 재빌드 필요
            auto& pipelineManager = needaimbot::PipelineManager::getInstance();
            if (pipelineManager.getPipeline()) {
                pipelineManager.getPipeline()->setGraphRebuildNeeded();
            }
        }
        ImGui::SameLine();
        UIHelpers::InfoTooltip("Enable CUDA Graph optimization for faster execution.\n"
                              "⚠️ Not compatible with all models (disable if inference fails).\n"
                              "✅ Works with: YOLO v8/v9/v10 without NMS\n"
                              "❌ May not work with: Models with dynamic shapes or built-in NMS");
    }
    
    UIHelpers::EndCard();
}

void draw_ai()
{
    auto& ctx = AppContext::getInstance();
    
    draw_model_settings();
    UIHelpers::Spacer();
    
    draw_detection_settings();
    UIHelpers::Spacer();
    
    draw_class_settings();
    UIHelpers::Spacer();
    
    draw_advanced_settings();
}
