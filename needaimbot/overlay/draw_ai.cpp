#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <vector>
#include <string>
#include <algorithm> 
#include <iterator> 
#include <filesystem>

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "AppContext.h"
#include "detector/detector.h"
#include "ui_helpers.h"

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

        UIHelpers::CompactCombo("Model", &currentModelIndex, modelsItems.data(), static_cast<int>(modelsItems.size()));
        if (ImGui::IsItemDeactivatedAfterEdit())
        {
            if (ctx.config.ai_model != availableModels[currentModelIndex])
            {
                ctx.config.ai_model = availableModels[currentModelIndex];
                ctx.config.saveConfig();
                detector_model_changed.store(true);
            }
        }
    }
    
    UIHelpers::CompactSpacer();
    
    const char* resolution_items[] = { "160", "320", "640" };
    int current_resolution_index = 0;
    if (ctx.config.onnx_input_resolution == 160)      current_resolution_index = 0;
    else if (ctx.config.onnx_input_resolution == 320) current_resolution_index = 1;
    else if (ctx.config.onnx_input_resolution == 640) current_resolution_index = 2;
    
    UIHelpers::CompactCombo("Input Resolution", &current_resolution_index, resolution_items, IM_ARRAYSIZE(resolution_items));
    UIHelpers::InfoTooltip("Select the input resolution for the ONNX model (e.g., 640 for 640x640 input).\nChanging this will require the .engine file to be rebuilt if it doesn't match.");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        int selected_resolution = 160; 
        if (current_resolution_index == 0)      selected_resolution = 160;
        else if (current_resolution_index == 1) selected_resolution = 320;
        else if (current_resolution_index == 2) selected_resolution = 640;

        if (ctx.config.onnx_input_resolution != selected_resolution)
        {
            ctx.config.onnx_input_resolution = selected_resolution;
            ctx.config.saveConfig();
            detector_model_changed.store(true); 
        }
    }
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulToggle("Enable FP16", &ctx.config.export_enable_fp16, "Enable FP16 precision for the exported TensorRT engine."))
    {
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    
    if (UIHelpers::BeautifulToggle("Enable FP8", &ctx.config.export_enable_fp8, "Enable FP8 precision for the exported TensorRT engine."))
    {
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulButton("Rebuild Engine", ImVec2(-1, 0)))
    {
        std::filesystem::path modelPath(std::string("models/") + ctx.config.ai_model);
        std::filesystem::path onnxPath = modelPath;
        if (modelPath.extension() == ".engine")
            onnxPath.replace_extension(".onnx");
        std::filesystem::path enginePath = onnxPath;
        enginePath.replace_extension(".engine");
        if (std::filesystem::exists(enginePath))
        {
            std::filesystem::remove(enginePath);
            if (ctx.config.verbose)
                std::cout << "[Overlay] Removed engine: " << enginePath.string() << std::endl;
        }
        ctx.config.ai_model = onnxPath.filename().string();
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    UIHelpers::WrappedTooltip("Force rebuild of the TensorRT engine from ONNX.");
    
    UIHelpers::EndCard();
}

static void draw_detection_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Detection Parameters");
    
    std::vector<std::string> postprocessOptions = { "yolo8", "yolo9", "yolo10", "yolo11", "yolo12" };
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

    UIHelpers::CompactCombo("Postprocess Algorithm", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size()));
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        ctx.config.postprocess = postprocessOptions[currentPostprocessIndex];
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::CompactSlider("Confidence Threshold", &ctx.config.confidence_threshold, 0.01f, 1.00f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    UIHelpers::CompactSlider("NMS Threshold", &ctx.config.nms_threshold, 0.01f, 1.00f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    
    ImGui::PushItemWidth(-1);
    if (ImGui::SliderInt("##max_detections", &ctx.config.max_detections, 1, 100)) {
        ctx.config.saveConfig();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("Max Detections");
    
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
        ctx.config.saveConfig();
    }
    
    UIHelpers::CompactSpacer();
    
    if (ImGui::BeginTable("class_settings_table", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Ignore", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < ctx.config.class_settings.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ClassSetting& setting = ctx.config.class_settings[i];

            ImGui::TableNextRow();
            
            ImGui::TableSetColumnIndex(0);
            if (ImGui::InputInt("##ID", &setting.id, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
                ctx.config.saveConfig();
            }

            ImGui::TableSetColumnIndex(1);
            char name_buf[128];
            strncpy_s(name_buf, sizeof(name_buf), setting.name.c_str(), _TRUNCATE);
            name_buf[sizeof(name_buf) - 1] = '\0';
            if (ImGui::InputText("##Name", name_buf, sizeof(name_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                setting.name = name_buf;
                ctx.config.saveConfig();
            }
            if (ImGui::IsItemDeactivatedAfterEdit() && setting.name != name_buf) { 
                setting.name = name_buf;
                ctx.config.saveConfig();
            }

            ImGui::TableSetColumnIndex(2);
            if (ImGui::Checkbox("##Ignore", &setting.ignore)) {
                ctx.config.saveConfig();
                if (ctx.detector) ctx.detector->m_ignore_flags_need_update = true;
            }

            ImGui::TableSetColumnIndex(3);
            if (UIHelpers::BeautifulButton("Remove", ImVec2(-1, 0))) {
                ctx.config.class_settings.erase(ctx.config.class_settings.begin() + i);
                ctx.config.saveConfig();
                if (ctx.detector) ctx.detector->m_ignore_flags_need_update = true;
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
    static bool new_class_ignore = false;

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
    
    ImGui::Checkbox("Ignore", &new_class_ignore);
    
    ImGui::Columns(1);
    
    UIHelpers::CompactSpacer();
    
    if (UIHelpers::BeautifulButton("Suggest Next ID", ImVec2(-1, 0))) {
        int max_id = -1;
        if (!ctx.config.class_settings.empty()) {
            for(const auto& cs : ctx.config.class_settings) {
                if (cs.id > max_id) max_id = cs.id;
            }
            new_class_id = max_id + 1;
        } else {
            new_class_id = 0;
        }
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
            ctx.config.class_settings.emplace_back(new_class_id, temp_name, new_class_ignore);
            ctx.config.saveConfig();
            if (ctx.detector) ctx.detector->m_ignore_flags_need_update = true;
            
            int max_id = -1;
            if (!ctx.config.class_settings.empty()) {
                for(const auto& cs : ctx.config.class_settings) {
                    if (cs.id > max_id) max_id = cs.id;
                }
                new_class_id = max_id + 1;
            } else {
                new_class_id = 0;
            }
            new_class_name_buf[0] = '\0'; 
            new_class_ignore = false;
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
            ctx.config.saveConfig();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Text("CUDA Device ID");
        UIHelpers::InfoTooltip("Set the CUDA device ID to use for detection (requires restart).");
    }
    
    UIHelpers::EndCard();
}

void draw_ai()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.6f);
    
    // Left column - Main settings
    draw_model_settings();
    UIHelpers::CompactSpacer();
    
    draw_detection_settings();
    UIHelpers::CompactSpacer();
    
    draw_advanced_settings();
    
    UIHelpers::NextColumn();
    
    // Right column - Class settings and info
    draw_class_settings();
    
    UIHelpers::EndTwoColumnLayout();
}
