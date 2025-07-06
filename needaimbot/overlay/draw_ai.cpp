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

void draw_ai()
{
    auto& ctx = AppContext::getInstance();
    
    std::vector<std::string> availableModels = getAvailableModels();
    if (availableModels.empty())
    {
        ImGui::Text("No models available in the 'models' folder.");
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

        if (ImGui::Combo("Model", &currentModelIndex, modelsItems.data(), static_cast<int>(modelsItems.size())))
        {
            if (ctx.config.ai_model != availableModels[currentModelIndex])
            {
                ctx.config.ai_model = availableModels[currentModelIndex];
                ctx.config.saveConfig();
                detector_model_changed.store(true);
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    
    const char* resolution_items[] = { "160", "320", "640" };
    int current_resolution_index = 0;
    if (ctx.config.onnx_input_resolution == 160)      current_resolution_index = 0;
    else if (ctx.config.onnx_input_resolution == 320) current_resolution_index = 1;
    else if (ctx.config.onnx_input_resolution == 640) current_resolution_index = 2;
    

    if (ImGui::Combo("ONNX Input Resolution", &current_resolution_index, resolution_items, IM_ARRAYSIZE(resolution_items)))
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
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Select the input resolution for the ONNX model (e.g., 640 for 640x640 input).\nChanging this will require the .engine file to be rebuilt if it doesn't match.");
    }

    // Add TensorRT precision options
    if (ImGui::Checkbox("Enable FP16", &ctx.config.export_enable_fp16))
    {
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Enable FP16 precision for the exported TensorRT engine.");

    if (ImGui::Checkbox("Enable FP8", &ctx.config.export_enable_fp8))
    {
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Enable FP8 precision for the exported TensorRT engine.");

    // Force rebuild button
    if (ImGui::Button("Rebuild Engine"))
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
        // switch to .onnx model to force rebuild from ONNX
        ctx.config.ai_model = onnxPath.filename().string();
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Force rebuild of the TensorRT engine from ONNX.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

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

    if (ImGui::Combo("Postprocess", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size())))
    {
        ctx.config.postprocess = postprocessOptions[currentPostprocessIndex];
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SliderFloat("Confidence Threshold", &ctx.config.confidence_threshold, 0.01f, 1.00f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.saveConfig();
    }
    if (ImGui::SliderFloat("NMS Threshold", &ctx.config.nms_threshold, 0.01f, 1.00f, "%.2f")) { ctx.config.saveConfig(); }
    if (ImGui::SliderInt("Max Detections", &ctx.config.max_detections, 1, 100)) { ctx.config.saveConfig(); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::InputInt("CUDA Device ID", &ctx.config.cuda_device_id))
    {
        if (ctx.config.cuda_device_id < 0) ctx.config.cuda_device_id = 0;
        ctx.config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Set the CUDA device ID to use for detection (requires restart).");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::SeparatorText("Class Definitions");
    ImGui::Spacing();

    
    static char head_class_name_buffer[128];
    strncpy_s(head_class_name_buffer, sizeof(head_class_name_buffer), ctx.config.head_class_name.c_str(), _TRUNCATE);
    head_class_name_buffer[sizeof(head_class_name_buffer) - 1] = '\0'; 
    ImGui::InputText("Head Class Identifier Name", head_class_name_buffer, sizeof(head_class_name_buffer));
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ctx.config.head_class_name = head_class_name_buffer;
        ctx.config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("The name of the class that should be treated as 'Head' for specific aiming logic (e.g., head_y_offset).");
    }
    ImGui::Spacing();

    
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
            if (ImGui::Button("Remove")) {
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

    ImGui::Spacing();

    
    ImGui::Separator();
    ImGui::Text("Add New Class:");
    static int new_class_id = 0; 
    static char new_class_name_buf[128] = "";
    static bool new_class_ignore = false;

    
    if (ImGui::Button("Suggest Next ID")) {
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
    ImGui::SameLine();
    ImGui::InputInt("New ID", &new_class_id);
    ImGui::InputText("New Name", new_class_name_buf, sizeof(new_class_name_buf));
    ImGui::Checkbox("Ignore New", &new_class_ignore);

    if (ImGui::Button("Add Class")) {
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

    ImGui::Spacing();
}
