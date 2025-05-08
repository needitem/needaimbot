#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <vector>
#include <string>
#include <algorithm> // For std::find
#include <iterator> // For std::distance

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "include/other_tools.h"
#include "overlay.h"

void draw_ai()
{
    std::vector<std::string> availableModels = getAvailableModels();
    if (availableModels.empty())
    {
        ImGui::Text("No models available in the 'models' folder.");
    }
    else
    {
        int currentModelIndex = 0;
        auto it = std::find(availableModels.begin(), availableModels.end(), config.ai_model);

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
            if (config.ai_model != availableModels[currentModelIndex])
            {
                config.ai_model = availableModels[currentModelIndex];
                config.saveConfig();
                detector_model_changed.store(true);
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- ONNX Input Resolution Selection ---
    const char* resolution_items[] = { "160", "320", "640" };
    int current_resolution_index = 0;
    if (config.onnx_input_resolution == 160)      current_resolution_index = 0;
    else if (config.onnx_input_resolution == 320) current_resolution_index = 1;
    else if (config.onnx_input_resolution == 640) current_resolution_index = 2;
    // else default to 0 (160) or handle error if value is unexpected

    if (ImGui::Combo("ONNX Input Resolution", &current_resolution_index, resolution_items, IM_ARRAYSIZE(resolution_items)))
    {
        int selected_resolution = 160; // Default
        if (current_resolution_index == 0)      selected_resolution = 160;
        else if (current_resolution_index == 1) selected_resolution = 320;
        else if (current_resolution_index == 2) selected_resolution = 640;

        if (config.onnx_input_resolution != selected_resolution)
        {
            config.onnx_input_resolution = selected_resolution;
            config.saveConfig();
            detector_model_changed.store(true); // Trigger model reload/rebuild
        }
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Select the input resolution for the ONNX model (e.g., 640 for 640x640 input).\nChanging this will require the .engine file to be rebuilt if it doesn't match.");
    }

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
        if (postprocessOptions[i] == config.postprocess)
        {
            currentPostprocessIndex = static_cast<int>(i);
            break;
        }
    }

    if (ImGui::Combo("Postprocess", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size())))
    {
        config.postprocess = postprocessOptions[currentPostprocessIndex];
        config.saveConfig();
        detector_model_changed.store(true);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.01f, 1.00f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        config.saveConfig();
    }
    if (ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.01f, 1.00f, "%.2f")) { config.saveConfig(); }
    if (ImGui::SliderInt("Max Detections", &config.max_detections, 1, 100)) { config.saveConfig(); }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::InputInt("CUDA Device ID", &config.cuda_device_id))
    {
        if (config.cuda_device_id < 0) config.cuda_device_id = 0;
        config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Set the CUDA device ID to use for detection (requires restart).");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::SeparatorText("Class Definitions");
    ImGui::Spacing();

    // Input for the head class name
    static char head_class_name_buffer[128];
    strncpy(head_class_name_buffer, config.head_class_name.c_str(), sizeof(head_class_name_buffer) - 1);
    head_class_name_buffer[sizeof(head_class_name_buffer) - 1] = '\0'; // Ensure null termination
    ImGui::InputText("Head Class Identifier Name", head_class_name_buffer, sizeof(head_class_name_buffer));
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        config.head_class_name = head_class_name_buffer;
        config.saveConfig();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("The name of the class that should be treated as 'Head' for specific aiming logic (e.g., head_y_offset).");
    }
    ImGui::Spacing();

    // Table for class settings for better alignment
    if (ImGui::BeginTable("class_settings_table", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Ignore", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < config.class_settings.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ClassSetting& setting = config.class_settings[i];

            ImGui::TableNextRow();
            
            ImGui::TableSetColumnIndex(0);
            if (ImGui::InputInt("##ID", &setting.id, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
                // ID changed, ensure it's unique or handle conflicts if necessary
                // For now, direct change and save.
                config.saveConfig();
            }

            ImGui::TableSetColumnIndex(1);
            char name_buf[128];
            strncpy(name_buf, setting.name.c_str(), sizeof(name_buf) - 1);
            name_buf[sizeof(name_buf) - 1] = '\0';
            if (ImGui::InputText("##Name", name_buf, sizeof(name_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                setting.name = name_buf;
                config.saveConfig();
            }
             if (ImGui::IsItemDeactivatedAfterEdit() && setting.name != name_buf) { // Handle focus loss too
                setting.name = name_buf;
                config.saveConfig();
            }

            ImGui::TableSetColumnIndex(2);
            if (ImGui::Checkbox("##Ignore", &setting.ignore)) {
                config.saveConfig();
            }

            ImGui::TableSetColumnIndex(3);
            if (ImGui::Button("Remove")) {
                config.class_settings.erase(config.class_settings.begin() + i);
                config.saveConfig();
                ImGui::PopID(); // Pop before potentially continuing loop with decremented i
                i--; // Adjust index due to removal
                continue; // Important to re-evaluate loop condition and avoid skipping next element
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();

    // --- Add new class --- 
    ImGui::Separator();
    ImGui::Text("Add New Class:");
    static int new_class_id = 0; // Start with 0 or suggest next available
    static char new_class_name_buf[128] = "";
    static bool new_class_ignore = false;

    // Suggest next available ID
    if (ImGui::Button("Suggest Next ID")) {
        int max_id = -1;
        if (!config.class_settings.empty()) {
            for(const auto& cs : config.class_settings) {
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
        for (const auto& cs : config.class_settings) {
            if (cs.id == new_class_id) {
                id_exists = true;
                break;
            }
        }
        std::string temp_name = new_class_name_buf;
        if (!id_exists && !temp_name.empty()) {
            config.class_settings.emplace_back(new_class_id, temp_name, new_class_ignore);
            config.saveConfig();
            // Reset for next entry
            int max_id = -1;
            if (!config.class_settings.empty()) {
                 for(const auto& cs : config.class_settings) {
                    if (cs.id > max_id) max_id = cs.id;
                }
                new_class_id = max_id + 1;
            } else {
                 new_class_id = 0;
            }
            new_class_name_buf[0] = '\0'; // Clear buffer
            new_class_ignore = false;
        }
        // TODO: else display error (e.g., ImGui::TextColored(ImVec4(1,0,0,1), "Error: ID exists or name empty."))
    }

    ImGui::Spacing();
}