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

    ImGui::Separator();

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

    ImGui::Separator();
    ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.01f, 1.00f, "%.2f");
    ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.01f, 1.00f, "%.2f");
    ImGui::SliderInt("Max Detections", &config.max_detections, 1, 100);
    
    ImGui::Separator();
    
    if (ImGui::InputInt("CUDA Device ID", &config.cuda_device_id))
    {
        if (config.cuda_device_id < 0) config.cuda_device_id = 0;
        config.saveConfig();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Set the CUDA device ID to use for detection (requires restart).");
    }

    ImGui::Separator();
    ImGui::SeparatorText("Class Filtering");

    // if (ImGui::Checkbox("Disable Headshot Aiming", &config.disable_headshot)) { config.saveConfig(); } // Removed
    // if (ImGui::IsItemHovered()){ // Removed
    //      ImGui::SetTooltip("Aim for the body instead of the head. Does not affect head *detection* unless 'Ignore Head' is also checked."); // Removed
    // } // Removed

    // ImGui::Spacing(); // Removed spacing as checkbox is gone

    // --- Add checkboxes for ignoring specific classes --- 
    struct ClassIgnoreInfo {
        const char* label;
        bool* config_flag;
    };

    std::vector<ClassIgnoreInfo> ignore_flags = {
        {"Ignore Player (0)", &config.ignore_class_0},
        {"Ignore Bot (1)", &config.ignore_class_1},
        {"Ignore Weapon (2)", &config.ignore_class_2},
        {"Ignore Outline (3)", &config.ignore_class_3},
        {"Ignore Dead Body (4)", &config.ignore_class_4},
        {"Ignore Hideout Human (5)", &config.ignore_class_5},
        {"Ignore Hideout Balls (6)", &config.ignore_class_6},
        {"Ignore Head (7)", &config.ignore_class_7},
        {"Ignore Smoke (8)", &config.ignore_class_8},
        {"Ignore Fire (9)", &config.ignore_class_9},
        {"Ignore Third Person (10)", &config.ignore_class_10}
    };

    // Display checkboxes in two columns for better layout
    ImGui::Columns(2, "ClassIgnoreColumns", false);
    int items_per_column = (ignore_flags.size() + 1) / 2; // Calculate items per column

    for (size_t i = 0; i < ignore_flags.size(); ++i) {
        if (ImGui::Checkbox(ignore_flags[i].label, ignore_flags[i].config_flag)) {
            config.saveConfig();
        }
        if (i == items_per_column - 1) {
            ImGui::NextColumn(); // Move to the next column after half the items
        }
    }
    ImGui::Columns(1); // Return to single column layout
}