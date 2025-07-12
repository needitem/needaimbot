#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

#include "imgui/imgui.h"
#include "AppContext.h"
#include "ui_helpers_new.h"
#include "overlay.h"
#include "other_tools.h"

void draw_ai_new()
{
    auto& ctx = AppContext::getInstance();
    
    UI::Space();
    
    // Model Selection
    UI::Section("Model Selection");
    
    std::vector<std::string> models = getAvailableModels();
    if (models.empty()) {
        UI::Warning("No models found in 'models' folder");
    } else {
        std::vector<const char*> model_items;
        model_items.reserve(models.size());
        for (const auto& m : models) {
            model_items.push_back(m.c_str());
        }
        
        int current = 0;
        auto it = std::find(models.begin(), models.end(), ctx.config.ai_model);
        if (it != models.end()) {
            current = static_cast<int>(std::distance(models.begin(), it));
        }
        
        if (UI::Combo("Model##ai", &current, model_items.data(), static_cast<int>(model_items.size()))) {
            ctx.config.ai_model = models[current];
            ctx.config.saveConfig();
            detector_model_changed.store(true);
        }
    }
    
    UI::Space();
    UI::Space();
    
    // Engine Settings
    UI::Section("Engine Settings");
    
    const char* resolutions[] = { "160", "320", "640" };
    int res_idx = 1; // default 320
    if (ctx.config.onnx_input_resolution == 160) res_idx = 0;
    else if (ctx.config.onnx_input_resolution == 640) res_idx = 2;
    
    if (UI::Combo("Resolution##ai", &res_idx, resolutions, 3)) {
        ctx.config.onnx_input_resolution = (res_idx == 0) ? 160 : (res_idx == 2) ? 640 : 320;
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    
    UI::SmallSpace();
    
    UI::BeginColumns(0.5f);
    
    if (UI::Toggle("FP16##ai", &ctx.config.export_enable_fp16)) {
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    
    UI::NextColumn();
    
    if (UI::Toggle("FP8##ai", &ctx.config.export_enable_fp8)) {
        ctx.config.saveConfig();
        detector_model_changed.store(true);
    }
    
    UI::EndColumns();
    
    UI::SmallSpace();
    
    if (UI::Button("Rebuild Engine##ai", -1)) {
        std::filesystem::path modelPath("models/" + ctx.config.ai_model);
        std::filesystem::path enginePath = modelPath;
        if (enginePath.extension() == ".engine") {
            enginePath.replace_extension(".onnx");
        }
        enginePath.replace_extension(".engine");
        
        if (std::filesystem::exists(enginePath)) {
            std::filesystem::remove(enginePath);
            detector_model_changed.store(true);
        }
    }
    
    UI::Space();
    UI::Space();
    
    // Detection Settings
    UI::Section("Detection Settings");
    
    if (UI::Slider("Confidence##ai", &ctx.config.confidence_threshold, 0.01f, 1.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    
    if (UI::Slider("NMS Threshold##ai", &ctx.config.nms_threshold, 0.01f, 1.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    
    float max_det = static_cast<float>(ctx.config.max_detections);
    if (UI::Slider("Max Detections##ai", &max_det, 1.0f, 50.0f, "%.0f")) {
        ctx.config.max_detections = static_cast<int>(max_det);
        ctx.config.saveConfig();
    }
    
    UI::Space();
    UI::Space();
    
    // Target Selection Weights
    UI::Section("Target Selection");
    
    if (UI::Slider("Confidence Weight##ai", &ctx.config.confidence_weight, 0.0f, 1.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    
    if (UI::Slider("Distance Weight##ai", &ctx.config.distance_weight, 0.0f, 1.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    
    float total = ctx.config.confidence_weight + ctx.config.distance_weight;
    if (total > 0) {
        ImGui::Text("Normalized: C=%.1f%% D=%.1f%%", 
                    (ctx.config.confidence_weight / total) * 100,
                    (ctx.config.distance_weight / total) * 100);
    }
    
    UI::Space();
}