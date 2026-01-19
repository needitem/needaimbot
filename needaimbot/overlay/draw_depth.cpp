#include "../core/windows_headers.h"

#include "../imgui/imgui.h"
#include <string>
#include <vector>
#include <filesystem>

#include "AppContext.h"
#include "config/config.h"
#include "needaimbot.h"
#include "draw_settings.h"
#include "ui_helpers.h"

#ifdef USE_CUDA
#include "../depth/depth_mask.h"
#endif

// Get available depth models (.engine files in models/ directory)
static std::vector<std::string> getAvailableDepthModels() {
    std::vector<std::string> models;
    
    try {
        std::string modelsDir = AppContext::getInstance().config.getExecutableDir() + "/models";
        
        if (std::filesystem::exists(modelsDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    // Look for depth-related engine files
                    if (filename.find(".engine") != std::string::npos &&
                        (filename.find("depth") != std::string::npos || 
                         filename.find("Depth") != std::string::npos)) {
                        models.push_back(filename);
                    }
                }
            }
        }
    } catch (...) {
        // Ignore filesystem errors
    }
    
    return models;
}

void draw_depth_settings()
{
    auto& ctx = AppContext::getInstance();

    UIHelpers::BeginCard("Depth Estimation");

    // Enable/disable depth
    if (UIHelpers::BeautifulToggle("Enable Depth Estimation", &ctx.config.profile().depth_enabled,
                                   "Use depth model to prioritize closer targets")) {
        SAVE_PROFILE();
    }

    if (ctx.config.profile().depth_enabled) {
        UIHelpers::CompactSpacer();

        // Model selection
        static std::vector<std::string> depthModels;
        static std::vector<const char*> depthModelPtrs;
        static auto lastRefresh = std::chrono::high_resolution_clock::now();
        
        auto now = std::chrono::high_resolution_clock::now();
        if (depthModels.empty() || 
            std::chrono::duration_cast<std::chrono::seconds>(now - lastRefresh).count() >= 5) {
            depthModels = getAvailableDepthModels();
            depthModelPtrs.clear();
            for (const auto& m : depthModels) {
                depthModelPtrs.push_back(m.c_str());
            }
            lastRefresh = now;
        }

        if (depthModels.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
                              "No depth models found in models/ folder");
            ImGui::TextDisabled("Place depth*.engine files in the models directory");
        } else {
            // Find current selection
            int currentIdx = -1;
            for (size_t i = 0; i < depthModels.size(); i++) {
                if (depthModels[i] == ctx.config.profile().depth_model_path) {
                    currentIdx = static_cast<int>(i);
                    break;
                }
            }

            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo("Depth Model", &currentIdx, depthModelPtrs.data(), 
                            static_cast<int>(depthModelPtrs.size()))) {
                if (currentIdx >= 0 && currentIdx < static_cast<int>(depthModels.size())) {
                    ctx.config.profile().depth_model_path = depthModels[currentIdx];
                    SAVE_PROFILE();
                }
            }
            UIHelpers::InfoTooltip("Select DepthAnything TensorRT model for depth estimation");
        }

        UIHelpers::CompactSpacer();

        // FPS setting
        int fps = ctx.config.profile().depth_fps;
        if (ImGui::SliderInt("Depth FPS", &fps, 1, 30, "%d fps")) {
            ctx.config.profile().depth_fps = fps;
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            SAVE_PROFILE();
        }
        UIHelpers::InfoTooltip("Update rate for depth estimation.\nLower = less GPU usage, Higher = more responsive");

        UIHelpers::CompactSpacer();

        // Near percent
        int nearPercent = ctx.config.profile().depth_near_percent;
        if (ImGui::SliderInt("Near Threshold %", &nearPercent, 1, 100, "%d%%")) {
            ctx.config.profile().depth_near_percent = nearPercent;
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            SAVE_PROFILE();
        }
        UIHelpers::InfoTooltip("Percentage of pixels to consider as 'near'.\n20% = prioritize closest 20% of objects");

        UIHelpers::CompactSpacer();

        // Invert
        if (UIHelpers::BeautifulToggle("Invert (Far Priority)", &ctx.config.profile().depth_invert,
                                       "Prioritize far objects instead of near")) {
            SAVE_PROFILE();
        }
    }

    UIHelpers::EndCard();

#ifdef USE_CUDA
    // Debug info
    if (ctx.config.profile().depth_enabled) {
        UIHelpers::CompactSpacer();
        UIHelpers::BeginCard("Depth Status");

        auto& generator = depth_anything::GetDepthMaskGenerator();
        auto debugState = generator.debugState();
        auto frameSize = generator.lastFrameSize();

        if (ImGui::BeginTable("##depth_status", 2, ImGuiTableFlags_NoBordersInBody)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            auto addRow = [](const char* label, const char* value, bool isError = false) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", label);
                ImGui::TableNextColumn();
                if (isError) {
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", value);
                } else {
                    ImGui::Text("%s", value);
                }
            };

            addRow("Initialized", debugState.initialized ? "Yes" : "No");
            addRow("Model Ready", debugState.model_ready ? "Yes" : "No");
            
            if (frameSize.first > 0 && frameSize.second > 0) {
                char sizeBuf[32];
                snprintf(sizeBuf, sizeof(sizeBuf), "%dx%d", frameSize.first, frameSize.second);
                addRow("Frame Size", sizeBuf);
            } else {
                addRow("Frame Size", "N/A");
            }

            std::string lastErr = generator.lastError();
            if (!lastErr.empty()) {
                // Truncate long error messages
                if (lastErr.length() > 50) {
                    lastErr = lastErr.substr(0, 47) + "...";
                }
                addRow("Last Error", lastErr.c_str(), true);
            }

            ImGui::EndTable();
        }

        UIHelpers::EndCard();
    }
#endif
}
