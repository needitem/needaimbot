#pragma once
#include "../config/config.h"
#include "../AppContext.h"
#include "../imgui/imgui.h"
#include <string>
#include <vector>

namespace CommonHelpers {
    // Find class name by ID
    inline std::string getClassNameById(int classId) {
        auto& ctx = AppContext::getInstance();
        for(const auto& cs : ctx.config.class_settings) {
            if (cs.id == classId) {
                return cs.name;
            }
        }
        return "Unknown";
    }
    
    // Get next available class ID
    inline int getNextClassId() {
        auto& ctx = AppContext::getInstance();
        int max_id = -1;
        for(const auto& cs : ctx.config.class_settings) {
            if (cs.id > max_id) max_id = cs.id;
        }
        return max_id + 1;
    }
    
    // Update config value and save
    template<typename T>
    inline void updateConfigAndSave(T& configValue, const T& newValue) {
        if (configValue != newValue) {
            configValue = newValue;
            AppContext::getInstance().config.saveActiveProfile();
        }
    }
    
    // Draw key binding UI
    inline bool drawKeyBinding(const char* label, std::string& keyName, const std::vector<std::string>& key_names, const std::vector<const char*>& key_names_cstrs) {
        bool changed = false;
        
        int current_index = -1;
        for (size_t k = 0; k < key_names.size(); ++k) {
            if (key_names[k] == keyName) {
                current_index = static_cast<int>(k);
                break;
            }
        }
        
        if (current_index == -1) {
            current_index = 0;
        }
        
        if (ImGui::Combo(label, &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size()))) {
            keyName = key_names[current_index];
            changed = true;
        }
        
        return changed;
    }
    
    // Draw key binding list with add/remove functionality
    inline void drawKeyBindingList(const char* label, std::vector<std::string>& bindings, const std::vector<std::string>& key_names, const std::vector<const char*>& key_names_cstrs) {
        auto& ctx = AppContext::getInstance();
        
        for (size_t i = 0; i < bindings.size(); ) {
            std::string& current_key_name = bindings[i];
            
            int current_index = -1;
            for (size_t k = 0; k < key_names.size(); ++k) {
                if (key_names[k] == current_key_name) {
                    current_index = static_cast<int>(k);
                    break;
                }
            }
            
            if (current_index == -1) {
                current_index = 0;
            }
            
            std::string combo_label = std::string(label) + " " + std::to_string(i);
            
            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size()))) {
                current_key_name = key_names[current_index];
                ctx.config.saveActiveProfile();
            }
            
            ImGui::SameLine();
            std::string remove_button_label = "Remove##" + std::string(label) + std::to_string(i);
            if (ImGui::Button(remove_button_label.c_str())) {
                if (bindings.size() <= 1) {
                    bindings[0] = std::string("None");
                    ctx.config.saveActiveProfile();
                    continue;
                } else {
                    bindings.erase(bindings.begin() + i);
                    ctx.config.saveActiveProfile();
                    continue;
                }
            }
            
            ++i;
        }
        
        std::string add_button_label = std::string("Add New ") + label;
        if (ImGui::Button(add_button_label.c_str())) {
            bindings.push_back("None");
            ctx.config.saveActiveProfile();
        }
    }
}
