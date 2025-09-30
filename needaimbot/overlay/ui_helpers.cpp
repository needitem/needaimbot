#include "ui_helpers.h"
#include "../imgui/imgui_internal.h"
#include <cmath>
#include "AppContext.h"
#include <vector>
#include <string>

namespace UIHelpers 
{
    ImVec4 GetAccentColor(float alpha) 
    {
        return ImVec4(0.20f, 0.60f, 0.90f, alpha);
    }
    
    ImVec4 GetSuccessColor(float alpha) 
    {
        return ImVec4(0.40f, 0.80f, 0.40f, alpha);
    }
    
    ImVec4 GetWarningColor(float alpha) 
    {
        return ImVec4(1.00f, 0.70f, 0.00f, alpha);
    }
    
    ImVec4 GetErrorColor(float alpha) 
    {
        return ImVec4(0.90f, 0.30f, 0.30f, alpha);
    }


    bool BeautifulToggle(const char* label, bool* value, const char* description)
    {
        bool changed = false;
        
        ImGui::PushID(label);
        
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        float height = ImGui::GetFrameHeight();
        float width = height * 1.8f;
        float radius = height * 0.5f;
        
        ImGui::InvisibleButton("##toggle", ImVec2(width, height));
        
        if (ImGui::IsItemClicked()) {
            *value = !*value;
            changed = true;
        }
        
        // No animation - instant toggle
        float t = *value ? 1.0f : 0.0f;
        
        ImU32 col_bg;
        if (ImGui::IsItemHovered()) {
            col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.25f, 0.25f, 0.30f, 1.0f), GetAccentColor(), t));
        } else {
            col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.20f, 0.20f, 0.25f, 1.0f), GetAccentColor(), t));
        }
        
        draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
        draw_list->AddCircleFilled(ImVec2(p.x + radius + t * (width - radius * 2.0f), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));
        
        ImGui::SameLine();
        ImGui::Text("%s", label);
        
        if (description && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            float max_width = ImGui::GetIO().DisplaySize.x * 0.5f;
            ImGui::PushTextWrapPos(max_width);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        
        ImGui::PopID();
        return changed;
    }

    bool BeautifulSlider(const char* label, float* value, float min, float max, const char* format)
    {
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, GetAccentColor(1.0f));
        bool changed = ImGui::SliderFloat(label, value, min, max, format);
        ImGui::PopStyleColor(2);
        return changed;
    }


    void BeautifulSeparator(const char* text)
    {
        if (text) {
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            
            float width = ImGui::GetContentRegionAvail().x;
            ImVec2 text_size = ImGui::CalcTextSize(text);
            float text_width = text_size.x + 20.0f;
            
            draw_list->AddLine(ImVec2(pos.x, pos.y + text_size.y * 0.5f),
                             ImVec2(pos.x + (width - text_width) * 0.5f, pos.y + text_size.y * 0.5f),
                             ImGui::GetColorU32(ImGuiCol_Separator));
            
            ImGui::SetCursorPosX((width - text_size.x) * 0.5f);
            ImGui::Text("%s", text);
            
            ImGui::SameLine();
            draw_list->AddLine(ImVec2(pos.x + (width + text_width) * 0.5f, pos.y + text_size.y * 0.5f),
                             ImVec2(pos.x + width, pos.y + text_size.y * 0.5f),
                             ImGui::GetColorU32(ImGuiCol_Separator));
        } else {
            ImGui::Separator();
        }
    }

    void BeautifulText(const char* text, ImVec4 color)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::Text("%s", text);
        ImGui::PopStyleColor();
    }

    void TextColored(const ImVec4& color, const char* text)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::Text("%s", text);
        ImGui::PopStyleColor();
    }

    void BeautifulSection(const char* title, bool* open)
    {
        ImGui::PushStyleColor(ImGuiCol_Header, GetAccentColor(0.3f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, GetAccentColor(0.4f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, GetAccentColor(0.5f));
        
        if (open) {
            *open = ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen);
        } else {
            ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen);
        }
        
        ImGui::PopStyleColor(3);
    }

    void StatusIndicator(const char* label, bool status, const char* description)
    {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        
        float radius = 6.0f;
        ImU32 color = status ? ImGui::GetColorU32(GetSuccessColor()) : ImGui::GetColorU32(GetErrorColor());
        
        draw_list->AddCircleFilled(ImVec2(pos.x + radius, pos.y + ImGui::GetTextLineHeight() * 0.5f), radius, color);
        
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + radius * 2.5f);
        ImGui::Text("%s", label);
        
        if (description && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            float max_width = ImGui::GetIO().DisplaySize.x * 0.5f;
            ImGui::PushTextWrapPos(max_width);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void InfoTooltip(const char* description)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            const float wrap_width = ImGui::GetFontSize() * 35.0f;
            ImGui::PushTextWrapPos(wrap_width);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }


    void PushStyleColors()
    {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.08f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.10f, 0.90f));
    }

    void PopStyleColors()
    {
        ImGui::PopStyleColor(2);
    }

    void WrappedTooltip(const char* description)
    {
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            float max_width = ImGui::GetIO().DisplaySize.x * 0.5f;
            ImGui::PushTextWrapPos(max_width);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    
    
    void BeginCard(const char* title)
    {
        if (title) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.9f));
            float original_scale = ImGui::GetFont()->Scale;
            ImGui::GetFont()->Scale *= 1.05f;
            ImGui::PushFont(ImGui::GetFont());
            ImGui::Text("%s", title);
            ImGui::GetFont()->Scale = original_scale;
            ImGui::PopFont();
            ImGui::PopStyleColor();
            ImGui::Separator();
            CompactSpacer();
        }
    }
    
    void EndCard()
    {
        // No-op now
    }
    
    
    void Spacer(float height)
    {
        ImGui::Dummy(ImVec2(0, height));
    }
    
    void CompactSpacer()
    {
        ImGui::Dummy(ImVec2(0, 3.0f));
    }
    
    bool BeautifulButton(const char* label, const ImVec2& size)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.18f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, GetAccentColor(0.9f));
        ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.5f, 0.5f));
        
        bool result = ImGui::Button(label, size);
        
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);
        
        return result;
    }
    
    void CompactSlider(const char* label, float* value, float min, float max, const char* format)
    {
        ImGui::PushItemWidth(-FLT_MIN);
        BeautifulSlider(label, value, min, max, format);
        ImGui::PopItemWidth();
    }
    
    void CompactCombo(const char* label, int* current_item, const char* const items[], int items_count)
    {
        ImGui::PushItemWidth(-FLT_MIN);
        ImGui::PushStyleColor(ImGuiCol_Header, GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, GetAccentColor(0.9f));
        ImGui::Combo(label, current_item, items, items_count);
        ImGui::PopStyleColor(3);
        ImGui::PopItemWidth();
    }
    
    void CompactCombo(const char* label, int* current_item, bool (*getter)(void*, int, const char**), void* data, int items_count)
    {
        ImGui::PushItemWidth(-FLT_MIN);
        ImGui::PushStyleColor(ImGuiCol_Header, GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, GetAccentColor(0.9f));
        ImGui::Combo(label, current_item, getter, data, items_count);
        ImGui::PopStyleColor(3);
        ImGui::PopItemWidth();
    }

    // Enhanced UI helpers for better organization
    void BeginSettingsSection(const char* title, const char* description)
    {
        // Section header
        if (title) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor());
            ImGui::Text("%s", title);
            ImGui::PopStyleColor();
            
            if (description) {
                ImGui::SameLine();
                HelpMarker(description);
            }
            
            ImGui::Separator();
            CompactSpacer();
        }
    }
    
    void EndSettingsSection()
    {
        CompactSpacer();
    }
    
    void SettingsHeader(const char* title)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.9f));
        ImGui::Text("%s", title);
        ImGui::PopStyleColor();
        ImGui::Separator();
        Spacer(3.0f);
    }
    
    void SettingsSubHeader(const char* title)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));
        float original_scale = ImGui::GetFont()->Scale;
        ImGui::GetFont()->Scale *= 0.95f;
        ImGui::PushFont(ImGui::GetFont());
        ImGui::Text("%s", title);
        ImGui::GetFont()->Scale = original_scale;
        ImGui::PopFont();
        ImGui::PopStyleColor();
        Spacer(1.5f);
    }
    
    void HelpMarker(const char* desc)
    {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
    
    void SettingsRow(const char* label, float label_width)
    {
        ImGui::Text("%s", label);
        ImGui::SameLine();
        ImGui::SetCursorPosX(label_width);
    }
    
    void SettingsValue(const char* value)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.8f));
        ImGui::Text("%s", value);
        ImGui::PopStyleColor();
    }
    
    bool EnhancedSliderFloat(const char* label, float* v, float v_min, float v_max, const char* format, const char* description)
    {
        // Enhanced styling for better visibility
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.30f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, GetAccentColor(1.0f));
        
        // Auto-width slider (fills available space)
        ImGui::SetNextItemWidth(-FLT_MIN);
        bool changed = ImGui::SliderFloat(label, v, v_min, v_max, format);
        
        ImGui::PopStyleColor(5);
        
        if (description && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        
        return changed;
    }
    
    bool EnhancedCombo(const char* label, int* current_item, const char* const items[], int items_count, const char* description)
    {
        // Enhanced styling for better visibility
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.30f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Button, GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_Header, GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, GetAccentColor(0.9f));
        
        // Auto-width combo (fills available space)
        ImGui::SetNextItemWidth(-FLT_MIN);
        bool changed = ImGui::Combo(label, current_item, items, items_count);
        
        ImGui::PopStyleColor(9);
        
        if (description && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        
        return changed;
    }
    
    bool EnhancedCheckbox(const char* label, bool* v, const char* description)
    {
        // Enhanced styling
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.30f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_CheckMark, GetAccentColor(1.0f));
        
        bool changed = ImGui::Checkbox(label, v);
        
        ImGui::PopStyleColor(4);
        
        if (description && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        
        return changed;
    }
    
    bool EnhancedButton(const char* label, const ImVec2& size, const char* description)
    {
        // Enhanced styling
        ImGui::PushStyleColor(ImGuiCol_Button, GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, GetAccentColor(1.0f));
        
        bool pressed = ImGui::Button(label, size);
        
        ImGui::PopStyleColor(3);
        
        if (description && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(description);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        
        return pressed;
    }
    
    bool ProfileDropdown(const char* combo_label, float width)
    {
        auto& ctx = AppContext::getInstance();
        static std::vector<std::string> profile_list;
        static int current_profile_index = -1;
        static bool initialized = false;
        
        // Initialize or refresh profile list
        static int last_refresh_frame = -1;
        bool should_refresh = !initialized;
        
        // Manual refresh check - only when dropdown is opened
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            int current_frame = ImGui::GetFrameCount();
            if (current_frame - last_refresh_frame > 180) { // Minimum 3 seconds between refreshes
                should_refresh = true;
                last_refresh_frame = current_frame;
            }
        }
        
        if (should_refresh) {
            profile_list = ctx.config.listProfiles();
            
            // Find current profile index
            current_profile_index = -1;
            for (size_t i = 0; i < profile_list.size(); ++i) {
                if (profile_list[i] == ctx.config.getActiveProfile()) {
                    current_profile_index = static_cast<int>(i);
                    break;
                }
            }
            
            // If current profile not found, add it
            if (current_profile_index == -1 && !ctx.config.getActiveProfile().empty()) {
                profile_list.insert(profile_list.begin(), ctx.config.getActiveProfile());
                current_profile_index = 0;
            }
            
            initialized = true;
        }
        
        // Profile dropdown
        ImGui::PushItemWidth(width);
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor());
        ImGui::Text("Profile:");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        
        bool changed = false;
        const char* current_profile_name = (current_profile_index >= 0 && current_profile_index < profile_list.size()) 
            ? profile_list[current_profile_index].c_str() : "Default";
            
        if (ImGui::BeginCombo(combo_label, current_profile_name))
        {
            for (int i = 0; i < profile_list.size(); i++)
            {
                const bool is_selected = (current_profile_index == i);
                if (ImGui::Selectable(profile_list[i].c_str(), is_selected))
                {
                    if (current_profile_index != i) {
                        current_profile_index = i;
                        // Load the profile immediately
                        ctx.config.setActiveProfile(profile_list[i]);
                        changed = true;
                    }
                }
                
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
        
        return changed;
    }
    
    bool WeaponProfileDropdown(const char* combo_label, float width)
    {
        auto& ctx = AppContext::getInstance();
        static std::vector<std::string> weapon_list;
        static int current_weapon_index = -1;
        static bool initialized = false;
        
        // Initialize or refresh weapon list
        static int last_weapon_refresh_frame = -1;
        bool should_refresh = !initialized;
        
        // Manual refresh check - only when dropdown is opened
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            int current_frame = ImGui::GetFrameCount();
            if (current_frame - last_weapon_refresh_frame > 180) { // Minimum 3 seconds between refreshes
                should_refresh = true;
                last_weapon_refresh_frame = current_frame;
            }
        }
        
        if (should_refresh) {
            weapon_list = ctx.config.getWeaponProfileNames();
            
            // Update current weapon index
            current_weapon_index = ctx.config.active_weapon_profile_index;
            if (current_weapon_index < 0 || current_weapon_index >= weapon_list.size()) {
                current_weapon_index = 0;
            }
            
            initialized = true;
        }
        
        // Weapon dropdown
        ImGui::PushItemWidth(width);
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor());
        ImGui::Text("Weapon:");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        
        bool changed = false;
        const char* current_weapon = (current_weapon_index >= 0 && current_weapon_index < weapon_list.size()) 
            ? weapon_list[current_weapon_index].c_str() : "Default";
            
        if (ImGui::BeginCombo(combo_label, current_weapon))
        {
            for (int i = 0; i < weapon_list.size(); i++)
            {
                const bool is_selected = (current_weapon_index == i);
                if (ImGui::Selectable(weapon_list[i].c_str(), is_selected))
                {
                    if (current_weapon_index != i) {
                        current_weapon_index = i;
                        // Set active weapon immediately
                        ctx.config.setActiveWeaponProfile(weapon_list[i]);
                        changed = true;
                    }
                }
                
                // Right-click context menu
                if (ImGui::BeginPopupContextItem())
                {
                    if (weapon_list[i] != "Default" && ImGui::MenuItem("Delete")) {
                        ctx.config.removeWeaponProfile(weapon_list[i]);
                        ctx.config.saveActiveProfile();
                        initialized = false; // Force refresh
                    }
                    if (ImGui::MenuItem("Duplicate")) {
                        std::string new_name = weapon_list[i] + "_copy";
                        WeaponRecoilProfile* original = ctx.config.getWeaponProfile(weapon_list[i]);
                        if (original) {
                            WeaponRecoilProfile copy = *original;
                            copy.weapon_name = new_name;
                            ctx.config.addWeaponProfile(copy);
                            ctx.config.saveActiveProfile();
                            initialized = false; // Force refresh
                        }
                    }
                    ImGui::EndPopup();
                }
                
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
        
        // Quick add weapon button
        ImGui::SameLine();
        if (ImGui::Button("+ Add")) {
            ImGui::OpenPopup("AddWeaponPopup");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Add new weapon profile");
        }
        
        // Add weapon popup
        if (ImGui::BeginPopup("AddWeaponPopup")) {
            static char new_weapon_name[64] = "";
            ImGui::Text("New Weapon Name:");
            ImGui::InputText("##NewWeaponName", new_weapon_name, sizeof(new_weapon_name));
            
            if (ImGui::Button("Add") && strlen(new_weapon_name) > 0) {
                WeaponRecoilProfile new_profile(new_weapon_name, 3.0f, 1.0f);
                if (ctx.config.addWeaponProfile(new_profile)) {
                    ctx.config.saveActiveProfile();
                    new_weapon_name[0] = '\0';
                    initialized = false; // Force refresh
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        
        return changed;
    }
}