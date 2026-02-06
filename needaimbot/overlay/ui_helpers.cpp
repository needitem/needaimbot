#include "ui_helpers.h"
#include "../imgui/imgui_internal.h"
#include <cmath>
#include "AppContext.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace UIHelpers 
{
    // Cyber Blue Theme Colors (#00D4FF based)
    ImVec4 GetAccentColor(float alpha) 
    {
        return ImVec4(0.00f, 0.83f, 1.00f, alpha);  // #00D4FF - Cyber Blue
    }
    
    ImVec4 GetSuccessColor(float alpha) 
    {
        return ImVec4(0.00f, 0.90f, 0.50f, alpha);  // Cyber Green
    }
    
    ImVec4 GetWarningColor(float alpha) 
    {
        return ImVec4(1.00f, 0.75f, 0.00f, alpha);  // Amber
    }
    
    ImVec4 GetErrorColor(float alpha) 
    {
        return ImVec4(1.00f, 0.30f, 0.35f, alpha);  // Cyber Red
    }
    
    // Secondary accent for gradients
    ImVec4 GetAccentColorDark(float alpha)
    {
        return ImVec4(0.00f, 0.55f, 0.75f, alpha);  // Darker Cyber Blue
    }


    // Animation state storage for toggles
    static std::unordered_map<ImGuiID, float> s_toggleAnimState;
    
    bool BeautifulToggle(const char* label, bool* value, const char* description)
    {
        bool changed = false;
        
        ImGui::PushID(label);
        ImGuiID id = ImGui::GetID("##toggle");
        
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
        
        // Smooth animation
        float target = *value ? 1.0f : 0.0f;
        float& animValue = s_toggleAnimState[id];
        float animSpeed = 8.0f * ImGui::GetIO().DeltaTime;
        animValue = animValue + (target - animValue) * ImClamp(animSpeed, 0.0f, 1.0f);
        float t = animValue;
        
        // Background color with glow effect when active
        ImVec4 bgOff = ImVec4(0.15f, 0.17f, 0.20f, 1.0f);
        ImVec4 bgOn = GetAccentColor(0.9f);
        ImU32 col_bg;
        
        if (ImGui::IsItemHovered()) {
            col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.20f, 0.22f, 0.26f, 1.0f), GetAccentColor(), t));
        } else {
            col_bg = ImGui::GetColorU32(ImLerp(bgOff, bgOn, t));
        }
        
        // Glow effect when enabled
        if (t > 0.1f) {
            ImVec4 glowColor = GetAccentColor(0.3f * t);
            draw_list->AddRectFilled(
                ImVec2(p.x - 2, p.y - 2), 
                ImVec2(p.x + width + 2, p.y + height + 2), 
                ImGui::GetColorU32(glowColor), 
                height * 0.5f + 2
            );
        }
        
        // Main track
        draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
        
        // Knob with subtle shadow
        float knobX = p.x + radius + t * (width - radius * 2.0f);
        float knobY = p.y + radius;
        draw_list->AddCircleFilled(ImVec2(knobX + 1, knobY + 1), radius - 1.5f, IM_COL32(0, 0, 0, 40));  // Shadow
        draw_list->AddCircleFilled(ImVec2(knobX, knobY), radius - 1.5f, IM_COL32(255, 255, 255, 255));
        
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
                if (profile_list[i] == ctx.config.getActiveProfileName()) {
                    current_profile_index = static_cast<int>(i);
                    break;
                }
            }
            
            // If current profile not found, add it
            if (current_profile_index == -1 && !ctx.config.getActiveProfileName().empty()) {
                profile_list.insert(profile_list.begin(), ctx.config.getActiveProfileName());
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
                        ctx.config.switchProfile(profile_list[i]);
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
    
    bool InputProfileDropdown(const char* combo_label, float width)
    {
        auto& ctx = AppContext::getInstance();
        static std::vector<std::string> profile_list;
        static int current_profile_index = -1;
        static bool initialized = false;
        
        // Initialize or refresh profile list
        static int last_profile_refresh_frame = -1;
        bool should_refresh = !initialized;
        
        // Manual refresh check - only when dropdown is opened
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            int current_frame = ImGui::GetFrameCount();
            if (current_frame - last_profile_refresh_frame > 180) { // Minimum 3 seconds between refreshes
                should_refresh = true;
                last_profile_refresh_frame = current_frame;
            }
        }
        
        if (should_refresh) {
            profile_list = ctx.config.getInputProfileNames();
            
            // Update current profile index
            current_profile_index = ctx.config.profile().active_input_profile_index;
            if (current_profile_index < 0 || current_profile_index >= profile_list.size()) {
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
        const char* current_profile = (current_profile_index >= 0 && current_profile_index < profile_list.size()) 
            ? profile_list[current_profile_index].c_str() : "Default";
            
        if (ImGui::BeginCombo(combo_label, current_profile))
        {
            for (int i = 0; i < profile_list.size(); i++)
            {
                const bool is_selected = (current_profile_index == i);
                if (ImGui::Selectable(profile_list[i].c_str(), is_selected))
                {
                    if (current_profile_index != i) {
                        current_profile_index = i;
                        // Set active profile immediately
                        ctx.config.setActiveInputProfile(profile_list[i]);
                        changed = true;
                    }
                }
                
                if (ImGui::BeginPopupContextItem())
                {
                    if (profile_list[i] != "Default" && ImGui::MenuItem("Delete")) {
                        ctx.config.removeInputProfile(profile_list[i]);
                        ctx.config.saveConfig();
                        initialized = false; // Force refresh
                    }
                    if (ImGui::MenuItem("Duplicate")) {
                        std::string new_name = profile_list[i] + "_copy";
                        InputProfile* original = ctx.config.getInputProfile(profile_list[i]);
                        if (original) {
                            InputProfile copy = *original;
                            copy.profile_name = new_name;
                            ctx.config.addInputProfile(copy);
                            ctx.config.saveConfig();
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
        
        // Quick add profile button
        ImGui::SameLine();
        if (ImGui::Button("+ Add")) {
            ImGui::OpenPopup("AddProfilePopup");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Add new input profile");
        }
        
        // Add profile popup
        if (ImGui::BeginPopup("AddProfilePopup")) {
            static char new_profile_name[64] = "";
            ImGui::Text("New Profile Name:");
            ImGui::InputText("##NewProfileName", new_profile_name, sizeof(new_profile_name));
            
            if (ImGui::Button("Add") && strlen(new_profile_name) > 0) {
                InputProfile new_profile(new_profile_name, 3.0f, 1.0f);
                if (ctx.config.addInputProfile(new_profile)) {
                    ctx.config.saveConfig();
                    new_profile_name[0] = '\0';
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
    
    // === NEW HIERARCHICAL UI COMPONENTS ===
    
    void StatusHeader(int targetCount, bool isPaused)
    {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 windowSize = ImGui::GetWindowSize();
        float padding = ImGui::GetStyle().WindowPadding.x;
        
        // Background for header
        ImVec2 headerStart = ImVec2(windowPos.x, windowPos.y + ImGui::GetCursorPosY());
        ImVec2 headerEnd = ImVec2(windowPos.x + windowSize.x, headerStart.y + 40);
        draw_list->AddRectFilled(headerStart, headerEnd, ImGui::GetColorU32(ImVec4(0.08f, 0.09f, 0.12f, 0.95f)));
        
        // Status text (centered vertically)
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
        
        if (isPaused) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetWarningColor());
            ImGui::Text("PAUSED (F3)");
            ImGui::PopStyleColor();
        } else if (targetCount > 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetSuccessColor());
            ImGui::Text("Tracking: %d target%s", targetCount, targetCount > 1 ? "s" : "");
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.7f));
            ImGui::Text("Running");
            ImGui::PopStyleColor();
        }
        
        // Profile dropdown (right-aligned)
        float dropdownWidth = 150.0f;
        ImGui::SameLine(ImGui::GetWindowWidth() - dropdownWidth - padding);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2);
        ImGui::PushItemWidth(dropdownWidth);
        ProfileDropdown("##HeaderProfile", dropdownWidth);
        ImGui::PopItemWidth();
        
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8);
        
        // Accent line under header
        ImVec2 lineStart = ImVec2(windowPos.x + padding, headerEnd.y);
        ImVec2 lineEnd = ImVec2(windowPos.x + windowSize.x - padding, headerEnd.y);
        draw_list->AddLine(lineStart, lineEnd, ImGui::GetColorU32(GetAccentColor(0.5f)), 2.0f);
        
        Spacer(8.0f);
    }
    
    bool BigToggle(const char* label, bool* value)
    {
        bool changed = false;
        
        ImGui::PushID(label);
        
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        float height = 32.0f;
        float width = 64.0f;
        float radius = height * 0.5f;
        
        ImGui::InvisibleButton("##bigtoggle", ImVec2(width, height));
        
        if (ImGui::IsItemClicked()) {
            *value = !*value;
            changed = true;
        }
        
        // Animation
        static std::unordered_map<ImGuiID, float> s_bigToggleAnim;
        ImGuiID id = ImGui::GetID("##bigtoggle");
        float target = *value ? 1.0f : 0.0f;
        float& animValue = s_bigToggleAnim[id];
        float animSpeed = 10.0f * ImGui::GetIO().DeltaTime;
        animValue = animValue + (target - animValue) * ImClamp(animSpeed, 0.0f, 1.0f);
        float t = animValue;
        
        // Glow when enabled
        if (t > 0.1f) {
            ImVec4 glowColor = GetAccentColor(0.4f * t);
            draw_list->AddRectFilled(
                ImVec2(p.x - 3, p.y - 3), 
                ImVec2(p.x + width + 3, p.y + height + 3), 
                ImGui::GetColorU32(glowColor), 
                radius + 3
            );
        }
        
        // Background
        ImVec4 bgOff = ImVec4(0.20f, 0.22f, 0.26f, 1.0f);
        ImVec4 bgOn = GetAccentColor(0.9f);
        ImU32 col_bg = ImGui::GetColorU32(ImLerp(bgOff, bgOn, t));
        
        draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, radius);
        
        // Knob
        float knobX = p.x + radius + t * (width - radius * 2.0f);
        float knobY = p.y + radius;
        draw_list->AddCircleFilled(ImVec2(knobX + 1, knobY + 1), radius - 3.0f, IM_COL32(0, 0, 0, 50));
        draw_list->AddCircleFilled(ImVec2(knobX, knobY), radius - 3.0f, IM_COL32(255, 255, 255, 255));
        
        ImGui::PopID();
        return changed;
    }
    
    bool CollapsibleSection(const char* label, bool* isOpen)
    {
        ImGui::PushID(label);
        
        static bool defaultOpen = false;
        bool* openState = isOpen ? isOpen : &defaultOpen;
        
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float width = ImGui::GetContentRegionAvail().x;
        float height = ImGui::GetFrameHeight() + 8;
        
        // Clickable area
        ImGui::InvisibleButton("##section", ImVec2(width, height));
        bool clicked = ImGui::IsItemClicked();
        bool hovered = ImGui::IsItemHovered();
        
        if (clicked) {
            *openState = !*openState;
        }
        
        // Background on hover
        if (hovered) {
            draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), 
                ImGui::GetColorU32(ImVec4(0.15f, 0.17f, 0.20f, 0.5f)), 4.0f);
        }
        
        // Arrow
        float arrowSize = 10.0f;
        ImVec2 arrowPos = ImVec2(p.x + 12, p.y + height * 0.5f);
        ImU32 arrowColor = ImGui::GetColorU32(GetAccentColor(hovered ? 1.0f : 0.7f));
        
        if (*openState) {
            // Down arrow
            draw_list->AddTriangleFilled(
                ImVec2(arrowPos.x - arrowSize * 0.5f, arrowPos.y - arrowSize * 0.3f),
                ImVec2(arrowPos.x + arrowSize * 0.5f, arrowPos.y - arrowSize * 0.3f),
                ImVec2(arrowPos.x, arrowPos.y + arrowSize * 0.4f),
                arrowColor
            );
        } else {
            // Right arrow
            draw_list->AddTriangleFilled(
                ImVec2(arrowPos.x - arrowSize * 0.3f, arrowPos.y - arrowSize * 0.5f),
                ImVec2(arrowPos.x - arrowSize * 0.3f, arrowPos.y + arrowSize * 0.5f),
                ImVec2(arrowPos.x + arrowSize * 0.4f, arrowPos.y),
                arrowColor
            );
        }
        
        // Label - use draw_list->AddText for precise positioning
        ImVec4 textColor = hovered ? GetAccentColor() : ImVec4(0.90f, 0.92f, 0.95f, 1.0f);
        float textY = p.y + (height - ImGui::GetTextLineHeight()) * 0.5f;
        draw_list->AddText(ImVec2(p.x + 28, textY), ImGui::GetColorU32(textColor), label);
        
        ImGui::PopID();
        
        return *openState;
    }
    
    bool QuickSlider(const char* label, float* value, float min, float max, const char* format)
    {
        ImGui::PushID(label);
        
        float totalWidth = ImGui::GetContentRegionAvail().x;
        float labelWidth = 110.0f;
        float valueWidth = 60.0f;
        float sliderWidth = totalWidth - labelWidth - valueWidth - 8.0f;
        if (sliderWidth < 50.0f) sliderWidth = 50.0f;
        
        // Label on left
        ImGui::AlignTextToFramePadding();
        ImGui::PushItemWidth(labelWidth);
        ImGui::Text("%s", label);
        ImGui::PopItemWidth();
        
        // Slider in middle
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(sliderWidth);
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, GetAccentColor(1.0f));
        bool changed = ImGui::SliderFloat("##slider", value, min, max, "");
        ImGui::PopStyleColor(2);
        ImGui::PopItemWidth();
        
        // Value on right
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.9f));
        ImGui::Text(format, *value);
        ImGui::PopStyleColor();
        
        ImGui::PopID();
        return changed;
    }
    
    bool QuickSliderInt(const char* label, int* value, int min, int max)
    {
        ImGui::PushID(label);
        
        float totalWidth = ImGui::GetContentRegionAvail().x;
        float labelWidth = 110.0f;
        float valueWidth = 60.0f;
        float sliderWidth = totalWidth - labelWidth - valueWidth - 8.0f;
        if (sliderWidth < 50.0f) sliderWidth = 50.0f;
        
        // Label on left
        ImGui::AlignTextToFramePadding();
        ImGui::PushItemWidth(labelWidth);
        ImGui::Text("%s", label);
        ImGui::PopItemWidth();
        
        // Slider in middle
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(sliderWidth);
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, GetAccentColor(0.9f));
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, GetAccentColor(1.0f));
        bool changed = ImGui::SliderInt("##slider", value, min, max, "");
        ImGui::PopStyleColor(2);
        ImGui::PopItemWidth();
        
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.9f));
        ImGui::Text("%d", *value);
        ImGui::PopStyleColor();
        
        ImGui::PopID();
        return changed;
    }
    
    void SectionHeader(const char* title)
    {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        float width = ImGui::GetContentRegionAvail().x;
        
        // Accent line
        draw_list->AddLine(
            ImVec2(p.x, p.y + ImGui::GetTextLineHeight() * 0.5f),
            ImVec2(p.x + 30, p.y + ImGui::GetTextLineHeight() * 0.5f),
            ImGui::GetColorU32(GetAccentColor(0.7f)), 2.0f
        );
        
        // Title
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 40);
        ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor(0.9f));
        ImGui::Text("%s", title);
        ImGui::PopStyleColor();
        
        // Accent line after
        ImGui::SameLine();
        float textEndX = ImGui::GetCursorPosX() + 10;
        draw_list->AddLine(
            ImVec2(p.x + textEndX, p.y + ImGui::GetTextLineHeight() * 0.5f),
            ImVec2(p.x + width, p.y + ImGui::GetTextLineHeight() * 0.5f),
            ImGui::GetColorU32(GetAccentColor(0.3f)), 1.0f
        );
        
        ImGui::NewLine();
        Spacer(4.0f);
    }
    
    bool TargetSelector(int* selected)
    {
        bool changed = false;
        const char* options[] = { "Head", "Body", "Auto" };
        
        ImGui::PushID("TargetSelector");
        
        float buttonWidth = (ImGui::GetContentRegionAvail().x - 20) / 3.0f;
        
        for (int i = 0; i < 3; i++) {
            if (i > 0) ImGui::SameLine();
            
            bool isSelected = (*selected == i);
            
            if (isSelected) {
                ImGui::PushStyleColor(ImGuiCol_Button, GetAccentColor(0.8f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, GetAccentColor(0.9f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, GetAccentColor(1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.17f, 0.20f, 0.9f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.22f, 0.26f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, GetAccentColor(0.5f));
            }
            
            if (ImGui::Button(options[i], ImVec2(buttonWidth, 28))) {
                *selected = i;
                changed = true;
            }
            
            ImGui::PopStyleColor(3);
        }
        
        ImGui::PopID();
        return changed;
    }
}