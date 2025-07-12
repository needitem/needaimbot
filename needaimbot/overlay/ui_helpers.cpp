#include "ui_helpers.h"
#include <imgui/imgui_internal.h>
#include <cmath>

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

    void BeautifulCombo(const char* label, int* current_item, const char* const items[], int items_count)
    {
        ImGui::PushStyleColor(ImGuiCol_Header, GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, GetAccentColor(0.9f));
        ImGui::Combo(label, current_item, items, items_count);
        ImGui::PopStyleColor(3);
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

    void BeautifulProgressBar(float fraction, const ImVec2& size, const char* overlay)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, GetAccentColor());
        ImGui::ProgressBar(fraction, size, overlay);
        ImGui::PopStyleColor();
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

    // Layout helpers
    static float s_leftColumnWidth = 0.0f;
    static float s_rightColumnWidth = 0.0f;
    
    void BeginTwoColumnLayout(float left_width_ratio)
    {
        float available_width = ImGui::GetContentRegionAvail().x;
        s_leftColumnWidth = available_width * left_width_ratio;
        s_rightColumnWidth = available_width * (1.0f - left_width_ratio) - ImGui::GetStyle().ItemSpacing.x;
        
        ImGui::BeginChild("##left_column", ImVec2(s_leftColumnWidth, 0), false);
    }
    
    void NextColumn()
    {
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("##right_column", ImVec2(s_rightColumnWidth, 0), false);
    }
    
    void EndTwoColumnLayout()
    {
        ImGui::EndChild();
    }
    
    void BeginGroupBox(const char* title)
    {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.10f, 0.13f, 0.90f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
        
        ImGui::BeginChild(title, ImVec2(0, 0), true);
        
        if (title) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor());
            ImGui::Text("%s", title);
            ImGui::PopStyleColor();
            ImGui::Separator();
            ImGui::Spacing();
        }
    }
    
    void EndGroupBox()
    {
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
    
    void BeginCard(const char* title)
    {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.15f, 0.95f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
        
        ImGui::BeginChild(title ? title : "##card", ImVec2(0, 0), true);
        
        if (title) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetAccentColor());
            ImGui::Text("%s", title);
            ImGui::PopStyleColor();
            ImGui::Separator();
            // Removed extra spacing after title
        }
    }
    
    void EndCard()
    {
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
    
    void BeginInfoPanel()
    {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.12f, 0.18f, 0.90f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
        
        ImGui::BeginChild("##info_panel", ImVec2(0, 0), true);
    }
    
    void EndInfoPanel()
    {
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
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
        ImGui::PushItemWidth(-1);
        BeautifulSlider(label, value, min, max, format);
        ImGui::PopItemWidth();
    }
    
    void CompactCombo(const char* label, int* current_item, const char* const items[], int items_count)
    {
        ImGui::PushItemWidth(-1);
        BeautifulCombo(label, current_item, items, items_count);
        ImGui::PopItemWidth();
    }
    
    void CompactCombo(const char* label, int* current_item, bool (*getter)(void*, int, const char**), void* data, int items_count)
    {
        ImGui::PushItemWidth(-1);
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
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.10f, 0.95f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 8.0f));
        
        ImGui::BeginChild(title, ImVec2(0, 0), true);
        
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
            Spacer(2.0f);
        }
    }
    
    void EndSettingsSection()
    {
        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor();
        Spacer(6.0f);
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
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
        ImGui::Text("%s", title);
        ImGui::PopStyleColor();
        Spacer(2.0f);
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
        
        // Make slider wider
        ImGui::SetNextItemWidth(-1);
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
        
        // Make combo wider
        ImGui::SetNextItemWidth(-1);
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
}