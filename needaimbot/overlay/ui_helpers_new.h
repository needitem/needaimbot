#pragma once
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <string>

namespace UI 
{
    // Colors
    inline ImVec4 AccentColor(float alpha = 1.0f) { return ImVec4(0.20f, 0.60f, 0.90f, alpha); }
    inline ImVec4 SuccessColor(float alpha = 1.0f) { return ImVec4(0.40f, 0.80f, 0.40f, alpha); }
    inline ImVec4 WarningColor(float alpha = 1.0f) { return ImVec4(1.00f, 0.70f, 0.00f, alpha); }
    inline ImVec4 ErrorColor(float alpha = 1.0f) { return ImVec4(0.90f, 0.30f, 0.30f, alpha); }
    
    // Minimal spacing
    inline void Space() { ImGui::Dummy(ImVec2(0, 4)); }
    inline void SmallSpace() { ImGui::Dummy(ImVec2(0, 2)); }
    
    // Compact section header
    inline void Section(const char* title) {
        ImGui::PushStyleColor(ImGuiCol_Text, AccentColor());
        ImGui::Text("%s", title);
        ImGui::PopStyleColor();
        
        // Draw underline
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetItemRectMin();
        ImVec2 pos2 = ImVec2(pos.x + ImGui::CalcTextSize(title).x, pos.y + ImGui::GetTextLineHeight());
        draw_list->AddLine(ImVec2(pos.x, pos2.y), ImVec2(pos2.x, pos2.y), ImGui::GetColorU32(AccentColor(0.3f)), 1.0f);
        SmallSpace();
    }
    
    // Compact toggle switch
    inline bool Toggle(const char* label, bool* value) {
        ImGui::PushID(label);
        
        float height = ImGui::GetFrameHeight() * 0.8f;
        float width = height * 1.6f;
        
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        ImGui::InvisibleButton("##toggle", ImVec2(width, height));
        bool clicked = ImGui::IsItemClicked();
        if (clicked) *value = !*value;
        
        float t = *value ? 1.0f : 0.0f;
        ImVec4 col_off = ImVec4(0.20f, 0.20f, 0.25f, 1.0f);
        ImVec4 col_on = AccentColor();
        ImVec4 col = ImVec4(
            col_off.x + (col_on.x - col_off.x) * t,
            col_off.y + (col_on.y - col_off.y) * t,
            col_off.z + (col_on.z - col_off.z) * t,
            col_off.w + (col_on.w - col_off.w) * t
        );
        ImU32 col_bg = ImGui::GetColorU32(col);
        
        draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
        draw_list->AddCircleFilled(ImVec2(p.x + height * 0.5f + t * (width - height), p.y + height * 0.5f), 
                                  height * 0.4f, IM_COL32(255, 255, 255, 255));
        
        ImGui::SameLine();
        ImGui::Text("%s", label);
        
        ImGui::PopID();
        return clicked;
    }
    
    // Compact slider that uses full width
    inline bool Slider(const char* label, float* value, float min, float max, const char* format = "%.2f") {
        ImGui::PushID(label);
        
        // Calculate sizes
        float label_width = ImGui::CalcTextSize(label).x;
        float value_width = ImGui::CalcTextSize("9999.99").x; // Max expected value display
        float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
        float total_width = ImGui::GetContentRegionAvail().x;
        float slider_width = total_width - label_width - value_width - spacing * 2;
        
        // Label
        ImGui::Text("%s", label);
        ImGui::SameLine();
        
        // Slider
        ImGui::SetNextItemWidth(slider_width);
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, AccentColor());
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, AccentColor(0.8f));
        bool changed = ImGui::SliderFloat("##slider", value, min, max, "");
        ImGui::PopStyleColor(2);
        
        // Value display
        ImGui::SameLine();
        ImGui::Text(format, *value);
        
        ImGui::PopID();
        return changed;
    }
    
    // Compact combo box
    inline bool Combo(const char* label, int* current, const char* const items[], int items_count) {
        ImGui::PushID(label);
        
        float label_width = ImGui::CalcTextSize(label).x;
        float total_width = ImGui::GetContentRegionAvail().x;
        float combo_width = total_width - label_width - ImGui::GetStyle().ItemInnerSpacing.x;
        
        ImGui::Text("%s", label);
        ImGui::SameLine();
        
        ImGui::SetNextItemWidth(combo_width);
        ImGui::PushStyleColor(ImGuiCol_Header, AccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, AccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, AccentColor(0.9f));
        bool changed = ImGui::Combo("##combo", current, items, items_count);
        ImGui::PopStyleColor(3);
        
        ImGui::PopID();
        return changed;
    }
    
    // Tooltip helper
    inline void Tip(const char* text) {
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(text);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
    
    // Warning text
    inline void Warning(const char* text) {
        ImGui::TextColored(WarningColor(), "%s", text);
    }
    
    // Compact button
    inline bool Button(const char* label, float width = 0) {
        if (width < 0) width = ImGui::GetContentRegionAvail().x;
        
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.18f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, AccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, AccentColor(0.9f));
        
        bool clicked = ImGui::Button(label, ImVec2(width, 0));
        
        ImGui::PopStyleColor(3);
        return clicked;
    }
    
    // Two column layout helpers
    inline void BeginColumns(float left_ratio = 0.5f) {
        ImGui::Columns(2, nullptr, false);
        ImGui::SetColumnWidth(0, ImGui::GetContentRegionAvail().x * left_ratio);
    }
    
    inline void NextColumn() {
        ImGui::NextColumn();
    }
    
    inline void EndColumns() {
        ImGui::Columns(1);
    }
}