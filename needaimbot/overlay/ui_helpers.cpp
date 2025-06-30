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

    void BeautifulButton(const char* label, const ImVec2& size, bool active)
    {
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, GetAccentColor(0.9f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, GetAccentColor(1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, GetAccentColor(0.8f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.18f, 0.90f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.20f, 0.25f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.25f, 0.30f, 1.00f));
        }
        
        ImGui::Button(label, size);
        ImGui::PopStyleColor(3);
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
}