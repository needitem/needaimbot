#ifndef UI_HELPERS_H
#define UI_HELPERS_H

#include <imgui/imgui.h>
#include <string>

namespace UIHelpers 
{
    bool BeautifulToggle(const char* label, bool* value, const char* description = nullptr);
    bool BeautifulSlider(const char* label, float* value, float min, float max, const char* format = "%.3f");
    void BeautifulCombo(const char* label, int* current_item, const char* const items[], int items_count);
    void BeautifulSeparator(const char* text = nullptr);
    void BeautifulText(const char* text, ImVec4 color = ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
    void BeautifulSection(const char* title, bool* open = nullptr);
    void StatusIndicator(const char* label, bool status, const char* description = nullptr);
    void InfoTooltip(const char* description);
    void WrappedTooltip(const char* description);
    void BeautifulProgressBar(float fraction, const ImVec2& size = ImVec2(-1, 0), const char* overlay = nullptr);
    
    // Layout helpers
    void BeginTwoColumnLayout(float left_width_ratio = 0.6f);
    void NextColumn();
    void EndTwoColumnLayout();
    void BeginGroupBox(const char* title);
    void EndGroupBox();
    void BeginCard(const char* title = nullptr);
    void EndCard();
    void BeginInfoPanel();
    void EndInfoPanel();
    void Spacer(float height = 10.0f);
    bool BeautifulButton(const char* label, const ImVec2& size = ImVec2(-1, 0));
    void CompactSlider(const char* label, float* value, float min, float max, const char* format = "%.2f");
    void CompactCombo(const char* label, int* current_item, const char* const items[], int items_count);
    void CompactCombo(const char* label, int* current_item, bool (*getter)(void*, int, const char**), void* data, int items_count);
    
    ImVec4 GetAccentColor(float alpha = 1.0f);
    ImVec4 GetSuccessColor(float alpha = 1.0f);
    ImVec4 GetWarningColor(float alpha = 1.0f);
    ImVec4 GetErrorColor(float alpha = 1.0f);
    
    void PushStyleColors();
    void PopStyleColors();
}

#endif