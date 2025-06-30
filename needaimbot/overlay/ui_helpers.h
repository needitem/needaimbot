#ifndef UI_HELPERS_H
#define UI_HELPERS_H

#include <imgui/imgui.h>
#include <string>

namespace UIHelpers 
{
    void BeautifulButton(const char* label, const ImVec2& size = ImVec2(0, 0), bool active = false);
    bool BeautifulToggle(const char* label, bool* value, const char* description = nullptr);
    void BeautifulSlider(const char* label, float* value, float min, float max, const char* format = "%.3f");
    void BeautifulCombo(const char* label, int* current_item, const char* const items[], int items_count);
    void BeautifulSeparator(const char* text = nullptr);
    void BeautifulText(const char* text, ImVec4 color = ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
    void BeautifulSection(const char* title, bool* open = nullptr);
    void StatusIndicator(const char* label, bool status, const char* description = nullptr);
    void InfoTooltip(const char* description);
    void WrappedTooltip(const char* description);
    void BeautifulProgressBar(float fraction, const ImVec2& size = ImVec2(-1, 0), const char* overlay = nullptr);
    
    ImVec4 GetAccentColor(float alpha = 1.0f);
    ImVec4 GetSuccessColor(float alpha = 1.0f);
    ImVec4 GetWarningColor(float alpha = 1.0f);
    ImVec4 GetErrorColor(float alpha = 1.0f);
    
    void PushStyleColors();
    void PopStyleColors();
}

#endif