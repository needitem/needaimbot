#ifndef UI_HELPERS_H
#define UI_HELPERS_H

#include "../imgui/imgui.h"
#include <string>

// Runtime string builder to avoid static strings in binary
namespace UIStrings {
    inline std::string TabMain() { 
        std::string s; s += 'T'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'i'; s += 'n'; s += 'g';
        return s;
    }
    inline std::string CardControls() {
        std::string s; s += 'T'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'i'; s += 'n'; s += 'g';
        s += ' '; s += 'C'; s += 'o'; s += 'n'; s += 't'; s += 'r'; s += 'o'; s += 'l'; s += 's';
        return s;
    }
    inline std::string OffsetDesc() {
        std::string s; s += 'A'; s += 'd'; s += 'j'; s += 'u'; s += 's'; s += 't'; s += ' ';
        s += 'w'; s += 'h'; s += 'e'; s += 'r'; s += 'e'; s += ' '; s += 't'; s += 'h'; s += 'e'; s += ' ';
        s += 't'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'e'; s += 'r'; s += ' ';
        s += 't'; s += 'a'; s += 'r'; s += 'g'; s += 'e'; s += 't'; s += 's';
        return s;
    }
    inline std::string HotkeyActivation() {
        std::string s; s += 'T'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'i'; s += 'n'; s += 'g';
        s += ' '; s += 'A'; s += 'c'; s += 't'; s += 'i'; s += 'v'; s += 'a'; s += 't'; s += 'i'; s += 'o'; s += 'n';
        return s;
    }
    inline std::string HotkeyActivationDesc() {
        std::string s; s += 'K'; s += 'e'; s += 'y'; s += 's'; s += ' '; s += 't'; s += 'h'; s += 'a'; s += 't'; s += ' ';
        s += 'a'; s += 'c'; s += 't'; s += 'i'; s += 'v'; s += 'a'; s += 't'; s += 'e'; s += ' ';
        s += 't'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'i'; s += 'n'; s += 'g';
        return s;
    }
    inline std::string HotkeyDisableUpward() {
        std::string s; s += 'P'; s += 'r'; s += 'e'; s += 'v'; s += 'e'; s += 'n'; s += 't'; s += ' ';
        s += 'u'; s += 'p'; s += 'w'; s += 'a'; s += 'r'; s += 'd'; s += ' '; s += 't'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'i'; s += 'n'; s += 'g';
        return s;
    }
    inline std::string HotkeyPause() {
        std::string s; s += 'T'; s += 'e'; s += 'm'; s += 'p'; s += 'o'; s += 'r'; s += 'a'; s += 'r'; s += 'i'; s += 'l'; s += 'y'; s += ' ';
        s += 'p'; s += 'a'; s += 'u'; s += 's'; s += 'e'; s += ' '; s += 't'; s += 'r'; s += 'a'; s += 'c'; s += 'k'; s += 'i'; s += 'n'; s += 'g';
        return s;
    }
}

namespace UIHelpers 
{
    bool BeautifulToggle(const char* label, bool* value, const char* description = nullptr);
    bool BeautifulSlider(const char* label, float* value, float min, float max, const char* format = "%.3f");
    void BeautifulSeparator(const char* text = nullptr);
    void BeautifulText(const char* text, ImVec4 color = ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
    void TextColored(const ImVec4& color, const char* text);
    void BeautifulSection(const char* title, bool* open = nullptr);
    void StatusIndicator(const char* label, bool status, const char* description = nullptr);
    void InfoTooltip(const char* description);
    void WrappedTooltip(const char* description);
    
    // Layout helpers
    void BeginCard(const char* title = nullptr);
    void EndCard();
    void Spacer(float height = 5.0f);
    void CompactSpacer();
    bool BeautifulButton(const char* label, const ImVec2& size = ImVec2(-1, 0));
    void CompactSlider(const char* label, float* value, float min, float max, const char* format = "%.2f");
    void CompactCombo(const char* label, int* current_item, const char* const items[], int items_count);
    void CompactCombo(const char* label, int* current_item, bool (*getter)(void*, int, const char**), void* data, int items_count);
    
    // Section helpers for better organization
    void BeginSettingsSection(const char* title, const char* description = nullptr);
    void EndSettingsSection();
    void SettingsHeader(const char* title);
    void SettingsSubHeader(const char* title);
    void HelpMarker(const char* desc);
    void SettingsRow(const char* label, float label_width = 120.0f);
    void SettingsValue(const char* value);
    
    // Enhanced controls with better visibility
    bool EnhancedSliderFloat(const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", const char* description = nullptr);
    bool EnhancedCombo(const char* label, int* current_item, const char* const items[], int items_count, const char* description = nullptr);
    bool EnhancedCheckbox(const char* label, bool* v, const char* description = nullptr);
    bool EnhancedButton(const char* label, const ImVec2& size = ImVec2(0, 0), const char* description = nullptr);
    
    ImVec4 GetAccentColor(float alpha = 1.0f);
    ImVec4 GetSuccessColor(float alpha = 1.0f);
    ImVec4 GetWarningColor(float alpha = 1.0f);
    ImVec4 GetErrorColor(float alpha = 1.0f);
    
    void PushStyleColors();
    void PopStyleColors();
    
    // Profile dropdown helper
    bool ProfileDropdown(const char* combo_label = "##ProfileDropdown", float width = 200.0f);
    
    // Weapon profile dropdown helper
    bool WeaponProfileDropdown(const char* combo_label = "##WeaponDropdown", float width = 200.0f);
}

#endif