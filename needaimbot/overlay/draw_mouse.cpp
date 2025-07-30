#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <shellapi.h>
#include <algorithm>

#include "AppContext.h"
#include "needaimbot.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "ui_helpers.h"
#include "common_helpers.h"
#include "draw_settings.h" 

// GHub version check removed - not needed

static void draw_error_scaling_controls()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Error-Based Scaling");
    
    UIHelpers::BeautifulText("Reduces jitter when error is large (recoil compensation)", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    // Static temporary rules for editing
    static std::vector<Config::ErrorScalingRule> temp_rules;
    static bool has_unsaved_changes = false;
    static size_t last_config_size = 0;
    
    // Check if config has changed (e.g., after reload or profile switch)
    // This ensures temp_rules stays in sync with loaded config
    if (!has_unsaved_changes && (temp_rules.empty() || last_config_size != ctx.config.error_scaling_rules.size())) {
        temp_rules = ctx.config.error_scaling_rules;
        last_config_size = ctx.config.error_scaling_rules.size();
        has_unsaved_changes = false;
    }
    
    // Also check if the actual values have changed (in case size is same but values differ)
    if (!has_unsaved_changes && temp_rules.size() == ctx.config.error_scaling_rules.size()) {
        bool values_differ = false;
        for (size_t i = 0; i < temp_rules.size(); i++) {
            if (temp_rules[i].error_threshold != ctx.config.error_scaling_rules[i].error_threshold ||
                temp_rules[i].scale_factor != ctx.config.error_scaling_rules[i].scale_factor) {
                values_differ = true;
                break;
            }
        }
        if (values_differ) {
            temp_rules = ctx.config.error_scaling_rules;
            has_unsaved_changes = false;
        }
    }
    
    // Display current rules
    ImGui::Text("Current Scaling Rules:");
    ImGui::SameLine();
    
    // Show unsaved changes indicator
    if (has_unsaved_changes) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "(Unsaved Changes)");
    }
    
    UIHelpers::CompactSpacer();
    
    // Sort rules by threshold for display
    std::sort(temp_rules.begin(), temp_rules.end(), 
        [](const Config::ErrorScalingRule& a, const Config::ErrorScalingRule& b) {
            return a.error_threshold > b.error_threshold;
        });
    
    // Table for rules
    if (ImGui::BeginTable("error_scaling_table", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Error Threshold", ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupColumn("Scale Factor", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();
        
        for (size_t i = 0; i < temp_rules.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            
            ImGui::PushID(static_cast<int>(i));
            
            float threshold = temp_rules[i].error_threshold;
            if (ImGui::InputFloat("##threshold", &threshold, 0.0f, 0.0f, "%.0f")) {
                temp_rules[i].error_threshold = threshold;
                has_unsaved_changes = true;
            }
            
            ImGui::TableSetColumnIndex(1);
            float scale = temp_rules[i].scale_factor * 100.0f; // Convert to percentage
            if (ImGui::SliderFloat("##scale", &scale, 0.0f, 100.0f, "%.0f%%")) {
                temp_rules[i].scale_factor = scale / 100.0f;
                has_unsaved_changes = true;
            }
            
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Remove##remove")) {
                temp_rules.erase(temp_rules.begin() + i);
                has_unsaved_changes = true;
            }
            
            ImGui::PopID();
        }
        
        ImGui::EndTable();
    }
    
    UIHelpers::Spacer();
    
    // Add new rule
    static float new_threshold = 200.0f;
    static float new_scale = 30.0f;
    
    ImGui::Text("Add New Rule:");
    ImGui::PushItemWidth(100);
    ImGui::InputFloat("Threshold", &new_threshold, 0.0f, 0.0f, "%.0f");
    ImGui::SameLine();
    ImGui::InputFloat("Scale %", &new_scale, 0.0f, 0.0f, "%.0f");
    ImGui::PopItemWidth();
    ImGui::SameLine();
    
    if (ImGui::Button("Add Rule")) {
        temp_rules.push_back(Config::ErrorScalingRule(new_threshold, new_scale / 100.0f));
        has_unsaved_changes = true;
    }
    
    UIHelpers::Spacer();
    
    // Apply and Cancel buttons
    if (has_unsaved_changes) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.7f, 0.0f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.8f, 0.0f, 1.0f));
        if (ImGui::Button("Apply Changes", ImVec2(120, 0))) {
            // Apply changes to actual config
            ctx.config.error_scaling_rules = temp_rules;
            SAVE_PROFILE();
            has_unsaved_changes = false;
        }
        ImGui::PopStyleColor(2);
        
        ImGui::SameLine();
        
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.0f, 0.0f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.0f, 0.0f, 1.0f));
        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            // Revert to saved config
            temp_rules = ctx.config.error_scaling_rules;
            has_unsaved_changes = false;
        }
        ImGui::PopStyleColor(2);
    } else {
        // Reset button when no changes
        if (ImGui::Button("Reset to Defaults", ImVec2(150, 0))) {
            temp_rules.clear();
            temp_rules.push_back(Config::ErrorScalingRule(150.0f, 0.3f));
            temp_rules.push_back(Config::ErrorScalingRule(100.0f, 0.5f));
            temp_rules.push_back(Config::ErrorScalingRule(50.0f, 0.8f));
            has_unsaved_changes = true;
        }
    }
    
    UIHelpers::Spacer();
    UIHelpers::BeautifulText("Tip: Higher error thresholds apply first. When error >= threshold, movement is scaled down.", UIHelpers::GetAccentColor(0.6f));
    
    UIHelpers::EndCard();
}

static void draw_pid_controls()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("PID Controller Settings");
    
    UIHelpers::BeautifulText("PID parameters control how the aimbot tracks targets.", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::CompactSpacer();
    
    // Combined PID settings in a more compact layout
    ImGui::Columns(2, "pid_columns", false);
    
    // X-axis
    UIHelpers::SettingsSubHeader("X-Axis (Horizontal)");
    
    float kp_x = static_cast<float>(ctx.config.kp_x);
    if (UIHelpers::EnhancedSliderFloat("Kp X", &kp_x, 0.0f, 2.0f, "%.3f", "Proportional gain for X-axis. Higher values = faster response but may cause oscillation.")) {
        ctx.config.kp_x = static_cast<double>(kp_x);
        SAVE_PROFILE();
    }
    
    float ki_x = static_cast<float>(ctx.config.ki_x);
    if (UIHelpers::EnhancedSliderFloat("Ki X", &ki_x, 0.0f, 0.02f, "%.4f", "Integral gain for X-axis. Helps eliminate steady-state error.")) {
        ctx.config.ki_x = static_cast<double>(ki_x);
        SAVE_PROFILE();
    }
    
    float kd_x = static_cast<float>(ctx.config.kd_x);
    if (UIHelpers::EnhancedSliderFloat("Kd X", &kd_x, 0.0f, 0.05f, "%.4f", "Derivative gain for X-axis. Reduces overshoot and oscillation.")) {
        ctx.config.kd_x = static_cast<double>(kd_x);
        SAVE_PROFILE();
    }
    
    ImGui::NextColumn();
    
    // Y-axis
    UIHelpers::SettingsSubHeader("Y-Axis (Vertical)");
    
    float kp_y = static_cast<float>(ctx.config.kp_y);
    if (UIHelpers::EnhancedSliderFloat("Kp Y", &kp_y, 0.0f, 2.0f, "%.3f", "Proportional gain for Y-axis. Higher values = faster response but may cause oscillation.")) {
        ctx.config.kp_y = static_cast<double>(kp_y);
        SAVE_PROFILE();
    }
    
    float ki_y = static_cast<float>(ctx.config.ki_y);
    if (UIHelpers::EnhancedSliderFloat("Ki Y", &ki_y, 0.0f, 0.02f, "%.4f", "Integral gain for Y-axis. Helps eliminate steady-state error.")) {
        ctx.config.ki_y = static_cast<double>(ki_y);
        SAVE_PROFILE();
    }
    
    float kd_y = static_cast<float>(ctx.config.kd_y);
    if (UIHelpers::EnhancedSliderFloat("Kd Y", &kd_y, 0.0f, 0.05f, "%.4f", "Derivative gain for Y-axis. Reduces overshoot and oscillation.")) {
        ctx.config.kd_y = static_cast<double>(kd_y);
        SAVE_PROFILE();
    }
    
    ImGui::Columns(1);
    
    UIHelpers::Spacer();
    UIHelpers::SettingsSubHeader("Quick Presets");
    
    if (ImGui::Button("Conservative", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3, 0))) {
        ctx.config.kp_x = 0.3; ctx.config.ki_x = 0.001; ctx.config.kd_x = 0.005;
        ctx.config.kp_y = 0.3; ctx.config.ki_y = 0.001; ctx.config.kd_y = 0.005;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Safe values for most situations");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Balanced", ImVec2((ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2, 0))) {
        ctx.config.kp_x = 0.5; ctx.config.ki_x = 0.003; ctx.config.kd_x = 0.01;
        ctx.config.kp_y = 0.5; ctx.config.ki_y = 0.003; ctx.config.kd_y = 0.01;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Balanced between speed and stability");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Aggressive", ImVec2(-1, 0))) {
        ctx.config.kp_x = 0.8; ctx.config.ki_x = 0.005; ctx.config.kd_x = 0.02;
        ctx.config.kp_y = 0.8; ctx.config.ki_y = 0.005; ctx.config.kd_y = 0.02;
        SAVE_PROFILE();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fast tracking, may oscillate");
    }
    
    UIHelpers::EndCard();
}

// Advanced settings removed - keeping code simple


static void draw_input_method_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Input Method Configuration");
    
    UIHelpers::BeautifulText("Choose how mouse movements are sent to the system", UIHelpers::GetAccentColor(0.8f));
    UIHelpers::Spacer(8.0f);
    
    // Method selection with cleaner layout
    std::vector<std::string> input_methods = { "WIN32", "GHUB", "ARDUINO", "RAZER", "KMBOX" };
    std::vector<const char*> method_items;
    method_items.reserve(input_methods.size());
    for (const auto& item : input_methods) {
        method_items.push_back(item.c_str());
    }
    
    int input_method_index = 0;
    for (size_t i = 0; i < input_methods.size(); ++i) {
        if (input_methods[i] == ctx.config.input_method) {
            input_method_index = static_cast<int>(i);
            break;
        }
    }
    
    // Enhanced method selector with cleaner styling
    UIHelpers::SettingsSubHeader("Input Method");
    if (UIHelpers::EnhancedCombo("##input_method", &input_method_index, method_items.data(), static_cast<int>(method_items.size()),
                                "WIN32: Standard API (detectable)\nGHUB: Logitech driver (safer)\nARDUINO: Hardware device\nRAZER: Razer driver\nKMBOX: Hardware solution")) {
        ctx.config.input_method = input_methods[input_method_index];
        SAVE_PROFILE();
        ctx.input_method_changed.store(true);
    }
    
    UIHelpers::Spacer(12.0f);
    
    // Method-specific settings in organized sections
    if (ctx.config.input_method == "GHUB") {
        UIHelpers::SettingsSubHeader("Logitech G HUB Status");
        
        // Version check removed - just show status
        UIHelpers::StatusIndicator("G HUB", true, "Selected as input method");
        UIHelpers::BeautifulText("Make sure Logitech G HUB is installed for this to work", UIHelpers::GetAccentColor(0.7f));
    }
    
    if (ctx.config.input_method == "ARDUINO") {
        UIHelpers::SettingsSubHeader("Arduino Configuration");
        
        // COM Port setting with cleaner layout
        ImGui::Text("COM Port");
        char port_buffer[64];
        strncpy_s(port_buffer, sizeof(port_buffer), ctx.config.arduino_port.c_str(), _TRUNCATE);
        
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##arduino_port", port_buffer, sizeof(port_buffer))) {
            ctx.config.arduino_port = port_buffer;
            SAVE_PROFILE();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enter the COM port your Arduino is connected to (e.g., COM3, /dev/ttyACM0)");
        }
        
        UIHelpers::CompactSpacer();
        
        // Baud rate with cleaner presentation
        ImGui::Text("Baud Rate");
        const char* baud_rates[] = { "9600", "115200", "250000", "500000", "1000000" };
        int current_baud_index = -1;
        char custom_baud[32];
        snprintf(custom_baud, sizeof(custom_baud), "%d", ctx.config.arduino_baudrate);
        
        for (int i = 0; i < IM_ARRAYSIZE(baud_rates); i++) {
            if (std::string(baud_rates[i]) == custom_baud) {
                current_baud_index = i;
                break;
            }
        }
        
        ImGui::SetNextItemWidth(-1);
        if (current_baud_index >= 0) {
            if (UIHelpers::EnhancedCombo("##arduino_baud_combo", &current_baud_index, baud_rates, IM_ARRAYSIZE(baud_rates),
                                        "115200: Standard speed\n250000-1000000: High speed (requires compatible firmware)")) {
                ctx.config.arduino_baudrate = std::stoi(baud_rates[current_baud_index]);
                SAVE_PROFILE();
                ctx.input_method_changed.store(true);
            }
        } else {
            if (ImGui::InputInt("##arduino_baud", &ctx.config.arduino_baudrate, 0)) {
                SAVE_PROFILE();
                ctx.input_method_changed.store(true);
            }
        }
        
        UIHelpers::Spacer(8.0f);
        
        if (UIHelpers::EnhancedCheckbox("Use 16-bit Mouse Movement", &ctx.config.arduino_16_bit_mouse,
                                       "Send mouse data as 16-bit values (requires compatible firmware). Higher precision but needs firmware support.")) {
            SAVE_PROFILE();
        }
    }
    
    if (ctx.config.input_method == "KMBOX") {
        UIHelpers::SettingsSubHeader("KmboxNet Hardware");
        UIHelpers::StatusIndicator("Hardware Required", true, "Ensure B-box hardware is connected and kmNet library is initialized");
        UIHelpers::BeautifulText("Hardware-based input simulation for maximum safety", UIHelpers::GetSuccessColor());
    }
    
    if (ctx.config.input_method == "RAZER") {
        UIHelpers::SettingsSubHeader("Razer Synapse");
        UIHelpers::StatusIndicator("Driver Required", true, "Requires Razer Synapse to be installed and running");
    }
    
    if (ctx.config.input_method == "WIN32") {
        UIHelpers::SettingsSubHeader("Windows API Warning");
        UIHelpers::StatusIndicator("Detection Risk", false, "This method uses standard Windows API which may be detected by anti-cheat systems");
        UIHelpers::BeautifulText("Consider using hardware-based methods for better safety", UIHelpers::GetWarningColor());
    }
    
    UIHelpers::EndCard();
}

static void draw_hotkey_section(const char* title, std::vector<std::string>& hotkeys, const char* add_id)
{
    // Create child window for better space management
    ImGui::BeginChild((std::string("hotkey_section_") + add_id).c_str(), ImVec2(0, 0), false);
    
    for (size_t i = 0; i < hotkeys.size(); )
    {
        std::string& current_key_name = hotkeys[i];
        
        int current_index = -1;
        for (size_t k = 0; k < key_names.size(); ++k)
        {
            if (key_names[k] == current_key_name)
            {
                current_index = static_cast<int>(k);
                break;
            }
        }
        
        if (current_index == -1)
        {
            current_index = 0;
        }
        
        // Use unique ID combining section name and index
        std::string unique_id = std::string(add_id) + "_" + std::to_string(i);
        ImGui::PushID(unique_id.c_str());
        
        // Calculate proper button width
        float remove_button_width = 80.0f;
        float available_width = ImGui::GetContentRegionAvail().x;
        float combo_width = available_width - remove_button_width - ImGui::GetStyle().ItemSpacing.x;
        
        // Enhanced key selector with better styling and wider width
        ImGui::SetNextItemWidth(combo_width);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Button, UIHelpers::GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, UIHelpers::GetAccentColor(0.8f));
        ImGui::PushStyleColor(ImGuiCol_Header, UIHelpers::GetAccentColor(0.7f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, UIHelpers::GetAccentColor(0.8f));
        
        std::string combo_label = "##hotkey_combo_" + unique_id;
        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            AppContext::getInstance().config.saveConfig();
        }
        ImGui::PopStyleColor(6);
        
        ImGui::SameLine();
        std::string remove_button_label = "Remove##" + unique_id;
        if (UIHelpers::BeautifulButton(remove_button_label.c_str(), ImVec2(remove_button_width, 0)))
        {
            if (hotkeys.size() <= 1)
            {
                hotkeys[0] = std::string("None");
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            }
            else
            {
                hotkeys.erase(hotkeys.begin() + i);
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            }
        }
        
        ImGui::PopID();
        ++i;
    }
    
    UIHelpers::CompactSpacer();
    std::string add_button_label = "Add Key##" + std::string(add_id);
    if (UIHelpers::BeautifulButton(add_button_label.c_str(), ImVec2(-1, 0)))
    {
        hotkeys.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
    
    ImGui::EndChild();
}

static void draw_aiming_settings()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginCard("Aiming Controls");
    
    // Targeting Section
    UIHelpers::SettingsSubHeader("Targeting Controls");
    UIHelpers::BeautifulText("Configure keys for aimbot activation", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    draw_hotkey_section("Aimbot Activation Keys", ctx.config.button_targeting, "targeting_keys");
    
    UIHelpers::Spacer(8.0f);
    
    // Auto Shoot Section
    UIHelpers::SettingsSubHeader("Auto Shooting");
    UIHelpers::BeautifulText("Automatically shoot when targeting enemies", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    draw_hotkey_section("Auto Shoot Keys", ctx.config.button_auto_shoot, "auto_shoot_keys");
    
    UIHelpers::Spacer(8.0f);
    
    // Movement Restrictions
    UIHelpers::SettingsSubHeader("Movement Restrictions");
    UIHelpers::BeautifulText("Control when aimbot should avoid certain movements", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    draw_hotkey_section("Disable Upward Aim Keys", ctx.config.button_disable_upward_aim, "disable_upward_keys");
    
    UIHelpers::EndCard();
    
    UIHelpers::CompactSpacer();
    
    // Separate card for Triggerbot to give more space
    UIHelpers::BeginCard("Triggerbot Configuration");
    
    UIHelpers::BeautifulText("Configure the screen area where triggerbot activates", UIHelpers::GetAccentColor(0.7f));
    UIHelpers::Spacer(6.0f);
    
    ImGui::Text("Area Size Multiplier");
    if (UIHelpers::EnhancedSliderFloat("##triggerbot_area", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f",
                                      "Defines the central screen area size where Triggerbot activates.\nSmaller value = larger area\nLarger value = smaller area\n(1.0 = default area)")) {
        SAVE_PROFILE();
    }
    
    UIHelpers::EndCard();
}

void draw_mouse()
{
    auto& ctx = AppContext::getInstance();
    
    UIHelpers::BeginTwoColumnLayout(0.65f);
    
    // Left column - PID settings and error scaling
    
    draw_pid_controls();
    UIHelpers::CompactSpacer();
    
    draw_error_scaling_controls();
    UIHelpers::CompactSpacer();
    
    draw_input_method_settings();
    
    UIHelpers::NextColumn();
    
    // Right column - Aiming settings and tips
    draw_aiming_settings();
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeginInfoPanel();
    
    UIHelpers::BeautifulText("PID Tuning Tips", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("Start with Kp (proportional) for basic tracking");
    ImGui::BulletText("Add Kd (derivative) to reduce oscillation");
    ImGui::BulletText("Use Ki (integral) sparingly to fix drift");
    ImGui::BulletText("Lower values = smoother, Higher values = faster");
    
    UIHelpers::CompactSpacer();
    
    UIHelpers::BeautifulText("Input Method Guide", UIHelpers::GetAccentColor());
    UIHelpers::CompactSpacer();
    
    ImGui::BulletText("WIN32: Standard method, may be detected");
    ImGui::BulletText("GHUB: Logitech driver, generally safer");
    ImGui::BulletText("ARDUINO: Hardware-based, requires setup");
    ImGui::BulletText("KMBOX: Hardware solution, very safe");
    
    UIHelpers::EndInfoPanel();
    
    UIHelpers::EndTwoColumnLayout();
}
