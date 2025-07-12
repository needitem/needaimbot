#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "AppContext.h"
#include "imgui/imgui.h"
#include "needaimbot.h"
#include "ui_helpers_new.h"
#include "overlay.h"

extern std::vector<std::string> key_names;
extern std::vector<const char*> key_names_cstrs;

static void draw_hotkeys(const char* title, std::vector<std::string>& keys)
{
    UI::Section(title);
    
    for (size_t i = 0; i < keys.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        
        int current = 0;
        for (size_t k = 0; k < key_names.size(); ++k) {
            if (key_names[k] == keys[i]) {
                current = static_cast<int>(k);
                break;
            }
        }
        
        float width = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth(width - 60);
        
        if (ImGui::Combo("##key", &current, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size()))) {
            keys[i] = key_names[current];
            AppContext::getInstance().config.saveConfig();
        }
        
        ImGui::SameLine();
        if (UI::Button("X", 50)) {
            if (keys.size() > 1) {
                keys.erase(keys.begin() + i);
                AppContext::getInstance().config.saveConfig();
                ImGui::PopID();
                continue;
            } else {
                keys[0] = "None";
                AppContext::getInstance().config.saveConfig();
            }
        }
        
        ImGui::PopID();
    }
    
    UI::SmallSpace();
    std::string button_label = std::string("Add Key##") + title;
    if (UI::Button(button_label.c_str(), -1)) {
        keys.push_back("None");
        AppContext::getInstance().config.saveConfig();
    }
}

void draw_mouse_new()
{
    auto& ctx = AppContext::getInstance();
    
    UI::Space();
    
    UI::BeginColumns(0.55f);
    
    // Left Column - Controller Settings
    UI::Section("Controller Type");
    
    if (UI::Toggle("Predictive Controller", &ctx.config.use_predictive_controller)) {
        ctx.config.saveConfig();
    }
    UI::Tip("Kalman filter for better tracking");
    
    UI::Space();
    UI::Space();
    
    // PID Settings
    UI::Section("PID Tuning");
    
    // X-axis
    ImGui::Text("X-Axis");
    float kp_x = static_cast<float>(ctx.config.kp_x);
    if (UI::Slider("Kp##X", &kp_x, 0.0f, 2.0f, "%.3f")) {
        ctx.config.kp_x = kp_x;
        ctx.config.saveConfig();
    }
    
    float ki_x = static_cast<float>(ctx.config.ki_x);
    if (UI::Slider("Ki##X", &ki_x, 0.0f, 0.1f, "%.3f")) {
        ctx.config.ki_x = ki_x;
        ctx.config.saveConfig();
    }
    
    float kd_x = static_cast<float>(ctx.config.kd_x);
    if (UI::Slider("Kd##X", &kd_x, 0.0f, 0.1f, "%.3f")) {
        ctx.config.kd_x = kd_x;
        ctx.config.saveConfig();
    }
    
    UI::SmallSpace();
    
    // Y-axis
    ImGui::Text("Y-Axis");
    float kp_y = static_cast<float>(ctx.config.kp_y);
    if (UI::Slider("Kp##Y", &kp_y, 0.0f, 2.0f, "%.3f")) {
        ctx.config.kp_y = kp_y;
        ctx.config.saveConfig();
    }
    
    float ki_y = static_cast<float>(ctx.config.ki_y);
    if (UI::Slider("Ki##Y", &ki_y, 0.0f, 0.1f, "%.3f")) {
        ctx.config.ki_y = ki_y;
        ctx.config.saveConfig();
    }
    
    float kd_y = static_cast<float>(ctx.config.kd_y);
    if (UI::Slider("Kd##Y", &kd_y, 0.0f, 0.1f, "%.3f")) {
        ctx.config.kd_y = kd_y;
        ctx.config.saveConfig();
    }
    
    UI::Space();
    UI::Space();
    
    // Advanced Settings
    UI::Section("Advanced");
    
    if (UI::Toggle("Adaptive PID", &ctx.config.enable_adaptive_pid)) {
        ctx.config.saveConfig();
    }
    
    if (UI::Slider("Smoothing##move", &ctx.config.movement_smoothing, 0.0f, 0.6f, "%.3f")) {
        ctx.config.saveConfig();
    }
    
    if (UI::Toggle("Sub-pixel Dither", &ctx.config.enable_subpixel_dithering)) {
        ctx.config.saveConfig();
    }
    
    UI::Space();
    UI::Space();
    
    // Input Method
    UI::Section("Input Method");
    
    const char* methods[] = { "WIN32", "GHUB", "ARDUINO", "RAZER", "KMBOX" };
    int method = 0;
    if (ctx.config.input_method == "WIN32") method = 0;
    else if (ctx.config.input_method == "GHUB") method = 1;
    else if (ctx.config.input_method == "ARDUINO") method = 2;
    else if (ctx.config.input_method == "RAZER") method = 3;
    else if (ctx.config.input_method == "KMBOX") method = 4;
    
    if (UI::Combo("Method##input", &method, methods, 5)) {
        ctx.config.input_method = methods[method];
        ctx.config.saveConfig();
        ctx.input_method_changed.store(true);
    }
    
    // Arduino specific
    if (ctx.config.input_method == "ARDUINO") {
        UI::SmallSpace();
        
        char port[64];
        strncpy_s(port, ctx.config.arduino_port.c_str(), _TRUNCATE);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##port", port, sizeof(port))) {
            ctx.config.arduino_port = port;
            ctx.config.saveConfig();
        }
        UI::Tip("COM port (e.g., COM3)");
        
        int baud = ctx.config.arduino_baudrate;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputInt("##baud", &baud, 0)) {
            ctx.config.arduino_baudrate = baud;
            ctx.config.saveConfig();
        }
        UI::Tip("Baud rate (e.g., 115200)");
    }
    
    UI::NextColumn();
    
    // Right Column - Hotkeys
    draw_hotkeys("Targeting Keys", ctx.config.button_targeting);
    UI::Space();
    
    draw_hotkeys("Auto Shoot Keys", ctx.config.button_auto_shoot);
    UI::Space();
    
    UI::Section("Triggerbot");
    
    if (UI::Slider("Area Size##trigger", &ctx.config.bScope_multiplier, 0.1f, 2.0f, "%.2f")) {
        ctx.config.saveConfig();
    }
    UI::Tip("Central area for triggerbot");
    
    UI::Space();
    
    draw_hotkeys("Disable Upward", ctx.config.button_disable_upward_aim);
    
    UI::EndColumns();
    
    UI::Space();
}