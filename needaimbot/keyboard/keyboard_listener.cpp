#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "../config/config.h"
#include "../AppContext.h"
#include "../mouse/input_drivers/SerialConnection.h"
#include "keyboard_listener.h"
#include "../mouse/mouse.h"
#include "keycodes.h"
#include "needaimbot.h"
#include "../capture/capture.h"



#include "../AppContext.h"
#include "keyboard_listener.h"
#include "../mouse/mouse.h"
#include <iostream>
#include <Windows.h>
#include <atomic>
#include <string>
#include <vector>

std::vector<int> get_vk_codes(const std::vector<std::string>& keys) {
    std::vector<int> vk_codes;
    for (const auto& key : keys) {
        vk_codes.push_back(KeyCodes::getKeyCode(key));
    }
    return vk_codes;
}

bool is_any_key_pressed(const std::vector<int>& vk_codes) {
    for (int code : vk_codes) {
        if (code != 0 && (GetAsyncKeyState(code) & 0x8000)) {
            return true;
        }
    }
    return false;
}

bool isAnyKeyPressed(const std::vector<std::string>& keys) {
    std::vector<int> vk_codes = get_vk_codes(keys);
    return is_any_key_pressed(vk_codes);
}

void keyboardListener() {
    auto& ctx = AppContext::getInstance();
    std::vector<int> aim_vk_codes = get_vk_codes(ctx.config.button_targeting);
    std::vector<int> pause_vk_codes = get_vk_codes(ctx.config.button_pause);
    std::vector<int> auto_shoot_vk_codes = get_vk_codes(ctx.config.button_auto_shoot);
    std::vector<int> exit_vk_codes = get_vk_codes(ctx.config.button_exit);


    static bool last_aiming_state = false;
    static bool last_shooting_state = false;

    while (!ctx.should_exit) {
        if (is_any_key_pressed(exit_vk_codes)) {
            ctx.should_exit = true;
            break;
        }

        bool current_aiming = is_any_key_pressed(aim_vk_codes);
        ctx.aiming = current_aiming;
        
        if (current_aiming != last_aiming_state) {
            last_aiming_state = current_aiming;
        }

        // Track auto_shoot button state
        // If button_auto_shoot is empty or "None", shooting is always false
        bool current_shooting = false;
        if (!ctx.config.button_auto_shoot.empty() && 
            ctx.config.button_auto_shoot[0] != "None") {
            current_shooting = is_any_key_pressed(auto_shoot_vk_codes);
        }
        ctx.shooting = current_shooting;
        
        if (current_shooting != last_shooting_state) {
            last_shooting_state = current_shooting;
        }

        if (is_any_key_pressed(pause_vk_codes)) {
            ctx.detectionPaused = !ctx.detectionPaused;
            Sleep(50); // Shorter debounce for faster response
        }

        // Auto shoot functionality removed
        // Add small delay to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
