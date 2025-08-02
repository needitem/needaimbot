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


const float MIN_OFFSET_Y = 0.0f;
const float MAX_OFFSET_Y = 1.0f;
const float MIN_NORECOIL_STRENGTH = 0.1f;
const float MAX_NORECOIL_STRENGTH = 500.0f;

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

    // Debug: Print targeting keys
    std::cout << "[Keyboard] Targeting keys: ";
    for (const auto& key : ctx.config.button_targeting) {
        std::cout << key << " ";
    }
    std::cout << std::endl;

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
        bool current_shooting = is_any_key_pressed(auto_shoot_vk_codes);
        ctx.shooting = current_shooting;
        
        if (current_shooting != last_shooting_state) {
            last_shooting_state = current_shooting;
        }

        if (is_any_key_pressed(pause_vk_codes)) {
            ctx.detectionPaused = !ctx.detectionPaused;
            Sleep(200); // Debounce
        }

        // Auto shoot functionality removed

        // Recoil strength adjustment with arrow keys
        static DWORD last_adjust_time = 0;
        DWORD current_time = GetTickCount();
        
        if (current_time - last_adjust_time > 100) {  // 100ms debounce
            if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
                if (ctx.config.easynorecoilstrength > MIN_NORECOIL_STRENGTH) {
                    ctx.config.easynorecoilstrength -= ctx.config.norecoil_step;
                    if (ctx.config.easynorecoilstrength < MIN_NORECOIL_STRENGTH) {
                        ctx.config.easynorecoilstrength = MIN_NORECOIL_STRENGTH;
                    }
                    std::cout << "[Keyboard] Recoil strength: " << ctx.config.easynorecoilstrength << std::endl;
                    ctx.config.saveConfig();
                    last_adjust_time = current_time;
                }
            }
            else if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
                if (ctx.config.easynorecoilstrength < MAX_NORECOIL_STRENGTH) {
                    ctx.config.easynorecoilstrength += ctx.config.norecoil_step;
                    if (ctx.config.easynorecoilstrength > MAX_NORECOIL_STRENGTH) {
                        ctx.config.easynorecoilstrength = MAX_NORECOIL_STRENGTH;
                    }
                    std::cout << "[Keyboard] Recoil strength: " << ctx.config.easynorecoilstrength << std::endl;
                    ctx.config.saveConfig();
                    last_adjust_time = current_time;
                }
            }
        }

        Sleep(1);
    }
}
