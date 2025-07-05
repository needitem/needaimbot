#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "../config/config.h"
#include "../mouse/input_drivers/SerialConnection.h"
#include "keyboard_listener.h"
#include "../mouse/mouse.h"
#include "keycodes.h"
#include "needaimbot.h"
#include "capture.h"


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
        vk_codes.push_back(get_vk_code(key));
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

void keyboardListener() {
    auto& ctx = AppContext::getInstance();
    std::vector<int> aim_vk_codes = get_vk_codes(ctx.config.aim_button);
    std::vector<int> shoot_vk_codes = get_vk_codes(ctx.config.shoot_button);
    std::vector<int> zoom_vk_codes = get_vk_codes(ctx.config.zoom_button);
    std::vector<int> pause_vk_codes = get_vk_codes(ctx.config.pause_button);
    std::vector<int> auto_shoot_vk_codes = get_vk_codes(ctx.config.auto_shoot_button);

    while (!ctx.shouldExit) {
        ctx.aiming = is_any_key_pressed(aim_vk_codes);
        ctx.shooting = is_any_key_pressed(shoot_vk_codes);
        ctx.zooming = is_any_key_pressed(zoom_vk_codes);
        ctx.auto_shoot_active = is_any_key_pressed(auto_shoot_vk_codes);

        if (is_any_key_pressed(pause_vk_codes)) {
            ctx.detectionPaused = !ctx.detectionPaused;
            Sleep(200); // Debounce
        }

        if (ctx.auto_shoot_active && ctx.globalMouseThread) {
            ctx.globalMouseThread->executeSilentAim();
        }

        Sleep(1);
    }
}
