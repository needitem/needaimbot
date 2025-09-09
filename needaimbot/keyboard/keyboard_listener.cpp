#include "../core/windows_headers.h"

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
// #include "../capture/capture.h" - removed, using GPU capture now



// Removed duplicate includes - already included above
// #include "../AppContext.h"
// #include "keyboard_listener.h"
// #include "../mouse/mouse.h"
// #include <iostream>
// #include <Windows.h> - using windows_headers.h instead
// #include <atomic>
// #include <string>
// #include <vector>

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
    static bool last_pause_state = false;

    while (!ctx.should_exit) {
        if (is_any_key_pressed(exit_vk_codes)) {
            ctx.should_exit = true;
            ctx.frame_cv.notify_all();  // Wake up main thread
            break;
        }

        bool current_aiming = is_any_key_pressed(aim_vk_codes);
        
        // Always update aiming state atomically
        ctx.aiming = current_aiming;
        
        // Notify pipeline thread on state change
        if (current_aiming != last_aiming_state) {
            // Wake up pipeline thread using event-driven mechanism
            ctx.pipeline_activation_cv.notify_one();  
            ctx.aiming_cv.notify_one();  // Keep for compatibility
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

        // Improved pause key handling - only toggle on key press, not while held
        bool current_pause = is_any_key_pressed(pause_vk_codes);
        if (current_pause && !last_pause_state) {
            // Key was just pressed (not held)
            if (ctx.detection_paused.load()) {
                ctx.detection_paused = false;
                std::cout << "[Keyboard] Detection RESUMED" << std::endl;
            } else {
                ctx.detection_paused = true;
                std::cout << "[Keyboard] Detection PAUSED" << std::endl;
            }
        }
        last_pause_state = current_pause;

        // Auto shoot functionality removed
        // Adaptive polling: faster when aiming, slower when idle
        if (current_aiming) {
            // Fast polling (500Hz) when aiming for better responsiveness
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } else {
            // Slow polling (100Hz) when not aiming to save CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
