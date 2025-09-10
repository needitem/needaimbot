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
    
    // Event-driven keyboard monitoring using Windows events
    HANDLE hKeyboardEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    
    while (!ctx.should_exit) {
        // Check for exit key
        if (is_any_key_pressed(exit_vk_codes)) {
            ctx.should_exit = true;
            ctx.frame_cv.notify_all();  // Wake up main thread
            break;
        }

        bool current_aiming = is_any_key_pressed(aim_vk_codes);
        
        // Always update aiming state atomically
        ctx.aiming = current_aiming;
        
        // Notify pipeline thread on state change (event-driven)
        if (current_aiming != last_aiming_state) {
            // Wake up pipeline thread using event-driven mechanism
            ctx.pipeline_activation_cv.notify_one();  
            ctx.aiming_cv.notify_one();  // Keep for compatibility
            last_aiming_state = current_aiming;
        }

        // Track auto_shoot button state - optimize branch
        bool current_shooting = (!ctx.config.button_auto_shoot.empty() && 
                                 ctx.config.button_auto_shoot[0] != "None") ?
                                 is_any_key_pressed(auto_shoot_vk_codes) : false;
        ctx.shooting = current_shooting;
        
        if (current_shooting != last_shooting_state) {
            last_shooting_state = current_shooting;
        }

        // Improved pause key handling - only toggle on key press, not while held
        bool current_pause = is_any_key_pressed(pause_vk_codes);
        if (current_pause && !last_pause_state) {
            // Key was just pressed (not held)
            // Toggle and print in one flow
            bool new_state = !ctx.detection_paused.load();
            ctx.detection_paused = new_state;
#ifdef _DEBUG
            std::cout << "[Keyboard] Detection " << (new_state ? "PAUSED" : "RESUMED") << std::endl;
#endif
        }
        last_pause_state = current_pause;

        // Event-driven adaptive delay - use wait with timeout for better CPU efficiency
        DWORD waitTime = current_aiming ? 2 : 10; // 2ms when aiming, 10ms when idle
        
        // Use WaitForSingleObject with timeout instead of sleep for better responsiveness
        // This allows the thread to wake immediately if signaled
        WaitForSingleObject(hKeyboardEvent, waitTime);
    }
    
    CloseHandle(hKeyboardEvent);
}
