#include "../core/windows_headers.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "../config/config.h"
#include "../AppContext.h"
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
    vk_codes.reserve(keys.size());  // Pre-allocate for better performance
    for (const auto& key : keys) {
        vk_codes.push_back(KeyCodes::getKeyCode(key));
    }
    return vk_codes;
}

// Parse key combo string (e.g., "LeftMouseButton+LeftShift") into vk codes
std::vector<int> parse_key_combo(const std::string& combo) {
    std::vector<int> vk_codes;
    std::string current;
    for (char c : combo) {
        if (c == '+') {
            if (!current.empty()) {
                vk_codes.push_back(KeyCodes::getKeyCode(current));
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        vk_codes.push_back(KeyCodes::getKeyCode(current));
    }
    return vk_codes;
}

// Check if all keys in combo are pressed (AND condition)
bool is_combo_pressed(const std::vector<int>& combo_vk_codes) {
    if (combo_vk_codes.empty()) return false;
    for (int code : combo_vk_codes) {
        if (!code || !(GetAsyncKeyState(code) & 0x8000)) {
            return false;
        }
    }
    return true;
}

bool is_any_key_pressed(const std::vector<int>& vk_codes) {
    // Check most likely keys first for early exit
    for (int code : vk_codes) {
        if (code && (GetAsyncKeyState(code) & 0x8000)) {
            return true;
        }
    }
    return false;
}

// Cached key combo structure to avoid repeated parsing
struct CachedKeyCombo {
    std::vector<std::vector<int>> combos;  // Each entry is a combo (single key = 1 element)
    size_t config_hash = 0;

    void update(const std::vector<std::string>& keys) {
        // Simple hash based on concatenated strings
        size_t new_hash = 0;
        for (const auto& k : keys) {
            for (char c : k) new_hash = new_hash * 31 + c;
        }

        if (new_hash == config_hash && !combos.empty()) return;  // No change

        config_hash = new_hash;
        combos.clear();
        combos.reserve(keys.size());

        for (const auto& key : keys) {
            if (key == "None" || key.empty()) continue;
            combos.push_back(parse_key_combo(key));
        }
    }

    bool isPressed() const {
        for (const auto& combo : combos) {
            if (is_combo_pressed(combo)) return true;
        }
        return false;
    }
};

// Check if any key combo is pressed (supports both single keys and combos with +)
// Returns true if ANY of the configured keys/combos is active
bool is_any_key_or_combo_pressed(const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        if (key == "None" || key.empty()) continue;

        // Check if this is a combo (contains +)
        if (key.find('+') != std::string::npos) {
            std::vector<int> combo_codes = parse_key_combo(key);
            if (is_combo_pressed(combo_codes)) {
                return true;
            }
        } else {
            // Single key
            int code = KeyCodes::getKeyCode(key);
            if (code && (GetAsyncKeyState(code) & 0x8000)) {
                return true;
            }
        }
    }
    return false;
}

bool isAnyKeyPressed(const std::vector<std::string>& keys) {
    return is_any_key_or_combo_pressed(keys);
}

// 2PC Architecture: Inference PC only handles control keys (exit, pause)
// Aim/shoot keys are read from Makcu device via serial communication
void keyboardListener() {
    auto& ctx = AppContext::getInstance();

    static bool last_aiming_state = false;
    static bool last_shooting_state = false;
    static bool last_pause_state = false;

    // Cached key combos for control keys only
    CachedKeyCombo cache_exit, cache_pause;

    // Event-driven keyboard monitoring
    HANDLE hKeyboardEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

    while (!ctx.should_exit) {
        // Update control key caches
        cache_exit.update(ctx.config.global().button_exit);
        cache_pause.update(ctx.config.global().button_pause);

        // Check for exit key (F2)
        if (cache_exit.isPressed()) {
            ctx.should_exit = true;
            ctx.frame_cv.notify_all();
            break;
        }

        // Pause key handling (F3) - toggle on key press
        bool current_pause = cache_pause.isPressed();
        if (current_pause && !last_pause_state) {
            bool new_state = !ctx.detection_paused.load();
            ctx.detection_paused = new_state;
            std::cout << "[Keyboard] Detection " << (new_state ? "PAUSED" : "RESUMED") << std::endl;
        }
        last_pause_state = current_pause;

        // 2PC: Read button states from Makcu device (connected to Game PC)
        // RMB (aim key) activates aimbot
        // RMB + LMB (aim + shoot) activates stabilizer (no recoil)
        if (ctx.makcu_connection) {
            bool current_aiming = ctx.makcu_connection->aiming_active;     // Right mouse button
            bool current_shooting = ctx.makcu_connection->shooting_active; // Left mouse button

            // Log button state changes
            if (current_aiming != last_aiming_state) {
                std::cout << "[Makcu] Right Mouse Button " << (current_aiming ? "PRESSED" : "RELEASED") << std::endl;
            }
            if (current_shooting != last_shooting_state) {
                std::cout << "[Makcu] Left Mouse Button " << (current_shooting ? "PRESSED" : "RELEASED") << std::endl;
            }

            // Update aiming state
            ctx.aiming = current_aiming;
            ctx.shooting = current_shooting;

            // Stabilizer (no recoil) activates when both RMB + LMB pressed
            bool current_stabilizer = current_aiming && current_shooting;
            bool last_stabilizer = last_aiming_state && last_shooting_state;

            if (current_stabilizer != last_stabilizer) {
                std::cout << "[Makcu] Stabilizer (No Recoil) " << (current_stabilizer ? "ACTIVATED" : "DEACTIVATED") << std::endl;
            }
            ctx.stabilizer_active = current_stabilizer;

            // Notify pipeline thread on aiming state change (event-driven)
            if (current_aiming != last_aiming_state) {
                ctx.pipeline_activation_cv.notify_one();
                ctx.aiming_cv.notify_one();
                last_aiming_state = current_aiming;
            }

            last_shooting_state = current_shooting;
        } else {
            // No Makcu connection - set to false
            ctx.aiming = false;
            ctx.shooting = false;
            ctx.stabilizer_active = false;
        }

        // Adaptive delay - 10ms polling for control keys
        WaitForSingleObject(hKeyboardEvent, 10);
    }

    CloseHandle(hKeyboardEvent);
}
