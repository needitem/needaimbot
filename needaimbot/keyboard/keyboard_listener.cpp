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

void keyboardListener() {
    auto& ctx = AppContext::getInstance();

    static bool last_aiming_state = false;
    static bool last_shooting_state = false;
    static bool last_pause_state = false;
    static bool last_single_shot_state = false;

    // No recoil timing
    auto last_recoil_time = std::chrono::steady_clock::now();

    // Cached key combos - parse once, check fast
    CachedKeyCombo cache_exit, cache_targeting, cache_auto_shoot;
    CachedKeyCombo cache_disable_upward, cache_norecoil, cache_pause, cache_single_shot;

    // Event-driven keyboard monitoring using Windows events
    HANDLE hKeyboardEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

    while (!ctx.should_exit) {
        // Update caches (only re-parses if config changed)
        cache_exit.update(ctx.config.button_exit);
        cache_targeting.update(ctx.config.button_targeting);
        cache_auto_shoot.update(ctx.config.button_auto_shoot);
        cache_disable_upward.update(ctx.config.button_disable_upward_aim);
        cache_norecoil.update(ctx.config.button_norecoil);
        cache_pause.update(ctx.config.button_pause);
        cache_single_shot.update(ctx.config.button_single_shot);

        // Check for exit key
        if (cache_exit.isPressed()) {
            ctx.should_exit = true;
            ctx.frame_cv.notify_all();
            break;
        }

        bool current_aiming = cache_targeting.isPressed();

        // Always update aiming state atomically
        ctx.aiming = current_aiming;

        // Notify pipeline thread on state change (event-driven)
        if (current_aiming != last_aiming_state) {
            // Wake up pipeline thread using event-driven mechanism
            ctx.pipeline_activation_cv.notify_one();
            ctx.aiming_cv.notify_one();  // Keep for compatibility
            last_aiming_state = current_aiming;
        }

        // Track auto_shoot button state
        bool current_shooting = cache_auto_shoot.isPressed();
        ctx.shooting = current_shooting;

        if (current_shooting != last_shooting_state) {
            last_shooting_state = current_shooting;
        }

        // Track disable upward aim state
        ctx.disable_upward_aim = cache_disable_upward.isPressed();

        // Track no recoil state
        bool current_norecoil = cache_norecoil.isPressed();
        ctx.norecoil_active = current_norecoil;

        // No recoil - apply mouse movement when active
        if (current_norecoil) {
            auto* profile = ctx.config.getCurrentWeaponProfile();
            if (profile) {
                auto now = std::chrono::steady_clock::now();
                float interval_ms = profile->recoil_ms;
                if (interval_ms < 1.0f) interval_ms = 1.0f;

                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_recoil_time).count();
                if (elapsed >= static_cast<long long>(interval_ms * 1000)) {
                    // Calculate strength with scope multiplier
                    float strength = profile->base_strength;
                    int scope = ctx.config.active_scope_magnification;
                    switch (scope) {
                        case 1: strength *= profile->scope_mult_1x; break;
                        case 2: strength *= profile->scope_mult_2x; break;
                        case 3: strength *= profile->scope_mult_3x; break;
                        case 4: strength *= profile->scope_mult_4x; break;
                        case 6: strength *= profile->scope_mult_6x; break;
                        case 8: strength *= profile->scope_mult_8x; break;
                        default: strength *= profile->scope_mult_1x; break;
                    }
                    strength *= profile->fire_rate_multiplier;

                    int dy = static_cast<int>(strength + 0.5f);
                    if (dy > 0) {
                        executeMouseMovement(0, dy);
                    }
                    last_recoil_time = now;
                }
            }
        }

        // Improved pause key handling - only toggle on key press, not while held
        bool current_pause = cache_pause.isPressed();
        if (current_pause && !last_pause_state) {
            bool new_state = !ctx.detection_paused.load();
            ctx.detection_paused = new_state;
#ifdef _DEBUG
            std::cout << "[Keyboard] Detection " << (new_state ? "PAUSED" : "RESUMED") << std::endl;
#endif
        }
        last_pause_state = current_pause;

        // Single shot trigger - only on key press, not while held
        bool current_single_shot = cache_single_shot.isPressed();
        if (current_single_shot && !last_single_shot_state) {
            ctx.single_shot_requested = true;
            ctx.pipeline_activation_cv.notify_one();
#ifdef _DEBUG
            std::cout << "[Keyboard] Single shot triggered" << std::endl;
#endif
        }
        last_single_shot_state = current_single_shot;

        // Event-driven adaptive delay - use wait with timeout for better CPU efficiency
        DWORD waitTime = current_aiming ? 2 : 10; // 2ms when aiming, 10ms when idle

        // Use WaitForSingleObject with timeout instead of sleep for better responsiveness
        // This allows the thread to wake immediately if signaled
        WaitForSingleObject(hKeyboardEvent, waitTime);
    }
    
    CloseHandle(hKeyboardEvent);
}
