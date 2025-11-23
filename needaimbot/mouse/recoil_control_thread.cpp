#include "recoil_control_thread.h"
#include "../core/windows_headers.h"
#include "../AppContext.h"
#include "../utils/input_state.h"
#include <iostream>

std::atomic<RecoilControlThread*> RecoilControlThread::instance_{nullptr};

RecoilControlThread::RecoilControlThread()
    : running_(false), enabled_(false) {
}

RecoilControlThread::~RecoilControlThread() {
    stop();
}

void RecoilControlThread::start() {
    if (running_) return;

    running_ = true;
    worker_thread_ = std::thread(&RecoilControlThread::threadLoop, this);

    // Set high thread priority for consistent timing and minimal jitter
    if (worker_thread_.joinable()) {
        HANDLE threadHandle = worker_thread_.native_handle();
        SetThreadPriority(threadHandle, THREAD_PRIORITY_ABOVE_NORMAL);

        #ifdef _WIN32
        SetThreadDescription(threadHandle, L"RecoilControl");
        #endif
    }
}

void RecoilControlThread::stop() {
    running_ = false;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void RecoilControlThread::setInputMethod(std::unique_ptr<InputMethod> method) {
    input_method_ = std::move(method);
}

void RecoilControlThread::threadLoop() {
    instance_.store(this, std::memory_order_release);

    while (running_) {
        auto& ctx = AppContext::getInstance();

        // Check if recoil control is enabled
        if (!ctx.config.easynorecoil || !enabled_) {
            // Optimized sleep when disabled
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // Check if both mouse buttons are pressed (shooting while ADS)
        // Use InputStateManager - supports both Makcu and regular mouse
        auto& inputState = InputStateManager::getInstance();
        bool left_mouse = inputState.isLeftButtonPressed();
        bool right_mouse = inputState.isRightButtonPressed();
        bool recoil_active = left_mouse && right_mouse;

        if (recoil_active) {
            // Track recoil start time
            if (!was_recoil_active_) {
                recoil_start_time_ = std::chrono::steady_clock::now();
                was_recoil_active_ = true;
            }

            // Apply recoil compensation
            applyRecoilCompensation();

            // Dynamic delay based on weapon profile or config
            float delay_ms = 10.0f; // Default delay
            if (ctx.config.active_weapon_profile_index >= 0 &&
                ctx.config.active_weapon_profile_index < ctx.config.weapon_profiles.size()) {
                const auto& profile = ctx.config.weapon_profiles[ctx.config.active_weapon_profile_index];
                delay_ms = profile.recoil_ms;
            }

            // Optimized delay: maintain responsiveness throughout
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - recoil_start_time_).count();
            if (elapsed_ms < 500) {
                // First 500ms: use configured delay for responsiveness
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(delay_ms * 1000)));
            } else {
                // After 500ms: slightly longer but still responsive
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(delay_ms * 1200)));
            }
        } else {
            was_recoil_active_ = false;
            // Optimized polling: more responsive
            auto now = std::chrono::steady_clock::now();
            auto time_since_active = std::chrono::duration_cast<std::chrono::milliseconds>(now - recoil_start_time_).count();
            if (time_since_active < 1000) {
                // Recently active: very responsive
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                // Idle: still responsive but save some CPU
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    instance_.store(nullptr, std::memory_order_release);
}

void RecoilControlThread::applyRecoilCompensation() {
    auto& ctx = AppContext::getInstance();

    if (!input_method_) return;

    float strength = calculateRecoilStrength();

    // Apply downward mouse movement to compensate for recoil
    int move_y = static_cast<int>(strength);
    if (move_y > 0) {
        input_method_->move(0, move_y);
    }
}

float RecoilControlThread::calculateRecoilStrength() {
    auto& ctx = AppContext::getInstance();

    float base_strength = 0.0f;
    float multiplier = 1.0f;

    // Get base strength from weapon profile or config
    if (ctx.config.active_weapon_profile_index >= 0 &&
        ctx.config.active_weapon_profile_index < ctx.config.weapon_profiles.size()) {
        const auto& profile = ctx.config.weapon_profiles[ctx.config.active_weapon_profile_index];
        base_strength = profile.base_strength;

        // Apply fire rate multiplier
        multiplier *= profile.fire_rate_multiplier;

        // Apply scope multiplier
        int scope = ctx.config.active_scope_magnification;
        if (scope <= 0) scope = 1;

        switch (scope) {
            case 1: multiplier *= profile.scope_mult_1x; break;
            case 2: multiplier *= profile.scope_mult_2x; break;
            case 3: multiplier *= profile.scope_mult_3x; break;
            case 4: multiplier *= profile.scope_mult_4x; break;
            case 6: multiplier *= profile.scope_mult_6x; break;
            case 8: multiplier *= profile.scope_mult_8x; break;
            default: multiplier *= profile.scope_mult_1x; break;
        }
    } else {
        base_strength = ctx.config.easynorecoilstrength;
    }

    // Apply crouch reduction if enabled
    if (ctx.config.crouch_recoil_enabled && (GetAsyncKeyState(VK_LCONTROL) & 0x8000)) {
        float crouch_multiplier = 1.0f + (ctx.config.crouch_recoil_reduction / 100.0f);
        crouch_multiplier = (std::max)(0.0f, crouch_multiplier);
        multiplier *= crouch_multiplier;
    }

    return base_strength * multiplier;
}
