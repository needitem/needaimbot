#include "recoil_control_thread.h"
#include "../core/windows_headers.h"
#include "../AppContext.h"
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
    
    // Set thread priority for consistent timing
    if (worker_thread_.joinable()) {
        SetThreadPriority(worker_thread_.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
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
    hook_installed_ = installMouseHook();

    while (running_) {
        processPendingMessages();

        auto& ctx = AppContext::getInstance();

        // Check if recoil control is enabled
        if (!ctx.config.easynorecoil || !enabled_) {
            // Optimized sleep when disabled
            waitForEventOrTimeout(std::chrono::milliseconds(5));
            continue;
        }

        // Check if both mouse buttons are pressed (shooting while ADS)
        bool left_mouse = hook_installed_ ? left_button_pressed_.load(std::memory_order_acquire)
                                          : (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
        bool right_mouse = hook_installed_ ? right_button_pressed_.load(std::memory_order_acquire)
                                           : (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
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
                waitForEventOrTimeout(std::chrono::microseconds(static_cast<int>(delay_ms * 1000)));
            } else {
                // After 500ms: slightly longer but still responsive
                waitForEventOrTimeout(std::chrono::microseconds(static_cast<int>(delay_ms * 1200)));
            }
        } else {
            was_recoil_active_ = false;
            // Optimized polling: more responsive
            auto now = std::chrono::steady_clock::now();
            auto time_since_active = std::chrono::duration_cast<std::chrono::milliseconds>(now - recoil_start_time_).count();
            if (time_since_active < 1000) {
                // Recently active: very responsive
                waitForEventOrTimeout(std::chrono::milliseconds(1));
            } else {
                // Idle: still responsive but save some CPU
                waitForEventOrTimeout(std::chrono::milliseconds(5));
            }
        }
    }

    uninstallMouseHook();
    instance_.store(nullptr, std::memory_order_release);
}

void RecoilControlThread::applyRecoilCompensation() {
    auto& ctx = AppContext::getInstance();
    
    if (!input_method_) return;
    
    // Calculate time since recoil started
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - recoil_start_time_).count();
    
    // Optimize: use branchless selection for common case
    const int profile_idx = ctx.config.active_weapon_profile_index;
    const bool has_profile = (profile_idx >= 0 && profile_idx < ctx.config.weapon_profiles.size());
    
    const int start_delay = has_profile ? 
        ctx.config.weapon_profiles[profile_idx].start_delay_ms : 
        
    const int end_delay = has_profile ? 
        ctx.config.weapon_profiles[profile_idx].end_delay_ms : 
        
    
    // Skip if in start delay period
    if (elapsed_ms < start_delay) {
        return;
    }
    
    // Stop if past end delay
    if (end_delay > 0 && elapsed_ms > end_delay) {
        return;
    }
    
    float strength = calculateRecoilStrength();
    
    // Apply downward mouse movement to compensate for recoil
    // Optimize: single branch instead of nested
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

bool RecoilControlThread::installMouseHook() {
    mouse_hook_ = SetWindowsHookExW(WH_MOUSE_LL, &RecoilControlThread::LowLevelMouseProc, GetModuleHandleW(nullptr), 0);
    if (!mouse_hook_) {
        std::cerr << "[Recoil] Failed to install low-level mouse hook, error=" << GetLastError() << std::endl;
        return false;
    }
    return true;
}

void RecoilControlThread::uninstallMouseHook() {
    if (mouse_hook_) {
        UnhookWindowsHookEx(mouse_hook_);
        mouse_hook_ = nullptr;
    }
}

void RecoilControlThread::waitForEventOrTimeout(std::chrono::milliseconds timeout) {
    if (!hook_installed_) {
        std::this_thread::sleep_for(timeout);
        return;
    }

    DWORD wait_ms = timeout.count() < 0 ? 0 : static_cast<DWORD>(timeout.count());
    MsgWaitForMultipleObjectsEx(0, nullptr, wait_ms, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
}

void RecoilControlThread::waitForEventOrTimeout(std::chrono::microseconds timeout) {
    if (!hook_installed_) {
        std::this_thread::sleep_for(timeout);
        return;
    }

    long long micros = timeout.count();
    if (micros < 0) micros = 0;
    DWORD wait_ms = micros == 0 ? 0 : static_cast<DWORD>((micros + 999) / 1000);
    MsgWaitForMultipleObjectsEx(0, nullptr, wait_ms, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
}

LRESULT CALLBACK RecoilControlThread::LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam) {
    auto* instance = instance_.load(std::memory_order_acquire);
    if (nCode >= HC_ACTION && instance) {
        switch (wParam) {
            case WM_LBUTTONDOWN:
            case WM_NCLBUTTONDOWN:
                instance->left_button_pressed_.store(true, std::memory_order_release);
                break;
            case WM_LBUTTONUP:
            case WM_NCLBUTTONUP:
                instance->left_button_pressed_.store(false, std::memory_order_release);
                break;
            case WM_RBUTTONDOWN:
            case WM_NCRBUTTONDOWN:
                instance->right_button_pressed_.store(true, std::memory_order_release);
                break;
            case WM_RBUTTONUP:
            case WM_NCRBUTTONUP:
                instance->right_button_pressed_.store(false, std::memory_order_release);
                break;
            default:
                break;
        }
    }

    return CallNextHookEx(instance ? instance->mouse_hook_ : nullptr, nCode, wParam, lParam);
}

void RecoilControlThread::processPendingMessages() {
    if (!hook_installed_) {
        return;
    }

    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            running_ = false;
            break;
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}
