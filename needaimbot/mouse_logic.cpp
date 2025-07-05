#include "mouse_logic.h"
#include "AppContext.h"
#include "needaimbot.h"

namespace MouseLogic {

void handle_recoil(MouseThread& mouseThread) {
    auto& ctx = AppContext::getInstance();
    bool current_shooting_state = ctx.shooting.load();
    bool currently_zooming = ctx.zooming.load();

    if (ctx.config.easynorecoil) {
        if (current_shooting_state && !ctx.was_shooting.load()) {
            ctx.shooting_key_press_time = std::chrono::steady_clock::now();
            ctx.end_delay_pending = false;
            if (ctx.config.easynorecoil_start_delay_ms == 0) {
                ctx.recoil_active = true;
                ctx.start_delay_pending = false;
            } else {
                ctx.recoil_active = false;
                ctx.start_delay_pending = true;
            }
        } else if (!current_shooting_state && ctx.was_shooting.load()) {
            auto now = std::chrono::steady_clock::now();
            if (ctx.config.easynorecoil_end_delay_ms == 0) {
                ctx.recoil_active = false;
                ctx.end_delay_pending = false;
            } else {
                ctx.shooting_key_release_time = now;
                ctx.end_delay_pending = true;
            }
            ctx.start_delay_pending = false;
        }
        if (ctx.start_delay_pending.load() && current_shooting_state) {
            auto elapsed_since_press = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - ctx.shooting_key_press_time
            ).count();
            
            WeaponRecoilProfile* current_profile = ctx.config.getCurrentWeaponProfile();
            int start_delay = current_profile ? current_profile->start_delay_ms : ctx.config.easynorecoil_start_delay_ms;
            
            if (elapsed_since_press >= start_delay) {
                ctx.recoil_active = true;
                ctx.start_delay_pending = false;
            }
        }
        
        if (ctx.end_delay_pending.load() && !current_shooting_state) {
            auto elapsed_since_release = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - ctx.shooting_key_release_time
            ).count();
            
            WeaponRecoilProfile* current_profile = ctx.config.getCurrentWeaponProfile();
            int end_delay = current_profile ? current_profile->end_delay_ms : ctx.config.easynorecoil_end_delay_ms;
            
            if (elapsed_since_release >= end_delay) {
                ctx.recoil_active = false;
                ctx.end_delay_pending = false;
            }
        }
        ctx.was_shooting = current_shooting_state;
        if (ctx.recoil_active.load() && currently_zooming && current_shooting_state) {
            WeaponRecoilProfile* current_profile = ctx.config.getCurrentWeaponProfile();
            if (current_profile) {
                mouseThread.applyWeaponRecoilCompensation(current_profile, ctx.config.active_scope_magnification);
            } else {
                mouseThread.applyRecoilCompensation(ctx.config.easynorecoilstrength);
            }
        }
        
        if (ctx.config.optical_flow_norecoil && current_shooting_state) {
            mouseThread.applyOpticalFlowRecoilCompensation();
        }
    } else {
        ctx.recoil_active = false;
        ctx.start_delay_pending = false;
        ctx.end_delay_pending = false;
    }
}

void handle_silent_aim(MouseThread& mouseThread) {
    auto& ctx = AppContext::getInstance();
    bool silent_trigger_active = ctx.silent_aim_trigger.load(std::memory_order_acquire);
    if (silent_trigger_active) {
        ctx.silent_aim_trigger.store(false, std::memory_order_release);

        Detection best_target_for_silent_aim;
        bool has_target_for_silent_aim = false;
        {
            std::unique_lock<std::mutex> lock(ctx.detector.detectionMutex);
            
            if (ctx.detector.detectionVersion > ctx.lastDetectionVersion) {
                ctx.lastDetectionVersion = ctx.detector.detectionVersion;
            }
            has_target_for_silent_aim = ctx.detector.m_hasBestTarget;
            if (has_target_for_silent_aim) {
                best_target_for_silent_aim = ctx.detector.m_bestTargetHost;
            }
        }

        if (has_target_for_silent_aim) {
            AimbotTarget silent_aim_target(
                best_target_for_silent_aim.box.x,
                best_target_for_silent_aim.box.y,
                best_target_for_silent_aim.box.width,
                best_target_for_silent_aim.box.height,
                best_target_for_silent_aim.classId
            );
            mouseThread.executeSilentAim(silent_aim_target);
        }
    }
}

void handle_aiming(MouseThread& mouseThread) {
    auto& ctx = AppContext::getInstance();
    bool is_aiming = ctx.aiming.load();
    bool hotkey_pressed_for_trigger = ctx.auto_shoot_active.load();

    bool newDetectionAvailable = false;
    bool has_target_from_detector = false;
    Detection best_target_from_detector;

    {
        std::unique_lock<std::mutex> lock(ctx.detector.detectionMutex);
        if (is_aiming) {
            ctx.detector.detectionCV.wait_for(lock, std::chrono::milliseconds(5), [&]() {
                return ctx.detector.detectionVersion > ctx.lastDetectionVersion || ctx.shouldExit;
            });
        } else {
            ctx.detector.detectionCV.wait(lock, [&]() {
                return ctx.detector.detectionVersion > ctx.lastDetectionVersion || ctx.shouldExit;
            });
        }
        if (ctx.shouldExit)
            return;
        if (ctx.detector.detectionVersion > ctx.lastDetectionVersion) {
            newDetectionAvailable = true;
            ctx.lastDetectionVersion = ctx.detector.detectionVersion;
            has_target_from_detector = ctx.detector.m_hasBestTarget;
            if (has_target_from_detector) {
                best_target_from_detector = ctx.detector.m_bestTargetHost;
            }
        }
    }

    if (newDetectionAvailable) {
        if (is_aiming && has_target_from_detector) {
            AimbotTarget best_target(
                best_target_from_detector.box.x,
                best_target_from_detector.box.y,
                best_target_from_detector.box.width,
                best_target_from_detector.box.height,
                best_target_from_detector.classId
            );
            
            auto pid_start = std::chrono::high_resolution_clock::now();
            mouseThread.moveMouse(best_target);
            auto pid_end = std::chrono::high_resolution_clock::now();
            
            float pid_time_ms = std::chrono::duration<float, std::milli>(pid_end - pid_start).count();
            add_to_history(ctx.g_pid_calc_time_history, pid_time_ms, ctx.g_pid_calc_history_mutex, 100);
            ctx.g_current_pid_calc_time_ms.store(pid_time_ms);

            if (hotkey_pressed_for_trigger) {
                mouseThread.pressMouse(best_target);
            } else {
                mouseThread.releaseMouse();
            }
        } else {
            mouseThread.releaseMouse();
        }
    } else {
        if (!hotkey_pressed_for_trigger || !is_aiming) {
            mouseThread.releaseMouse();
        }
    }
}

}
