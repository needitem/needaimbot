// Event-based mouse thread implementation
#include "AppContext.h"
#include "mouse/mouse.h"
#include "mouse/aimbot_components/AimbotTarget.h"
#include "needaimbot.h"
#include "keyboard/keyboard_listener.h"
#include "core/logger.h"
#include <chrono>
#include <Windows.h>

void mouseThreadFunctionEventBased(MouseThread& mouseThread)
{
    auto& ctx = AppContext::getInstance();
    
    // Reduce thread priority to prevent excessive CPU usage
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    
    LOG_INFO("MouseThread", "Mouse thread started - Event-based mode");
    
    // Track state for smooth transitions
    static bool had_target_before = false;
    static bool last_aiming_state = false;
    static int event_log_count = 0;
    
    // Key cache for performance
    struct KeyStateCache {
        bool left_mouse = false;
        bool right_mouse = false;
        
        void update() {
            left_mouse = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
            right_mouse = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
        }
    } key_cache;
    
    while (!ctx.should_exit)
    {
        // Wait for GPU-calculated mouse movement data
        std::unique_lock<std::mutex> lock(ctx.mouseDataMutex);
        
        // Wait for events - use timeout only when recoil compensation is active
        bool needs_recoil_check = ctx.config.easynorecoil && 
                                  ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) && (GetAsyncKeyState(VK_RBUTTON) & 0x8000));
        
        if (needs_recoil_check) {
            // Use timeout for recoil compensation
            ctx.mouseDataCV.wait_for(lock, 
                std::chrono::milliseconds(16), // 60Hz for recoil updates when shooting
                [&ctx] {
                    return ctx.mouseDataReady.load() || ctx.mouse_events_available.load() || ctx.should_exit;
                });
        } else {
            // Pure event-based wait when not shooting
            ctx.mouseDataCV.wait(lock,
                [&ctx] {
                    return ctx.mouseDataReady.load() || ctx.mouse_events_available.load() || ctx.should_exit;
                });
        }
        
        if (ctx.should_exit) break;
        
        // Process GPU-calculated movement if available
        if (ctx.mouseDataReady.load()) {
            ctx.mouseDataReady.store(false);
            
            // Get GPU-calculated movement
            auto gpuMovement = ctx.latestMouseMovement;
            
            // Release lock while processing
            lock.unlock();
            
            // Check if we should aim
            bool current_aiming = ctx.aiming.load();
            
            if (gpuMovement.hasTarget && current_aiming) {
                // Apply movement directly from GPU calculation
                int final_dx = gpuMovement.dx;
                int final_dy = gpuMovement.dy;
                
                // Apply PID error smoothing if needed
                if (ctx.config.pid_error_smoothing > 0) {
                    static float smooth_x = 0.0f;
                    static float smooth_y = 0.0f;
                    float alpha = ctx.config.pid_error_smoothing;
                    smooth_x = alpha * final_dx + (1.0f - alpha) * smooth_x;
                    smooth_y = alpha * final_dy + (1.0f - alpha) * smooth_y;
                    final_dx = static_cast<int>(smooth_x);
                    final_dy = static_cast<int>(smooth_y);
                }
                
                // Send mouse movement directly through input method
                if (final_dx != 0 || final_dy != 0) {
                    // Create a minimal AimbotTarget for mouse movement
                    // Target(x, y, width, height, confidence, classId)
                    float centerX = ctx.config.detection_resolution / 2.0f;
                    float centerY = ctx.config.detection_resolution / 2.0f;
                    
                    AimbotTarget moveTarget(
                        static_cast<int>(centerX + final_dx),  // x
                        static_cast<int>(centerY + final_dy),  // y
                        10,  // width
                        10,  // height
                        gpuMovement.confidence,  // confidence
                        0  // classId
                    );
                    mouseThread.moveMouse(moveTarget);
                }
            }
            
            // Re-acquire lock for next iteration
            lock.lock();
        }
        
        // Process all pending events
        while (!ctx.mouse_event_queue.empty()) {
            MouseEvent event = ctx.mouse_event_queue.front();
            ctx.mouse_event_queue.pop();
            
            
            
            // Release lock while processing
            lock.unlock();
            
            // Measure processing time
            auto cycle_start = std::chrono::high_resolution_clock::now();
            
            // Use event data directly
            bool current_aiming = ctx.aiming;
            bool current_has_target = event.has_target;
            Target current_target = event.target;
            
            // Update overlay data
            {
                std::lock_guard<std::mutex> overlay_lock(ctx.overlay_target_mutex);
                ctx.overlay_has_target.store(current_has_target);
                if (current_has_target) {
                    ctx.overlay_target_info = current_target;
                }
            }
            
            // Process target if we have one
            if (current_has_target && current_aiming) {
                // Convert from detection coordinates to screen offset
                float dx = event.dx;
                float dy = event.dy;
                
                // Create AimbotTarget from Target
                AimbotTarget target(
                    current_target.x,
                    current_target.y,
                    current_target.width,
                    current_target.height,
                    current_target.confidence,
                    current_target.classId
                );
                
                // Move mouse to target if aimbot is enabled
                if (ctx.config.enable_aimbot) {
                    
                    mouseThread.moveMouse(target);
                }
                
                // Auto-shoot if triggerbot is enabled
                bool auto_shoot_active = ctx.config.button_auto_shoot.empty() || 
                                       ctx.config.button_auto_shoot[0] == "None" ||
                                       isAnyKeyPressed(ctx.config.button_auto_shoot);
                
                key_cache.update();
                bool manual_mouse_down = key_cache.left_mouse;
                
                if (ctx.config.enable_triggerbot && auto_shoot_active && !manual_mouse_down) {
                    mouseThread.pressMouse(target);
                } else if (!manual_mouse_down) {
                    mouseThread.releaseMouse();
                }
                
                had_target_before = true;
            } else {
                // No target or not aiming
                mouseThread.releaseMouse();
                
                if (had_target_before) {
                    mouseThread.resetAccumulatedStates();
                    had_target_before = false;
                }
            }
            
            // Track aiming state changes for PID reset
            if (last_aiming_state && !current_aiming) {
                mouseThread.resetAccumulatedStates();
            }
            last_aiming_state = current_aiming;
            
            // Measure end of cycle
            auto cycle_end = std::chrono::high_resolution_clock::now();
            float cycle_time_ms = std::chrono::duration<float, std::milli>(cycle_end - cycle_start).count();
            
            // Store the total cycle time
            ctx.g_current_total_cycle_time_ms.store(cycle_time_ms);
            ctx.add_to_history(ctx.g_total_cycle_time_history, cycle_time_ms, ctx.g_total_cycle_history_mutex);
            
            // Re-acquire lock for next iteration
            lock.lock();
        }
        
        // Clear flag if queue is empty
        if (ctx.mouse_event_queue.empty()) {
            ctx.mouse_events_available = false;
        }
        
        // Handle recoil compensation (runs even without events)
        lock.unlock();
        key_cache.update();
        
        if (ctx.config.easynorecoil) {
            bool recoil_active = key_cache.left_mouse && key_cache.right_mouse;
            bool is_crouching = ctx.config.crouch_recoil_enabled && (GetAsyncKeyState(VK_LCONTROL) & 0x8000);
            
            if (recoil_active) {
                float recoil_multiplier = 1.0f;
                if (is_crouching) {
                    recoil_multiplier = 1.0f + (ctx.config.crouch_recoil_reduction / 100.0f);
                    recoil_multiplier = (std::max)(0.0f, recoil_multiplier);
                }
                
                if (ctx.config.active_weapon_profile_index >= 0 && 
                    ctx.config.active_weapon_profile_index < ctx.config.weapon_profiles.size()) {
                    
                    WeaponRecoilProfile profile = ctx.config.weapon_profiles[ctx.config.active_weapon_profile_index];
                    profile.base_strength *= recoil_multiplier;
                    mouseThread.applyWeaponRecoilCompensation(&profile, ctx.config.active_scope_magnification);
                } else {
                    float adjusted_strength = ctx.config.easynorecoilstrength * recoil_multiplier;
                    mouseThread.applyRecoilCompensation(adjusted_strength);
                }
            }
        }
    }
    
    mouseThread.releaseMouse();
    LOG_INFO("MouseThread", "Mouse thread exiting");
}