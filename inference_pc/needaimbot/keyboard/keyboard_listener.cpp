// 2PC Architecture: Inference PC (Jetson) has no keyboard/mouse
// All input comes from Makcu device connected to Game PC

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "../config/config.h"
#include "../AppContext.h"
#include "keyboard_listener.h"
#include "../mouse/mouse.h"

// 2PC Architecture: Inference PC only reads button states from Makcu
// No local keyboard - exit is handled via signal or Makcu disconnect
void keyboardListener() {
    auto& ctx = AppContext::getInstance();

    bool last_aiming_state = false;
    bool last_shooting_state = false;

    std::cout << "[Input] 2PC mode - reading button states from Makcu" << std::endl;

    while (!ctx.should_exit) {
        // 2PC: Read button states from Makcu device (connected to Game PC)
        // RMB (aim key) activates aimbot
        // RMB + LMB (aim + shoot) activates stabilizer (no recoil)
        if (ctx.makcu_connection) {
            bool current_aiming = ctx.makcu_connection->aiming_active;     // Right mouse button
            bool current_shooting = ctx.makcu_connection->shooting_active; // Left mouse button

            // Button state change logging removed for cleaner output

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

        // 10ms polling interval
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
