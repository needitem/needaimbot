#include "rapidfire.h"
#include "input_drivers/InputMethod.h"
#include <iostream>
#include <algorithm>
#include <cstring>

RapidFire::RapidFire() 
    : enabled(false), running(false), firing(false), clicks_per_second(10), just_sent_click(false), ui_active(false) {
    // Initialize with default WIN32 input method
    input_method = std::make_shared<Win32InputMethod>();
}

RapidFire::~RapidFire() {
    stop();
}

void RapidFire::start() {
    if (running.load()) return;
    
    running = true;
    worker_thread = std::thread(&RapidFire::workerLoop, this);
}

void RapidFire::stop() {
    running = false;
    firing = false;
    
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void RapidFire::setInputMethod(std::shared_ptr<InputMethod> method) {
    std::lock_guard<std::mutex> lock(input_method_mutex);
    if (method && method->isValid()) {
        input_method = method;
    }
}

void RapidFire::workerLoop() {
    bool was_pressed = false;
    bool is_firing = false;
    
    while (running.load()) {
        if (enabled.load() && !ui_active.load()) {
            // Always check the current physical button state
            bool currently_pressed = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
            
            // Start firing when button is first pressed
            if (currently_pressed && !was_pressed) {
                is_firing = true;
                firing = true;
                last_click_time = std::chrono::steady_clock::now();
            }
            // Stop firing immediately when button is released
            else if (!currently_pressed && was_pressed) {
                is_firing = false;
                firing = false;
            }
            
            // Perform rapid fire only while button is physically held
            if (is_firing && currently_pressed) {
                auto now = std::chrono::steady_clock::now();
                auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_click_time).count();
                
                int delay_ms = 1000 / clicks_per_second.load();
                
                if (time_since_last >= delay_ms) {
                    // Double check button is still pressed before clicking
                    if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0) {
                        performClick();
                        last_click_time = now;
                    } else {
                        // Button was released, stop firing
                        is_firing = false;
                        firing = false;
                    }
                }
            }
            
            was_pressed = currently_pressed;
        } else {
            // Rapid fire disabled or UI active, reset states
            firing = false;
            is_firing = false;
            was_pressed = false;
        }
        
        // Small delay to prevent CPU overuse while maintaining responsiveness
        std::this_thread::sleep_for(std::chrono::microseconds(500));  // 0.5ms delay, 2000Hz polling
    }
}

void RapidFire::performClick() {
    std::lock_guard<std::mutex> lock(input_method_mutex);
    
    if (input_method && input_method->isValid()) {
        // Send release and press immediately
        input_method->release();
        input_method->press();
    } else {
        // Fallback to WIN32 if no valid input method
        INPUT inputs[2] = {};
        
        // Release
        inputs[0].type = INPUT_MOUSE;
        inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTUP;
        
        // Press
        inputs[1].type = INPUT_MOUSE;
        inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        
        SendInput(2, inputs, sizeof(INPUT));
    }
}

void RapidFire::setEnabled(bool enable) {
    enabled = enable;
    if (!enable) {
        firing = false;
    }
}

void RapidFire::setClicksPerSecond(int cps) {
    clicks_per_second = (std::max)(1, (std::min)(cps, 50)); // Limit between 1-50 CPS
}

void RapidFire::startFiring() {
    if (enabled.load()) {
        firing = true;
        last_click_time = std::chrono::steady_clock::now();
    }
}

void RapidFire::stopFiring() {
    firing = false;
}