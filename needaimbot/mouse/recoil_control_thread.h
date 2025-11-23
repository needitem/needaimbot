#pragma once

#include <thread>
#include <atomic>
#include <chrono>
#include <memory>

#include "../core/windows_headers.h"
#include "../config/config.h"
#include "input_drivers/InputMethod.h"

class RecoilControlThread {
public:
    RecoilControlThread();
    ~RecoilControlThread();

    void start();
    void stop();

    // Set the input method for mouse control
    void setInputMethod(std::unique_ptr<InputMethod> method);

    // Enable/disable recoil control
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    void threadLoop();
    void applyRecoilCompensation();
    float calculateRecoilStrength();

    static std::atomic<RecoilControlThread*> instance_;

    std::thread worker_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> enabled_;

    std::unique_ptr<InputMethod> input_method_;
    std::chrono::steady_clock::time_point last_recoil_time_;

    // Recoil state tracking
    bool was_recoil_active_ = false;
    std::chrono::steady_clock::time_point recoil_start_time_;
};
