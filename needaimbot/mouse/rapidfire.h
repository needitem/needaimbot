#ifndef RAPIDFIRE_H
#define RAPIDFIRE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <memory>
#include <mutex>

// Forward declaration to avoid including the full header
class InputMethod;

class RapidFire {
private:
    std::atomic<bool> enabled;
    std::atomic<bool> running;
    std::atomic<bool> firing;
    std::atomic<int> clicks_per_second;
    std::thread worker_thread;
    
    std::chrono::steady_clock::time_point last_click_time;
    std::chrono::steady_clock::time_point last_programmatic_click_time;
    
    // Input method for sending clicks
    std::shared_ptr<InputMethod> input_method;
    std::mutex input_method_mutex;
    
    // Flag to track if we just sent a programmatic click
    std::atomic<bool> just_sent_click;
    
    // Flag to pause rapidfire when UI is active
    std::atomic<bool> ui_active;
    
    void workerLoop();
    void performClick();
    
public:
    RapidFire();
    ~RapidFire();
    
    void start();
    void stop();
    
    void setEnabled(bool enable);
    bool isEnabled() const { return enabled.load(); }
    
    void setClicksPerSecond(int cps);
    int getClicksPerSecond() const { return clicks_per_second.load(); }
    
    void setInputMethod(std::shared_ptr<InputMethod> method);
    
    void setUIActive(bool active) { ui_active = active; }
    
    void startFiring();
    void stopFiring();
    bool isFiring() const { return firing.load(); }
};

#endif // RAPIDFIRE_H