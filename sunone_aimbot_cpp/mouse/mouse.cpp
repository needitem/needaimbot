#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "PIDController2D.h"
#include "mouse.h"
#include "AimbotTarget.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "config.h"
#include "keyboard/keyboard_listener.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;

constexpr float SCOPE_MARGIN = 0.15f;

MouseThread::MouseThread(
    int resolution,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    bool auto_shoot,
    float bScope_multiplier,
    float norecoil_ms,
    SerialConnection *serialConnection,
    GhubMouse *gHub) : tracking_errors(false)
{
    initializeScreen(resolution, auto_shoot, bScope_multiplier, norecoil_ms);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
    initializeInputMethod(serialConnection, gHub);
}

MouseThread::~MouseThread() = default;

void MouseThread::initializeInputMethod(SerialConnection *serialConnection, GhubMouse *gHub)
{
    if (serialConnection && serialConnection->isOpen())
    {
        input_method = std::make_unique<SerialInputMethod>(serialConnection);
    }
    else if (gHub)
    {
        input_method = std::make_unique<GHubInputMethod>(gHub);
    }
    else
    {
        input_method = std::make_unique<Win32InputMethod>();
    }
}

void MouseThread::initializeScreen(int resolution, bool auto_shoot, float bScope_multiplier, float norecoil_ms)
{
    this->screen_width = static_cast<float>(resolution);
    this->screen_height = static_cast<float>(resolution); 
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->norecoil_ms = norecoil_ms; 
    this->center_x = screen_width / 2.0f;
    this->center_y = screen_height / 2.0f;

    const float SENSITIVITY_FACTOR = 0.05f; // Example: Adjust this base sensitivity
    float base_scale_x = SENSITIVITY_FACTOR;
    float base_scale_y = SENSITIVITY_FACTOR;

    if (this->bScope_multiplier > 1.0f) {
        this->move_scale_x = base_scale_x / this->bScope_multiplier;
        this->move_scale_y = base_scale_y / this->bScope_multiplier;
    } else {
        this->move_scale_x = base_scale_x;
        this->move_scale_y = base_scale_y;
    }

    this->last_recoil_compensation_time = std::chrono::steady_clock::now();
}

void MouseThread::updateConfig(
    int resolution,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    bool auto_shoot,
    float bScope_multiplier,
    float norecoil_ms
    )
{
    initializeScreen(resolution, auto_shoot, bScope_multiplier, norecoil_ms);
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y);
}

Eigen::Vector2f MouseThread::calculateMovement(const Eigen::Vector2f &target_pos)
{
    float error_x = target_pos[0] - center_x;
    float error_y = target_pos[1] - center_y;
    
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    float result_x = pid_output[0] * move_scale_x;
    float result_y = pid_output[1] * move_scale_y;
    
    return Eigen::Vector2f(result_x, result_y);
}

bool MouseThread::checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor)
{
    static const float screen_margin_x = screen_width * SCOPE_MARGIN;
    static const float screen_margin_y = screen_height * SCOPE_MARGIN;
    
    float target_center_x = target_x + target_w * 0.5f;
    float target_center_y = target_y + target_h * 0.5f;
    
    float diff_x = std::abs(target_center_x - center_x);
    float diff_y = std::abs(target_center_y - center_y);
    
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    float reduced_half_w = target_w * reduction_factor * 0.5f;
    float reduced_half_h = target_h * reduction_factor * 0.5f;
    
    float min_x = target_center_x - reduced_half_w;
    float max_x = target_center_x + reduced_half_w;
    float min_y = target_center_y - reduced_half_h;
    float max_y = target_center_y + reduced_half_h;
    
    return (center_x >= min_x && center_x <= max_x && 
            center_y >= min_y && center_y <= max_y);
}

float MouseThread::calculateTargetDistanceSquared(const AimbotTarget &target) const
{
    float dx = target.x + target.w * 0.5f - center_x;
    float target_center_y;
    /*
     Model Class IDs (from user):
      0: player
      1: bot
      2: weapon
      3: outline
      4: dead_body
      5: hideout_target_human
      6: hideout_target_balls
      7: head
      8: smoke
      9: fire
      10: third_person
    */
    constexpr int HEAD_CLASS_ID = 7; // Class ID for head

    if (!config.ignore_class_7 && target.classId == HEAD_CLASS_ID) {
        target_center_y = target.y + target.h * config.head_y_offset;
    } else {
        target_center_y = target.y + target.h * config.body_y_offset;
    }
    float dy = target_center_y - center_y;
    return dx * dx + dy * dy;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    const float local_center_x = center_x;
    const float local_center_y = center_y;

    float target_center_x = target.x + target.w * 0.5f;
    float target_center_y;
    /*
     Model Class IDs (from user):
      0: player
      1: bot
      2: weapon
      3: outline
      4: dead_body
      5: hideout_target_human
      6: hideout_target_balls
      7: head
      8: smoke
      9: fire
      10: third_person
    */
    constexpr int HEAD_CLASS_ID = 7; // Class ID for head

    {
        std::lock_guard<std::mutex> lock(configMutex);
        if (!config.ignore_class_7 && target.classId == HEAD_CLASS_ID) {
            // Headshot aiming logic
            target_center_y = target.y + target.h * config.head_y_offset;
            // Optional: Add logging for headshot case too if needed
            // std::cout << "[Aim Debug] Headshot Aim: target.h = " << target.h 
            //           << ", head_offset = " << config.head_y_offset 
            //           << ", target_center_y = " << target_center_y << std::endl;
        } else {
            // Body aiming logic (or disabled headshot)
            target_center_y = target.y + target.h * config.body_y_offset;
            
            // --- Logging Removed --- 
            // Logging was here
            // --- End Logging --- 
        }
    }

    float error_x = target_center_x - local_center_x;
    float error_y = target_center_y - local_center_y;

    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y);
        }
    }

    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    float move_x = pid_output.x() * move_scale_x;
    float move_y = pid_output.y() * move_scale_y;

    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    // Check if the disable upward aim button is pressed and movement is upward
    if (isAnyKeyPressed(config.button_disable_upward_aim) && dy_int < 0)
    {
        dy_int = 0; // Disable upward movement
    }

    if (dx_int != 0 || dy_int != 0)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            input_method->move(dx_int, dy_int);
        }
    }
}

void MouseThread::pressMouse(const AimbotTarget &target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    auto bScope = checkTargetInScope(target.x, target.y, target.w, target.h, bScope_multiplier);

    if (bScope && !mouse_pressed)
    {
        if (input_method)
        {
            input_method->press();
        }
        mouse_pressed = true;
    }
}

void MouseThread::releaseMouse()
{
    if (!mouse_pressed)
        return;

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (input_method)
    {
        input_method->release();
    }
    mouse_pressed = false;
}

void MouseThread::applyRecoilCompensation(float strength)
{
    if (!input_method || !input_method->isValid()) {
        return;
    }

    if (std::abs(strength) < 1e-3f) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_recoil_compensation_time);
    auto required_delay = std::chrono::milliseconds(static_cast<long long>(this->norecoil_ms));

    if (elapsed >= required_delay)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex); 
        
        int dy_recoil = static_cast<int>(std::round(strength));
        if (dy_recoil != 0) {
             input_method->move(0, dy_recoil);
        }
        
        last_recoil_compensation_time = now;
    }
}

void MouseThread::enableErrorTracking(const ErrorTrackingCallback &callback)
{
    std::lock_guard<std::mutex> lock(callback_mutex);
    error_callback = callback;
    tracking_errors = true;
}

void MouseThread::disableErrorTracking()
{
    std::lock_guard<std::mutex> lock(callback_mutex);
    tracking_errors = false;
    error_callback = nullptr; 
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}