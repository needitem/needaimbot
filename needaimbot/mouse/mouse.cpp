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
#include <random>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "aimbot_components/PIDController2D.h"
#include "mouse.h"
#include "aimbot_components/AimbotTarget.h"
#include "capture/capture.h"
#include "input_drivers/SerialConnection.h"
#include "needaimbot.h"
#include "input_drivers/ghub.h"
#include "config/config.h"
#include "keyboard/keyboard_listener.h"


extern std::atomic<bool> aiming;
extern std::mutex configMutex;
extern Config config;


std::random_device rd;
std::mt19937 gen(rd());

constexpr float SCOPE_MARGIN = 0.15f;


void WindMouse(float target_x, float target_y, float G, float W, float M, float D,
               std::function<void(int, int)> move_func)
{
    float current_x = 0, current_y = 0;
    float v_x = 0, v_y = 0;
    float W_sqrt = sqrt(W);
    float D_sqrt = sqrt(D);

    float dist, random_dist, new_hypot;
    float new_x_vel, new_y_vel;
    float new_x, new_y;

    while (true) {
        dist = hypot(target_x - current_x, target_y - current_y);
        if (dist <= 1.0f) { 
            break;
        }

        random_dist = std::uniform_real_distribution<float>(0.0f, dist / D_sqrt)(gen);
        new_hypot = std::uniform_real_distribution<float>(0.0f, dist / M)(gen);

        v_x += W_sqrt * (std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen)) + G * (target_x - current_x - v_x) / dist;
        v_y += W_sqrt * (std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen)) + G * (target_y - current_y - v_y) / dist;

        
        float v_magnitude = hypot(v_x, v_y);
        if (v_magnitude > new_hypot) {
            v_x = (new_hypot / v_magnitude) * v_x;
            v_y = (new_hypot / v_magnitude) * v_y;
        }
        
        new_x = current_x + v_x;
        new_y = current_y + v_y;

        
        if (hypot(target_x - new_x, target_y - new_y) > dist) {
             new_x = current_x + (target_x - current_x) * (dist - random_dist) / dist;
             new_y = current_y + (target_y - current_y) * (dist - random_dist) / dist;
        }
        
        int move_dx = static_cast<int>(std::round(new_x - current_x));
        int move_dy = static_cast<int>(std::round(new_y - current_y));

        if (move_dx != 0 || move_dy != 0) {
            move_func(move_dx, move_dy);
        }
        
        current_x = new_x;
        current_y = new_y;
        
    }
}


thread_local static LARGE_INTEGER freq;
thread_local static bool freq_initialized = false;

void InitializeHighPrecisionTimer() {
    if (!freq_initialized) {
        QueryPerformanceFrequency(&freq);
        freq_initialized = true;
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        SetThreadAffinityMask(GetCurrentThread(), 1 << 1);
    }
}

void QueueMove(int dx, int dy, std::function<void(int, int)> move_func) {
    InitializeHighPrecisionTimer();
    
    if (dx != 0 || dy != 0) {
        LARGE_INTEGER start;
        QueryPerformanceCounter(&start);
        
        move_func(dx, dy);
        
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        float elapsed_us = ((end.QuadPart - start.QuadPart) * 1000000.0f) / freq.QuadPart;
        
        add_to_history(g_input_send_time_history, elapsed_us / 1000.0f, g_input_send_history_mutex, 100);
        g_current_input_send_time_ms.store(elapsed_us / 1000.0f);
    }
}

MouseThread::MouseThread(
    int resolution,
    float kp_x,
    float ki_x,
    float kd_x,
    float kp_y,
    float ki_y,
    float kd_y,
    float bScope_multiplier,
    float norecoil_ms,
    float derivative_smoothing_factor,
    SerialConnection *serialConnection,
    GhubMouse *gHub) : tracking_errors(false), silent_aim_click_duration_ms(50)
{
    initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    pid_controller = std::make_unique<PIDController2D>(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, derivative_smoothing_factor);
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

void MouseThread::initializeScreen(int resolution, float bScope_multiplier, float norecoil_ms)
{
    this->screen_width = static_cast<float>(resolution);
    this->screen_height = static_cast<float>(resolution); 
    this->bScope_multiplier = bScope_multiplier;
    this->norecoil_ms = norecoil_ms; 
    this->center_x = screen_width / 2.0f;
    this->center_y = screen_height / 2.0f;

    const float SENSITIVITY_FACTOR = 0.05f; 
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
    float bScope_multiplier,
    float norecoil_ms,
    float derivative_smoothing_factor
    )
{
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    }
    pid_controller->updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, derivative_smoothing_factor);
    
}



Eigen::Vector2f MouseThread::calculateMovement(const Eigen::Vector2f &target_pos)
{
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

    float error_x = target_pos[0] - current_center_x;
    float error_y = target_pos[1] - current_center_y;
    
    Eigen::Vector2f error(error_x, error_y);
    Eigen::Vector2f pid_output = pid_controller->calculate(error);

    float result_x = pid_output[0] * current_move_scale_x;
    float result_y = pid_output[1] * current_move_scale_y;
    
    return Eigen::Vector2f(result_x, result_y);
}

bool MouseThread::checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor)
{
    float current_screen_width, current_screen_height, current_center_x, current_center_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_screen_width = this->screen_width;
        current_screen_height = this->screen_height;
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }

    const float screen_margin_x = current_screen_width * SCOPE_MARGIN; 
    const float screen_margin_y = current_screen_height * SCOPE_MARGIN; 
    
    float target_center_x_val = target_x + target_w * 0.5f;
    float target_center_y_val = target_y + target_h * 0.5f;
    
    float diff_x = std::abs(target_center_x_val - current_center_x); 
    float diff_y = std::abs(target_center_y_val - current_center_y); 
    
    if (diff_x > screen_margin_x || diff_y > screen_margin_y)
    {
        return false;
    }
    
    float reduced_half_w = target_w * reduction_factor * 0.5f;
    float reduced_half_h = target_h * reduction_factor * 0.5f;
    
    float min_x = target_center_x_val - reduced_half_w;
    float max_x = target_center_x_val + reduced_half_w;
    float min_y = target_center_y_val - reduced_half_h;
    float max_y = target_center_y_val + reduced_half_h;
    
    return (current_center_x >= min_x && current_center_x <= max_x && 
            current_center_y >= min_y && current_center_y <= max_y); 
}

float MouseThread::calculateTargetDistanceSquared(const AimbotTarget &target) const
{
    float current_center_x, current_center_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }

    float dx = target.x + target.w * 0.5f - current_center_x;
    float target_center_y_val;

    
    int local_head_class_id_to_use = -1;
    bool local_apply_head_offset = false;
    float local_head_y_offset_val;
    float local_body_y_offset_val;
    std::string local_head_class_name_val;

    {
        std::lock_guard<std::mutex> lock(configMutex); 
        local_head_class_name_val = config.head_class_name;
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == local_head_class_name_val) {
                local_head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    local_apply_head_offset = true;
                }
                break;
            }
        }
        local_head_y_offset_val = config.head_y_offset;
        local_body_y_offset_val = config.body_y_offset;
    }

    if (local_apply_head_offset && target.classId == local_head_class_id_to_use) {
        target_center_y_val = target.y + target.h * local_head_y_offset_val;
    } else {
        target_center_y_val = target.y + target.h * local_body_y_offset_val;
    }
    float dy = target_center_y_val - current_center_y;
    return dx * dx + dy * dy;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;

    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

    
    Point2D raw_target_pos;
    raw_target_pos.x = target.x + target.w * 0.5f;
    
    
    float local_y_offset_multiplier_val;
    int local_head_class_id_to_use = -1;
    bool local_apply_head_offset = false;
    bool local_disable_upward_aim_active = false;
    std::vector<std::string> local_button_disable_upward_aim;


    {
        std::lock_guard<std::mutex> lock(configMutex); 
        
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == config.head_class_name) {
                local_head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    local_apply_head_offset = true;
                }
                break;
            }
        }

        if (local_apply_head_offset && target.classId == local_head_class_id_to_use) {
            local_y_offset_multiplier_val = config.head_y_offset;
        } else {
            local_y_offset_multiplier_val = config.body_y_offset;
        }
        
        
        local_button_disable_upward_aim = config.button_disable_upward_aim; 
    }
    raw_target_pos.y = target.y + target.h * local_y_offset_multiplier_val;

    
    
    local_disable_upward_aim_active = isAnyKeyPressed(local_button_disable_upward_aim);


    
    Point2D predicted_target_pos = raw_target_pos;

    
    float error_x = predicted_target_pos.x - current_center_x;
    float error_y = predicted_target_pos.y - current_center_y;

    if (tracking_errors)
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (error_callback)
        {
            error_callback(error_x, error_y);
        }
    }

    
    Eigen::Vector2f error(error_x, error_y);
    auto pid_start_time = std::chrono::steady_clock::now();
    Eigen::Vector2f pid_output = pid_controller->calculate(error);
    auto pid_end_time = std::chrono::steady_clock::now();
    float pid_duration_ms = std::chrono::duration<float, std::milli>(pid_end_time - pid_start_time).count();
    g_current_pid_calc_time_ms.store(pid_duration_ms, std::memory_order_relaxed);
    add_to_history(g_pid_calc_time_history, pid_duration_ms, g_pid_calc_history_mutex);

    
    float move_x = pid_output.x() * current_move_scale_x;
    float move_y = pid_output.y() * current_move_scale_y;

    int dx_int = static_cast<int>(std::round(move_x));
    int dy_int = static_cast<int>(std::round(move_y));

    if (local_disable_upward_aim_active && dy_int < 0)
    {
        dy_int = 0; 
    }

    
    if (dx_int != 0 || dy_int != 0)
    {        
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            auto input_send_start_time = std::chrono::steady_clock::now();
            input_method->move(dx_int, dy_int);
            auto input_send_end_time = std::chrono::steady_clock::now();
            float input_send_duration_ms = std::chrono::duration<float, std::milli>(input_send_end_time - input_send_start_time).count();
            g_current_input_send_time_ms.store(input_send_duration_ms, std::memory_order_relaxed);
            add_to_history(g_input_send_time_history, input_send_duration_ms, g_input_send_history_mutex);
        }
    }
}

void MouseThread::pressMouse(const AimbotTarget &target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    auto bScope = checkTargetInScope(target.x, target.y, target.w, target.h, bScope_multiplier);

    if (bScope && !mouse_pressed)
    {
        if (input_method && input_method->isValid())
        {
            auto input_press_start_time = std::chrono::steady_clock::now();
            input_method->press();
            auto input_press_end_time = std::chrono::steady_clock::now();
            float press_duration_ms = std::chrono::duration<float, std::milli>(input_press_end_time - input_press_start_time).count();
            g_current_input_send_time_ms.store(press_duration_ms, std::memory_order_relaxed); 
            add_to_history(g_input_send_time_history, press_duration_ms, g_input_send_history_mutex);
        }
        mouse_pressed = true;
    }
}

void MouseThread::releaseMouse()
{
    if (!mouse_pressed)
        return;

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (input_method && input_method->isValid())
    {
        auto input_release_start_time = std::chrono::steady_clock::now();
        input_method->release();
        auto input_release_end_time = std::chrono::steady_clock::now();
        float release_duration_ms = std::chrono::duration<float, std::milli>(input_release_end_time - input_release_start_time).count();
        g_current_input_send_time_ms.store(release_duration_ms, std::memory_order_relaxed); 
        add_to_history(g_input_send_time_history, release_duration_ms, g_input_send_history_mutex);
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

    auto now_chrono_recoil = std::chrono::steady_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_chrono_recoil - last_recoil_compensation_time);
    
    long long current_norecoil_ms;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_norecoil_ms = static_cast<long long>(this->norecoil_ms);
    }
    auto required_delay = std::chrono::milliseconds(current_norecoil_ms);

    if (elapsed >= required_delay)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex); 
        
        int dy_recoil = static_cast<int>(std::round(strength));
        if (dy_recoil != 0) {
             auto recoil_send_start_time = std::chrono::steady_clock::now();
             input_method->move(0, dy_recoil);
             auto recoil_send_end_time = std::chrono::steady_clock::now();
             float recoil_send_duration_ms = std::chrono::duration<float, std::milli>(recoil_send_end_time - recoil_send_start_time).count();
             g_current_input_send_time_ms.store(recoil_send_duration_ms, std::memory_order_relaxed);
             add_to_history(g_input_send_time_history, recoil_send_duration_ms, g_input_send_history_mutex);
        }
        
        last_recoil_compensation_time = now_chrono_recoil;
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

void MouseThread::executeSilentAim(const AimbotTarget& target)
{
    float current_center_x, current_center_y;
    {
        std::lock_guard<std::mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
    }

    
    float target_actual_x = target.x + target.w * 0.5f;
    float target_actual_y;

    
    float local_y_offset_multiplier_val;
    int local_head_class_id_to_use = -1;
    bool local_apply_head_offset = false;
    bool local_disable_upward_aim_active = false;
    std::vector<std::string> local_button_disable_upward_aim;


    {
        std::lock_guard<std::mutex> lock(configMutex); 
        for (const auto& class_setting : config.class_settings) {
            if (class_setting.name == config.head_class_name) {
                local_head_class_id_to_use = class_setting.id;
                if (!class_setting.ignore) {
                    local_apply_head_offset = true;
                }
                break;
            }
        }

        if (local_apply_head_offset && target.classId == local_head_class_id_to_use) {
            local_y_offset_multiplier_val = config.head_y_offset;
        } else {
            local_y_offset_multiplier_val = config.body_y_offset;
        }
        local_button_disable_upward_aim = config.button_disable_upward_aim; 
    }
    target_actual_y = target.y + target.h * local_y_offset_multiplier_val;
    
    
    
    local_disable_upward_aim_active = isAnyKeyPressed(local_button_disable_upward_aim);


    
    float delta_x_float = target_actual_x - current_center_x;
    float delta_y_float = target_actual_y - current_center_y;

    int dx = static_cast<int>(std::round(delta_x_float));
    int dy = static_cast<int>(std::round(delta_y_float));
    
    if (local_disable_upward_aim_active && dy < 0) {
        dy = 0;
    }

    
    if (dx != 0 || dy != 0)
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method && input_method->isValid())
        {
            auto action_start_time = std::chrono::steady_clock::now();

            
            input_method->move(dx, dy);
            
            
            

            
            input_method->press();

            
            std::this_thread::sleep_for(silent_aim_click_duration_ms);

            
            input_method->release();

            auto action_end_time = std::chrono::steady_clock::now();
            float action_duration_ms = std::chrono::duration<float, std::milli>(action_end_time - action_start_time).count();
            g_current_input_send_time_ms.store(action_duration_ms, std::memory_order_relaxed);
            add_to_history(g_input_send_time_history, action_duration_ms, g_input_send_history_mutex);
        }
    }
}
