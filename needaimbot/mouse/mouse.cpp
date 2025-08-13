#include "../AppContext.h"
#include "mouse.h"
#include "aimbot_components/AimbotTarget.h"
// #include "../capture/capture.h" - removed, using GPU capture now
#include "input_drivers/SerialConnection.h"
#include "input_drivers/MakcuConnection.h"
#include "../needaimbot.h"
#include "input_drivers/ghub.h"
#include "../config/config.h"
#include "../keyboard/keyboard_listener.h"

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
#ifdef _WIN32
#include <intrin.h>  // For _mm_prefetch
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "aimbot_components/BezierController.h"


extern std::atomic<bool> aiming;
extern std::mutex configMutex;

std::random_device rd;
std::mt19937 gen(rd());

constexpr float SCOPE_MARGIN = 0.15f;




thread_local static LARGE_INTEGER freq;
thread_local static bool freq_initialized = false;

void InitializeHighPrecisionTimer() {
    if (!freq_initialized) {
        QueryPerformanceFrequency(&freq);
        freq_initialized = true;
    }
}

void QueueMove(int dx, int dy, std::function<void(int, int)> move_func) {
    auto& ctx = AppContext::getInstance();
    InitializeHighPrecisionTimer();
    
    if (dx != 0 || dy != 0) {
        // Skip timing measurements if not needed (performance optimization)
        if (ctx.config.show_metrics) {
            LARGE_INTEGER start;
            QueryPerformanceCounter(&start);
            
            move_func(dx, dy);
            
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            float elapsed_us = ((end.QuadPart - start.QuadPart) * 1000000.0f) / freq.QuadPart;
            
            ctx.add_to_history(ctx.g_input_send_time_history, elapsed_us / 1000.0f, ctx.g_input_send_history_mutex);
            ctx.g_current_input_send_time_ms.store(elapsed_us / 1000.0f);
        } else {
            move_func(dx, dy);
        }
    }
}

MouseThread::MouseThread(
    int resolution,
    float bScope_multiplier,
    float norecoil_ms,
    SerialConnection *serialConnection,
    MakcuConnection *makcuConnection,
    GhubMouse *gHub)
{
    initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    bezier_controller = std::make_unique<BezierController>();
    
    initializeInputMethod(serialConnection, makcuConnection, gHub);
    
    // Initialize RapidFire
    rapid_fire = std::make_unique<RapidFire>();
    rapid_fire->start();
    updateRapidFire(); // Set initial state from config
    
    // Start async input worker thread
    async_input_thread_ = std::thread(&MouseThread::asyncInputWorker, this);
}

MouseThread::~MouseThread() {
    // Stop RapidFire
    if (rapid_fire) {
        rapid_fire->stop();
    }
    
    // Stop the async worker thread
    should_stop_thread_ = true;
    
    if (async_input_thread_.joinable()) {
        async_input_thread_.join();
    }
}

void MouseThread::initializeInputMethod(SerialConnection *serialConnection, MakcuConnection *makcuConnection, GhubMouse *gHub)
{
    std::shared_ptr<InputMethod> shared_input_method;
    
    if (serialConnection && serialConnection->isOpen())
    {
        input_method = std::make_unique<SerialInputMethod>(serialConnection);
        shared_input_method = std::make_shared<SerialInputMethod>(serialConnection);
    }
    else if (makcuConnection && makcuConnection->isOpen())
    {
        input_method = std::make_unique<MakcuInputMethod>(makcuConnection);
        shared_input_method = std::make_shared<MakcuInputMethod>(makcuConnection);
    }
    else if (gHub)
    {
        input_method = std::make_unique<GHubInputMethod>(gHub);
        shared_input_method = std::make_shared<GHubInputMethod>(gHub);
    }
    else
    {
        input_method = std::make_unique<Win32InputMethod>();
        shared_input_method = std::make_shared<Win32InputMethod>();
    }
    
    // Update RapidFire with the same input method
    if (rapid_fire) {
        rapid_fire->setInputMethod(shared_input_method);
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

    const float SENSITIVITY_FACTOR = 1.0f; // Default sensitivity factor
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
    this->accumulated_x_ = 0.0f;
    this->accumulated_y_ = 0.0f;
}

void MouseThread::updateConfig(
    int resolution,
    float bScope_multiplier,
    float norecoil_ms
    )
{
    auto& ctx = AppContext::getInstance();
    {
        std::unique_lock<std::shared_mutex> lock(member_data_mutex_); 
        initializeScreen(resolution, bScope_multiplier, norecoil_ms);
    }
    // Bezier parameters are now managed through the controller's Parameters struct
}





bool MouseThread::checkTargetInScope(float target_x, float target_y, float target_w, float target_h, float reduction_factor)
{
    // Cache screen values with reduced mutex access
    static thread_local struct ScopeCache {
        float screen_width, screen_height, center_x, center_y;
        float margin_x, margin_y;
        // Time tracking removed - using frame counter
        
        void update(float sw, float sh, float cx, float cy) {
            screen_width = sw;
            screen_height = sh;
            center_x = cx;
            center_y = cy;
            margin_x = sw * 0.1f;  // Pre-calculate margins
            margin_y = sh * 0.1f;
            // No time tracking needed
        }
    } cache;
    
    // Update cache every 8 frames (no time check)
    static thread_local uint32_t cache_frame_counter = 0;
    if ((++cache_frame_counter & 7) == 0) {
        std::shared_lock<std::shared_mutex> lock(member_data_mutex_);
        cache.update(this->screen_width, this->screen_height, this->center_x, this->center_y);
    }
    
    // Fast calculations without divisions
    float target_center_x = target_x + target_w * 0.5f;
    float target_center_y = target_y + target_h * 0.5f;
    
    // Early exit with squared distance check (no abs/sqrt needed)
    float dx = target_center_x - cache.center_x;
    float dy = target_center_y - cache.center_y;
    
    // Quick margin check without abs()
    if (dx > cache.margin_x || dx < -cache.margin_x || 
        dy > cache.margin_y || dy < -cache.margin_y) {
        return false;
    }
    
    // Precise box check
    float half_w = target_w * reduction_factor * 0.5f;
    float half_h = target_h * reduction_factor * 0.5f;
    
    return (dx >= -half_w && dx <= half_w && 
            dy >= -half_h && dy <= half_h);
}



void MouseThread::moveMouse(const AimbotTarget &target)
{
#ifdef _WIN32
    // Prefetch target data into cache for faster access
    _mm_prefetch(reinterpret_cast<const char*>(&target), _MM_HINT_T0);
#endif
    auto& ctx = AppContext::getInstance();
    
    // Fast path: early exit checks (most likely failures first)
    [[unlikely]] if ((target.width | target.height) <= 0) return;  // Bitwise OR for single comparison
    
    float current_center_x, current_center_y;
    float current_move_scale_x, current_move_scale_y;
    
    // Cache for config values to reduce mutex locks
    alignas(64) static thread_local struct ConfigCache {
        int head_class_id_to_use = -1;
        bool apply_head_offset = false;
        float head_y_offset, body_y_offset;
        std::string head_class_name;
        // Time checks removed - using frame counter instead
        
        void update(const Config& config) {
            // Frame-based update without time check
            static uint32_t update_counter = 0;
            if ((++update_counter & 15) == 0) {  // Every 16 frames
                head_class_name = config.head_class_name;
                head_y_offset = config.head_y_offset;
                body_y_offset = config.body_y_offset;
                
                head_class_id_to_use = -1;
                apply_head_offset = false;
                for (const auto& class_setting : config.class_settings) {
                    if (class_setting.name == head_class_name) {
                        head_class_id_to_use = class_setting.id;
                        apply_head_offset = class_setting.allow;
                        break;
                    }
                }
                // Frame counter updated, no time tracking
            }
        }
    } config_cache;

    {
        std::shared_lock<std::shared_mutex> lock(member_data_mutex_); 
        current_center_x = this->center_x;
        current_center_y = this->center_y;
        current_move_scale_x = this->move_scale_x;
        current_move_scale_y = this->move_scale_y;
    }

    // Update config cache if needed
    {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        config_cache.update(ctx.config);
    }

    // Calculate target position with cached offsets (branchless)
    float target_center_x = target.x + target.width * 0.5f;
    // Use arithmetic to select offset without branching
    int use_head = (config_cache.apply_head_offset && target.classId == config_cache.head_class_id_to_use);
    float target_center_y = target.y + target.height * 
        (config_cache.head_y_offset * use_head + config_cache.body_y_offset * (1 - use_head));

    // Early distance check for optimization
    float error_x = target_center_x - current_center_x;
    float error_y = target_center_y - current_center_y;
    
    // Quick exit if error is too small
    float error_magnitude_sq = error_x * error_x + error_y * error_y;
    [[unlikely]] if (error_magnitude_sq < 0.25f) { // Less than 0.5 pixels
        return;
    }
    
    // Check disable upward aim - no time check, just counter
    static thread_local bool local_disable_upward_aim_active = false;
    static thread_local uint32_t frame_counter = 0;
    
    // Update every 8 frames (branchless modulo)
    uint32_t should_update = ((++frame_counter & 7) == 0);
    if (should_update) {
        std::lock_guard<std::mutex> lock(ctx.configMutex);
        local_disable_upward_aim_active = isAnyKeyPressed(ctx.config.button_disable_upward_aim);
    }
    
    // Bezier curve calculation
    LA::Vector2f error(error_x, error_y);
    LA::Vector2f bezier_output;
    
    // Calculate movement using Bezier curve
    bezier_output = bezier_controller->calculate(error);
    
    float move_x = bezier_output.x() * current_move_scale_x;
    float move_y = bezier_output.y() * current_move_scale_y;
    
    // Process accumulated movement
    auto [dx_int, dy_int] = processAccumulatedMovement(move_x, move_y);
    
    // Apply upward aim restriction without branching (branchless)
    // If upward aim is disabled and dy is negative, set to 0
    int upward_mask = -(local_disable_upward_aim_active && dy_int < 0);  // -1 if true, 0 if false
    dy_int = dy_int & ~upward_mask;  // Clear dy_int if mask is -1
    
    // Always enqueue - let the worker filter zero movements
    // This removes branch misprediction cost
    MouseCommand cmd(MouseCommand::MOVE, dx_int, dy_int);
    mouse_command_queue_.enqueue(std::move(cmd));
}

void MouseThread::pressMouse(const AimbotTarget &target)
{
    auto& ctx = AppContext::getInstance();

    // For triggerbot, use a more lenient scope check (1.5x the normal multiplier)
    float triggerbot_scope_multiplier = bScope_multiplier * 1.5f;
    auto bScope = checkTargetInScope(static_cast<float>(target.x), static_cast<float>(target.y), static_cast<float>(target.width), static_cast<float>(target.height), triggerbot_scope_multiplier);

    [[unlikely]] if (bScope && !mouse_pressed)
    {
        // Add a small delay after release to prevent rapid fire
        auto now = std::chrono::steady_clock::now();
        auto time_since_release = std::chrono::duration<float, std::milli>(now - last_mouse_release_time).count();
        
        // Only press if at least 50ms have passed since last release
        if (time_since_release > 50.0f) {
            // Use async queue instead of direct call
            enqueueMouseCommand(MouseCommand::PRESS);
            mouse_pressed = true;
            last_mouse_press_time = now;
        }
    }
}

void MouseThread::releaseMouse()
{
    auto& ctx = AppContext::getInstance();
    [[likely]] if (!mouse_pressed)
        return;

    // Add a small delay to prevent rapid press/release cycles
    auto now = std::chrono::steady_clock::now();
    auto time_since_press = std::chrono::duration<float, std::milli>(now - last_mouse_press_time).count();
    
    // Only release if at least 100ms have passed since press (prevents rapid clicking)
    if (time_since_press < 100.0f)
        return;

    // Use async queue instead of direct call
    enqueueMouseCommand(MouseCommand::RELEASE);
    mouse_pressed = false;
    last_mouse_release_time = now;
}

void MouseThread::applyRecoilCompensation(float strength)
{
    // Always perform direct recoil control
    float current_norecoil_ms;
    {
        std::shared_lock<std::shared_mutex> lock(member_data_mutex_); 
        current_norecoil_ms = this->norecoil_ms;
    }
    
    applyRecoilCompensationInternal(strength, current_norecoil_ms);
}

void MouseThread::applyWeaponRecoilCompensation(const WeaponRecoilProfile* profile, int scope_magnification)
{
    // Always perform direct recoil control
    if (!profile) {
        return;
    }

    float scope_multiplier = 1.0f;
    switch (scope_magnification) {
        case 1: scope_multiplier = profile->scope_mult_1x; break;
        case 2: scope_multiplier = profile->scope_mult_2x; break;
        case 3: scope_multiplier = profile->scope_mult_3x; break;
        case 4: scope_multiplier = profile->scope_mult_4x; break;
        case 6: scope_multiplier = profile->scope_mult_6x; break;
        case 8: scope_multiplier = profile->scope_mult_8x; break;
        default: scope_multiplier = profile->scope_mult_1x; break;
    }

    float adjusted_strength = profile->base_strength * profile->fire_rate_multiplier * scope_multiplier;
    
    applyRecoilCompensationInternal(adjusted_strength, profile->recoil_ms);
}




void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    
    // Create a shared_ptr version for RapidFire
    // We need to determine the actual type to create a proper shared_ptr
    auto& ctx = AppContext::getInstance();
    std::shared_ptr<InputMethod> shared_method;
    
    if (ctx.config.input_method == "ARDUINO") {
        // Note: We can't directly share the unique_ptr, need to create a new instance
        // This is a limitation - we'd need to refactor to use shared_ptr throughout
        shared_method = std::make_shared<Win32InputMethod>();
    } else if (ctx.config.input_method == "GHUB") {
        shared_method = std::make_shared<Win32InputMethod>();
    } else if (ctx.config.input_method == "KMBOX") {
        shared_method = std::make_shared<KmboxInputMethod>();
    } else if (ctx.config.input_method == "RAZER") {
        shared_method = std::make_shared<RZInputMethod>();
    } else {
        shared_method = std::make_shared<Win32InputMethod>();
    }
    
    input_method = std::move(new_method);
    
    // Update RapidFire with the new input method
    if (rapid_fire) {
        rapid_fire->setInputMethod(shared_method);
    }
}




inline std::pair<int, int> MouseThread::processAccumulatedMovement(float move_x, float move_y)
{
    // Accumulate sub-pixel movements
    accumulated_x_ += move_x;
    accumulated_y_ += move_y;
    
    // Use round instead of floor to reduce bias and stick-slip near zero
    int dx_int = static_cast<int>(std::round(accumulated_x_));
    int dy_int = static_cast<int>(std::round(accumulated_y_));
    
    // Subtract the integer part from accumulated values
    accumulated_x_ -= dx_int;
    accumulated_y_ -= dy_int;
    
    return {dx_int, dy_int};
}




void MouseThread::applyRecoilCompensationInternal(float strength, float delay_ms)
{
    auto& ctx = AppContext::getInstance();

    if (std::abs(strength) < 1e-3f) {
        return;
    }

    auto now_chrono_recoil = std::chrono::steady_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_chrono_recoil - last_recoil_compensation_time);
    auto required_delay = std::chrono::milliseconds(static_cast<long long>(delay_ms));

    if (elapsed >= required_delay) {
        // Apply strength directly without accumulation to prevent over-compensation
        int dy_recoil = static_cast<int>(std::round(strength));
        
        // Debug logging removed
        
        if (dy_recoil != 0) {
            // Use async queue instead of direct call
            enqueueMouseCommand(MouseCommand::MOVE, 0, dy_recoil);
        }
        
        last_recoil_compensation_time = now_chrono_recoil;
    }
}


void MouseThread::resetAccumulatedStates()
{
    // Reset Bezier controller
    if (bezier_controller) {
        bezier_controller->reset();
    }
    
    // Reset movement accumulation
    accumulated_x_ = 0.0f;
    accumulated_y_ = 0.0f;
}

void MouseThread::updateRapidFire()
{
    auto& ctx = AppContext::getInstance();
    
    if (rapid_fire) {
        // Enable/disable based on config
        rapid_fire->setEnabled(ctx.config.enable_rapidfire);
        
        // Set CPS (clicks per second) from config
        rapid_fire->setClicksPerSecond(ctx.config.rapidfire_cps);
    }
}

void MouseThread::asyncInputWorker()
{
    auto& ctx = AppContext::getInstance();
    
    // Set thread priority for better responsiveness
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    
    // Metrics removed for performance
    
    [[likely]] while (!should_stop_thread_.load(std::memory_order_relaxed)) {
        MouseCommand cmd;
        
        // Batch dequeue for better throughput
        constexpr size_t BATCH_SIZE = 4;
        MouseCommand batch[BATCH_SIZE];
        size_t batch_count = 0;
        
        // Try to dequeue multiple commands at once
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            if (mouse_command_queue_.tryDequeue(batch[i], i == 0 ? 1 : 0)) {
                batch_count++;
            } else {
                break;
            }
        }
        
        [[unlikely]] if (batch_count == 0) {
            if (should_stop_thread_.load(std::memory_order_relaxed)) {
                break;
            }
            continue;
        }
        
        // Metrics completely removed - no periodic checks
        
        // Process batch
        {
            std::lock_guard<std::mutex> input_lock(input_method_mutex);
            [[likely]] if (input_method && input_method->isValid()) {
                for (size_t i = 0; i < batch_count; ++i) {
                    const auto& cmd = batch[i];
                    // Remove timing branch - metrics are rarely used anyway
                    
                    switch (cmd.type) {
                        case MouseCommand::MOVE:
                            // Skip zero movements (branchless filtering from moveMouse)
                            if (cmd.dx != 0 || cmd.dy != 0) {
                                input_method->move(cmd.dx, cmd.dy);
                            }
                            break;
                        case MouseCommand::PRESS:
                            input_method->press();
                            break;
                        case MouseCommand::RELEASE:
                            input_method->release();
                            break;
                    }
                    
                    // Metrics removed for performance - no branching
                }
            }
        }
    }
}

void MouseThread::enqueueMouseCommand(MouseCommand::Type type, int dx, int dy)
{
    // Direct enqueue to lockless queue - no locking needed
    MouseCommand cmd(type, dx, dy);
    mouse_command_queue_.enqueue(std::move(cmd));
}
