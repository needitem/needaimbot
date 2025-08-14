// Simplified mouse.h - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline

#ifndef MOUSE_SIMPLE_H
#define MOUSE_SIMPLE_H

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "optimized_mouse_queue.h"
#include "input_drivers/InputMethod.h"
#include "../core/Target.h"

// Forward declarations
class SerialConnection;
class MakcuConnection;
class GhubMouse;
class RapidFire;

// Alias for compatibility with old code
using AimbotTarget = Target;

class MouseThread {
private:
    // Input method for sending mouse commands
    std::unique_ptr<InputMethod> input_method;
    std::mutex input_method_mutex;
    
    // Async command queue
    MouseCommandQueue mouse_command_queue_;
    std::thread async_input_thread_;
    std::atomic<bool> should_stop_thread_{false};
    
    // RapidFire support (stub for now)
    std::unique_ptr<RapidFire> rapid_fire;
    
    // Screen configuration
    float screen_width;
    float screen_height;
    float bScope_multiplier;
    float norecoil_ms;
    
    // Initialize input method based on configuration
    void initializeInputMethod(
        SerialConnection *serialConnection,
        MakcuConnection *makcuConnection,
        GhubMouse *gHub
    );
    
    // Worker thread for processing commands
    void asyncInputWorker();

public:
    // Original constructor for compatibility
    MouseThread(
        int resolution,
        float bScope_multiplier,
        float norecoil_ms,
        SerialConnection *serialConnection = nullptr,
        MakcuConnection *makcuConnection = nullptr,
        GhubMouse *gHub = nullptr
    );
    
    // Simplified constructor
    MouseThread(
        SerialConnection *serialConnection = nullptr,
        MakcuConnection *makcuConnection = nullptr,
        GhubMouse *gHub = nullptr
    );
    
    ~MouseThread();
    
    // Set new input method
    void setInputMethod(std::unique_ptr<InputMethod> new_method);
    
    // Execute commands calculated by GPU
    void executeMovement(int dx, int dy);
    void executePress();
    void executeRelease();
    
    // Configuration update
    void updateConfig(int resolution, float bScope_multiplier, float norecoil_ms);
    
    // RapidFire methods (stubs for now)
    void updateRapidFire();
    RapidFire* getRapidFire() { return rapid_fire.get(); }
};

// C interface for GPU to call
extern "C" {
    void executeMouseMovement(int dx, int dy);
    void executeMouseClick(bool press);
}

#endif // MOUSE_SIMPLE_H