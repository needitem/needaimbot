// Simplified mouse.cpp - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline

#include "mouse.h"
#include "rapidfire.h"
#include "input_drivers/SerialConnection.h"
#include "input_drivers/MakcuConnection.h"
#include "input_drivers/ghub.h"
#include "input_drivers/kmboxNet.h"
#include "input_drivers/rzctl.h"
#include "../config/config.h"
#include "../AppContext.h"

#include "../core/windows_headers.h"
#include <thread>
#include <atomic>
#include <memory>

// Original constructor for compatibility
MouseThread::MouseThread(
    int resolution,
    float bScope_multiplier,
    float norecoil_ms,
    SerialConnection *serialConnection,
    MakcuConnection *makcuConnection,
    GhubMouse *gHub)
    : screen_width(resolution), screen_height(resolution),
      bScope_multiplier(bScope_multiplier), norecoil_ms(norecoil_ms)
{
    initializeInputMethod(serialConnection, makcuConnection, gHub);
    
    // Direct input - no worker thread needed
    should_stop_thread_ = false;
}

// Simplified constructor
MouseThread::MouseThread(
    SerialConnection *serialConnection,
    MakcuConnection *makcuConnection,
    GhubMouse *gHub)
    : screen_width(1920), screen_height(1080),
      bScope_multiplier(1.0f), norecoil_ms(0.0f)
{
    initializeInputMethod(serialConnection, makcuConnection, gHub);
    
    // Direct input - no worker thread needed
    should_stop_thread_ = false;
}

MouseThread::~MouseThread() 
{
    // No worker thread to stop in direct input mode
    should_stop_thread_ = true;
}

void MouseThread::initializeInputMethod(
    SerialConnection *serialConnection, 
    MakcuConnection *makcuConnection, 
    GhubMouse *gHub)
{
    if (serialConnection && serialConnection->isOpen()) {
        input_method = std::make_unique<SerialInputMethod>(serialConnection);
    }
    else if (makcuConnection && makcuConnection->isOpen()) {
        input_method = std::make_unique<MakcuInputMethod>(makcuConnection);
    }
    else if (gHub) {
        input_method = std::make_unique<GHubInputMethod>(gHub);
    }
    else {
        input_method = std::make_unique<Win32InputMethod>();
    }
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}

// Main function - execute movement calculated by GPU (direct)
void MouseThread::executeMovement(int dx, int dy)
{
    if (dx != 0 || dy != 0) {
        directMouseMove(dx, dy);
    }
}

// Execute mouse press (calculated by GPU) (direct)
void MouseThread::executePress()
{
    directMouseClick(true);
}

// Execute mouse release (calculated by GPU) (direct)
void MouseThread::executeRelease()
{
    directMouseClick(false);
}

// Direct input helpers - no queue, immediate execution
void MouseThread::directMouseMove(int dx, int dy)
{
    std::lock_guard<std::mutex> input_lock(input_method_mutex);
    if (input_method && input_method->isValid()) {
        input_method->move(dx, dy);
    }
}

void MouseThread::directMouseClick(bool press)
{
    std::lock_guard<std::mutex> input_lock(input_method_mutex);
    if (input_method && input_method->isValid()) {
        if (press) {
            input_method->press();
        } else {
            input_method->release();
        }
    }
}

// Configuration update
void MouseThread::updateConfig(int resolution, float bScope_multiplier, float norecoil_ms)
{
    this->screen_width = resolution;
    this->screen_height = resolution;
    this->bScope_multiplier = bScope_multiplier;
    this->norecoil_ms = norecoil_ms;
}

void MouseThread::updateRapidFire()
{
    // Stub implementation - RapidFire not yet integrated
    // TODO: Implement when RapidFire is needed
}

// Simplified interface for GPU to call
// Global input method for direct GPU->mouse control
static std::unique_ptr<InputMethod> g_globalInputMethod;
static std::mutex g_globalInputMutex;

// Call this from main initialization to set the correct input method
void setGlobalInputMethod(std::unique_ptr<InputMethod> method) {
    std::lock_guard<std::mutex> lock(g_globalInputMutex);
    g_globalInputMethod = std::move(method);
}

// Public API to configure input from SerialConnection, MakcuConnection, or GhubMouse
void configureGlobalInput(SerialConnection* serial, MakcuConnection* makcu, GhubMouse* ghub) {
    std::unique_ptr<InputMethod> method;
    
    if (serial && serial->isOpen()) {
        method = std::make_unique<SerialInputMethod>(serial);
    } else if (makcu && makcu->isOpen()) {
        method = std::make_unique<MakcuInputMethod>(makcu);
    } else if (ghub) {
        method = std::make_unique<GHubInputMethod>(ghub);
    } else {
        method = std::make_unique<Win32InputMethod>();
    }
    
    setGlobalInputMethod(std::move(method));
}

extern "C" {
    void executeMouseMovement(int dx, int dy) {
        std::lock_guard<std::mutex> lock(g_globalInputMutex);
        if (g_globalInputMethod && g_globalInputMethod->isValid()) {
            g_globalInputMethod->move(dx, dy);
        }
    }
    
    void executeMouseClick(bool press) {
        std::lock_guard<std::mutex> lock(g_globalInputMutex);
        if (g_globalInputMethod && g_globalInputMethod->isValid()) {
            if (press) {
                g_globalInputMethod->press();
            } else {
                g_globalInputMethod->release();
            }
        }
    }
}