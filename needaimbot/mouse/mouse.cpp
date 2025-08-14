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
    
    // Start async input worker thread
    should_stop_thread_ = false;
    async_input_thread_ = std::thread(&MouseThread::asyncInputWorker, this);
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
    
    // Start async input worker thread
    should_stop_thread_ = false;
    async_input_thread_ = std::thread(&MouseThread::asyncInputWorker, this);
}

MouseThread::~MouseThread() 
{
    // Stop the async worker thread
    should_stop_thread_ = true;
    
    if (async_input_thread_.joinable()) {
        async_input_thread_.join();
    }
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

// Main function - execute movement calculated by GPU
void MouseThread::executeMovement(int dx, int dy)
{
    if (dx != 0 || dy != 0) {
        MouseCommand cmd(MouseCommand::MOVE, dx, dy);
        mouse_command_queue_.enqueue(std::move(cmd));
    }
}

// Execute mouse press (calculated by GPU)
void MouseThread::executePress()
{
    MouseCommand cmd(MouseCommand::PRESS, 0, 0);
    mouse_command_queue_.enqueue(std::move(cmd));
}

// Execute mouse release (calculated by GPU)
void MouseThread::executeRelease()
{
    MouseCommand cmd(MouseCommand::RELEASE, 0, 0);
    mouse_command_queue_.enqueue(std::move(cmd));
}

// Worker thread - processes queued commands
void MouseThread::asyncInputWorker()
{
    // Set thread priority for better responsiveness
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    
    while (!should_stop_thread_.load(std::memory_order_relaxed)) {
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
        
        if (batch_count == 0) {
            if (should_stop_thread_.load(std::memory_order_relaxed)) {
                break;
            }
            continue;
        }
        
        // Process batch
        {
            std::lock_guard<std::mutex> input_lock(input_method_mutex);
            if (input_method && input_method->isValid()) {
                for (size_t i = 0; i < batch_count; ++i) {
                    const auto& cmd = batch[i];
                    
                    switch (cmd.type) {
                        case MouseCommand::MOVE:
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
                }
            }
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
extern "C" {
    void executeMouseMovement(int dx, int dy) {
        auto& ctx = AppContext::getInstance();
        if (ctx.mouseThread) {
            ctx.mouseThread->executeMovement(dx, dy);
        }
    }
    
    void executeMouseClick(bool press) {
        auto& ctx = AppContext::getInstance();
        if (ctx.mouseThread) {
            if (press) {
                ctx.mouseThread->executePress();
            } else {
                ctx.mouseThread->executeRelease();
            }
        }
    }
}