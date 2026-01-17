// Simplified mouse.cpp - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline
// Lock-free queue for zero-contention between GPU and CPU

#include "mouse.h"
#include "lockfree_mouse_queue.h"

#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

#include "input_drivers/InputMethod.h"
#include "../AppContext.h"

namespace {
    // Lock-free queue - GPU writes, CPU reads
    LockFreeMouseQueue<2048> g_mouseQueue;

    // Consumer thread
    std::thread g_consumerThread;
    std::atomic<bool> g_running{false};

    // Stabilizer thread
    std::thread g_stabilizerThread;
    std::atomic<bool> g_stabilizerRunning{false};

    // Global input method for direct GPU->mouse control
    std::unique_ptr<InputMethod> g_globalInputMethod;

    InputMethod* getActiveInputMethod() {
        if (g_globalInputMethod && g_globalInputMethod->isValid()) {
            return g_globalInputMethod.get();
        }
        return nullptr;
    }

    void moveWithInput(int dx, int dy) {
        if (InputMethod* active = getActiveInputMethod()) {
            active->move(dx, dy);
        }
        // No fallback in 2PC - KMBOX or MAKCU required
    }

    void pressWithInput() {
        if (InputMethod* active = getActiveInputMethod()) {
            active->press();
        }
    }

    void releaseWithInput() {
        if (InputMethod* active = getActiveInputMethod()) {
            active->release();
        }
    }

    // Stabilizer thread - moves mouse down using input profile settings
    void stabilizerThread() {
        auto& ctx = AppContext::getInstance();

        while (g_stabilizerRunning.load(std::memory_order_acquire)) {
            if (ctx.stabilizer_active.load(std::memory_order_acquire)) {
                // Get current input profile
                auto* profile = ctx.config.getCurrentInputProfile();
                if (profile) {
                    // Get base strength and apply scope multiplier
                    float strength = profile->base_strength;
                    int scope = ctx.config.profile().active_scope_magnification;

                    switch (scope) {
                        case 1: strength *= profile->scope_mult_1x; break;
                        case 2: strength *= profile->scope_mult_2x; break;
                        case 3: strength *= profile->scope_mult_3x; break;
                        case 4: strength *= profile->scope_mult_4x; break;
                        case 6: strength *= profile->scope_mult_6x; break;
                        case 8: strength *= profile->scope_mult_8x; break;
                        default: strength *= profile->scope_mult_1x; break;
                    }

                    // Apply fire rate multiplier
                    strength *= profile->fire_rate_multiplier;

                    int dy = static_cast<int>(strength + 0.5f);
                    if (dy > 0) {
                        moveWithInput(0, dy);
                    }

                    // Use interval from profile
                    float interval_ms = profile->interval_ms;
                    if (interval_ms < 1.0f) interval_ms = 1.0f;
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(interval_ms * 1000)));
                    continue;
                }
            }

            // Default sleep when not active
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Consumer thread - processes mouse commands
    void mouseConsumerThread() {
        MouseCommand cmd;
        int emptyCount = 0;

        while (g_running.load(std::memory_order_acquire)) {
            if (g_mouseQueue.pop(cmd)) {
                emptyCount = 0; // Reset when we get data

                switch (cmd.type) {
                    case MouseCommandType::MOVE:
                        moveWithInput(cmd.dx, cmd.dy);
                        break;
                    case MouseCommandType::PRESS:
                        pressWithInput();
                        break;
                    case MouseCommandType::RELEASE:
                        releaseWithInput();
                        break;
                    default:
                        break;
                }
            } else {
                // Queue empty - use adaptive busy-wait for minimal jitter
                emptyCount++;
                if (emptyCount > 100) {
                    // After some empty iterations, sleep to save CPU
                    // With timeBeginPeriod(1), this will likely be ~1ms
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } else if (emptyCount > 5) {
                    // Yield quickly to allow other threads (game) to run
                    std::this_thread::yield();
                }
                // For first 5 iterations: pure spin for ultra-low latency
            }
        }
    }
}

// Call this from main initialization to set the correct input method
void setGlobalInputMethod(std::unique_ptr<InputMethod> method) {
    g_globalInputMethod = std::move(method);
}

// Start the consumer thread with optimized priority
void startMouseConsumer() {
    if (!g_running.exchange(true, std::memory_order_release)) {
        g_consumerThread = std::thread(mouseConsumerThread);

#ifdef _WIN32
        // Set priority to ABOVE_NORMAL instead of HIGHEST to prevent game starvation
        HANDLE threadHandle = g_consumerThread.native_handle();
        SetThreadPriority(threadHandle, THREAD_PRIORITY_ABOVE_NORMAL);
        SetThreadDescription(threadHandle, L"MouseConsumer");
#endif
        // Linux: thread priority can be set via pthread if needed
    }
}

// Stop the consumer thread
void stopMouseConsumer() {
    if (g_running.exchange(false, std::memory_order_release)) {
        if (g_consumerThread.joinable()) {
            g_consumerThread.join();
        }
    }
}

// Start the stabilizer thread
void startStabilizer() {
    if (!g_stabilizerRunning.exchange(true, std::memory_order_release)) {
        g_stabilizerThread = std::thread(stabilizerThread);

        #ifdef _WIN32
        HANDLE threadHandle = g_stabilizerThread.native_handle();
        SetThreadDescription(threadHandle, L"InputStabilizer");
        #endif
    }
}

// Stop the stabilizer thread
void stopStabilizer() {
    if (g_stabilizerRunning.exchange(false, std::memory_order_release)) {
        if (g_stabilizerThread.joinable()) {
            g_stabilizerThread.join();
        }
    }
}

extern "C" {
    // GPU calls this - just pushes to lock-free queue
    void executeMouseMovement(int dx, int dy) {
        if (dx == 0 && dy == 0) {
            return;
        }

        // No mutex - just atomic push
        g_mouseQueue.push(MouseCommandType::MOVE, dx, dy);
    }

    void executeMouseClick(bool press) {
        // No mutex - just atomic push
        if (press) {
            g_mouseQueue.push(MouseCommandType::PRESS);
        } else {
            g_mouseQueue.push(MouseCommandType::RELEASE);
        }
    }
}