// Simplified mouse.cpp - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline
// Lock-free queue for zero-contention between GPU and CPU

#include "mouse.h"
#include "lockfree_mouse_queue.h"

#include <memory>
#include <thread>
#include <atomic>

#include "input_drivers/InputMethod.h"

namespace {
    // Lock-free queue - GPU writes, CPU reads
    LockFreeMouseQueue<2048> g_mouseQueue;

    // Consumer thread
    std::thread g_consumerThread;
    std::atomic<bool> g_running{false};

    // Global input method for direct GPU->mouse control
    std::unique_ptr<InputMethod> g_globalInputMethod;

    InputMethod* getActiveInputMethod() {
        if (g_globalInputMethod && g_globalInputMethod->isValid()) {
            return g_globalInputMethod.get();
        }
        return nullptr;
    }

    InputMethod& getFallbackInputMethod() {
        static Win32InputMethod fallback;
        return fallback;
    }

    void moveWithFallback(int dx, int dy) {
        if (InputMethod* active = getActiveInputMethod()) {
            active->move(dx, dy);
            return;
        }
        getFallbackInputMethod().move(dx, dy);
    }

    void pressWithFallback() {
        if (InputMethod* active = getActiveInputMethod()) {
            active->press();
            return;
        }
        getFallbackInputMethod().press();
    }

    void releaseWithFallback() {
        if (InputMethod* active = getActiveInputMethod()) {
            active->release();
            return;
        }
        getFallbackInputMethod().release();
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
                        moveWithFallback(cmd.dx, cmd.dy);
                        break;
                    case MouseCommandType::PRESS:
                        pressWithFallback();
                        break;
                    case MouseCommandType::RELEASE:
                        releaseWithFallback();
                        break;
                    default:
                        break;
                }
            } else {
                // Queue empty - use adaptive sleep
                emptyCount++;
                if (emptyCount > 10) {
                    // Only sleep if queue has been empty for a while
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                } else {
                    // Spin briefly to catch new data faster
                    std::this_thread::yield();
                }
            }
        }
    }
}

// Call this from main initialization to set the correct input method
void setGlobalInputMethod(std::unique_ptr<InputMethod> method) {
    g_globalInputMethod = std::move(method);
}

// Start the consumer thread
void startMouseConsumer() {
    if (!g_running.exchange(true, std::memory_order_release)) {
        g_consumerThread = std::thread(mouseConsumerThread);
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