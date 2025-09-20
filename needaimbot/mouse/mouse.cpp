// Simplified mouse.cpp - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline

#include "mouse.h"

#include <memory>
#include <mutex>

#include "input_drivers/InputMethod.h"

namespace {
    // Simplified interface for GPU to call
    // Global input method for direct GPU->mouse control
    std::unique_ptr<InputMethod> g_globalInputMethod;
    std::mutex g_globalInputMutex;

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
}

// Call this from main initialization to set the correct input method
void setGlobalInputMethod(std::unique_ptr<InputMethod> method) {
    std::lock_guard<std::mutex> lock(g_globalInputMutex);
    g_globalInputMethod = std::move(method);
}

extern "C" {
    void executeMouseMovement(int dx, int dy) {
        if (dx == 0 && dy == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(g_globalInputMutex);
        moveWithFallback(dx, dy);
    }

    void executeMouseClick(bool press) {
        std::lock_guard<std::mutex> lock(g_globalInputMutex);
        if (press) {
            pressWithFallback();
        } else {
            releaseWithFallback();
        }
    }
}