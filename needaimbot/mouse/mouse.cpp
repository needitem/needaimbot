// Simplified mouse.cpp - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline

#include "mouse.h"

#include <memory>
#include <mutex>

#include "input_drivers/InputMethod.h"

// Simplified interface for GPU to call
// Global input method for direct GPU->mouse control
static std::unique_ptr<InputMethod> g_globalInputMethod;
static std::mutex g_globalInputMutex;

// Call this from main initialization to set the correct input method
void setGlobalInputMethod(std::unique_ptr<InputMethod> method) {
    std::lock_guard<std::mutex> lock(g_globalInputMutex);
    g_globalInputMethod = std::move(method);
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