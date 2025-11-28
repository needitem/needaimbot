// Simplified mouse.h - Only handles input execution
// All calculations are done on GPU via CUDA Graph pipeline

#ifndef MOUSE_SIMPLE_H
#define MOUSE_SIMPLE_H

#include <memory>

#include "input_drivers/InputMethod.h"

// Global input configuration
void setGlobalInputMethod(std::unique_ptr<InputMethod> method);

// Consumer thread management
void startMouseConsumer();
void stopMouseConsumer();

// No recoil feature
void startNoRecoil();
void stopNoRecoil();

// C interface for GPU to call
extern "C" {
    void executeMouseMovement(int dx, int dy);
    void executeMouseClick(bool press);
}

#endif // MOUSE_SIMPLE_H