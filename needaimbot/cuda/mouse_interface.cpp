#include "mouse_interface.h"
#include <iostream>

// Forward declare the C interface functions without including mouse.h
extern "C" {
    void executeMouseMovement(int dx, int dy);
    void executeMouseClick(bool press);
}

namespace needaimbot {
namespace cuda {

void executeMouseMovementFromGPU(int dx, int dy) {
    // Call the actual mouse movement function
    executeMouseMovement(dx, dy);
}

void executeMouseClickFromGPU(bool press) {
    // Call the actual mouse click function
    executeMouseClick(press);
}

} // namespace cuda
} // namespace needaimbot