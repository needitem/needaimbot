#pragma once

// Mouse interface for CUDA code
// This header avoids Windows.h inclusion to prevent conflicts in CUDA files

namespace needaimbot {
namespace cuda {

// Mouse movement interface for GPU pipeline
// These functions are implemented in mouse_interface.cpp
void executeMouseMovementFromGPU(int dx, int dy);
void executeMouseClickFromGPU(bool press);

} // namespace cuda
} // namespace needaimbot