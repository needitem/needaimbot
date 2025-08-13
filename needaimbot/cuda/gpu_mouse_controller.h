#pragma once

// Forward declaration only
struct Target;

// Forward declaration for GPU Mouse Controller
class GpuMouseController;

// C-style interface functions
extern "C" {
    GpuMouseController* createGpuMouseController(int width, int height);
    void destroyGpuMouseController(GpuMouseController* controller);
    void calculateMouseAsync(
        GpuMouseController* controller,
        const Target* d_best_target,
        bool has_target,
        float kp_x, float ki_x, float kd_x,
        float kp_y, float ki_y, float kd_y
    );
    bool waitForMouseMovement(GpuMouseController* controller, float* dx, float* dy);
}