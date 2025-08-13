// Stub implementation for GPU Mouse Controller
// This is a temporary CPU implementation until CUDA file is properly added to project

#include "gpu_mouse_controller.h"
#include <cstring>

class GpuMouseController {
public:
    float dx = 0, dy = 0;
    bool should_move = false;
    int screen_width, screen_height;
    
    // PID state
    float integral_x = 0, integral_y = 0;
    float prev_error_x = 0, prev_error_y = 0;
    
    GpuMouseController(int width, int height) 
        : screen_width(width), screen_height(height) {}
};

extern "C" {
    GpuMouseController* createGpuMouseController(int width, int height) {
        return new GpuMouseController(width, height);
    }
    
    void destroyGpuMouseController(GpuMouseController* controller) {
        delete controller;
    }
    
    void calculateMouseAsync(
        GpuMouseController* controller,
        const Target* best_target,
        bool has_target,
        float kp_x, float ki_x, float kd_x,
        float kp_y, float ki_y, float kd_y
    ) {
        if (!controller || !has_target || !best_target) {
            controller->should_move = false;
            return;
        }
        
        // Simple PID calculation on CPU (temporary)
        float center_x = controller->screen_width * 0.5f;
        float center_y = controller->screen_height * 0.5f;
        
        float target_center_x = best_target->x + best_target->width * 0.5f;
        float target_center_y = best_target->y + best_target->height * 0.5f;
        
        
        
        float error_x = target_center_x - center_x;
        float error_y = target_center_y - center_y;
        
        // Deadzone
        if (fabs(error_x) < 1.5f && fabs(error_y) < 1.5f) {
            controller->should_move = false;
            return;
        }
        
        // P
        float p_x = kp_x * error_x;
        float p_y = kp_y * error_y;
        
        // I
        controller->integral_x += error_x;
        controller->integral_y += error_y;
        controller->integral_x = fmin(fmax(controller->integral_x, -50.0f), 50.0f);
        controller->integral_y = fmin(fmax(controller->integral_y, -50.0f), 50.0f);
        float i_x = ki_x * controller->integral_x;
        float i_y = ki_y * controller->integral_y;
        
        // D
        float d_x = kd_x * (error_x - controller->prev_error_x);
        float d_y = kd_y * (error_y - controller->prev_error_y);
        
        controller->prev_error_x = error_x;
        controller->prev_error_y = error_y;
        
        // Output
        controller->dx = p_x + i_x + d_x;
        controller->dy = p_y + i_y + d_y;
        
        // Limit
        controller->dx = fmin(fmax(controller->dx, -100.0f), 100.0f);
        controller->dy = fmin(fmax(controller->dy, -100.0f), 100.0f);
        
        controller->should_move = true;
    }
    
    bool waitForMouseMovement(GpuMouseController* controller, float* dx, float* dy) {
        if (!controller || !controller->should_move) {
            return false;
        }
        
        *dx = controller->dx;
        *dy = controller->dy;
        controller->should_move = false;
        return true;
    }
}