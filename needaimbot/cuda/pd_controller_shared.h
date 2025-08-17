// Shared definitions between CUDA and C++ for PD Controller
#pragma once

namespace needaimbot {
namespace cuda {

// Structure to store previous error for derivative calculation
struct PDState {
    float prev_error_x;
    float prev_error_y;
    float filtered_derivative_x;
    float filtered_derivative_y;
    bool initialized;
};

// PD Controller configuration
struct PDConfig {
    // PD gains
    float kp_x;  // Proportional gain for X
    float kp_y;  // Proportional gain for Y
    float kd_x;  // Derivative gain for X
    float kd_y;  // Derivative gain for Y
    
    // Deadzone
    float deadzone_x;
    float deadzone_y;
    
    // Derivative filter (to reduce noise)
    float derivative_filter_alpha;  // 0.0 = no filtering, 1.0 = full filtering
    
    // Output limits
    float max_output_x;
    float max_output_y;
};

// Host function declarations
void calculatePDControlWithID(
    int target_id,
    float error_x,
    float error_y,
    float& output_x,
    float& output_y,
    const PDConfig& config,
    float dt = 0.016f);

} // namespace cuda
} // namespace needaimbot