// PD Controller for Mouse Movement (CUDA implementation)
// Supports per-target tracking with ID or global tracking without ID

#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "pd_controller_shared.h"

namespace needaimbot {
namespace cuda {

// Global state for when no ID is available
// Initialize with aggregate initialization (C-style)
__device__ PDState g_global_pd_state = {0.0f, 0.0f, 0.0f, 0.0f, false};

// Forward declaration for host-side state management
// Implementation is in pd_controller.cpp
class PDControllerHost;

// Device-side PD calculation
__device__ __host__ inline void calculatePDControl(
    float error_x,
    float error_y,
    float& output_x,
    float& output_y,
    PDState& state,
    const PDConfig& config,
    float dt = 0.016f)  // Default 60 FPS
{
    // Apply deadzone
    if (fabsf(error_x) < config.deadzone_x) error_x = 0.0f;
    if (fabsf(error_y) < config.deadzone_y) error_y = 0.0f;
    
    // Calculate proportional terms
    float p_term_x = config.kp_x * error_x;
    float p_term_y = config.kp_y * error_y;
    
    // Calculate derivative terms
    float d_term_x = 0.0f;
    float d_term_y = 0.0f;
    
    if (state.initialized && dt > 0.0f) {
        // Raw derivative
        float raw_derivative_x = (error_x - state.prev_error_x) / dt;
        float raw_derivative_y = (error_y - state.prev_error_y) / dt;
        
        // Apply exponential moving average filter to derivative
        if (config.derivative_filter_alpha > 0.0f) {
            // Filter the derivative using exponential moving average
            state.filtered_derivative_x = config.derivative_filter_alpha * state.filtered_derivative_x + 
                                         (1.0f - config.derivative_filter_alpha) * raw_derivative_x;
            state.filtered_derivative_y = config.derivative_filter_alpha * state.filtered_derivative_y + 
                                         (1.0f - config.derivative_filter_alpha) * raw_derivative_y;
            
            d_term_x = config.kd_x * state.filtered_derivative_x;
            d_term_y = config.kd_y * state.filtered_derivative_y;
        } else {
            // No filtering
            d_term_x = config.kd_x * raw_derivative_x;
            d_term_y = config.kd_y * raw_derivative_y;
            
            // Store unfiltered derivative for consistency
            state.filtered_derivative_x = raw_derivative_x;
            state.filtered_derivative_y = raw_derivative_y;
        }
    }
    
    // Calculate total output
    output_x = p_term_x + d_term_x;
    output_y = p_term_y + d_term_y;
    
    // Apply output limits
    if (output_x > config.max_output_x) output_x = config.max_output_x;
    if (output_x < -config.max_output_x) output_x = -config.max_output_x;
    if (output_y > config.max_output_y) output_y = config.max_output_y;
    if (output_y < -config.max_output_y) output_y = -config.max_output_y;
    
    // Update state for next iteration
    state.prev_error_x = error_x;
    state.prev_error_y = error_y;
    state.initialized = true;
}

// Simplified interface for GPU kernel usage (no ID)
__device__ inline void calculatePDControlGPU(
    float error_x,
    float error_y,
    float& output_x,
    float& output_y,
    const PDConfig& config,
    float dt = 0.016f)
{
    calculatePDControl(error_x, error_y, output_x, output_y, 
                      g_global_pd_state, config, dt);
}

// Note: Host-side functions (calculatePDControlWithID, resetTargetPDState, resetAllPDStates)
// are implemented in pd_controller.cpp to avoid CUDA/C++ linking issues

} // namespace cuda
} // namespace needaimbot