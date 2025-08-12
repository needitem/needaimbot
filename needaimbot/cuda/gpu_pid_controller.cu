#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// GPU PID Controller for unified pipeline
// Processes target directly in GPU memory without CPU transfer

struct PIDState {
    float2 prev_error;
    float2 integral;
    float2 derivative;
    float2 filtered_error;
    float2 prev_output;
    float last_time;
    bool first_error;
};

struct PIDConfig {
    float kp_x, ki_x, kd_x;
    float kp_y, ki_y, kd_y;
    float integral_clamp;
    float output_saturation;
    float error_smoothing;
    float max_velocity;
    float max_jerk;
    bool use_error_filter;
    bool use_jerk_limit;
};

// Constant memory for PID configuration (faster access)
__constant__ PIDConfig d_pid_config;
__constant__ float d_screen_center_x;
__constant__ float d_screen_center_y;

// Device function for median of three
__device__ float median3(float a, float b, float c) {
    if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
    return c;
}

// Main PID calculation kernel
__global__ void pidCalculateKernel(
    float target_x, float target_y,      // Input: target position
    float* output_dx, float* output_dy,  // Output: mouse movement
    PIDState* state,                     // PID state
    float current_time                   // Current timestamp
) {
    // Calculate error from screen center
    float2 error;
    error.x = target_x - d_screen_center_x;
    error.y = target_y - d_screen_center_y;
    
    // Calculate delta time
    float dt = current_time - state->last_time;
    dt = fmaxf(0.001f, fminf(dt, 0.2f)); // Clamp between 1ms and 200ms
    state->last_time = current_time;
    
    // Error filtering (smooth sudden target changes)
    if (state->first_error) {
        state->filtered_error = error;
        state->first_error = false;
    } else {
        float alpha = d_pid_config.error_smoothing;
        state->filtered_error.x = alpha * error.x + (1.0f - alpha) * state->filtered_error.x;
        state->filtered_error.y = alpha * error.y + (1.0f - alpha) * state->filtered_error.y;
    }
    
    float2 working_error = d_pid_config.use_error_filter ? state->filtered_error : error;
    
    // Update integral with anti-windup
    state->integral.x = fmaxf(-d_pid_config.integral_clamp, 
                              fminf(d_pid_config.integral_clamp, 
                                   state->integral.x + working_error.x * dt));
    state->integral.y = fmaxf(-d_pid_config.integral_clamp, 
                              fminf(d_pid_config.integral_clamp, 
                                   state->integral.y + working_error.y * dt));
    
    // Calculate derivative with filtering
    float2 delta;
    delta.x = (working_error.x - state->prev_error.x) / dt;
    delta.y = (working_error.y - state->prev_error.y) / dt;
    
    // Low-pass filter for derivative
    float alpha = dt / (0.03f + dt); // 30ms time constant
    state->derivative.x = (1.0f - alpha) * state->derivative.x + alpha * delta.x;
    state->derivative.y = (1.0f - alpha) * state->derivative.y + alpha * delta.y;
    
    // PID calculation
    float2 output;
    output.x = d_pid_config.kp_x * working_error.x + 
               d_pid_config.ki_x * state->integral.x + 
               d_pid_config.kd_x * state->derivative.x;
    output.y = d_pid_config.kp_y * working_error.y + 
               d_pid_config.ki_y * state->integral.y + 
               d_pid_config.kd_y * state->derivative.y;
    
    // Velocity limiting
    output.x = fmaxf(-d_pid_config.max_velocity, fminf(d_pid_config.max_velocity, output.x));
    output.y = fmaxf(-d_pid_config.max_velocity, fminf(d_pid_config.max_velocity, output.y));
    
    // Jerk limiting (smooth acceleration changes)
    if (d_pid_config.use_jerk_limit) {
        float accel_change_x = output.x - state->prev_output.x;
        float accel_change_y = output.y - state->prev_output.y;
        
        if (fabsf(accel_change_x) > d_pid_config.max_jerk) {
            output.x = state->prev_output.x + copysignf(d_pid_config.max_jerk, accel_change_x);
        }
        if (fabsf(accel_change_y) > d_pid_config.max_jerk) {
            output.y = state->prev_output.y + copysignf(d_pid_config.max_jerk, accel_change_y);
        }
    }
    
    // Update state for next iteration
    state->prev_error = working_error;
    state->prev_output = output;
    
    // Write output
    *output_dx = output.x;
    *output_dy = output.y;
}

// Batch processing kernel for multiple targets
__global__ void pidCalculateBatchKernel(
    float* targets_x, float* targets_y,   // Input: array of target positions
    float* outputs_dx, float* outputs_dy, // Output: array of mouse movements
    PIDState* states,                     // Array of PID states
    float current_time,
    int num_targets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_targets) return;
    
    // Same PID logic as single kernel but for array element idx
    float2 error;
    error.x = targets_x[idx] - d_screen_center_x;
    error.y = targets_y[idx] - d_screen_center_y;
    
    PIDState* state = &states[idx];
    
    float dt = current_time - state->last_time;
    dt = fmaxf(0.001f, fminf(dt, 0.2f));
    state->last_time = current_time;
    
    if (state->first_error) {
        state->filtered_error = error;
        state->first_error = false;
    } else {
        float alpha = d_pid_config.error_smoothing;
        state->filtered_error.x = alpha * error.x + (1.0f - alpha) * state->filtered_error.x;
        state->filtered_error.y = alpha * error.y + (1.0f - alpha) * state->filtered_error.y;
    }
    
    float2 working_error = d_pid_config.use_error_filter ? state->filtered_error : error;
    
    state->integral.x = fmaxf(-d_pid_config.integral_clamp, 
                              fminf(d_pid_config.integral_clamp, 
                                   state->integral.x + working_error.x * dt));
    state->integral.y = fmaxf(-d_pid_config.integral_clamp, 
                              fminf(d_pid_config.integral_clamp, 
                                   state->integral.y + working_error.y * dt));
    
    float2 delta;
    delta.x = (working_error.x - state->prev_error.x) / dt;
    delta.y = (working_error.y - state->prev_error.y) / dt;
    
    float alpha_deriv = dt / (0.03f + dt);
    state->derivative.x = (1.0f - alpha_deriv) * state->derivative.x + alpha_deriv * delta.x;
    state->derivative.y = (1.0f - alpha_deriv) * state->derivative.y + alpha_deriv * delta.y;
    
    float2 output;
    output.x = d_pid_config.kp_x * working_error.x + 
               d_pid_config.ki_x * state->integral.x + 
               d_pid_config.kd_x * state->derivative.x;
    output.y = d_pid_config.kp_y * working_error.y + 
               d_pid_config.ki_y * state->integral.y + 
               d_pid_config.kd_y * state->derivative.y;
    
    output.x = fmaxf(-d_pid_config.max_velocity, fminf(d_pid_config.max_velocity, output.x));
    output.y = fmaxf(-d_pid_config.max_velocity, fminf(d_pid_config.max_velocity, output.y));
    
    if (d_pid_config.use_jerk_limit) {
        float accel_change_x = output.x - state->prev_output.x;
        float accel_change_y = output.y - state->prev_output.y;
        
        if (fabsf(accel_change_x) > d_pid_config.max_jerk) {
            output.x = state->prev_output.x + copysignf(d_pid_config.max_jerk, accel_change_x);
        }
        if (fabsf(accel_change_y) > d_pid_config.max_jerk) {
            output.y = state->prev_output.y + copysignf(d_pid_config.max_jerk, accel_change_y);
        }
    }
    
    state->prev_error = working_error;
    state->prev_output = output;
    
    outputs_dx[idx] = output.x;
    outputs_dy[idx] = output.y;
}

// Reset PID state kernel
__global__ void pidResetKernel(PIDState* state) {
    state->prev_error = make_float2(0.0f, 0.0f);
    state->integral = make_float2(0.0f, 0.0f);
    state->derivative = make_float2(0.0f, 0.0f);
    state->filtered_error = make_float2(0.0f, 0.0f);
    state->prev_output = make_float2(0.0f, 0.0f);
    state->first_error = true;
}

// Host-side API functions
extern "C" {
    
// Initialize PID configuration in constant memory
void initializeGpuPID(
    float kp_x, float ki_x, float kd_x,
    float kp_y, float ki_y, float kd_y,
    float screen_center_x, float screen_center_y,
    cudaStream_t stream
) {
    PIDConfig config;
    config.kp_x = kp_x;
    config.ki_x = ki_x;
    config.kd_x = kd_x;
    config.kp_y = kp_y;
    config.ki_y = ki_y;
    config.kd_y = kd_y;
    config.integral_clamp = 100.0f;
    config.output_saturation = 100.0f;
    config.error_smoothing = 0.3f;
    config.max_velocity = 50.0f;
    config.max_jerk = 10.0f;
    config.use_error_filter = true;
    config.use_jerk_limit = true;
    
    cudaMemcpyToSymbolAsync(d_pid_config, &config, sizeof(PIDConfig), 0, 
                            cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(d_screen_center_x, &screen_center_x, sizeof(float), 0,
                            cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(d_screen_center_y, &screen_center_y, sizeof(float), 0,
                            cudaMemcpyHostToDevice, stream);
}

// Allocate PID state in GPU memory
PIDState* allocateGpuPIDState(cudaStream_t stream) {
    PIDState* d_state;
    cudaMalloc(&d_state, sizeof(PIDState));
    pidResetKernel<<<1, 1, 0, stream>>>(d_state);
    return d_state;
}

// Calculate PID on GPU (single target)
void calculateGpuPID(
    float target_x, float target_y,
    float* d_output_dx, float* d_output_dy,
    PIDState* d_state,
    float current_time,
    cudaStream_t stream
) {
    pidCalculateKernel<<<1, 1, 0, stream>>>(
        target_x, target_y,
        d_output_dx, d_output_dy,
        d_state, current_time
    );
}

// Calculate PID on GPU (batch)
void calculateGpuPIDBatch(
    float* d_targets_x, float* d_targets_y,
    float* d_outputs_dx, float* d_outputs_dy,
    PIDState* d_states,
    float current_time,
    int num_targets,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_targets + threadsPerBlock - 1) / threadsPerBlock;
    
    pidCalculateBatchKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_targets_x, d_targets_y,
        d_outputs_dx, d_outputs_dy,
        d_states, current_time, num_targets
    );
}

// Reset PID state
void resetGpuPID(PIDState* d_state, cudaStream_t stream) {
    pidResetKernel<<<1, 1, 0, stream>>>(d_state);
}

// Free PID state
void freeGpuPIDState(PIDState* d_state) {
    cudaFree(d_state);
}

} // extern "C"