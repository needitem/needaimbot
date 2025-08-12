#ifndef GPU_PID_CONTROLLER_H
#define GPU_PID_CONTROLLER_H

#include <cuda_runtime.h>

// Forward declaration
struct PIDState;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize PID configuration in GPU constant memory
void initializeGpuPID(
    float kp_x, float ki_x, float kd_x,
    float kp_y, float ki_y, float kd_y,
    float screen_center_x, float screen_center_y,
    cudaStream_t stream
);

// Allocate PID state in GPU memory
PIDState* allocateGpuPIDState(cudaStream_t stream);

// Calculate PID on GPU (single target)
// Input: target position in GPU coordinates
// Output: mouse movement delta (dx, dy) in GPU memory
void calculateGpuPID(
    float target_x, float target_y,
    float* d_output_dx, float* d_output_dy,
    PIDState* d_state,
    float current_time,
    cudaStream_t stream
);

// Calculate PID on GPU (batch processing for multiple targets)
void calculateGpuPIDBatch(
    float* d_targets_x, float* d_targets_y,
    float* d_outputs_dx, float* d_outputs_dy,
    PIDState* d_states,
    float current_time,
    int num_targets,
    cudaStream_t stream
);

// Reset PID state
void resetGpuPID(PIDState* d_state, cudaStream_t stream);

// Free PID state
void freeGpuPIDState(PIDState* d_state);

#ifdef __cplusplus
}
#endif

// C++ wrapper class for easier integration
#ifdef __cplusplus
class GpuPIDController {
private:
    PIDState* d_state_;
    float* d_output_dx_;
    float* d_output_dy_;
    float h_output_dx_;
    float h_output_dy_;
    cudaStream_t stream_;
    
public:
    GpuPIDController(cudaStream_t stream = 0) : stream_(stream) {
        d_state_ = allocateGpuPIDState(stream);
        cudaMalloc(&d_output_dx_, sizeof(float));
        cudaMalloc(&d_output_dy_, sizeof(float));
    }
    
    ~GpuPIDController() {
        freeGpuPIDState(d_state_);
        cudaFree(d_output_dx_);
        cudaFree(d_output_dy_);
    }
    
    void initialize(float kp_x, float ki_x, float kd_x,
                   float kp_y, float ki_y, float kd_y,
                   float screen_center_x, float screen_center_y) {
        initializeGpuPID(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y,
                        screen_center_x, screen_center_y, stream_);
    }
    
    // Calculate and return results in GPU memory (no transfer)
    void calculateGpu(float target_x, float target_y, float current_time) {
        calculateGpuPID(target_x, target_y, d_output_dx_, d_output_dy_,
                       d_state_, current_time, stream_);
    }
    
    // Get results to CPU (only when needed for mouse control)
    void getResults(float& dx, float& dy) {
        cudaMemcpyAsync(&h_output_dx_, d_output_dx_, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(&h_output_dy_, d_output_dy_, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        dx = h_output_dx_;
        dy = h_output_dy_;
    }
    
    // Direct GPU memory pointers for zero-copy access
    float* getGpuOutputDx() { return d_output_dx_; }
    float* getGpuOutputDy() { return d_output_dy_; }
    
    void reset() {
        resetGpuPID(d_state_, stream_);
    }
};
#endif

#endif // GPU_PID_CONTROLLER_H