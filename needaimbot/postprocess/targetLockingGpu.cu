#include "targetLockingGpu.h"
#include <device_launch_parameters.h> 
#include <algorithm> 



__device__ inline float calculateIoUForLockingKernel(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2) {
    int xA = max(x1, x2);
    int yA = max(y1, y2);
    int xB = min(x1 + w1, x2 + w2);
    int yB = min(y1 + h1, y2 + h2);

    int interWidth = max(0, xB - xA);
    int interHeight = max(0, yB - yA);
    int interArea = interWidth * interHeight;

    int box1Area = w1 * h1;
    int box2Area = w2 * h2;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}

__global__ void reacquireLockedTargetKernel(
    const Detection* __restrict__ d_current_detections,
    int num_current_detections,
    cv::Rect previous_locked_target_box, 
    float iou_threshold,
    int* __restrict__ d_final_best_index_output)
{
    
    
    extern __shared__ char s_data[]; 
    int* s_best_indices = reinterpret_cast<int*>(s_data);
    float* s_best_ious = reinterpret_cast<float*>(&s_best_indices[blockDim.x]);

    int tid = threadIdx.x;
    int idx_global = blockIdx.x * blockDim.x + tid;

    
    s_best_indices[tid] = -1;
    s_best_ious[tid] = 0.0f; 

    if (idx_global < num_current_detections) {
        float iou = calculateIoUForLockingKernel(
            d_current_detections[idx_global].x, 
            d_current_detections[idx_global].y,
            d_current_detections[idx_global].width, 
            d_current_detections[idx_global].height,
            previous_locked_target_box.x,
            previous_locked_target_box.y,
            previous_locked_target_box.width,
            previous_locked_target_box.height);
        if (iou >= iou_threshold) {
            s_best_indices[tid] = idx_global;
            s_best_ious[tid] = iou;
        }
    }
    __syncthreads(); 

    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            
            if (s_best_ious[tid + s] > s_best_ious[tid]) {
                s_best_ious[tid] = s_best_ious[tid + s];
                s_best_indices[tid] = s_best_indices[tid + s];
            } 
            
            
            
            
            
            
        }
        __syncthreads();
    }

    
    
    
    if (tid == 0) {
        *d_final_best_index_output = s_best_indices[0];
    }
}

cudaError_t reacquireLockedTargetGpu(
    const Detection* d_current_detections,
    int num_current_detections,
    const cv::Rect& previous_locked_target_box,
    float iou_threshold,
    int* d_reacquired_target_index_output,
    cudaStream_t stream)
{
    if (num_current_detections <= 0) {
        int minus_one = -1;
        
        cudaError_t err = cudaMemcpy(d_reacquired_target_index_output, &minus_one, sizeof(int), cudaMemcpyHostToDevice);
        return err; 
    }

    
    int initial_index_val = -1;
    cudaError_t err = cudaMemcpyAsync(d_reacquired_target_index_output, &initial_index_val, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    
    
    unsigned int block_size;
    if (num_current_detections <= 32) block_size = 32;
    else if (num_current_detections <= 64) block_size = 64;
    else if (num_current_detections <= 128) block_size = 128;
    else block_size = 256; 
    
    
    block_size = std::min((unsigned int)num_current_detections, block_size);
    if (block_size == 0) block_size = 1; 
    
    unsigned int grid_size = 1; 

    size_t shared_mem_size = block_size * (sizeof(int) + sizeof(float));

    reacquireLockedTargetKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_current_detections,
        num_current_detections,
        previous_locked_target_box,
        iou_threshold,
        d_reacquired_target_index_output
    );
    
    return cudaGetLastError();
} 