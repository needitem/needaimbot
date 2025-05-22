#include "targetLockingGpu.h"
#include <device_launch_parameters.h> // For __global__, __shared__, etc.
#include <algorithm> // For std::min, std::max (used by block_size logic host-side)

// __device__ function for IoU calculation
// cv::Rect is expected to be POD-like for GPU usage here.
__device__ inline float calculateIoUForLockingKernel(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = max(box1.x, box2.x);
    int yA = max(box1.y, box2.y);
    int xB = min(box1.x + box1.width, box2.x + box2.width);
    int yB = min(box1.y + box1.height, box2.y + box2.height);

    int interWidth = max(0, xB - xA);
    int interHeight = max(0, yB - yA);
    int interArea = interWidth * interHeight;

    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}

__global__ void reacquireLockedTargetKernel(
    const Detection* d_current_detections,
    int num_current_detections,
    cv::Rect previous_locked_target_box, // Passed by value to kernel
    float iou_threshold,
    int* d_final_best_index_output)
{
    // Shared memory for block-level reduction
    // One index and one IoU value per thread in the block
    extern __shared__ char s_data[]; // Untyped shared memory
    int* s_best_indices = reinterpret_cast<int*>(s_data);
    float* s_best_ious = reinterpret_cast<float*>(&s_best_indices[blockDim.x]);

    int tid = threadIdx.x;
    int idx_global = blockIdx.x * blockDim.x + tid;

    // Initialize this thread's entry in shared memory
    s_best_indices[tid] = -1;
    s_best_ious[tid] = 0.0f; // IoU cannot be negative, 0.0 means no qualifying candidate yet

    if (idx_global < num_current_detections) {
        float iou = calculateIoUForLockingKernel(d_current_detections[idx_global].box, previous_locked_target_box);
        if (iou >= iou_threshold) {
            s_best_indices[tid] = idx_global;
            s_best_ious[tid] = iou;
        }
    }
    __syncthreads(); // Wait for all threads to calculate their initial candidates

    // Perform reduction in shared memory to find the best IoU in this block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // If candidate at (tid + s) has higher IoU, it becomes the new best for position tid
            if (s_best_ious[tid + s] > s_best_ious[tid]) {
                s_best_ious[tid] = s_best_ious[tid + s];
                s_best_indices[tid] = s_best_indices[tid + s];
            } 
            // Optional tie-breaking: if IoUs are equal, could prefer smaller index, etc.
            // else if (s_best_ious[tid + s] == s_best_ious[tid] && s_best_ious[tid] > 0.0f) {
            //    if (s_best_indices[tid + s] < s_best_indices[tid]) {
            //        s_best_indices[tid] = s_best_indices[tid + s];
            //    }
            // }
        }
        __syncthreads();
    }

    // Thread 0 of the block writes the final result for this block.
    // This kernel is designed for a single block (grid_size = 1).
    // If multiple blocks were used, a global atomic operation or a second reduction kernel would be needed.
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
        // For small critical init, blocking copy is fine.
        cudaError_t err = cudaMemcpy(d_reacquired_target_index_output, &minus_one, sizeof(int), cudaMemcpyHostToDevice);
        return err; // Return result of memcpy
    }

    // Initialize the output index on GPU to -1 (no target found yet)
    int initial_index_val = -1;
    cudaError_t err = cudaMemcpyAsync(d_reacquired_target_index_output, &initial_index_val, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    // Determine block size. Max 100 detections, so one block is sufficient.
    // Choose a power of 2 up to num_current_detections or a common size like 128, 256.
    unsigned int block_size;
    if (num_current_detections <= 32) block_size = 32;
    else if (num_current_detections <= 64) block_size = 64;
    else if (num_current_detections <= 128) block_size = 128;
    else block_size = 256; // Cap at 256, good for up to 100-ish items too.
    
    // Ensure block_size does not exceed num_current_detections if it's very small, but keep it a reasonable minimum for warp efficiency if possible.
    block_size = std::min((unsigned int)num_current_detections, block_size);
    if (block_size == 0) block_size = 1; // Should not happen if num_current_detections > 0
    
    unsigned int grid_size = 1; // Single block approach

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