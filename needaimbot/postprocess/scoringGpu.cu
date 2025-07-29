#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/distance.h>
#include <limits>
#include <cmath>
#include <cfloat>

#include "scoringGpu.h"
#include "postProcess.h" 
// OpenCV removed - using custom types

__device__ inline float calculateIoU(const Detection& det1, const Detection& det2) {
    int xA = max(det1.x, det2.x);
    int yA = max(det1.y, det2.y);
    int xB = min(det1.x + det1.width, det2.x + det2.width);
    int yB = min(det1.y + det1.height, det2.y + det2.height);

    int interArea = max(0, xB - xA) * max(0, yB - yA);

    int box1Area = det1.width * det1.height;
    int box2Area = det2.width * det2.height;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}

__global__ void calculateTargetScoresGpuKernel(
    const Detection* __restrict__ d_detections,
    int num_detections,
    float* __restrict__ d_scores,
    int frame_width,
    int frame_height,
    float distance_weight,
    float confidence_weight,
    int head_class_id,             
    float head_class_score_multiplier,
    float crosshairX,
    float crosshairY
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_detections) return;
    
    const Detection& det = d_detections[idx];
    
    // Calculate center of detection
    float center_x = det.x + det.width * 0.5f;
    float center_y = det.y + det.height * 0.5f;
    
    // Calculate distance from crosshair
    float dx = center_x - crosshairX;
    float dy = center_y - crosshairY;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // Normalize distance (0-1, where 0 is best)
    // Pre-calculate max_distance as it's constant for all detections
    float max_distance = sqrtf(float(frame_width * frame_width + frame_height * frame_height)) * 0.5f;
    float normalized_distance = fminf(distance / max_distance, 1.0f); // Clamp to [0, 1]
    
    // Distance score (closer is better, so 1 - normalized_distance)
    float distance_score = 1.0f - normalized_distance;
    
    // Confidence score (already 0-1)
    float confidence_score = det.confidence;
    
    // Class bonus for head shots
    float class_bonus = (det.classId == head_class_id) ? head_class_score_multiplier : 1.0f;
    
    // Combined score
    float final_score = (distance_score * distance_weight + 
                        confidence_score * confidence_weight) * class_bonus;
    
    d_scores[idx] = final_score;
}

__global__ void findBestTargetKernel(
    const float* __restrict__ d_scores,
    int num_detections,
    int* __restrict__ d_best_index,
    float* __restrict__ d_block_max_scores,
    int* __restrict__ d_block_max_indices
) {
    extern __shared__ char shared_mem[];
    float* shared_scores = (float*)shared_mem;
    int* shared_indices = (int*)&shared_scores[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory with thread's data or invalid values
    if (idx < num_detections) {
        shared_scores[tid] = d_scores[idx];
        shared_indices[tid] = idx;
    } else {
        shared_scores[tid] = -FLT_MAX;
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Parallel reduction to find maximum within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_scores[tid + stride] > shared_scores[tid]) {
                shared_scores[tid] = shared_scores[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // First thread of each block writes its block's maximum
    if (tid == 0) {
        if (d_block_max_scores && d_block_max_indices) {
            // Multi-block case
            d_block_max_scores[blockIdx.x] = shared_scores[0];
            d_block_max_indices[blockIdx.x] = shared_indices[0];
        } else {
            // Single block case - write directly to output
            *d_best_index = shared_indices[0];
        }
    }
}

__global__ void findGlobalMaxKernel(
    const float* __restrict__ d_block_max_scores,
    const int* __restrict__ d_block_max_indices,
    int num_blocks,
    int* __restrict__ d_best_index
) {
    extern __shared__ char shared_mem[];
    float* shared_scores = (float*)shared_mem;
    int* shared_indices = (int*)&shared_scores[blockDim.x];
    
    int tid = threadIdx.x;
    
    // Load block maximums
    if (tid < num_blocks) {
        shared_scores[tid] = d_block_max_scores[tid];
        shared_indices[tid] = d_block_max_indices[tid];
    } else {
        shared_scores[tid] = -FLT_MAX;
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Parallel reduction to find global maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_scores[tid + stride] > shared_scores[tid]) {
                shared_scores[tid] = shared_scores[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write final result - always update even if it's -1
    if (tid == 0) {
        *d_best_index = shared_indices[0];
    }
}

// Host function implementations
cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight_config,
    float confidence_weight_config,
    int head_class_id,
    float crosshair_offset_x,
    float crosshair_offset_y,
    cudaStream_t stream
) {
    if (num_detections <= 0) return cudaSuccess;
    
    int block_size = 256;
    int grid_size = (num_detections + block_size - 1) / block_size;
    
    calculateTargetScoresGpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_detections,
        num_detections,
        d_scores,
        frame_width,
        frame_height,
        distance_weight_config,
        confidence_weight_config,
        head_class_id,
        2.0f, // head_class_score_multiplier
        crosshair_offset_x,
        crosshair_offset_y
    );
    
    return cudaGetLastError();
}

cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream,
    float* d_temp_scores,
    int* d_temp_indices
) {
    if (num_detections <= 0) {
        cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
        return cudaSuccess;
    }
    
#ifdef USE_THRUST_IMPLEMENTATION
    // Thrust implementation - more reliable but potentially slower
    thrust::device_ptr<const float> scores_ptr(d_scores);
    thrust::device_ptr<const float> max_iter = thrust::max_element(thrust::cuda::par.on(stream), 
                                                                    scores_ptr, 
                                                                    scores_ptr + num_detections);
    
    // Calculate the index
    int best_index = thrust::distance(scores_ptr, max_iter);
    
    // Always write the result, even if it's -1
    cudaMemcpyAsync(d_best_index_gpu, &best_index, sizeof(int), cudaMemcpyHostToDevice, stream);
#else
    // Custom kernel implementation - supports multiple blocks for large datasets
    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;
    
    if (grid_size == 1) {
        // Single block - can write directly to output
        size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));
        
        // Initialize to -1 first
        cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
        
        findBestTargetKernel<<<1, block_size, shared_mem_size, stream>>>(
            d_scores,
            num_detections,
            d_best_index_gpu,
            nullptr,  // Not needed for single block
            nullptr   // Not needed for single block
        );
    } else {
        // Multiple blocks - need temporary storage and second kernel
        if (!d_temp_scores || !d_temp_indices) {
            // If temporary buffers not provided, fall back to thrust
            thrust::device_ptr<const float> scores_ptr(d_scores);
            thrust::device_ptr<const float> max_iter = thrust::max_element(thrust::cuda::par.on(stream), 
                                                                            scores_ptr, 
                                                                            scores_ptr + num_detections);
            int best_index = thrust::distance(scores_ptr, max_iter);
            cudaMemcpyAsync(d_best_index_gpu, &best_index, sizeof(int), cudaMemcpyHostToDevice, stream);
        } else {
            size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));
            
            // First pass: find maximum in each block
            findBestTargetKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
                d_scores,
                num_detections,
                d_best_index_gpu,  // Not used in multi-block first pass
                d_temp_scores,
                d_temp_indices
            );
            
            // Second pass: find global maximum from block maximums
            int final_block_size = min(grid_size, 256);
            shared_mem_size = final_block_size * (sizeof(float) + sizeof(int));
            
            findGlobalMaxKernel<<<1, final_block_size, shared_mem_size, stream>>>(
                d_temp_scores,
                d_temp_indices,
                grid_size,
                d_best_index_gpu
            );
        }
    }
#endif
    
    return cudaGetLastError();
}