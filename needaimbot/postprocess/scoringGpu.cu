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

#include "scoringGpu.h"
#include "postProcess.h" 
#include <opencv2/cudaarithm.hpp>

__device__ inline float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = max(box1.x, box2.x);
    int yA = max(box1.y, box2.y);
    int xB = min(box1.x + box1.width, box2.x + box2.width);
    int yB = min(box1.y + box1.height, box2.y + box2.height);

    int interArea = max(0, xB - xA) * max(0, yB - yA);

    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
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
    float max_distance = sqrtf(frame_width * frame_width + frame_height * frame_height) * 0.5f;
    float normalized_distance = distance / max_distance;
    
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
    int* __restrict__ d_best_index
) {
    extern __shared__ float shared_scores[];
    int* shared_indices = (int*)&shared_scores[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory
    if (idx < num_detections) {
        shared_scores[tid] = d_scores[idx];
        shared_indices[tid] = idx;
    } else {
        shared_scores[tid] = -1.0f; // Invalid score
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Parallel reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < blockDim.x) {
            if (shared_scores[tid + stride] > shared_scores[tid]) {
                shared_scores[tid] = shared_scores[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // First thread writes result
    if (tid == 0 && shared_indices[0] >= 0) {
        atomicMax(d_best_index, shared_indices[0]);
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
    cudaStream_t stream
) {
    if (num_detections <= 0) return cudaSuccess;
    
    // Initialize best index to -1
    cudaMemsetAsync(d_best_index_gpu, -1, sizeof(int), stream);
    
    int block_size = 256;
    int grid_size = (num_detections + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));
    
    findBestTargetKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_scores,
        num_detections,
        d_best_index_gpu
    );
    
    return cudaGetLastError();
}