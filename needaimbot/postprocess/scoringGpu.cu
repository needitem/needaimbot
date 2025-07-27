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
    // Simple linear search for now - more reliable
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        float best_score = -1.0f;
        int best_idx = -1;
        
        for (int i = 0; i < num_detections; i++) {
            if (d_scores[i] > best_score) {
                best_score = d_scores[i];
                best_idx = i;
            }
        }
        
        *d_best_index = best_idx;
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
    if (num_detections <= 0) {
        cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
        return cudaSuccess;
    }
    
    // Use thrust to find the maximum element
    thrust::device_ptr<const float> scores_ptr(d_scores);
    thrust::device_ptr<const float> max_iter = thrust::max_element(thrust::cuda::par.on(stream), 
                                                                    scores_ptr, 
                                                                    scores_ptr + num_detections);
    
    // Calculate the index
    int best_index = thrust::distance(scores_ptr, max_iter);
    
    // Verify the index is valid
    if (best_index >= 0 && best_index < num_detections) {
        cudaMemcpyAsync(d_best_index_gpu, &best_index, sizeof(int), cudaMemcpyHostToDevice, stream);
    } else {
        // No valid target found
        int invalid_index = -1;
        cudaMemcpyAsync(d_best_index_gpu, &invalid_index, sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    
    return cudaGetLastError();
}