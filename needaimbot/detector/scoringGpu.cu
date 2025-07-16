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
    float crosshairX,  // Crosshair position X (including offset)
    float crosshairY   // Crosshair position Y (including offset)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections) {
        const Detection& det = d_detections[idx];

        float centerX = det.x + det.width * 0.5f;
        float centerY = det.y + det.height * 0.5f;

        float dx = centerX - crosshairX;
        float dy = centerY - crosshairY;
        
        // Simple distance-only calculation for closest target selection
        float distance = sqrtf(dx * dx + dy * dy);
        
        // Lower score = better target (closest to crosshair)
        d_scores[idx] = distance;
    }
}

cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight_config,
    float confidence_weight_config,
    int head_class_id_param,
    float crosshair_offset_x,
    float crosshair_offset_y,
    cudaStream_t stream) {
    if (num_detections <= 0) {
        return cudaSuccess;
    }

    const float head_bonus_multiplier_val = 0.8f; 

    // Calculate crosshair position with offset
    const float crosshairX = frame_width * 0.5f + crosshair_offset_x;
    const float crosshairY = frame_height * 0.5f + crosshair_offset_y;
    
    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;

    calculateTargetScoresGpuKernel<<<grid_size, block_size, 0, stream>>>( 
        d_detections,
        num_detections,
        d_scores,
        frame_width,
        frame_height,
        distance_weight_config,
        confidence_weight_config,
        head_class_id_param,
        head_bonus_multiplier_val,
        crosshairX,
        crosshairY
    );

    return cudaGetLastError();
}


cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream)
{
    if (num_detections <= 0) {
         
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaSuccess;
    }
    try {
        thrust::device_ptr<const float> d_scores_ptr(d_scores);

        
        auto min_iter = thrust::min_element(
            thrust::cuda::par.on(stream),
            d_scores_ptr,
            d_scores_ptr + num_detections
        );

        
        int best_index = thrust::distance(d_scores_ptr, min_iter);

        
        cudaMemcpyAsync(
            d_best_index_gpu,
            &best_index,
            sizeof(int),
            cudaMemcpyHostToDevice,
            stream
        );
        return cudaGetLastError();
    } catch (const std::exception& e) {
         fprintf(stderr, "[Thrust Error] findBestTargetGpu: %s\n", e.what());
         
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaErrorUnknown;
    }
}

// Kernel to find the detection that best matches the previous target
__global__ void findMatchingTargetKernel(
    const Detection* d_detections,
    int num_detections,
    const Detection previous_target,
    int* d_matching_index,
    float* d_matching_score
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;
    
    const Detection& det = d_detections[idx];
    
    // Skip if not the same class
    if (det.classId != previous_target.classId) return;
    
    // Calculate center points
    float det_cx = det.x + det.width * 0.5f;
    float det_cy = det.y + det.height * 0.5f;
    float prev_cx = previous_target.x + previous_target.width * 0.5f;
    float prev_cy = previous_target.y + previous_target.height * 0.5f;
    
    // Calculate distance between centers
    float dx = det_cx - prev_cx;
    float dy = det_cy - prev_cy;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // Calculate size difference (IoU-like metric)
    float size_diff = fabsf(det.width - previous_target.width) + fabsf(det.height - previous_target.height);
    
    // Combined score (lower is better) - prioritize spatial proximity
    float score = distance + size_diff * 0.1f;
    
    // Use atomicMin to find the best match
    unsigned int* score_as_uint = (unsigned int*)d_matching_score;
    unsigned int my_score_uint = __float_as_uint(score);
    unsigned int old_score_uint = atomicMin(score_as_uint, my_score_uint);
    
    // If we updated the score, also update the index
    if (old_score_uint != my_score_uint && my_score_uint == *score_as_uint) {
        *d_matching_index = idx;
    }
}

cudaError_t findMatchingTargetGpu(
    const Detection* d_detections,
    int num_detections,
    const Detection& previous_target,
    int* d_matching_index_gpu,
    float* d_matching_score_gpu,
    cudaStream_t stream
) {
    if (num_detections <= 0) {
        cudaMemsetAsync(d_matching_index_gpu, 0xFF, sizeof(int), stream);
        float max_float = FLT_MAX;
        cudaMemcpyAsync(d_matching_score_gpu, &max_float, sizeof(float), cudaMemcpyHostToDevice, stream);
        return cudaSuccess;
    }
    
    // Initialize with max values
    cudaMemsetAsync(d_matching_index_gpu, 0xFF, sizeof(int), stream);
    float max_float = FLT_MAX;
    cudaMemcpyAsync(d_matching_score_gpu, &max_float, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;
    
    findMatchingTargetKernel<<<grid_size, block_size, 0, stream>>>(
        d_detections,
        num_detections,
        previous_target,
        d_matching_index_gpu,
        d_matching_score_gpu
    );
    
    return cudaGetLastError();
}

