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
#include "postProcess.h" // For Detection struct

__device__ const float STICKY_TARGET_SCORE = -1000.0f; // Score for a sticky target (lower is better)

// Simple IoU calculation for __device__ function
__device__ inline float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = max(box1.x, box2.x);
    int yA = max(box1.y, box2.y);
    int xB = min(box1.x + box1.width, box2.x + box2.width);
    int yB = min(box1.y + box1.height, box2.y + box2.height);

    // Intersection area
    int interArea = max(0, xB - xA) * max(0, yB - yA);

    // Union area
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    // Compute IoU
    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}

// GPU Kernel to calculate scores for each detection
__global__ void calculateTargetScoresGpuKernel(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight,       // Parameter for distance weighting
    // Stickiness parameters
    bool has_previous_target_kernel,
    int prev_target_box_x_kernel,
    int prev_target_box_y_kernel,
    int prev_target_box_width_kernel,
    int prev_target_box_height_kernel,
    float stickiness_radius_sq_kernel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections) {
        const Detection& det = d_detections[idx];
        const cv::Rect& box = det.box; // cv::Rect should be fine if it's a simple struct

        // Calculate center of the current box
        float current_centerX = static_cast<float>(box.x) + static_cast<float>(box.width) / 2.0f;
        float current_centerY = static_cast<float>(box.y) + static_cast<float>(box.height) / 2.0f;

        // Stickiness check
        if (has_previous_target_kernel) {
            float prev_target_centerX = static_cast<float>(prev_target_box_x_kernel) + static_cast<float>(prev_target_box_width_kernel) / 2.0f;
            float prev_target_centerY = static_cast<float>(prev_target_box_y_kernel) + static_cast<float>(prev_target_box_height_kernel) / 2.0f;

            float dx_to_prev = current_centerX - prev_target_centerX;
            float dy_to_prev = current_centerY - prev_target_centerY;
            float dist_to_prev_sq = dx_to_prev * dx_to_prev + dy_to_prev * dy_to_prev;

            if (dist_to_prev_sq < stickiness_radius_sq_kernel) {
                d_scores[idx] = STICKY_TARGET_SCORE;
                return; // This is our sticky target
            }
        }

        // Calculate distance from the center of the frame (original scoring)
        float frameCenterX = static_cast<float>(frame_width) / 2.0f;
        float frameCenterY = static_cast<float>(frame_height) / 2.0f;
        float dx_to_center = current_centerX - frameCenterX;
        float dy_to_center = current_centerY - frameCenterY;
        float distance_from_center_score = sqrtf(dx_to_center * dx_to_center + dy_to_center * dy_to_center) * distance_weight; // Apply distance weight

        // Final score (lower is better)
        d_scores[idx] = distance_from_center_score;
    }
}

cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight_config,  // Renamed for clarity
    // Stickiness parameters
    bool has_previous_target,
    int prev_target_box_x,
    int prev_target_box_y,
    int prev_target_box_width,
    int prev_target_box_height,
    float stickiness_radius_sq,
    cudaStream_t stream) {
    if (num_detections <= 0) {
        return cudaSuccess; // Nothing to score
    }

    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;

    calculateTargetScoresGpuKernel<<<grid_size, block_size, 0, stream>>>( 
        d_detections,
        num_detections,
        d_scores,
        frame_width,
        frame_height,
        distance_weight_config,       // Pass distance weight parameter
        // Pass stickiness parameters
        has_previous_target,
        prev_target_box_x,
        prev_target_box_y,
        prev_target_box_width,
        prev_target_box_height,
        stickiness_radius_sq
    );

    return cudaGetLastError();
}

// Function to find the best target index using Thrust
cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream)
{
    if (num_detections <= 0) {
         // Set index to -1 (0xFFFFFFFF) if no detections
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaSuccess;
    }
    try {
        thrust::device_ptr<const float> d_scores_ptr(d_scores);

        // Use min_element because lower scores are better
        auto min_iter = thrust::min_element(
            thrust::cuda::par.on(stream),
            d_scores_ptr,
            d_scores_ptr + num_detections
        );

        // Calculate the index of the minimum element
        int best_index = thrust::distance(d_scores_ptr, min_iter);

        // Copy the best index to the output GPU buffer
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
         // Set index to -1 on error
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaErrorUnknown;
    }
}
