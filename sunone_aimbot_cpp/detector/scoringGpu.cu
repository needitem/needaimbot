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
    bool disable_headshot,
    int head_class_id,
    cv::Rect previous_target_box, // Pass by value
    bool had_target_last_frame,
    float distance_weight,       // New parameter
    float headshot_bonus,        // New parameter
    float sticky_bonus_config,   // New parameter (avoid name clash)
    float sticky_iou_threshold   // New parameter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections) {
        const Detection& det = d_detections[idx];
        const cv::Rect& box = det.box;

        // Calculate center of the box
        float centerX = box.x + box.width / 2.0f;
        float centerY = box.y + box.height / 2.0f;

        // Calculate distance from the center of the frame
        float frameCenterX = frame_width / 2.0f;
        float frameCenterY = frame_height / 2.0f;
        float dx = centerX - frameCenterX;
        float dy = centerY - frameCenterY;
        float distance_score = sqrtf(dx * dx + dy * dy) * distance_weight; // Use parameter

        // Apply headshot bonus if enabled and class matches
        float head_bonus_applied = 0.0f;
        if (!disable_headshot && det.classId == head_class_id) {
            head_bonus_applied = headshot_bonus; // Use parameter
        }

        // Apply sticky bonus if applicable
        float sticky_bonus_applied = 0.0f;
        if (had_target_last_frame) {
            float iou = calculateIoU(box, previous_target_box);
            if (iou > sticky_iou_threshold) { // Use parameter
                sticky_bonus_applied = sticky_bonus_config; // Use parameter
            }
        }

        // Final score (lower is better)
        d_scores[idx] = distance_score + head_bonus_applied + sticky_bonus_applied;
    }
}

cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    bool disable_headshot,
    int head_class_id,
    const cv::Rect& previous_target_box,
    bool had_target_last_frame,
    float sticky_bonus,            // Added
    float sticky_iou_threshold,    // Added
    cudaStream_t stream) {
    if (num_detections <= 0) {
        return cudaSuccess; // Nothing to score
    }

    // Use hardcoded distance_weight and headshot_bonus for now, or pass them too?
    // For now, hardcode them here as they are less likely to change frequently.
    // If needed, add them to config and pass them down like sticky params.
    const float distance_weight = 1.0f;
    const float headshot_bonus = -20.0f;

    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;

    calculateTargetScoresGpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_detections,
        num_detections,
        d_scores,
        frame_width,
        frame_height,
        disable_headshot,
        head_class_id,
        previous_target_box,
        had_target_last_frame,
        distance_weight,       // Pass hardcoded/local value
        headshot_bonus,        // Pass hardcoded/local value
        sticky_bonus,          // Pass parameter
        sticky_iou_threshold   // Pass parameter
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
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaSuccess;
    }
    try {
        thrust::device_ptr<const float> d_scores_ptr(d_scores);

        auto max_iter = thrust::max_element(
            thrust::cuda::par.on(stream),
            d_scores_ptr,
            d_scores_ptr + num_detections
        );

        int best_index = thrust::distance(d_scores_ptr, max_iter);

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
         return cudaErrorUnknown;
    }
}
