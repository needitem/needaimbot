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
    float distance_weight       // Parameter for distance weighting
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
        float distance_score = sqrtf(dx * dx + dy * dy) * distance_weight; // Apply distance weight

        // Final score (lower is better) - Only distance
        d_scores[idx] = distance_score;
    }
}

cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight_config,  // Renamed for clarity
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
        distance_weight_config       // Pass distance weight parameter
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
