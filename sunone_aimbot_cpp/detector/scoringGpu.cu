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

#include "scoringGpu.h"

// CUDA kernel to calculate scores
__global__ void calculateTargetScoresKernel(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int resolution_x,
    int resolution_y,
    bool disable_headshot,
    int class_head)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections) {
        const Detection& det = d_detections[idx];
        const cv::Rect& box = det.box; // cv::Rect is plain data, accessible in kernel

        // Calculate center (relative to detection resolution, assuming box coords are scaled)
        const float center_x = box.x + box.width * 0.5f;
        const float center_y = box.y + box.height * 0.5f;

        // Calculate difference from screen center
        const float half_res_x = resolution_x * 0.5f;
        const float half_res_y = resolution_y * 0.5f;
        const float diff_x = center_x - half_res_x;
        const float diff_y = center_y - half_res_y;

        // Calculate squared distance and score
        const float squared_distance = diff_x * diff_x + diff_y * diff_y;

        float distance_score;
        // These thresholds might need adjustment based on typical distances
        if (squared_distance < 100.0f) { // Very close
            distance_score = 1.0f;
        } else if (squared_distance > 500000.0f) { // Very far
            distance_score = 0.0001f;
        } else {
            // Inverse relationship: closer is better score
            distance_score = 1.0f / (1.0f + sqrtf(squared_distance));
        }

        // Calculate class score bonus
        float class_score = (!disable_headshot && det.classId == class_head) ? 1.5f : 1.0f;

        // Final score
        d_scores[idx] = distance_score * class_score;
    }
}

// Wrapper function to launch the kernel
void calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int resolution_x,
    int resolution_y,
    bool disable_headshot,
    int class_head,
    cudaStream_t stream)
{
    if (num_detections <= 0) return;

    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;

    calculateTargetScoresKernel<<<grid_size, block_size, 0, stream>>>(
        d_detections,
        num_detections,
        d_scores,
        resolution_x,
        resolution_y,
        disable_headshot,
        class_head
    );
}


// Function to find the best target index using Thrust
cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream)
{
    if (num_detections <= 0) {
         // Set index to -1 or handle appropriately if no detections
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream); // Set to -1
         return cudaSuccess;
    }
    try {
        // Wrap raw pointers for Thrust
        thrust::device_ptr<const float> d_scores_ptr(d_scores);

        // Find iterator to the max element
        auto max_iter = thrust::max_element(
            thrust::cuda::par.on(stream), // Execute on the specified stream
            d_scores_ptr,
            d_scores_ptr + num_detections
        );

        // Calculate the index using thrust::distance
        int best_index = thrust::distance(d_scores_ptr, max_iter);

        // Copy the index to the output GPU buffer
        cudaMemcpyAsync(
            d_best_index_gpu,
            &best_index,
            sizeof(int),
            cudaMemcpyHostToDevice, // Copying calculated index from host temp variable to GPU buffer
            stream
        );
        return cudaGetLastError(); // Check for errors in async operations
    } catch (const std::exception& e) {
         fprintf(stderr, "[Thrust Error] findBestTargetGpu: %s\n", e.what());
         return cudaErrorUnknown; // Or a more specific error
    }
}
