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
        const cv::Rect& box = det.box;

        const float center_x = box.x + box.width * 0.5f;
        const float center_y = box.y + box.height * 0.5f;

        const float half_res_x = resolution_x * 0.5f;
        const float half_res_y = resolution_y * 0.5f;
        const float diff_x = center_x - half_res_x;
        const float diff_y = center_y - half_res_y;

        const float squared_distance = diff_x * diff_x + diff_y * diff_y;

        float distance_score;

        if (squared_distance < 100.0f) {
            distance_score = 1.0f;
        } else if (squared_distance > 500000.0f) {
            distance_score = 0.0001f;
        } else {
            distance_score = 1.0f / (1.0f + sqrtf(squared_distance));
        }

        float class_score = (!disable_headshot && det.classId == class_head) ? 1.5f : 1.0f;

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
