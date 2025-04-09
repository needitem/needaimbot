#ifndef SCORING_GPU_H
#define SCORING_GPU_H

#include <cuda_runtime.h>
#include "postProcess.h" // For Detection struct

/**
 * @brief Calculates target scores on the GPU based on distance and class ID.
 *
 * @param d_detections Pointer to the Detection array on the GPU.
 * @param num_detections Number of detections in the array.
 * @param d_scores Output buffer for scores on the GPU (size should be num_detections).
 * @param resolution_x Screen width.
 * @param resolution_y Screen height.
 * @param disable_headshot Whether to disable headshot bonus.
 * @param class_head The class ID representing the head.
 * @param stream CUDA stream for asynchronous execution.
 */
void calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int resolution_x,
    int resolution_y,
    bool disable_headshot,
    int class_head,
    cudaStream_t stream
);

/**
 * @brief Finds the index of the detection with the maximum score on the GPU.
 *
 * @param d_scores Pointer to the scores array on the GPU.
 * @param num_detections Number of scores/detections.
 * @param d_best_index_gpu Output buffer (size 1) for the best index on the GPU.
 * @param stream CUDA stream for asynchronous execution.
 * @return cudaError_t Error code from Thrust/CUDA operations.
 */
cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream
);


#endif // SCORING_GPU_H