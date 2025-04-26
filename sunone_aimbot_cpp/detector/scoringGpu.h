#ifndef SCORING_GPU_H
#define SCORING_GPU_H

#include <cuda_runtime.h>
#include "postProcess.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>

struct Detection; // Forward declaration

/**
 * @brief Calculates target scores on the GPU based on distance.
 *
 * @param d_detections Pointer to the Detection array on the GPU.
 * @param num_detections Number of detections in the array.
 * @param d_scores Output buffer for scores on the GPU (size should be num_detections).
 * @param frame_width Width of the detection frame/ROI.
 * @param frame_height Height of the detection frame/ROI.
 * @param distance_weight_config Weighting factor for the distance score.
 * @param stream CUDA stream for asynchronous execution.
 */
cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight_config,
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

#endif