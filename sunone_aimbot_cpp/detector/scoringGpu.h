#ifndef SCORING_GPU_H
#define SCORING_GPU_H

#include <cuda_runtime.h>
#include "postProcess.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>

struct Detection; // Forward declaration

/**
 * @brief Calculates target scores on the GPU based on distance and class ID.
 *
 * @param d_detections Pointer to the Detection array on the GPU.
 * @param num_detections Number of detections in the array.
 * @param d_scores Output buffer for scores on the GPU (size should be num_detections).
 * @param frame_width Width of the detection frame/ROI.
 * @param frame_height Height of the detection frame/ROI.
 * @param disable_headshot Whether to disable headshot bonus.
 * @param head_class_id The class ID representing the head.
 * @param previous_target_box Previous frame's target box.
 * @param had_target_last_frame Flag indicating if there was a target last frame.
 * @param sticky_bonus Score bonus (negative) applied if IoU > threshold.
 * @param sticky_iou_threshold IoU threshold to trigger the bonus.
 * @param stream CUDA stream for asynchronous execution.
 */
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
    float sticky_bonus,
    float sticky_iou_threshold,
    cudaStream_t stream = 0
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