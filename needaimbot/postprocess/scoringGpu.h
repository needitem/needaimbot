#ifndef SCORING_GPU_H
#define SCORING_GPU_H

#include <cuda_runtime.h>
#include "postProcess.h"
// OpenCV removed - using custom types

struct Detection; 


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
    cudaStream_t stream);


cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream,
    float* d_temp_scores = nullptr,
    int* d_temp_indices = nullptr
);

// Find the detection that best matches the previous target (for tracking continuity)
cudaError_t findMatchingTargetGpu(
    const Detection* d_detections,
    int num_detections,
    const Detection& previous_target,
    int* d_matching_index_gpu,
    float* d_matching_score_gpu,
    cudaStream_t stream
);

#endif