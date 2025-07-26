#pragma once

#include "postProcess.h" 
#include <cuda_runtime.h>

cudaError_t filterDetectionsByClassIdGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
    int* d_output_count,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    const unsigned char* d_color_mask,
    int mask_pitch,
    int min_color_pixels,
    bool remove_color_matches,
    int max_output_detections,
    cudaStream_t stream
); 