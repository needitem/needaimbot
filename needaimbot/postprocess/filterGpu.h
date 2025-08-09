#pragma once

#include "postProcess.h" 
#include <cuda_runtime.h>

// Class ID filtering function (optimized and separated)
cudaError_t filterTargetsByClassIdGpu(
    const Target* d_input_detections,
    int num_input_detections,
    Target* d_output_detections,
    int* d_output_count,
    const unsigned char* d_allowed_class_ids,
    int max_check_id,
    int max_output_detections,
    cudaStream_t stream
);

// RGB color filtering function (runs after class filtering, before NMS)
cudaError_t filterTargetsByColorGpu(
    const Target* d_input_detections,
    int num_input_detections,
    Target* d_output_detections,
    int* d_output_count,
    const unsigned char* d_color_mask,
    int mask_pitch,
    int min_color_pixels,
    bool remove_color_matches,
    int max_output_detections,
    cudaStream_t stream
); 