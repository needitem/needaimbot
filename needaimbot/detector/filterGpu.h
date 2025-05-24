#pragma once

#include "postProcess.h" // For Detection struct
#include <cuda_runtime.h>

// Filters detections by class (using ignore flags array) and optional HSV mask.
// d_ignored_class_ids: array of size max_check_id, 1 to ignore class, 0 to keep.
// d_hsv_mask: pointer to binary mask image on GPU (uchar per pixel), or nullptr to skip HSV filtering.
// mask_pitch: number of columns (stride) in d_hsv_mask row (in bytes).
// min_hsv_pixels: minimum number of mask pixels (within bounding box) required to pass HSV filter.
// Both filters applied before writing to output.
// d_output_count must be zeroed before call.
cudaError_t filterDetectionsByClassIdGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
    int* d_output_count,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    const unsigned char* d_hsv_mask,
    int mask_pitch,
    int min_hsv_pixels,
    bool remove_hsv_matches,
    int max_output_detections,
    cudaStream_t stream
); 