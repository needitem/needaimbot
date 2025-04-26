#pragma once

#include "postProcess.h" // For Detection struct
#include <cuda_runtime.h>

cudaError_t filterDetectionsByClassIdGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
    int* d_output_count, // Must be initialized to 0 before calling kernel!
    int class_id_person,
    int class_id_head,
    bool disable_headshot,
    int max_output_detections,
    cudaStream_t stream
); 