#pragma once

#include "postProcess.h" // For Detection struct
#include <cuda_runtime.h>

cudaError_t filterDetectionsByClassIdGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
    int* d_output_count, // Must be initialized to 0 before calling kernel!
    bool ignore_class_0,
    bool ignore_class_1,
    bool ignore_class_2,
    bool ignore_class_3,
    bool ignore_class_4,
    bool ignore_class_5,
    bool ignore_class_6,
    bool ignore_class_7,
    bool ignore_class_8,
    bool ignore_class_9,
    bool ignore_class_10,
    int max_output_detections,
    cudaStream_t stream
); 