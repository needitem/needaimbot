#ifndef TARGET_LOCKING_GPU_H
#define TARGET_LOCKING_GPU_H

#include <cuda_runtime.h>
#include "postProcess.h" 


cudaError_t reacquireLockedTargetGpu(
    const Detection* d_current_detections,
    int num_current_detections,
    const cv::Rect& previous_locked_target_box,
    float iou_threshold,
    int* d_reacquired_target_index_output,
    cudaStream_t stream);

#endif 