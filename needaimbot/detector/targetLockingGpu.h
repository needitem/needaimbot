#ifndef TARGET_LOCKING_GPU_H
#define TARGET_LOCKING_GPU_H

#include <cuda_runtime.h>
#include "postProcess.h" // For Detection struct

/**
 * @brief Attempts to reacquire a locked target from a list of current detections on the GPU.
 *
 * This kernel assumes a single block will be launched and is suitable when the number
 * of current_detections is relatively small (e.g., <= 1024, typically <= 256 for good occupancy).
 * It finds the detection with the highest IoU above the threshold.
 *
 * @param d_current_detections Pointer to the array of current frame's detections on the GPU.
 * @param num_current_detections Number of detections in d_current_detections.
 * @param previous_locked_target_box The bounding box of the target locked in the previous frame.
 * @param iou_threshold Minimum IoU to consider a detection as a match.
 * @param d_reacquired_target_index_output Output buffer (size 1 on GPU) to store the index of the
 *                                         reacquired target in d_current_detections. Will be -1 if no
 *                                         target is reacquired meeting the criteria.
 * @param stream CUDA stream for asynchronous execution.
 * @return cudaError_t Error code from CUDA operations.
 */
cudaError_t reacquireLockedTargetGpu(
    const Detection* d_current_detections,
    int num_current_detections,
    const cv::Rect& previous_locked_target_box,
    float iou_threshold,
    int* d_reacquired_target_index_output,
    cudaStream_t stream);

#endif // TARGET_LOCKING_GPU_H 