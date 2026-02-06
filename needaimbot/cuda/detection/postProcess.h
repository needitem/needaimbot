#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <cuda_runtime.h>
#include <NvInferRuntimeCommon.h> 
#include "../core/Target.h"


// Validate and clean detections
void validateTargetsGpu(
    Target* d_detections,
    int n,
    cudaStream_t stream = 0);

// Final validation to remove extreme values
void finalValidateTargetsGpu(
    Target* d_targets,
    int* d_count,
    int max_targets,
    cudaStream_t stream = 0);

// Validate single best target before host copy
void validateBestTargetGpu(
    Target* d_best_target,
    cudaStream_t stream = 0);



cudaError_t decodeYolo10Gpu(
    const void* d_raw_output,
    nvinfer1::DataType output_type,
    const std::vector<int64_t>& shape,
    int num_classes,
    float conf_threshold,
    float img_scale,
    Target* d_decoded_detections,
    int* d_decoded_count,
    int max_detections,
    int max_candidates,
    const unsigned char* d_class_filter = nullptr,
    int max_class_filter_size = 0,
    cudaStream_t stream = 0);


cudaError_t decodeYolo11Gpu(
    const void* d_raw_output,
    nvinfer1::DataType output_type,
    const std::vector<int64_t>& shape,
    int num_classes,
    float conf_threshold,
    float img_scale,
    Target* d_decoded_detections,
    int* d_decoded_count,
    int max_detections,
    int max_candidates,
    const unsigned char* d_class_filter = nullptr,
    int max_class_filter_size = 0,
    cudaStream_t stream = 0);

// NMS (Non-Maximum Suppression) functions
// Marks which detections to keep based on IoU threshold
cudaError_t performNmsGpu(
    Target* d_detections,
    int* d_count,
    bool* d_keep_flags,
    float iou_threshold,
    int max_detections,
    cudaStream_t stream);

// Optimized NMS: runs on input buffer, compacts to separate output buffer.
// Eliminates initial D2D copy (3 CUDA ops instead of 5).
// Caller must swap finalTargets pointer to d_output after call.
cudaError_t performNmsCompactGpu(
    const Target* d_input,
    int* d_input_count,
    bool* d_keep_flags,
    Target* d_output,
    int* d_output_count,
    float iou_threshold,
    int max_detections,
    cudaStream_t stream);


#endif