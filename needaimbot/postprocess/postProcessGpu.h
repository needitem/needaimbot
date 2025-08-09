#pragma once

#include "postProcess.h"
#include "../detector/detector.h"
#include <cuda_runtime.h>
#include <cstdint>

// YOLO decoding functions
void decodeYolo10Gpu(
    const float* output, const int64_t* shape,
    int numClasses, float confThreshold, float img_scale,
    Target* d_detections, int* d_count, int max_detections,
    cudaStream_t stream);

void decodeYolo11Gpu(
    const float* output, const int64_t* shape,
    int numClasses, float confThreshold, float img_scale,
    Target* d_detections, int* d_count, int max_detections,
    cudaStream_t stream);

// NMS function
void performNMSGpu(
    Target* d_detections,
    int numDetections,
    float nmsThreshold,
    Target* d_nmsDetections,
    int* d_nmsCount,
    int* d_x1, int* d_y1, int* d_x2, int* d_y2,
    float* d_areas, float* d_scores, int* d_classIds,
    float* d_iou_matrix, bool* d_keep, int* d_indices,
    cudaStream_t stream);

// CUDA kernel declarations
cudaError_t decodeYolo10Cuda(
    const float* output,
    const int64_t* shape,
    int numClasses,
    float confThreshold,
    float img_scale,
    Target* d_detections,
    int* d_count,
    int max_detections,
    cudaStream_t stream);

cudaError_t decodeYolo11Cuda(
    const float* output,
    const int64_t* shape,
    int numClasses,
    float confThreshold,
    float img_scale,
    Target* d_detections,
    int* d_count,
    int max_detections,
    cudaStream_t stream);

cudaError_t performNMSCuda(
    Target* d_detections,
    int numDetections,
    float nmsThreshold,
    Target* d_nmsDetections,
    int* d_nmsCount,
    int* d_x1, int* d_y1, int* d_x2, int* d_y2,
    float* d_areas, float* d_scores, int* d_classIds,
    float* d_iou_matrix, bool* d_keep, int* d_indices,
    cudaStream_t stream);