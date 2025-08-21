#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
// OpenCV removed - using custom structures
#include <cuda_runtime.h>
#include <NvInferRuntimeCommon.h> 
#include "../core/Target.h"


// Validate and clean detections
void validateTargetsGpu(
    Target* d_detections,
    int n,
    cudaStream_t stream = 0);

void NMSGpu(
    const Target* d_input_detections, 
    int input_num_detections,          
    Target* d_output_detections,       
    int* d_output_count_gpu,           
    int max_output_detections,         
    float nmsThreshold,
    int frame_width,
    int frame_height,
    
    int* d_x1,
    int* d_y1,
    int* d_x2,
    int* d_y2,
    float* d_areas,
    float* d_scores_nms,      
    int* d_classIds_nms,      
    float* d_iou_matrix,
    bool* d_keep,
    int* d_indices,
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
    cudaStream_t stream);


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
    cudaStream_t stream);

// GPU function to find closest target to crosshair
// Accepts d_num_detections as device pointer to avoid CPU-GPU sync
cudaError_t findClosestTargetGpu(
    const Target* d_detections,
    int* d_num_detections,  // Device pointer
    float crosshairX,
    float crosshairY,
    int* d_best_index,
    Target* d_best_target,
    cudaStream_t stream);


#endif 