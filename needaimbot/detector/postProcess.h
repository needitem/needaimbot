#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInferRuntimeCommon.h> 



struct Detection
{
    cv::Rect box;
    float confidence;
    int classId;
};


void NMS(std::vector<Detection>& detections, float nmsThreshold);


void NMSGpu(
    const Detection* d_input_detections, 
    int input_num_detections,          
    Detection* d_output_detections,       
    int* d_output_count_gpu,           
    int max_output_detections,         
    float nmsThreshold,
    
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
    Detection* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections,                
    cudaStream_t stream);


cudaError_t decodeYolo11Gpu(
    const void* d_raw_output,          
    nvinfer1::DataType output_type,    
    const std::vector<int64_t>& shape, 
    int num_classes,                   
    float conf_threshold,              
    float img_scale,                   
    Detection* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections,                
    cudaStream_t stream);


std::vector<Detection> decodeYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float img_scale);

std::vector<Detection> decodeYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float img_scale);


std::vector<Detection> postProcessYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold);

std::vector<Detection> postProcessYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold);

#endif 
