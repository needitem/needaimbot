#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInferRuntimeCommon.h> 



// Cache-optimized Detection structure
struct Detection
{
    // Most frequently accessed members first
    float confidence;     // 4 bytes
    int classId;         // 4 bytes
    
    // Box coordinates packed together
    int x;               // 4 bytes
    int y;               // 4 bytes
    int width;           // 4 bytes
    int height;          // 4 bytes
    
    // Helper methods for cv::Rect compatibility
    cv::Rect box() const { return cv::Rect(x, y, width, height); }
    void setBox(const cv::Rect& rect) {
        x = rect.x;
        y = rect.y;
        width = rect.width;
        height = rect.height;
    }
    
    // Constructor for compatibility
    Detection() : confidence(0), classId(0), x(0), y(0), width(0), height(0) {}
    Detection(const cv::Rect& rect, float conf, int cls) 
        : confidence(conf), classId(cls), x(rect.x), y(rect.y), 
          width(rect.width), height(rect.height) {}
};


void NMS(std::vector<Detection>& detections, float nmsThreshold);


void NMSGpu(
    const Detection* d_input_detections, 
    int input_num_detections,          
    Detection* d_output_detections,       
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
    Detection* d_decoded_detections,
    int* d_decoded_count,
    int max_candidates,
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
    int max_candidates,
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