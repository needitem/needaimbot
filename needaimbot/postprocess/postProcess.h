#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
// OpenCV removed - using custom structures
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
    
    // Helper methods for box access
    void getBox(int& outX, int& outY, int& outWidth, int& outHeight) const {
        outX = x;
        outY = y;
        outWidth = width;
        outHeight = height;
    }
    
    // Constructors
    Detection() : confidence(-1.0f), classId(-1), x(-1), y(-1), width(-1), height(-1) {}
    Detection(int x_, int y_, int width_, int height_, float conf, int cls) 
        : confidence(conf), classId(cls), x(x_), y(y_), 
          width(width_), height(height_) {}
};




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

// GPU function to find closest target to crosshair
cudaError_t findClosestTargetGpu(
    const Detection* d_detections,
    int num_detections,
    float crosshairX,
    float crosshairY,
    int* d_best_index,
    Detection* d_best_target,
    cudaStream_t stream);


#endif 