#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInferRuntimeCommon.h> // Include for nvinfer1::DataType

// #include "detector.h" // Removed potential circular dependency

struct Detection
{
    cv::Rect box;
    float confidence;
    int classId;
};

// CPU implementation
void NMS(std::vector<Detection>& detections, float nmsThreshold);

// GPU implementation - NMS
void NMSGpu(
    const Detection* d_input_detections, // Input detections (GPU)
    int input_num_detections,          // Number of input detections
    Detection* d_output_detections,       // Output buffer for filtered detections (GPU)
    int* d_output_count_gpu,           // Output count of filtered detections (GPU, single int)
    int max_output_detections,         // Max size of the output buffer
    float nmsThreshold,
    cudaStream_t stream = 0);

// --- GPU Decoding Functions ---
// Decodes YOLOv10 output directly on GPU
cudaError_t decodeYolo10Gpu(
    const void* d_raw_output,          // Raw output buffer (GPU, float* or half*)
    nvinfer1::DataType output_type,    // Data type of the raw output
    const std::vector<int64_t>& shape, // Shape of the raw output tensor
    int num_classes,                   // Number of classes
    float conf_threshold,              // Confidence threshold
    float img_scale,                   // Image scale factor
    Detection* d_decoded_detections,   // Output buffer for decoded detections (GPU)
    int* d_decoded_count,              // Output/Input counter for decoded detections (GPU, atomic)
    int max_detections,                // Maximum number of detections allowed in output buffer
    cudaStream_t stream);

// Decodes YOLOv11 (YOLOv8/9 format) output directly on GPU
cudaError_t decodeYolo11Gpu(
    const void* d_raw_output,          // Raw output buffer (GPU, float* or half*)
    nvinfer1::DataType output_type,    // Data type of the raw output
    const std::vector<int64_t>& shape, // Shape of the raw output tensor
    int num_classes,                   // Number of classes
    float conf_threshold,              // Confidence threshold
    float img_scale,                   // Image scale factor
    Detection* d_decoded_detections,   // Output buffer for decoded detections (GPU)
    int* d_decoded_count,              // Output/Input counter for decoded detections (GPU, atomic)
    int max_detections,                // Maximum number of detections allowed in output buffer
    cudaStream_t stream);

// --- CPU Decoding Functions (NMS separated) ---
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

// --- Original Combined Functions (kept for potential compatibility or reference) ---
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

#endif // POSTPROCESS_H