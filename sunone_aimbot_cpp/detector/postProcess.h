#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// #include "detector.h" // Removed potential circular dependency

struct Detection
{
    cv::Rect box;
    float confidence;
    int classId;
};

// CPU implementation
void NMS(std::vector<Detection>& detections, float nmsThreshold);

// GPU implementation - Modified Signature
void NMSGpu(
    const Detection* d_input_detections, // Input detections (GPU)
    int input_num_detections,          // Number of input detections
    Detection* d_output_detections,       // Output buffer for filtered detections (GPU)
    int* d_output_count_gpu,           // Output count of filtered detections (GPU, single int)
    int max_output_detections,         // Max size of the output buffer
    float nmsThreshold,
    cudaStream_t stream = 0);

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