#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/distance.h>
#include <limits>
#include <cmath>

#include "scoringGpu.h"
#include "postProcess.h" 


__device__ inline float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = max(box1.x, box2.x);
    int yA = max(box1.y, box2.y);
    int xB = min(box1.x + box1.width, box2.x + box2.width);
    int yB = min(box1.y + box1.height, box2.y + box2.height);

    
    int interArea = max(0, xB - xA) * max(0, yB - yA);

    
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    float unionArea = static_cast<float>(box1Area + box2Area - interArea);

    
    return (unionArea > 0.0f) ? static_cast<float>(interArea) / unionArea : 0.0f;
}


__global__ void calculateTargetScoresGpuKernel(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight,
    float confidence_weight,
    int head_class_id,             
    float head_class_score_multiplier 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections) {
        const Detection& det = d_detections[idx];
        const cv::Rect& box = det.box;

        float centerX = box.x + box.width / 2.0f;
        float centerY = box.y + box.height / 2.0f;

        float frameCenterX = frame_width / 2.0f;
        float frameCenterY = frame_height / 2.0f;
        float dx = centerX - frameCenterX;
        float dy = centerY - frameCenterY;
        float distance_score = sqrtf(dx * dx + dy * dy) * distance_weight;

        float confidence_penalty_factor = (1.0f - det.confidence) * confidence_weight;
        d_scores[idx] = distance_score * (1.0f + confidence_penalty_factor);

        
        if (head_class_id != -1 && det.classId == head_class_id) {
            d_scores[idx] *= head_class_score_multiplier; 
        }
    }
}

cudaError_t calculateTargetScoresGpu(
    const Detection* d_detections,
    int num_detections,
    float* d_scores,
    int frame_width,
    int frame_height,
    float distance_weight_config,
    float confidence_weight_config,
    int head_class_id_param         
    ,cudaStream_t stream) {
    if (num_detections <= 0) {
        return cudaSuccess;
    }

    const float head_bonus_multiplier_val = 0.8f; 

    const int block_size = 256;
    const int grid_size = (num_detections + block_size - 1) / block_size;

    calculateTargetScoresGpuKernel<<<grid_size, block_size, 0, stream>>>( 
        d_detections,
        num_detections,
        d_scores,
        frame_width,
        frame_height,
        distance_weight_config,
        confidence_weight_config,
        head_class_id_param,
        head_bonus_multiplier_val 
    );

    return cudaGetLastError();
}


cudaError_t findBestTargetGpu(
    const float* d_scores,
    int num_detections,
    int* d_best_index_gpu,
    cudaStream_t stream)
{
    if (num_detections <= 0) {
         
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaSuccess;
    }
    try {
        thrust::device_ptr<const float> d_scores_ptr(d_scores);

        
        auto min_iter = thrust::min_element(
            thrust::cuda::par.on(stream),
            d_scores_ptr,
            d_scores_ptr + num_detections
        );

        
        int best_index = thrust::distance(d_scores_ptr, min_iter);

        
        cudaMemcpyAsync(
            d_best_index_gpu,
            &best_index,
            sizeof(int),
            cudaMemcpyHostToDevice,
            stream
        );
        return cudaGetLastError();
    } catch (const std::exception& e) {
         fprintf(stderr, "[Thrust Error] findBestTargetGpu: %s\n", e.what());
         
         cudaMemsetAsync(d_best_index_gpu, 0xFF, sizeof(int), stream);
         return cudaErrorUnknown;
    }
}

