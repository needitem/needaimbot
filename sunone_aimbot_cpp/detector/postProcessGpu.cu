#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "postProcess.h"

// CUDA kernel to calculate IoU between boxes
__global__ void calculateIoUKernel(
    const int* d_x1, const int* d_y1, const int* d_x2, const int* d_y2,
    const float* d_areas, float* d_iou_matrix,
    int num_boxes, float nms_threshold) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < num_boxes && idy < num_boxes && idx < idy) {
        int x1 = max(d_x1[idx], d_x1[idy]);
        int y1 = max(d_y1[idx], d_y1[idy]);
        int x2 = min(d_x2[idx], d_x2[idy]);
        int y2 = min(d_y2[idx], d_y2[idy]);
        
        // Check if boxes overlap
        float width = (float)max(0, x2 - x1);
        float height = (float)max(0, y2 - y1);
        
        if (width > 0 && height > 0) {
            float intersection_area = width * height;
            float union_area = d_areas[idx] + d_areas[idy] - intersection_area;
            float iou = intersection_area / union_area;
            
            // Store IoU value
            d_iou_matrix[idx * num_boxes + idy] = iou;
            d_iou_matrix[idy * num_boxes + idx] = iou; // Symmetric matrix
        } else {
            d_iou_matrix[idx * num_boxes + idy] = 0.0f;
            d_iou_matrix[idy * num_boxes + idx] = 0.0f;
        }
    }
}

// CUDA kernel to perform NMS suppression
__global__ void nmsKernel(
    bool* d_keep, const float* d_iou_matrix,
    const float* d_scores, int num_boxes, float nms_threshold) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        if (!d_keep[idx]) return; // Already suppressed
        
        for (int i = 0; i < num_boxes; i++) {
            if (idx == i) continue; // Skip self
            
            if (d_scores[idx] > d_scores[i] && d_iou_matrix[idx * num_boxes + i] > nms_threshold) {
                d_keep[i] = false; // Suppress box i
            }
        }
    }
}

void NMSGpu(std::vector<Detection>& detections, float nmsThreshold, cudaStream_t stream) {
    if (detections.empty()) return;
    
    int num_boxes = detections.size();
    
    // Sort detections by confidence (highest first)
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    
    // Prepare host data
    thrust::host_vector<int> h_x1(num_boxes);
    thrust::host_vector<int> h_y1(num_boxes);
    thrust::host_vector<int> h_x2(num_boxes);
    thrust::host_vector<int> h_y2(num_boxes);
    thrust::host_vector<float> h_areas(num_boxes);
    thrust::host_vector<float> h_scores(num_boxes);
    
    for (int i = 0; i < num_boxes; i++) {
        const cv::Rect& box = detections[i].box;
        h_x1[i] = box.x;
        h_y1[i] = box.y;
        h_x2[i] = box.x + box.width;
        h_y2[i] = box.y + box.height;
        h_areas[i] = box.area();
        h_scores[i] = detections[i].confidence;
    }
    
    // Copy data to device
    thrust::device_vector<int> d_x1 = h_x1;
    thrust::device_vector<int> d_y1 = h_y1;
    thrust::device_vector<int> d_x2 = h_x2;
    thrust::device_vector<int> d_y2 = h_y2;
    thrust::device_vector<float> d_areas = h_areas;
    thrust::device_vector<float> d_scores = h_scores;
    thrust::device_vector<float> d_iou_matrix(num_boxes * num_boxes, 0.0f);
    thrust::device_vector<bool> d_keep(num_boxes, true);
    
    // Configure kernel execution
    dim3 block_size(16, 16);
    dim3 grid_size((num_boxes + block_size.x - 1) / block_size.x, 
                  (num_boxes + block_size.y - 1) / block_size.y);
    
    // Calculate IoU matrix
    calculateIoUKernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_x1.data()),
        thrust::raw_pointer_cast(d_y1.data()),
        thrust::raw_pointer_cast(d_x2.data()),
        thrust::raw_pointer_cast(d_y2.data()),
        thrust::raw_pointer_cast(d_areas.data()),
        thrust::raw_pointer_cast(d_iou_matrix.data()),
        num_boxes,
        nmsThreshold
    );
    
    // Perform NMS
    nmsKernel<<<(num_boxes + 255) / 256, 256, 0, stream>>>(
        thrust::raw_pointer_cast(d_keep.data()),
        thrust::raw_pointer_cast(d_iou_matrix.data()),
        thrust::raw_pointer_cast(d_scores.data()),
        num_boxes,
        nmsThreshold
    );
    
    // Copy results back to host
    thrust::host_vector<bool> h_keep = d_keep;
    
    // Update detections list
    std::vector<Detection> result;
    result.reserve(num_boxes);
    
    for (int i = 0; i < num_boxes; i++) {
        if (h_keep[i]) {
            result.push_back(detections[i]);
        }
    }
    
    detections = std::move(result);
} 