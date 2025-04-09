#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>       // For thrust::copy_if
#include <thrust/iterator/counting_iterator.h> // For thrust::counting_iterator
#include <thrust/gather.h>     // Added for thrust::gather

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

// Functor for Thrust copy_if
struct is_kept {
    const bool* d_keep_ptr;
    is_kept(const bool* ptr) : d_keep_ptr(ptr) {}
    __host__ __device__
    bool operator()(const int& i) const {
        return d_keep_ptr[i];
    }
};

// Modified NMSGpu function for direct GPU processing
void NMSGpu(
    const Detection* d_input_detections,
    int input_num_detections,
    Detection* d_output_detections,
    int* d_output_count_gpu,
    int max_output_detections,
    float nmsThreshold,
    cudaStream_t stream)
{
    if (input_num_detections <= 0) {
        cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        return;
    }

    // --- Temporary GPU Buffers --- 
    // Need buffers for coordinates, areas, scores, IoU matrix, and keep flags.
    // Consider using a memory pool for better performance if NMS is called frequently.
    int* d_x1 = nullptr;
    int* d_y1 = nullptr;
    int* d_x2 = nullptr;
    int* d_y2 = nullptr;
    float* d_areas = nullptr;
    float* d_scores = nullptr;
    float* d_iou_matrix = nullptr;
    bool* d_keep = nullptr;
    int* d_indices = nullptr; // For copy_if

    cudaError_t err;
    err = cudaMallocAsync(&d_x1, input_num_detections * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_y1, input_num_detections * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_x2, input_num_detections * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_y2, input_num_detections * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_areas, input_num_detections * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_scores, input_num_detections * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_iou_matrix, (size_t)input_num_detections * input_num_detections * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_keep, input_num_detections * sizeof(bool), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;
    err = cudaMallocAsync(&d_indices, input_num_detections * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup_and_exit;

    // --- Kernel to Extract Data from Detection Struct --- 
    // (Need to write this kernel: extractDataKernel)
    // extractDataKernel<<<...>>> (d_input_detections, input_num_detections, d_x1, d_y1, d_x2, d_y2, d_areas, d_scores, stream);
    // Placeholder: For now, assume data is already in separate arrays (this won't work yet)
    // In a real scenario, you MUST extract data from d_input_detections here.
    // Let's add a simple placeholder kernel.
    
    /*
    __global__ void extractDataKernel(
        const Detection* input, int n, int* x1, int* y1, int* x2, int* y2, float* areas, float* scores
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            const cv::Rect& box = input[idx].box;
            x1[idx] = box.x;
            y1[idx] = box.y;
            x2[idx] = box.x + box.width;
            y2[idx] = box.y + box.height;
            areas[idx] = (float)box.width * box.height; // area might be 0 if width/height is 0
            scores[idx] = input[idx].confidence;
        }
    }
    // Launch configuration needs calculation
    // extractDataKernel<<<grid, block, 0, stream>>>(...);
    */
    
    // --- Initialize Keep Flags and IoU Matrix --- 
    // Initialize d_keep to true using Thrust or a simple kernel
    // thrust::fill(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_keep), thrust::device_pointer_cast(d_keep) + input_num_detections, true);
    cudaMemsetAsync(d_keep, 1, input_num_detections * sizeof(bool), stream); // Set all to true (1)
    cudaMemsetAsync(d_iou_matrix, 0, (size_t)input_num_detections * input_num_detections * sizeof(float), stream); // Zero out IoU matrix

    // --- Calculate IoU Matrix --- 
    {
        dim3 block_size(16, 16);
        dim3 grid_size((input_num_detections + block_size.x - 1) / block_size.x,
                       (input_num_detections + block_size.y - 1) / block_size.y);
        calculateIoUKernel<<<grid_size, block_size, 0, stream>>>(
            d_x1, d_y1, d_x2, d_y2, d_areas, d_iou_matrix, input_num_detections, nmsThreshold // Pass nmsThreshold here? Original kernel didn't use it directly
        );
    }

    // --- Perform NMS --- 
    {
        const int block_nms = 256;
        const int grid_nms = (input_num_detections + block_nms - 1) / block_nms;
        nmsKernel<<<grid_nms, block_nms, 0, stream>>>(
            d_keep, d_iou_matrix, d_scores, input_num_detections, nmsThreshold
        );
    }

    // --- Compact Results using Thrust copy_if --- 
    {
        thrust::device_ptr<const Detection> d_input_ptr(d_input_detections);
        thrust::device_ptr<Detection> d_output_ptr(d_output_detections);
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + input_num_detections;
        
        // Create temporary buffer for indices to keep
        thrust::device_ptr<int> d_indices_ptr(d_indices);

        // 1. Filter indices: Keep indices where d_keep is true
        auto end_indices_iter = thrust::copy_if(
            thrust::cuda::par.on(stream),
            first, last,
            d_indices_ptr,
            is_kept(d_keep) // Use the functor with the keep flags
        );
        int final_count = end_indices_iter - d_indices_ptr;

        // Ensure final_count doesn't exceed output buffer size
        final_count = min(final_count, max_output_detections);

        // 2. Gather data: Use the filtered indices to copy Detection objects
        thrust::gather(
            thrust::cuda::par.on(stream),
            d_indices_ptr, d_indices_ptr + final_count, // Indices to gather from
            d_input_ptr,                               // Input Detection array
            d_output_ptr                               // Output Detection array
        );

        // Copy the final count to the output count buffer
        cudaMemcpyAsync(d_output_count_gpu, &final_count, sizeof(int), cudaMemcpyHostToDevice, stream);
    }

cleanup_and_exit:
    // Free temporary buffers (consider using RAII or smart pointers for robustness)
    cudaFreeAsync(d_x1, stream);
    cudaFreeAsync(d_y1, stream);
    cudaFreeAsync(d_x2, stream);
    cudaFreeAsync(d_y2, stream);
    cudaFreeAsync(d_areas, stream);
    cudaFreeAsync(d_scores, stream);
    cudaFreeAsync(d_iou_matrix, stream);
    cudaFreeAsync(d_keep, stream);
    cudaFreeAsync(d_indices, stream);

    // Check for errors that might have occurred during cleanup or earlier
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NMSGpu] CUDA error during processing or cleanup: %s\n", cudaGetErrorString(err));
        // Optionally, try to set the output count to 0 if an error occurred before that point
        // cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
    }
}

/* Original NMSGpu implementation (operates on std::vector)
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
*/ 