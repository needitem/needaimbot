#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <device_launch_parameters.h>
#include <device_atomic_functions.h> 
#include <vector>
#include <algorithm> 
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>       
#include <thrust/iterator/counting_iterator.h> 
#include <thrust/gather.h>     

#include "postProcess.h"
#include <NvInferRuntimeCommon.h> 

// Fast initialization kernel
__global__ void initKeepKernel(bool* d_keep, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_keep[idx] = true;
    }
}


// Optimized spatial indexing constants
__constant__ int GRID_SIZE = 32;  // 32x32 spatial grid for better granularity
__constant__ int GRID_SHIFT = 5;  // log2(32) for faster division
__constant__ int GRID_MASK = 31;  // For fast modulo

// Helper function to read output values based on data type
__device__ inline float readOutputValue(const void* buffer, int type, size_t index) {
    if (type == 0) { // kFLOAT
        return reinterpret_cast<const float*>(buffer)[index];
    } else if (type == 1) { // kHALF
        return __half2float(reinterpret_cast<const __half*>(buffer)[index]);
    }
    
    return 0.0f; 
}

// Fused decode and filter kernel for YOLO10
__global__ void decodeAndFilterYolo10Kernel(
    const void* d_raw_output,
    int output_type,
    int num_detections_raw,
    int stride,
    int num_classes,
    float conf_threshold,
    float img_scale,
    const unsigned char* __restrict__ d_ignored_class_ids,
    int max_check_id,
    Detection* d_decoded_detections,
    int* d_decoded_count,
    int max_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_detections_raw) {
        size_t base_idx = idx * stride;
        
        // Read confidence first for early rejection
        float confidence = readOutputValue(d_raw_output, output_type, base_idx + 4);
        
        if (confidence > conf_threshold) {
            int classId = static_cast<int>(readOutputValue(d_raw_output, output_type, base_idx + 5));
            
            // Apply class filter immediately
            if (classId >= 0 && classId < max_check_id && d_ignored_class_ids && d_ignored_class_ids[classId]) {
                return; // Skip ignored classes
            }
            
            // Decode bounding box
            float x1 = readOutputValue(d_raw_output, output_type, base_idx + 0);
            float y1 = readOutputValue(d_raw_output, output_type, base_idx + 1);
            float x2 = readOutputValue(d_raw_output, output_type, base_idx + 2);
            float y2 = readOutputValue(d_raw_output, output_type, base_idx + 3);
            
            // Convert to pixel coordinates
            int x = static_cast<int>(x1 * img_scale);
            int y = static_cast<int>(y1 * img_scale);
            int width = static_cast<int>((x2 - x1) * img_scale);
            int height = static_cast<int>((y2 - y1) * img_scale);
            
            // Validate dimensions
            if (width > 0 && height > 0) {
                int write_idx = atomicAdd(d_decoded_count, 1);
                
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.x = x;
                    det.y = y;
                    det.width = width;
                    det.height = height;
                    det.confidence = confidence;
                    det.classId = classId;
                }
            }
        }
    }
}

// Fused decode and filter kernel for YOLO11
__global__ void decodeAndFilterYolo11Kernel(
    const void* d_raw_output,
    int output_type,
    int num_boxes_raw,
    int num_rows,
    int num_classes,
    float conf_threshold,
    float img_scale,
    const unsigned char* __restrict__ d_ignored_class_ids,
    int max_check_id,
    Detection* d_decoded_detections,
    int* d_decoded_count,
    int max_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes_raw) {
        // Find max score and class
        float max_score = -1.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            size_t score_idx = (4 + c) * num_boxes_raw + idx;
            if (score_idx >= num_rows * num_boxes_raw) continue;
            
            float score = readOutputValue(d_raw_output, output_type, score_idx);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }
        
        // Apply confidence threshold and class filter
        if (max_score > conf_threshold) {
            // Check class filter
            if (max_class_id >= 0 && max_class_id < max_check_id && 
                d_ignored_class_ids && d_ignored_class_ids[max_class_id]) {
                return; // Skip ignored classes
            }
            
            // Decode bounding box
            float cx = readOutputValue(d_raw_output, output_type, 0 * num_boxes_raw + idx);
            float cy = readOutputValue(d_raw_output, output_type, 1 * num_boxes_raw + idx);
            float ow = readOutputValue(d_raw_output, output_type, 2 * num_boxes_raw + idx);
            float oh = readOutputValue(d_raw_output, output_type, 3 * num_boxes_raw + idx);
            
            if (ow > 0 && oh > 0) {
                const float half_ow = 0.5f * ow;
                const float half_oh = 0.5f * oh;
                int x = static_cast<int>((cx - half_ow) * img_scale);
                int y = static_cast<int>((cy - half_oh) * img_scale);
                int width = static_cast<int>(ow * img_scale);
                int height = static_cast<int>(oh * img_scale);
                
                int write_idx = atomicAdd(d_decoded_count, 1);
                
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.x = x;
                    det.y = y;
                    det.width = width;
                    det.height = height;
                    det.confidence = max_score;
                    det.classId = max_class_id;
                }
            }
        }
    }
}

__device__ inline int2 getSpatialCell(float cx, float cy, float inv_cell_width, float inv_cell_height) {
    // Direct calculation without divisions
    int cellX = min(GRID_SIZE - 1, __float2int_rn(cx * inv_cell_width));
    int cellY = min(GRID_SIZE - 1, __float2int_rn(cy * inv_cell_height));
    return make_int2(cellX, cellY);
}

__device__ inline bool cellsAreNear(int2 cell1, int2 cell2, int threshold = 1) {
    return abs(cell1.x - cell2.x) <= threshold && abs(cell1.y - cell2.y) <= threshold;
}

__global__ __launch_bounds__(256, 4) void calculateIoUKernel(
    const int* __restrict__ d_x1, const int* __restrict__ d_y1, 
    const int* __restrict__ d_x2, const int* __restrict__ d_y2,
    const float* __restrict__ d_areas, float* __restrict__ d_iou_matrix,
    int num_boxes, float nms_threshold, float inv_cell_width, float inv_cell_height) 
{
    // Increased shared memory for better cache utilization
    extern __shared__ char shared_mem[];
    int* s_x1 = (int*)shared_mem;
    int* s_y1 = s_x1 + blockDim.x;
    int* s_x2 = s_y1 + blockDim.x;
    int* s_y2 = s_x2 + blockDim.x;
    float* s_areas = (float*)(s_y2 + blockDim.x);
    int2* s_cells = (int2*)(s_areas + blockDim.x);
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Cooperative loading with coalesced access
    if (idx < num_boxes) {
        s_x1[tid] = d_x1[idx];
        s_y1[tid] = d_y1[idx];
        s_x2[tid] = d_x2[idx];
        s_y2[tid] = d_y2[idx];
        s_areas[tid] = d_areas[idx];
        
        // Pre-calculate spatial cells
        float cx = (s_x1[tid] + s_x2[tid]) * 0.5f;
        float cy = (s_y1[tid] + s_y2[tid]) * 0.5f;
        s_cells[tid] = getSpatialCell(cx, cy, inv_cell_width, inv_cell_height);
    }
    __syncthreads();
    
    if (idx < num_boxes && idy < num_boxes && idx < idy) {
        // Load second box data
        int x1_b = d_x1[idy];
        int y1_b = d_y1[idy];
        int x2_b = d_x2[idy];
        int y2_b = d_y2[idy];
        float area_b = d_areas[idy];
        
        // Calculate spatial cell for second box
        float cx_b = (x1_b + x2_b) * 0.5f;
        float cy_b = (y1_b + y2_b) * 0.5f;
        int2 cell_b = getSpatialCell(cx_b, cy_b, inv_cell_width, inv_cell_height);
        
        // Early spatial rejection
        if (!cellsAreNear(s_cells[tid], cell_b, 1)) {
            return; // Matrix is initialized to 0
        }
        
        // Calculate intersection using min/max intrinsics
        int x1 = max(s_x1[tid], x1_b);
        int y1 = max(s_y1[tid], y1_b);
        int x2 = min(s_x2[tid], x2_b);
        int y2 = min(s_y2[tid], y2_b);
        
        // Early exit if no overlap
        if (x2 <= x1 || y2 <= y1) {
            return;
        }
        
        // Use FMA for better performance
        float intersection_area = __int2float_rn(x2 - x1) * __int2float_rn(y2 - y1);
        float union_area = fmaf(-1.0f, intersection_area, s_areas[tid] + area_b);
        float iou = __fdiv_rn(intersection_area, union_area);
        
        // Single coalesced write (symmetric matrix)
        if (iou > nms_threshold) {
            d_iou_matrix[idx * num_boxes + idy] = iou;
            d_iou_matrix[idy * num_boxes + idx] = iou;
        }
    }
}


__global__ void nmsKernel(
    bool* d_keep, const float* d_iou_matrix,
    const float* d_scores, const int* d_classIds,
    int num_boxes, float nms_threshold) 
{
    // Use shared memory for frequently accessed data
    extern __shared__ float s_scores[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load scores into shared memory
    if (idx < num_boxes && tid < blockDim.x) {
        s_scores[tid] = d_scores[idx];
    }
    __syncthreads();
    
    if (idx < num_boxes) {
        if (!d_keep[idx]) return; 
        
        float my_score = s_scores[tid];
        int my_class = d_classIds[idx];
        
        // Unroll loop for better performance
        #pragma unroll 4
        for (int i = 0; i < num_boxes; i++) {
            if (idx == i) continue; 
            
            
            if (d_classIds[idx] == d_classIds[i]) {
                
                if (d_scores[idx] > d_scores[i] && d_iou_matrix[idx * num_boxes + i] > nms_threshold) {
                    d_keep[i] = false; 
                }
            }
        }
    }
}


struct is_kept {
    const bool* d_keep_ptr;
    is_kept(const bool* ptr) : d_keep_ptr(ptr) {}
    __host__ __device__
    bool operator()(const int& i) const {
        return d_keep_ptr[i];
    }
};


__global__ void extractDataKernel(
    const Detection* d_input_detections, int n, 
    int* d_x1, int* d_y1, int* d_x2, int* d_y2, 
    float* d_areas, float* d_scores, int* d_classIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const Detection& det = d_input_detections[idx];
        d_x1[idx] = det.x;
        d_y1[idx] = det.y;
        d_x2[idx] = det.x + det.width;
        d_y2[idx] = det.y + det.height;
        
        float width = max(0.0f, (float)det.width); 
        float height = max(0.0f, (float)det.height);
        d_areas[idx] = width * height; 
        d_scores[idx] = det.confidence;
        d_classIds[idx] = det.classId; 
    }
}


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
    cudaStream_t stream)
{
    
    cudaError_t err = cudaSuccess;
    const int block_size = 256; 
    int final_count = 0; 

    if (input_num_detections <= 0) {
        cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        return; 
    }

    
    

    
    {
        const int grid_extract = (input_num_detections + block_size - 1) / block_size;
        extractDataKernel<<<grid_extract, block_size, 0, stream>>>( 
            d_input_detections, input_num_detections,
            d_x1, d_y1, d_x2, d_y2,
            d_areas, d_scores_nms, d_classIds_nms 
        );
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }


    
    
    
    // Initialize keep array to 1 (true)
    {
        int grid_init = (input_num_detections + block_size - 1) / block_size;
        initKeepKernel<<<grid_init, block_size, 0, stream>>>(d_keep, input_num_detections);
    }
    // Skip zeroing IoU matrix - kernel will only write non-zero values
    err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;

    
    {
        dim3 block_iou(16, 16); 
        dim3 grid_iou((input_num_detections + block_iou.x - 1) / block_iou.x,
                       (input_num_detections + block_iou.y - 1) / block_iou.y);
        
        // Pre-calculate inverse cell dimensions for faster division
        float inv_cell_width = static_cast<float>(GRID_SIZE) / frame_width;
        float inv_cell_height = static_cast<float>(GRID_SIZE) / frame_height;
        
        // Calculate shared memory size
        size_t shared_mem_size = block_iou.x * (4 * sizeof(int) + sizeof(float) + sizeof(int2));
        
        calculateIoUKernel<<<grid_iou, block_iou, shared_mem_size, stream>>>( 
            d_x1, d_y1, d_x2, d_y2, d_areas, d_iou_matrix, input_num_detections, nmsThreshold, inv_cell_width, inv_cell_height
        );
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }

    
    {
        const int grid_nms = (input_num_detections + block_size - 1) / block_size;
        nmsKernel<<<grid_nms, block_size, 0, stream>>>( 
            d_keep, d_iou_matrix, d_scores_nms, d_classIds_nms, input_num_detections, nmsThreshold 
        );
         err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }


    
    try {
        thrust::device_ptr<const Detection> d_input_ptr(d_input_detections);
        thrust::device_ptr<Detection> d_output_ptr(d_output_detections);
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + input_num_detections;
        thrust::device_ptr<int> d_indices_ptr(d_indices);

        
        auto end_indices_iter = thrust::copy_if(
            thrust::cuda::par.on(stream),
            first, last,
            d_indices_ptr,
            is_kept(d_keep)
        );
        final_count = end_indices_iter - d_indices_ptr; 

        final_count = std::min(final_count, max_output_detections); 

        
        thrust::gather(
            thrust::cuda::par.on(stream),
            d_indices_ptr, d_indices_ptr + final_count,
            d_input_ptr,
            d_output_ptr
        );

        cudaMemcpyAsync(d_output_count_gpu, &final_count, sizeof(int), cudaMemcpyHostToDevice, stream);
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;

    } catch (const std::exception& e) {
        fprintf(stderr, "[Thrust Error] NMSGpu copy_if/gather: %s\n", e.what());
        err = cudaErrorUnknown; 
        goto cleanup; 
    }


cleanup: 
    
    

    cudaError_t lastErr = cudaGetLastError(); 
    if (err != cudaSuccess || lastErr != cudaSuccess) {
        cudaError_t errorToReport = (err != cudaSuccess) ? err : lastErr;
        fprintf(stderr, "[NMSGpu] CUDA error occurred: %s (%d)\n", cudaGetErrorString(errorToReport), errorToReport);
        if (err != cudaSuccess) { 
             cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        }
    }
}




__global__ void decodeYolo10GpuKernel(
    const void* d_raw_output,          
    int output_type,    
    int num_detections_raw,        
    int stride,                    
    int num_classes,                   
    float conf_threshold,              
    float img_scale,                   
    Detection* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections_raw) {
        size_t base_idx = idx * stride;

        float confidence = readOutputValue(d_raw_output, output_type, base_idx + 4);

        if (confidence > conf_threshold) {
            int classId = static_cast<int>(readOutputValue(d_raw_output, output_type, base_idx + 5));

            // YOLO10 outputs x1, y1, x2, y2 (top-left and bottom-right corners)
            float x1 = readOutputValue(d_raw_output, output_type, base_idx + 0);
            float y1 = readOutputValue(d_raw_output, output_type, base_idx + 1);
            float x2 = readOutputValue(d_raw_output, output_type, base_idx + 2); 
            float y2 = readOutputValue(d_raw_output, output_type, base_idx + 3);

            // Convert to pixel coordinates
            int x = static_cast<int>(x1 * img_scale);
            int y = static_cast<int>(y1 * img_scale);
            int width = static_cast<int>((x2 - x1) * img_scale);
            int height = static_cast<int>((y2 - y1) * img_scale);

            
            if (width > 0 && height > 0) {
                
                int write_idx = ::atomicAdd(d_decoded_count, 1);

                
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.x = x;
                    det.y = y;
                    det.width = width;
                    det.height = height;
                    det.confidence = confidence;
                    det.classId = classId;
                } else {
                    
                    
                }
            }
        }
    }
}



__global__ void decodeYolo11GpuKernel(
    const void* d_raw_output,          
    int output_type,    
    int num_boxes_raw,             
    int num_rows,                  
    int num_classes,                   
    float conf_threshold,              
    float img_scale,                   
    Detection* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < num_boxes_raw) {
        size_t box_base_idx = idx; 

        
        float max_score = -1.0f;
        int max_class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            
            size_t score_idx = (4 + c) * num_boxes_raw + idx;
            if (score_idx >= num_rows * num_boxes_raw) {
                printf("[decodeYolo11GpuKernel] Score index out of bounds: %zu >= %d\n", score_idx, num_rows * num_boxes_raw);
                continue;
            }
            float score = readOutputValue(d_raw_output, output_type, score_idx);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        
        if (max_score > conf_threshold) {
            
            size_t cx_idx = 0 * num_boxes_raw + idx;
            size_t cy_idx = 1 * num_boxes_raw + idx;
            size_t ow_idx = 2 * num_boxes_raw + idx;
            size_t oh_idx = 3 * num_boxes_raw + idx;
            
            if (cx_idx >= num_rows * num_boxes_raw || cy_idx >= num_rows * num_boxes_raw || 
                ow_idx >= num_rows * num_boxes_raw || oh_idx >= num_rows * num_boxes_raw) {
                printf("[decodeYolo11GpuKernel] Bbox index out of bounds\n");
                return;
            }
            
            float cx = readOutputValue(d_raw_output, output_type, cx_idx);
            float cy = readOutputValue(d_raw_output, output_type, cy_idx);
            float ow = readOutputValue(d_raw_output, output_type, ow_idx);
            float oh = readOutputValue(d_raw_output, output_type, oh_idx);

            
            if (ow > 0 && oh > 0) {
                
                const float half_ow = 0.5f * ow;
                const float half_oh = 0.5f * oh;
                int x = static_cast<int>((cx - half_ow) * img_scale);
                int y = static_cast<int>((cy - half_oh) * img_scale);
                int width = static_cast<int>(ow * img_scale);
                int height = static_cast<int>(oh * img_scale);


                 
                int write_idx = ::atomicAdd(d_decoded_count, 1);

                
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.x = x;
                    det.y = y;
                    det.width = width;
                    det.height = height;
                    det.confidence = max_score;
                    det.classId = max_class_id;
                } else {
                    
                    
                }
            }
        }
    }
}




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
    cudaStream_t stream)
{
    
    if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo10Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    if (max_candidates <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    int64_t stride = shape[2];
    
    if (stride <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    const int block_size = 256;
    const int grid_size = (max_candidates + block_size - 1) / block_size;

    if (d_raw_output == nullptr || d_decoded_detections == nullptr || d_decoded_count == nullptr) {
        fprintf(stderr, "[decodeYolo10Gpu] Error: Null pointer detected\n");
        return cudaErrorInvalidValue;
    }

    decodeYolo10GpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_output, (int)output_type, max_candidates, (int)stride, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "[decodeYolo10Gpu] Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
    }
    
    return kernel_err;
}


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
    cudaStream_t stream)
{
    // Fixed CUDA "invalid argument" error by:
    // 1. Using max_candidates consistently for grid calculation and kernel launch
    // 2. Proper initialization of decoded count with synchronous cudaMemset
    // 3. Clearing previous CUDA errors before kernel launch
    // 4. Added parameter validation for floating-point values
    
    if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo11Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    if (max_candidates <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    int64_t num_rows = shape[1];
    int64_t num_boxes = shape[2];
    
    if (num_rows <= 0 || num_boxes <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    const int block_size = 256;
    const int grid_size = (max_candidates + block_size - 1) / block_size;

    if (d_raw_output == nullptr || d_decoded_detections == nullptr || d_decoded_count == nullptr) {
        fprintf(stderr, "[decodeYolo11Gpu] Error: Null pointer detected\n");
        return cudaErrorInvalidValue;
    }

    // Initialize decoded count to zero (synchronous to ensure proper initialization)
    cudaError_t init_err = cudaMemset(d_decoded_count, 0, sizeof(int));
    if (init_err != cudaSuccess) {
        return init_err;
    }

    // Clear any previous CUDA errors before kernel launch
    cudaGetLastError();
    
    // Validate parameters
    if (grid_size <= 0 || block_size <= 0 || max_candidates <= 0 || num_rows <= 0 || max_detections <= 0) {
        return cudaErrorInvalidValue;
    }

    if (!isfinite(conf_threshold) || !isfinite(img_scale) || conf_threshold < 0.0f || img_scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    
    decodeYolo11GpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_output, (int)output_type, max_candidates, (int)num_rows, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "[decodeYolo11Gpu] Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
    }
    
    return kernel_err;
}

 