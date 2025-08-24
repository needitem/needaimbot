#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <vector>
#include <algorithm> 
#include <cmath>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>       
#include <thrust/iterator/counting_iterator.h> 
#include <thrust/gather.h>     

#include "postProcess.h"
#include <NvInferRuntimeCommon.h>
#include "../../utils/cuda_utils.h"

// For min/max functions
#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif 

// Fast initialization kernel
__global__ void initKeepKernel(bool* d_keep, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_keep[idx] = true;
    }
}

// Kernel to conditionally copy the best target based on d_best_index
__global__ void copyBestTargetKernel(
    const Target* d_detections,
    const int* d_best_index,
    Target* d_best_target,
    int num_detections)
{
    int best_idx = *d_best_index;
    if (best_idx >= 0 && best_idx < num_detections) {
        *d_best_target = d_detections[best_idx];
    } else {
        // Initialize to invalid target when no target is selected
        d_best_target->classId = -1;
        d_best_target->x = -1;
        d_best_target->y = -1;
        d_best_target->width = -1;
        d_best_target->height = -1;
        d_best_target->confidence = 0.0f;
    }
}

// Optimized compaction kernel with warp-level primitives
__global__ void compactTargetsKernel(
    const Target* d_input_detections,
    const bool* d_keep,
    Target* d_output_detections,
    int* d_output_count,
    int input_num_detections,
    int max_output_detections)
{
    // Use shared memory for block-level scan
    extern __shared__ int shared_data[];
    int* warp_sums = shared_data;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    // Load keep flag
    int local_val = (idx < input_num_detections && d_keep[idx]) ? 1 : 0;
    
    // Warp-level inclusive scan using shuffle operations
    int warp_scan = local_val;
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, warp_scan, offset);
        if (lane_id >= offset) warp_scan += n;
    }
    
    // Store warp sum to shared memory
    if (lane_id == 31) {
        warp_sums[warp_id] = warp_scan;
    }
    __syncthreads();
    
    // Block-level scan of warp sums (for up to 32 warps = 1024 threads)
    if (tid < (blockDim.x / 32)) {
        int val = warp_sums[tid];
        // Simple sequential scan for small number of warps
        for (int i = 0; i < tid; i++) {
            val += warp_sums[i];
        }
        warp_sums[tid] = val;
    }
    __syncthreads();
    
    // Compute write position
    int write_pos = warp_scan - local_val; // Convert to exclusive scan
    if (warp_id > 0) {
        write_pos += warp_sums[warp_id - 1];
    }
    
    // Get global offset for this block
    __shared__ int global_offset;
    if (tid == 0) {
        int block_total = (blockDim.x / 32 > 0) ? warp_sums[blockDim.x / 32 - 1] : 0;
        if (block_total > 0) {
            global_offset = atomicAdd(d_output_count, block_total);
        } else {
            global_offset = 0;
        }
    }
    __syncthreads();
    
    // Write compacted output
    if (local_val && idx < input_num_detections) {
        int final_pos = global_offset + write_pos;
        if (final_pos < max_output_detections) {
            d_output_detections[final_pos] = d_input_detections[idx];
        }
    }
}

// Kernel to validate and clean detections
__global__ void validateTargetsKernel(
    Target* d_detections,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Target& det = d_detections[idx];
        
        // Log if target is outside boundaries (before fixing)
        if (det.x < 0 || det.y < 0 || 
            det.x >= 640 || det.y >= 640 ||
            (det.x + det.width) > 640 || 
            (det.y + det.height) > 640) {
            if (idx == 0) { // Only log from first thread to avoid spam
                printf("[BOUNDARY WARNING] Target %d out of bounds: x=%d, y=%d, w=%d, h=%d (x+w=%d, y+h=%d)\n", 
                       idx, det.x, det.y, det.width, det.height, 
                       det.x + det.width, det.y + det.height);
            }
        }
        
        // Ensure positive dimensions
        if (det.width <= 0) det.width = 1;
        if (det.height <= 0) det.height = 1;
        // Ensure non-negative position
        if (det.x < 0) det.x = 0;
        if (det.y < 0) det.y = 0;
    }
}

// Export function for validation
void validateTargetsGpu(
    Target* d_detections,
    int n,
    cudaStream_t stream)
{
    if (n <= 0 || !d_detections) return;
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    validateTargetsKernel<<<grid_size, block_size, 0, stream>>>(d_detections, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[validateTargetsGpu] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

// Simple kernel to count kept detections
__global__ void countKeptTargetsKernel(
    const bool* d_keep,
    int* d_count,
    int n)
{
    // Fixed shared memory for CUDA Graph compatibility
    __shared__ int shared_counts[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread counts its element
    int local_count = 0;
    if (idx < n && d_keep[idx]) {
        local_count = 1;
    }
    
    // Reduce within block
    shared_counts[tid] = local_count;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_counts[tid] += shared_counts[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(d_count, shared_counts[0]);
    }
}

// More efficient kernel using atomic operations for gathering
__global__ void gatherKeptTargetsAtomicKernel(
    const Target* d_input_detections,
    const bool* d_keep,
    Target* d_output_detections,
    int* d_write_index,  // Global write index
    int n,
    int max_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && d_keep[idx]) {
        int output_idx = atomicAdd(d_write_index, 1);
        if (output_idx < max_output) {
            // Copy detection and ensure valid dimensions
            Target& output_det = d_output_detections[output_idx];
            const Target& input_det = d_input_detections[idx];
            
            // Copy all fields
            output_det = input_det;
            
            // Ensure valid dimensions
            if (output_det.width <= 0) {
                output_det.width = 1;
            }
            if (output_det.height <= 0) {
                output_det.height = 1;
            }
        }
    }
}


// Optimized spatial indexing constants
#define HOST_GRID_SIZE 64  // Increased for better spatial partitioning
__constant__ int GRID_SIZE = 64;  // 64x64 spatial grid for finer granularity
__constant__ int GRID_SHIFT = 6;  // log2(64) for faster division
__constant__ int GRID_MASK = 63;  // For fast modulo

// Warp-level primitives for faster reduction
#define WARP_SIZE 32

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
    const unsigned char* __restrict__ d_allowed_class_ids,
    int max_check_id,
    Target* d_decoded_detections,
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
            if (classId >= 0 && classId < max_check_id && d_allowed_class_ids && !d_allowed_class_ids[classId]) {
                return; // Skip non-allowed classes
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
            
            // Detect abnormal values (likely memory corruption or bad decoding)
            if (width > 1000 || height > 1000 || x > 1000 || y > 1000 || 
                x < -1000 || y < -1000) {
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("[YOLO DECODE ERROR] Abnormal values detected: x=%d, y=%d, w=%d, h=%d (raw: x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f, conf=%.3f, cls=%d)\n",
                           x, y, width, height, x1, y1, x2, y2, confidence, classId);
                }
                return; // Skip this detection
            }
            
            // Validate dimensions (reasonable range: 1-640 pixels)
            if (width > 0 && height > 0 && width <= 640 && height <= 640) {
                int write_idx = atomicAdd(d_decoded_count, 1);
                
                if (write_idx < max_detections) {
                    Target& det = d_decoded_detections[write_idx];
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
    const unsigned char* __restrict__ d_allowed_class_ids,
    int max_check_id,
    Target* d_decoded_detections,
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
                d_allowed_class_ids && !d_allowed_class_ids[max_class_id]) {
                return; // Skip non-allowed classes
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
                    Target& det = d_decoded_detections[write_idx];
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
    // No shared memory for CUDA Graph compatibility - read directly from global
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < num_boxes && idy < num_boxes && idx < idy) {
        // Load first box data
        int x1_a = d_x1[idx];
        int y1_a = d_y1[idx];
        int x2_a = d_x2[idx];
        int y2_a = d_y2[idx];
        float area_a = d_areas[idx];
        
        // Calculate spatial cell for first box
        float cx_a = (x1_a + x2_a) * 0.5f;
        float cy_a = (y1_a + y2_a) * 0.5f;
        int2 cell_a = getSpatialCell(cx_a, cy_a, inv_cell_width, inv_cell_height);
        
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
        if (!cellsAreNear(cell_a, cell_b, 1)) {
            return; // Matrix is initialized to 0
        }
        
        // Calculate intersection using min/max intrinsics
        int x1 = max(x1_a, x1_b);
        int y1 = max(y1_a, y1_b);
        int x2 = min(x2_a, x2_b);
        int y2 = min(y2_a, y2_b);
        
        // Early exit if no overlap
        if (x2 <= x1 || y2 <= y1) {
            return;
        }
        
        // Use FMA for better performance
        float intersection_area = __int2float_rn(x2 - x1) * __int2float_rn(y2 - y1);
        float union_area = fmaf(-1.0f, intersection_area, area_a + area_b);
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
    // No shared memory needed - reading directly from global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        if (!d_keep[idx]) return; 
        
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
    const Target* d_input_detections, int n, 
    int* d_x1, int* d_y1, int* d_x2, int* d_y2, 
    float* d_areas, float* d_scores, int* d_classIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const Target& det = d_input_detections[idx];
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
    cudaStream_t stream)
{
    
    cudaError_t err = cudaSuccess;
    const int block_size = 256;
    // Use fixed grid sizes for CUDA Graph compatibility
    const int MAX_GRID_SIZE = 256; // Fixed maximum grid size 

    // Validate input parameters
    if (!d_input_detections || !d_output_detections || !d_output_count_gpu ||
        !d_x1 || !d_y1 || !d_x2 || !d_y2 || !d_areas || !d_scores_nms || 
        !d_classIds_nms || !d_iou_matrix || !d_keep || !d_indices) {
        fprintf(stderr, "[NMSGpu] Error: NULL pointer passed to NMSGpu\n");
        if (d_output_count_gpu) cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        return;
    }

    if (input_num_detections <= 0 || max_output_detections <= 0) {
        cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        return; 
    }
    
    // Use actual input count for processing, output will be limited later
    int effective_detections = input_num_detections;

    
    

    
    {
        // Use fixed grid size for CUDA Graph compatibility
        // Ensure at least 1 block even if effective_detections is 0
        const int grid_extract = max(1, min((effective_detections + block_size - 1) / block_size, MAX_GRID_SIZE));
        extractDataKernel<<<grid_extract, block_size, 0, stream>>>( 
            d_input_detections, effective_detections,
            d_x1, d_y1, d_x2, d_y2,
            d_areas, d_scores_nms, d_classIds_nms 
        );
        err = cudaGetLastError(); 
        if (err != cudaSuccess) {
            fprintf(stderr, "[NMSGpu] extractDataKernel failed: %s\n", cudaGetErrorString(err));
            fprintf(stderr, "[NMSGpu] CUDA error occurred: %s (%d) - input_num_detections=%d, effective=%d, max_output=%d, grid=%d, block=%d\n", 
                    cudaGetErrorString(err), err, input_num_detections, effective_detections, max_output_detections, grid_extract, block_size);
            goto cleanup;
        }
    }


    
    
    
    // Initialize keep array to 1 (true)
    {
        // Use fixed grid size for CUDA Graph compatibility
        // Ensure at least 1 block
        int grid_init = max(1, min((effective_detections + block_size - 1) / block_size, MAX_GRID_SIZE));
        initKeepKernel<<<grid_init, block_size, 0, stream>>>(d_keep, effective_detections);
    }
    // Skip zeroing IoU matrix - kernel will only write non-zero values
    err = cudaGetLastError(); 
    if (err != cudaSuccess) {
        fprintf(stderr, "[NMSGpu] initKeepKernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    
    {
        dim3 block_iou(16, 16); 
        // Fixed grid size for CUDA Graph compatibility
        // Ensure at least 1 block
        int grid_dim = max(1, min((effective_detections + block_iou.x - 1) / block_iou.x, 64));
        dim3 grid_iou(grid_dim, grid_dim);
        
        // Pre-calculate inverse cell dimensions for faster division
        float inv_cell_width = static_cast<float>(HOST_GRID_SIZE) / frame_width;
        float inv_cell_height = static_cast<float>(HOST_GRID_SIZE) / frame_height;
        
        // No dynamic shared memory for CUDA Graph compatibility
        calculateIoUKernel<<<grid_iou, block_iou, 0, stream>>>( 
            d_x1, d_y1, d_x2, d_y2, d_areas, d_iou_matrix, effective_detections, nmsThreshold, inv_cell_width, inv_cell_height
        );
        err = cudaGetLastError(); 
        if (err != cudaSuccess) {
            fprintf(stderr, "[NMSGpu] calculateIoUKernel failed: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }

    
    {
        // Use fixed grid size for CUDA Graph compatibility
        // Ensure at least 1 block
        const int grid_nms = max(1, min((effective_detections + block_size - 1) / block_size, MAX_GRID_SIZE));
        nmsKernel<<<grid_nms, block_size, 0, stream>>>( 
            d_keep, d_iou_matrix, d_scores_nms, d_classIds_nms, effective_detections, nmsThreshold 
        );
        err = cudaGetLastError(); 
        if (err != cudaSuccess) {
            fprintf(stderr, "[NMSGpu] nmsKernel failed: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }


    // Replace Thrust with CUDA Graph compatible kernels
    {
        // Reset output count to use as write index
        cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        
        // Single pass: gather kept detections and count simultaneously
        // Use fixed grid size for CUDA Graph compatibility
        // Ensure at least 1 block
        const int gather_blocks = max(1, min((effective_detections + block_size - 1) / block_size, MAX_GRID_SIZE));
        gatherKeptTargetsAtomicKernel<<<gather_blocks, block_size, 0, stream>>>(
            d_input_detections, d_keep, d_output_detections, 
            d_output_count_gpu,  // Use as atomic write index
            effective_detections, max_output_detections
        );
        err = cudaGetLastError(); 
        if (err != cudaSuccess) {
            fprintf(stderr, "[NMSGpu] gatherKeptTargetsAtomicKernel failed: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }


cleanup: 
    
    cudaError_t lastErr = cudaGetLastError(); 
    if (err != cudaSuccess || lastErr != cudaSuccess) {
        cudaError_t errorToReport = (err != cudaSuccess) ? err : lastErr;
        fprintf(stderr, "[NMSGpu] CUDA error occurred: %s (%d) - input_num_detections=%d, effective=%d, max_output=%d\n", 
                cudaGetErrorString(errorToReport), errorToReport, input_num_detections, effective_detections, max_output_detections);
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
    Target* d_decoded_detections,   
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

            // CRITICAL: Validate bbox values before processing
            if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2)) {
                return;  // Skip NaN or infinity values
            }
            
            // Reasonable bounds check
            const float MAX_COORD = 10000.0f;
            if (x1 < 0 || x1 > MAX_COORD || y1 < 0 || y1 > MAX_COORD ||
                x2 < 0 || x2 > MAX_COORD || y2 < 0 || y2 > MAX_COORD) {
                return;  // Skip out-of-bounds values
            }

            // Convert to pixel coordinates
            int x = static_cast<int>(x1 * img_scale);
            int y = static_cast<int>(y1 * img_scale);
            int width = static_cast<int>((x2 - x1) * img_scale);
            int height = static_cast<int>((y2 - y1) * img_scale);

            // Additional validation after scaling
            const int MAX_SCALED_COORD = 10000;
            if (x < -1000 || x > MAX_SCALED_COORD || 
                y < -1000 || y > MAX_SCALED_COORD ||
                width <= 0 || width > MAX_SCALED_COORD ||
                height <= 0 || height > MAX_SCALED_COORD) {
                return;  // Skip invalid scaled values
            }
            
            if (width > 0 && height > 0) {
                
                // Check current count before incrementing
                int current_count = *d_decoded_count;
                if (current_count >= max_detections) {
                    return;  // Stop decoding when limit reached
                }
                
                int write_idx = ::atomicAdd(d_decoded_count, 1);
                
                // Double-check after atomic operation (race condition protection)
                if (write_idx < max_detections) {  // Buffer overflow protection
                    Target& det = d_decoded_detections[write_idx];
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



__global__ void decodeYolo11GpuKernel(
    const void* d_raw_output,          
    int output_type,    
    int num_boxes_raw,             
    int num_rows,                  
    int num_classes,                   
    float conf_threshold,              
    float img_scale,                   
    Target* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < num_boxes_raw) {
        // box_base_idx variable removed as it was unused 

        
        float max_score = -1.0f;
        int max_class_id = -1;
        
        // YOLO12 always has 15 channels: 4 bbox + 11 classes (no objectness)
        int class_start_idx = 4;
        
        for (int c = 0; c < num_classes; ++c) {
            // Back to channel-first layout: [batch, channel, anchor]
            size_t score_idx = (class_start_idx + c) * num_boxes_raw + idx;
            if (score_idx >= num_rows * num_boxes_raw) {
                continue;
            }
            float score = readOutputValue(d_raw_output, output_type, score_idx);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // YOLO12: No objectness channel, use class confidence only
        float final_confidence = max_score;
        
        if (final_confidence > conf_threshold) {
            
            // Back to channel-first layout: [batch, channel, anchor]
            size_t cx_idx = 0 * num_boxes_raw + idx;
            size_t cy_idx = 1 * num_boxes_raw + idx;
            size_t ow_idx = 2 * num_boxes_raw + idx;
            size_t oh_idx = 3 * num_boxes_raw + idx;
            
            if (cx_idx >= num_rows * num_boxes_raw || cy_idx >= num_rows * num_boxes_raw || 
                ow_idx >= num_rows * num_boxes_raw || oh_idx >= num_rows * num_boxes_raw) {
                return;
            }
            
            float cx = readOutputValue(d_raw_output, output_type, cx_idx);
            float cy = readOutputValue(d_raw_output, output_type, cy_idx);
            float ow = readOutputValue(d_raw_output, output_type, ow_idx);
            float oh = readOutputValue(d_raw_output, output_type, oh_idx);

            
            // CRITICAL: Validate bbox values before processing
            // Check for NaN, infinity, or unreasonable values
            if (!isfinite(cx) || !isfinite(cy) || !isfinite(ow) || !isfinite(oh)) {
                return;  // Skip invalid values
            }
            
            // Reasonable bounds check (model output should be within input resolution)
            const float MAX_COORD = 10000.0f;  // Very generous upper bound
            if (cx < 0 || cx > MAX_COORD || cy < 0 || cy > MAX_COORD ||
                ow <= 0 || ow > MAX_COORD || oh <= 0 || oh > MAX_COORD) {
                return;  // Skip out-of-bounds values
            }
            
            if (ow > 0 && oh > 0) {
                
                const float half_ow = 0.5f * ow;
                const float half_oh = 0.5f * oh;
                int x = static_cast<int>((cx - half_ow) * img_scale);
                int y = static_cast<int>((cy - half_oh) * img_scale);
                int width = static_cast<int>(ow * img_scale);
                int height = static_cast<int>(oh * img_scale);

                // Additional validation after scaling
                const int MAX_SCALED_COORD = 10000;  // Reasonable upper bound for scaled coordinates
                if (x < -1000 || x > MAX_SCALED_COORD || 
                    y < -1000 || y > MAX_SCALED_COORD ||
                    width <= 0 || width > MAX_SCALED_COORD ||
                    height <= 0 || height > MAX_SCALED_COORD) {
                    return;  // Skip invalid scaled values
                }
                 
                // Check current count before incrementing
                int current_count = *d_decoded_count;
                if (current_count >= max_detections) {
                    return;  // Stop decoding when limit reached
                }
                
                int write_idx = ::atomicAdd(d_decoded_count, 1);
                
                // Double-check after atomic operation (race condition protection)
                if (write_idx < max_detections) {  // Buffer overflow protection
                    Target& det = d_decoded_detections[write_idx];
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
    cudaStream_t stream)
{
    
    if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo10Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    int64_t stride = shape[2];
    
    if (stride <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    // Use shape[1] as the actual number of candidates for YOLO10
    int actual_candidates = static_cast<int>(shape[1]);
    if (actual_candidates <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    const int block_size = 256;
    const int grid_size = (actual_candidates + block_size - 1) / block_size;

    if (d_raw_output == nullptr || d_decoded_detections == nullptr || d_decoded_count == nullptr) {
        fprintf(stderr, "[decodeYolo10Gpu] Error: Null pointer detected\n");
        return cudaErrorInvalidValue;
    }

    // Initialize decoded count to zero (asynchronous for better performance)
    cudaError_t init_err = cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
    if (init_err != cudaSuccess) {
        fprintf(stderr, "[decodeYolo10Gpu] Failed to initialize d_decoded_count: %s\n", cudaGetErrorString(init_err));
        return init_err;
    }

    // Clear any previous CUDA errors before kernel launch
    cudaGetLastError();

    // Validate parameters
    if (grid_size <= 0 || block_size <= 0 || actual_candidates <= 0 || stride <= 0 || max_detections <= 0) {
        fprintf(stderr, "[decodeYolo10Gpu] Invalid parameters: grid_size=%d, block_size=%d, actual_candidates=%d, stride=%d, max_detections=%d\n",
                grid_size, block_size, actual_candidates, (int)stride, max_detections);
        return cudaErrorInvalidValue;
    }

    decodeYolo10GpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_output, (int)output_type, actual_candidates, (int)stride, num_classes,
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
    Target* d_decoded_detections,
    int* d_decoded_count, 
    int max_detections,
    int max_candidates,
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

    int64_t num_rows = shape[1];
    int64_t num_boxes = shape[2];
    
    if (num_rows <= 0 || num_boxes <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    // Use num_boxes (shape[2]) as the actual number of anchor points for grid calculation
    int actual_candidates = static_cast<int>(num_boxes);
    if (actual_candidates <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    const int block_size = 256;
    const int grid_size = (actual_candidates + block_size - 1) / block_size;

    if (d_raw_output == nullptr || d_decoded_detections == nullptr || d_decoded_count == nullptr) {
        fprintf(stderr, "[decodeYolo11Gpu] Error: Null pointer detected\n");
        return cudaErrorInvalidValue;
    }

    // Initialize decoded count to zero (asynchronous for better performance)
    cudaError_t init_err = cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
    if (init_err != cudaSuccess) {
        return init_err;
    }

    // Clear any previous CUDA errors before kernel launch
    cudaGetLastError();
    
    // Validate parameters
    if (grid_size <= 0 || block_size <= 0 || actual_candidates <= 0 || num_rows <= 0 || max_detections <= 0) {
        return cudaErrorInvalidValue;
    }

    if (!isfinite(conf_threshold) || !isfinite(img_scale) || conf_threshold < 0.0f || img_scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    
    decodeYolo11GpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_output, (int)output_type, actual_candidates, (int)num_rows, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "[decodeYolo11Gpu] Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
    }
    
    return kernel_err;
}

// Helper function for atomic min on float with associated index - declared before use
__device__ void atomicMinFloat(float* addr, float value, int index, int* index_addr) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (value >= old_val) break;
        
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(value));
        if (old == assumed) {
            atomicExch(index_addr, index);
        }
    } while (assumed != old);
}

// GPU kernel for finding the closest target to crosshair

// Kernel with head-in-body priority selection
__global__ void findBestTargetWithHeadPriority(
    const Target* d_detections,
    const int* d_num_detections,
    float crosshairX,
    float crosshairY,
    int head_class_id,
    int* d_best_index,
    Target* d_best_target)
{
    int num = *d_num_detections;
    if (num <= 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_best_index = -1;
        }
        return;
    }
    
    // Only thread 0 does the work (since we have sequential dependencies)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Step 1: Check if any Head is inside a Body
        for (int i = 0; i < num; i++) {
            if (d_detections[i].classId == head_class_id) {
                const Target& head = d_detections[i];
                
                // Check if this head is inside any body
                for (int j = 0; j < num; j++) {
                    if (i != j && d_detections[j].classId != head_class_id) {
                        const Target& body = d_detections[j];
                        
                        // Check if head bounding box is completely inside body bounding box
                        if (head.x >= body.x && 
                            head.y >= body.y &&
                            (head.x + head.width) <= (body.x + body.width) &&
                            (head.y + head.height) <= (body.y + body.height)) {
                            
                            // Found a head inside a body - select it
                            *d_best_index = i;
                            *d_best_target = head;
                            return;
                        }
                    }
                }
            }
        }
        
        // Step 2: No head inside body found, select closest target to crosshair
        float min_distance = FLT_MAX;
        int best_idx = -1;
        
        for (int i = 0; i < num; i++) {
            const Target& det = d_detections[i];
            
            // Skip invalid detections
            if (det.width > 0 && det.height > 0 && det.confidence > 0) {
                float centerX = det.x + det.width * 0.5f;
                float centerY = det.y + det.height * 0.5f;
                
                // Log suspicious targets (center outside screen bounds)
                if (centerX < 0 || centerX >= 640 || centerY < 0 || centerY >= 640) {
                    printf("[TARGET SELECT WARNING] Target %d center out of bounds: centerX=%.1f, centerY=%.1f (box: x=%d,y=%d,w=%d,h=%d)\n",
                           i, centerX, centerY, det.x, det.y, det.width, det.height);
                }
                
                float dx = fabsf(centerX - crosshairX);
                float dy = fabsf(centerY - crosshairY);
                float distance = dx + dy;
                
                if (distance < min_distance) {
                    min_distance = distance;
                    best_idx = i;
                }
            }
        }
        
        if (best_idx >= 0) {
            *d_best_index = best_idx;
            *d_best_target = d_detections[best_idx];
            
            // Log when selecting a target with large offset
            const Target& selected = d_detections[best_idx];
            float selectedCenterX = selected.x + selected.width * 0.5f;
            float selectedCenterY = selected.y + selected.height * 0.5f;
            float offsetX = fabsf(selectedCenterX - crosshairX);
            float offsetY = fabsf(selectedCenterY - crosshairY);
            
            if (offsetX > 200 || offsetY > 200) {
                printf("[TARGET SELECT] Large offset target selected: idx=%d, centerX=%.1f, centerY=%.1f, offsetX=%.1f, offsetY=%.1f\n",
                       best_idx, selectedCenterX, selectedCenterY, offsetX, offsetY);
            }
        } else {
            *d_best_index = -1;
        }
    }
}

// New kernel that reads count from device memory
__global__ void findClosestTargetKernelWithDeviceCount(
    const Target* d_detections,
    const int* d_num_detections,
    float crosshairX,
    float crosshairY,
    float* d_best_distance,
    int* d_best_index)
{
    // Read the count from device memory
    int num_detections = *d_num_detections;
    if (num_detections <= 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_best_index = -1;
            *d_best_distance = FLT_MAX;
        }
        return;
    }
    
    extern __shared__ char shared[];
    float* s_distances = (float*)shared;
    int* s_indices = (int*)&s_distances[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    float min_distance = FLT_MAX;
    int min_index = -1;
    
    // Each thread computes distance for one detection
    if (idx < num_detections) {
        const Target& det = d_detections[idx];
        
        // Skip invalid detections
        if (det.width > 0 && det.height > 0 && det.confidence > 0) {
            float centerX = det.x + det.width * 0.5f;
            float centerY = det.y + det.height * 0.5f;
            
            float dx = fabsf(centerX - crosshairX);
            float dy = fabsf(centerY - crosshairY);
            float distance = dx + dy;
            
            min_distance = distance;
            min_index = idx;
        }
    }
    
    // Store in shared memory
    s_distances[tid] = min_distance;
    s_indices[tid] = min_index;
    __syncthreads();
    
    // Block-level reduction to find minimum distance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_distances[tid + s] < s_distances[tid]) {
                s_distances[tid] = s_distances[tid + s];
                s_indices[tid] = s_indices[tid + s];
            }
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        atomicMinFloat(d_best_distance, s_distances[0], s_indices[0], d_best_index);
    }
}

// New host function that accepts device pointer for count
cudaError_t findClosestTargetGpu(
    const Target* d_detections,
    int* d_num_detections,  // Device pointer
    float crosshairX,
    float crosshairY,
    int* d_best_index,
    Target* d_best_target,
    cudaStream_t stream)
{
    if (!d_detections || !d_num_detections || !d_best_index || !d_best_target) {
        return cudaErrorInvalidValue;
    }
    
    // Allocate temporary buffer for best distance using RAII
    // Use a static unique_ptr for reuse across calls to avoid allocation overhead
    static thread_local std::unique_ptr<CudaMemory<float>> best_distance_buffer;
    if (!best_distance_buffer) {
        best_distance_buffer = std::make_unique<CudaMemory<float>>(1);
    }
    float* d_best_distance = best_distance_buffer->get();
    
    // Initialize best distance and index
    cudaMemsetAsync(d_best_distance, 0x7F, sizeof(float), stream);  // Set to large value
    cudaMemsetAsync(d_best_index, 0xFF, sizeof(int), stream);      // Set to -1
    
    // Use maximum possible detections for grid size
    const int max_detections = 100;
    const int block_size = 256;
    const int grid_size = (max_detections + block_size - 1) / block_size;
    const size_t shared_size = block_size * (sizeof(float) + sizeof(int));
    
    // Launch kernel that reads count from device
    findClosestTargetKernelWithDeviceCount<<<grid_size, block_size, shared_size, stream>>>(
        d_detections,
        d_num_detections,
        crosshairX,
        crosshairY,
        d_best_distance,
        d_best_index
    );
    
    // Check for kernel errors
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        return kernel_err;
    }
    
    // Use the copyBestTargetKernel to conditionally copy based on d_best_index
    // This avoids needing to read d_best_index on the host
    copyBestTargetKernel<<<1, 1, 0, stream>>>(
        d_detections, d_best_index, d_best_target, max_detections);
    
    // RAII handles cleanup automatically
    return cudaSuccess;
}

// New function with head priority selection
cudaError_t findBestTargetWithHeadPriorityGpu(
    const Target* d_detections,
    int* d_num_detections,
    float crosshairX,
    float crosshairY,
    int head_class_id,
    int* d_best_index,
    Target* d_best_target,
    cudaStream_t stream)
{
    if (!d_detections || !d_num_detections || !d_best_index || !d_best_target) {
        return cudaErrorInvalidValue;
    }
    
    // Initialize best index
    cudaMemsetAsync(d_best_index, 0xFF, sizeof(int), stream);  // Set to -1
    
    // Launch kernel with a single thread block
    // Since we have sequential dependencies, we use only one thread
    findBestTargetWithHeadPriority<<<1, 1, 0, stream>>>(
        d_detections,
        d_num_detections,
        crosshairX,
        crosshairY,
        head_class_id,
        d_best_index,
        d_best_target
    );
    
    // Check for kernel errors
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        return kernel_err;
    }
    
    return cudaSuccess;
}


 