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

// Define GRID_SIZE for spatial hashing
#define GRID_SIZE 10

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
        
        // Skip already invalidated targets
        if (det.classId < 0) {
            return;
        }
        
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

// Final validation kernel to remove extreme values after NMS
__global__ void finalValidateAndCleanKernel(
    Target* d_targets,
    int* d_count,
    int max_targets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_targets) {
        Target& target = d_targets[idx];
        
        // Skip already invalidated targets
        if (target.classId < 0) {
            return;
        }
        
        // Check for extreme values and mark as invalid
        if (abs(target.x) > 1000000 || abs(target.y) > 1000000 ||
            target.x < -100 || target.x > 2000 || target.y < -100 || target.y > 2000 ||
            target.width <= 0 || target.width > 1000 || target.height <= 0 || target.height > 1000 ||
            target.confidence <= 0.0f || target.confidence > 1.0f ||
            target.classId < 0 || target.classId > 100) {
            
            // Log the garbage value before cleaning it
            if (abs(target.x) > 1000000 || abs(target.y) > 1000000) {
                printf("[FINAL VALIDATION] Cleaning garbage target %d: x=%d, y=%d, w=%d, h=%d, conf=%.3f, cls=%d\n",
                       idx, target.x, target.y, target.width, target.height, target.confidence, target.classId);
            }
            
            // Mark as invalid by setting negative class ID
            target.classId = -1;
            target.confidence = 0.0f;
            target.x = -1;
            target.y = -1;
            target.width = 0;
            target.height = 0;
        }
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
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[validateTargetsGpu] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

// Kernel to validate single best target
__global__ void validateBestTargetKernel(
    Target* d_best_target)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Target& target = *d_best_target;
        
        // Optimize: combine checks to reduce branches
        bool invalid_position = (target.x < -100) | (target.x > 2000) | 
                                (target.y < -100) | (target.y > 2000);
        bool invalid_size = (target.width <= 0) | (target.width > 1000) |
                            (target.height <= 0) | (target.height > 1000);
        bool invalid_meta = (target.classId < 0) | (target.confidence <= 0.0f) | (target.confidence > 1.0f);
        
        if (invalid_position | invalid_size | invalid_meta) {
            
            // Clear invalid target
            target.classId = -1;
            target.confidence = 0.0f;
            target.x = -1;
            target.y = -1;
            target.width = 0;
            target.height = 0;
        }
    }
}

// Final validation function to clean extreme values
void finalValidateTargetsGpu(
    Target* d_targets,
    int* d_count,
    int max_targets,
    cudaStream_t stream)
{
    if (!d_targets || max_targets <= 0) return;
    
    const int block_size = 256;
    const int grid_size = (max_targets + block_size - 1) / block_size;
    
    finalValidateAndCleanKernel<<<grid_size, block_size, 0, stream>>>(d_targets, d_count, max_targets);
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[finalValidateTargetsGpu] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

// Validate single best target before host copy
void validateBestTargetGpu(
    Target* d_best_target,
    cudaStream_t stream)
{
    if (!d_best_target) return;
    
    validateBestTargetKernel<<<1, 1, 0, stream>>>(d_best_target);
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[validateBestTargetGpu] Kernel launch failed: %s\n", cudaGetErrorString(err));  
    }
#endif
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
// NMS gather kernel removed


// Optimized spatial indexing constants
#define HOST_GRID_SIZE 64  // Increased for better spatial partitioning
__constant__ int CUDA_GRID_SIZE = 64;  // 64x64 spatial grid for finer granularity
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
            
            // STRICT validation to prevent garbage values like 1061249024
            if (width > 2000 || height > 2000 || x > 2000 || y > 2000 || 
                x < -100 || y < -100 || width <= 0 || height <= 0 ||
                abs(x) > 1000000 || abs(y) > 1000000 ||  // Catch 10억대 극단값
                abs(width) > 1000000 || abs(height) > 1000000 ||
                !isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2) ||
                !isfinite(static_cast<float>(x)) || !isfinite(static_cast<float>(y)) ||
                !isfinite(static_cast<float>(width)) || !isfinite(static_cast<float>(height))) {
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("[YOLO DECODE ERROR] Extreme/invalid values detected: x=%d, y=%d, w=%d, h=%d (raw: x1=%.6f, y1=%.6f, x2=%.6f, y2=%.6f, conf=%.3f, cls=%d)\n",
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
                
                // STRICT validation to prevent garbage values like 1061249024
                if (width > 2000 || height > 2000 || x > 2000 || y > 2000 || 
                    x < -100 || y < -100 || width <= 0 || height <= 0 ||
                    abs(x) > 1000000 || abs(y) > 1000000 ||  // Catch 10억대 극단값
                    abs(width) > 1000000 || abs(height) > 1000000 ||
                    !isfinite(cx) || !isfinite(cy) || !isfinite(ow) || !isfinite(oh) ||
                    !isfinite(static_cast<float>(x)) || !isfinite(static_cast<float>(y)) ||
                    !isfinite(static_cast<float>(width)) || !isfinite(static_cast<float>(height))) {
                    if (threadIdx.x == 0 && blockIdx.x == 0) {
                        printf("[YOLO11 DECODE ERROR] Extreme/invalid values detected: x=%d, y=%d, w=%d, h=%d (raw: cx=%.6f, cy=%.6f, ow=%.6f, oh=%.6f, score=%.3f, cls=%d)\n",
                               x, y, width, height, cx, cy, ow, oh, max_score, max_class_id);
                    }
                    return; // Skip this detection
                }
                
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

// IoU calculation kernel removed


// NMS kernel removed


// NMS helper struct and extract kernel removed

// NMS removed for performance - not needed for aimbot




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
    int max_detections,
    const unsigned char* d_class_filter,
    int max_class_filter_size)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections_raw) {
        size_t base_idx = idx * stride;

        float confidence = readOutputValue(d_raw_output, output_type, base_idx + 4);

        if (confidence > conf_threshold) {
            int classId = static_cast<int>(readOutputValue(d_raw_output, output_type, base_idx + 5));
            
            // Apply class filter if provided
            if (d_class_filter && max_class_filter_size > 0) {
                if (classId >= max_class_filter_size || d_class_filter[classId] == 0) {
                    return;  // Skip this class
                }
            }

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
                
                // Atomic increment first, then check (thread-safe)
                int write_idx = ::atomicAdd(d_decoded_count, 1);
                
                // Only proceed if we got a valid index
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
    int max_detections,
    const unsigned char* d_class_filter,
    int max_class_filter_size)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < num_boxes_raw) {
        float max_score = -1.0f;
        int max_class_id = -1;
        
        // YOLO12 always has 15 channels: 4 bbox + 11 classes (no objectness)
        int class_start_idx = 4;
        
        // Early class filtering during score search
        for (int c = 0; c < num_classes; ++c) {
            // Skip non-allowed classes early
            if (d_class_filter && max_class_filter_size > 0) {
                if (c >= max_class_filter_size || d_class_filter[c] == 0) {
                    continue;  // Skip this class entirely
                }
            }
            
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

        // Early exit if no valid class or confidence too low
        if (max_class_id < 0 || max_score <= conf_threshold) {
            return;
        }
        
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
             
            // Atomic increment first, then check (thread-safe)
            int write_idx = ::atomicAdd(d_decoded_count, 1);
            
            // Only proceed if we got a valid index
            if (write_idx < max_detections) {
                Target& det = d_decoded_detections[write_idx];
                det.x = x;
                det.y = y;
                det.width = width;
                det.height = height;
                det.confidence = max_score;
                det.classId = max_class_id;
                
                // Debug assertion to catch any remaining issues
                #ifdef DEBUG
                if (abs(x + width/2) > 1000000 || abs(y + height/2) > 1000000) {
                    printf("[DEBUG] Warning: Large center coordinates detected - x:%d y:%d w:%d h:%d\n", 
                           x, y, width, height);
                }
                #endif
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
    const unsigned char* d_class_filter,
    int max_class_filter_size,
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
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections,
        d_class_filter, max_class_filter_size);

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
    const unsigned char* d_class_filter,
    int max_class_filter_size,
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
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections,
        d_class_filter, max_class_filter_size);

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
            if (d_detections[i].classId == head_class_id && d_detections[i].classId >= 0) {
                const Target& head = d_detections[i];
                
                // Check if this head is inside any body
                for (int j = 0; j < num; j++) {
                    if (i != j && d_detections[j].classId != head_class_id && d_detections[j].classId >= 0) {
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
            
            // Skip invalid detections and extreme values
            if (det.width > 0 && det.height > 0 && det.confidence > 0 && det.classId >= 0 &&
                det.x > -100 && det.x < 2000 && det.y > -100 && det.y < 2000 &&
                det.width < 1000 && det.height < 1000 &&
                abs(det.x) < 1000000 && abs(det.y) < 1000000) {
                float centerX = det.x + det.width * 0.5f;
                float centerY = det.y + det.height * 0.5f;
                
                // Additional validation for center coordinates
                if (!isfinite(centerX) || !isfinite(centerY) ||
                    abs(centerX) > 10000 || abs(centerY) > 10000) {
                    continue; // Skip targets with extreme center coordinates
                }
                
                float dx = fabsf(centerX - crosshairX);
                float dy = fabsf(centerY - crosshairY);
                
                // Validate dx, dy calculations
                if (!isfinite(dx) || !isfinite(dy) || dx > 10000 || dy > 10000) {
                    continue; // Skip targets with extreme distance values
                }
                
                float distance = dx + dy;
                
                // Final distance validation
                if (!isfinite(distance) || distance > 20000) {
                    continue; // Skip targets with extreme distances
                }
                
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
        
        // Skip invalid detections and extreme values
        if (det.width > 0 && det.height > 0 && det.confidence > 0 && det.classId >= 0 &&
            det.x > -100 && det.x < 2000 && det.y > -100 && det.y < 2000 &&
            det.width < 1000 && det.height < 1000 &&
            abs(det.x) < 1000000 && abs(det.y) < 1000000) {
            float centerX = det.x + det.width * 0.5f;
            float centerY = det.y + det.height * 0.5f;
            
            // Additional validation for center coordinates
            if (!isfinite(centerX) || !isfinite(centerY) ||
                abs(centerX) > 10000 || abs(centerY) > 10000) {
                // Skip this target - keep default min_distance = FLT_MAX
            } else {
                float dx = fabsf(centerX - crosshairX);
                float dy = fabsf(centerY - crosshairY);
                
                // Validate dx, dy calculations  
                if (isfinite(dx) && isfinite(dy) && dx < 10000 && dy < 10000) {
                    float distance = dx + dy;
                    
                    // Final distance validation
                    if (isfinite(distance) && distance < 20000) {
                        min_distance = distance;
                        min_index = idx;
                    }
                }
            }
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

// processNMSOutputKernel removed - NMS not needed for aimbot

// processNMSOutputGpu removed - NMS not needed for aimbot


 