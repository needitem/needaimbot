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

// Final validation kernel to remove extreme values
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
// Helper function to read output values based on data type
__device__ inline float readOutputValue(const void* buffer, int type, size_t index) {
    if (type == 0) { // kFLOAT
        return reinterpret_cast<const float*>(buffer)[index];
    } else if (type == 1) { // kHALF
        return __half2float(reinterpret_cast<const __half*>(buffer)[index]);
    }

    return 0.0f;
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

        // Optimized class score search using register caching and loop unrolling
        // Pre-load up to 16 class scores into registers to minimize loop overhead
        // and enable instruction-level parallelism
        float scores[16];
        int num_to_check = min(num_classes, 16);

        // Batch-read class scores into register array
        #pragma unroll 4
        for (int c = 0; c < num_to_check; ++c) {
            // Early class filter skip
            if (d_class_filter && max_class_filter_size > 0) {
                if (c >= max_class_filter_size || d_class_filter[c] == 0) {
                    scores[c] = -1.0f;
                    continue;
                }
            }

            // Back to channel-first layout: [batch, channel, anchor]
            size_t score_idx = (size_t)(class_start_idx + c) * num_boxes_raw + idx;
            if (score_idx < (size_t)num_rows * num_boxes_raw) {
                scores[c] = readOutputValue(d_raw_output, output_type, score_idx);
            } else {
                scores[c] = -1.0f;
            }
        }

        // Find max score in register array
        #pragma unroll 4
        for (int c = 0; c < num_to_check; ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                max_class_id = c;
            }
        }

        // Handle remaining classes beyond 16 (rare case for models with >16 classes)
        for (int c = 16; c < num_classes; ++c) {
            // Skip non-allowed classes early
            if (d_class_filter && max_class_filter_size > 0) {
                if (c >= max_class_filter_size || d_class_filter[c] == 0) {
                    continue;
                }
            }

            size_t score_idx = (size_t)(class_start_idx + c) * num_boxes_raw + idx;
            if (score_idx >= (size_t)num_rows * num_boxes_raw) {
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

    int output_type_int = static_cast<int>(output_type);

    decodeYolo11GpuKernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_output, output_type_int, actual_candidates, (int)num_rows, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections,
        d_class_filter, max_class_filter_size);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "[decodeYolo11Gpu] Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
    }
    
    return kernel_err;
}

// ============================================================================
// NMS (Non-Maximum Suppression) Implementation
// ============================================================================

// Compute IoU (Intersection over Union) between two boxes
__device__ float computeIoU(const Target& a, const Target& b) {
    float x1 = max(static_cast<float>(a.x), static_cast<float>(b.x));
    float y1 = max(static_cast<float>(a.y), static_cast<float>(b.y));
    float x2 = min(static_cast<float>(a.x + a.width), static_cast<float>(b.x + b.width));
    float y2 = min(static_cast<float>(a.y + a.height), static_cast<float>(b.y + b.height));
    
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float areaA = static_cast<float>(a.width * a.height);
    float areaB = static_cast<float>(b.width * b.height);
    float unionArea = areaA + areaB - intersection;
    
    return (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;
}

// NMS kernel - marks suppressed detections
// Each thread handles one detection and checks against higher-confidence detections
__global__ void nmsKernel(
    Target* d_detections,
    const int* d_count,
    bool* d_keep,
    float iou_threshold,
    int max_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = *d_count;
    
    if (idx >= count || idx >= max_detections) {
        return;
    }
    
    Target& current = d_detections[idx];
    
    // Skip invalid detections
    if (current.width <= 0 || current.height <= 0 || current.confidence <= 0.0f) {
        d_keep[idx] = false;
        return;
    }
    
    // Assume we keep this detection unless suppressed
    d_keep[idx] = true;
    
    // Check against all other detections with higher confidence
    for (int j = 0; j < count && j < max_detections; ++j) {
        if (j == idx) continue;
        
        const Target& other = d_detections[j];
        
        // Skip invalid detections
        if (other.width <= 0 || other.height <= 0 || other.confidence <= 0.0f) {
            continue;
        }
        
        // Only suppress if other has higher confidence (or same confidence but lower index)
        if (other.confidence > current.confidence || 
            (other.confidence == current.confidence && j < idx)) {
            float iou = computeIoU(current, other);
            if (iou > iou_threshold) {
                d_keep[idx] = false;
                return;
            }
        }
    }
}

// Compact kernel - removes suppressed detections
// Host function to perform NMS
cudaError_t performNmsGpu(
    Target* d_detections,
    int* d_count,
    bool* d_keep_flags,
    float iou_threshold,
    int max_detections,
    cudaStream_t stream)
{
    if (!d_detections || !d_count || !d_keep_flags) {
        return cudaErrorInvalidValue;
    }
    
    // Validate threshold
    if (!isfinite(iou_threshold) || iou_threshold < 0.0f || iou_threshold > 1.0f) {
        iou_threshold = 0.45f;  // Default
    }
    
    const int block_size = 128;
    const int grid_size = (max_detections + block_size - 1) / block_size;
    
    // Clear previous errors
    cudaGetLastError();
    
    // Run NMS kernel to mark which detections to keep
    nmsKernel<<<grid_size, block_size, 0, stream>>>(
        d_detections, d_count, d_keep_flags, iou_threshold, max_detections);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NMS] Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    return err;
}

// Compact kernel that uses max_detections as bound (for CUDA graph compatibility)
__global__ void compactNmsResultsKernelFixed(
    const Target* d_input,
    const bool* d_keep,
    Target* d_output,
    int* d_output_count,
    int max_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= max_detections) {
        return;
    }

    // Check if this detection is valid and should be kept
    const Target& t = d_input[idx];
    if (d_keep[idx] && t.width > 0 && t.height > 0 && t.confidence > 0.0f) {
        int out_idx = atomicAdd(d_output_count, 1);
        if (out_idx < max_detections) {
            d_output[out_idx] = t;
        }
    }
}


// In-place NMS that compacts results
// Optimized NMS: runs directly on input, compacts to separate output.
// 3 CUDA ops (NMS + memset + compact) instead of the old 5-op in-place approach.
cudaError_t performNmsCompactGpu(
    const Target* d_input,
    int* d_input_count,
    bool* d_keep_flags,
    Target* d_output,
    int* d_output_count,
    float iou_threshold,
    int max_detections,
    cudaStream_t stream)
{
    if (!d_input || !d_input_count || !d_keep_flags || !d_output || !d_output_count) {
        return cudaErrorInvalidValue;
    }

    // Step 1: Run NMS directly on input buffer (marks keep_flags)
    cudaError_t err = performNmsGpu(
        const_cast<Target*>(d_input), d_input_count, d_keep_flags,
        iou_threshold, max_detections, stream);
    if (err != cudaSuccess) {
        return err;
    }

    // Step 2: Reset output count for atomicAdd in compact kernel
    cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);

    // Step 3: Compact kept targets from input to output
    const int block_size = 128;
    const int grid_size = (max_detections + block_size - 1) / block_size;
    compactNmsResultsKernelFixed<<<grid_size, block_size, 0, stream>>>(
        d_input, d_keep_flags, d_output, d_output_count, max_detections);

    return cudaGetLastError();
}


