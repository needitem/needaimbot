#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <algorithm>

#include "filterGpu.h"
#include "postProcess.h" 

// Warp-level reduction for color pixel counting
__device__ inline int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Simple copy kernel for pass-through when no filtering is needed
__global__ void copyDetectionsKernel(
    const Target* __restrict__ input_detections,
    int num_input_detections,
    Target* __restrict__ output_detections,
    int* __restrict__ output_count,
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First thread in first block sets the output count
    if (idx == 0) {
        *output_count = min(num_input_detections, max_output_detections);
    }
    
    // All threads copy their respective detections
    if (idx < num_input_detections && idx < max_output_detections) {
        output_detections[idx] = input_detections[idx];
    }
}

// Optimized class filtering kernel (separated from color filtering)
__global__ __launch_bounds__(256, 4) void filterTargetsByClassIdKernel(
    const Target* __restrict__ input_detections,
    const int* __restrict__ d_input_count,
    Target* __restrict__ output_detections,
    int* __restrict__ output_count,
    const unsigned char* __restrict__ d_allowed_class_ids,
    int max_check_id,
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Read input count from GPU memory
    int num_input_detections = *d_input_count;
    if (num_input_detections <= 0) return;
    
    for (; idx < num_input_detections; idx += stride) {
        const Target det = input_detections[idx];
        
        // Class ID filtering only
        bool should_keep = true;
        if (det.classId >= 0 && det.classId < max_check_id && !d_allowed_class_ids[det.classId]) {
            should_keep = false;
        }
        
        // If detection passes class filter, add to output
        if (should_keep) {
            int output_idx = atomicAdd(output_count, 1);
            if (output_idx < max_output_detections) {
                output_detections[output_idx] = det;
            }
        }
    }
}

// Separate RGB color filtering kernel (to be run after class filtering, before NMS)
__global__ __launch_bounds__(256, 4) void filterTargetsByColorKernel(
    const Target* __restrict__ input_detections,
    int num_input_detections,
    Target* __restrict__ output_detections,
    int* __restrict__ output_count,
    const unsigned char* __restrict__ d_color_mask,
    int mask_pitch,
    int min_color_pixels,
    bool remove_color_matches,
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < num_input_detections; idx += stride) {
        const Target det = input_detections[idx];
        
        bool should_keep = true;
        
        // Color filtering with optimized pixel counting
        if (d_color_mask != nullptr) {
            int x0 = det.x;
            int y0 = det.y;
            int x1 = x0 + det.width;
            int y1 = y0 + det.height;
            
            // Count color matching pixels with early exit
            int color_pixel_count = 0;
            
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    if (d_color_mask[y * mask_pitch + x]) {
                        color_pixel_count++;
                        
                        // Early exit optimization for remove_color_matches
                        if (remove_color_matches && color_pixel_count >= min_color_pixels) {
                            should_keep = false;
                            goto early_exit;
                        }
                    }
                }
                
                // Early exit optimization for !remove_color_matches
                if (!remove_color_matches) {
                    int remaining_pixels = (y1 - y - 1) * det.width + (x1 - x0);
                    if (color_pixel_count + remaining_pixels < min_color_pixels) {
                        should_keep = false;
                        goto early_exit;
                    }
                }
            }
            
            // Final check for color filtering logic
            if (remove_color_matches) {
                if (color_pixel_count >= min_color_pixels) {
                    should_keep = false;
                }
            } else {
                if (color_pixel_count < min_color_pixels) {
                    should_keep = false;
                }
            }
        }
        
        early_exit:
        // If detection passes color filter, add to output
        if (should_keep) {
            int output_idx = atomicAdd(output_count, 1);
            if (output_idx < max_output_detections) {
                output_detections[output_idx] = det;
            }
        }
    }
}

// Host function for class ID filtering only
cudaError_t filterTargetsByClassIdGpu(
    const Target* d_input_detections,
    const int* d_input_count,
    Target* d_output_detections,
    int* d_output_count,
    const unsigned char* d_allowed_class_ids,
    int max_check_id,
    int max_output_detections,
    cudaStream_t stream
) {
    
    // Initialize output count to 0
    cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
    
    // Launch class filtering kernel - use sufficient grid size
    int block_size = 256;
    int grid_size = (max_output_detections + block_size - 1) / block_size;
    
    filterTargetsByClassIdKernel<<<grid_size, block_size, 0, stream>>>(
        d_input_detections,
        d_input_count,
        d_output_detections,
        d_output_count,
        d_allowed_class_ids,
        max_check_id,
        max_output_detections
    );
    
    return cudaGetLastError();
}

// New separate host function for RGB color filtering
cudaError_t filterTargetsByColorGpu(
    const Target* d_input_detections,
    int num_input_detections,
    Target* d_output_detections,
    int* d_output_count,
    const unsigned char* d_color_mask,
    int mask_pitch,
    int min_color_pixels,
    bool remove_color_matches,
    int max_output_detections,
    cudaStream_t stream
) {
    if (num_input_detections <= 0) return cudaSuccess;
    
    // If no color mask, just copy input to output
    if (d_color_mask == nullptr) {
        int block_size = 256;
        int grid_size = (num_input_detections + block_size - 1) / block_size;
        
        copyDetectionsKernel<<<grid_size, block_size, 0, stream>>>(
            d_input_detections,
            num_input_detections,
            d_output_detections,
            d_output_count,
            max_output_detections
        );
        
        return cudaGetLastError();
    }
    
    // Launch color filtering kernel
    int block_size = 256;
    int grid_size = (num_input_detections + block_size - 1) / block_size;
    
    filterTargetsByColorKernel<<<grid_size, block_size, 0, stream>>>(
        d_input_detections,
        num_input_detections,
        d_output_detections,
        d_output_count,
        d_color_mask,
        mask_pitch,
        min_color_pixels,
        remove_color_matches,
        max_output_detections
    );
    
    return cudaGetLastError();
}