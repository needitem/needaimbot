#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "filterGpu.h"
#include "postProcess.h" 

// Warp-level reduction for HSV pixel counting
__device__ inline int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized HSV filtering kernel
__global__ __launch_bounds__(256, 8) void filterDetectionsByClassIdKernel(
    const Detection* __restrict__ input_detections,
    int num_input_detections,
    Detection* __restrict__ output_detections,
    int* __restrict__ output_count,
    const unsigned char* __restrict__ d_ignored_class_ids,
    int max_check_id,
    const unsigned char* __restrict__ d_hsv_mask,
    int mask_pitch,
    int min_hsv_pixels,
    bool remove_hsv_matches,
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < num_input_detections; idx += stride) {
        const Detection det = input_detections[idx];
        
        // Early rejection based on class ID
        if (det.classId >= 0 && det.classId < max_check_id && d_ignored_class_ids[det.classId]) {
            continue;
        }
        
        bool should_keep = true;
        
        // HSV filtering with simple pixel counting
        if (d_hsv_mask != nullptr) {
            int x0 = det.x;
            int y0 = det.y;
            int x1 = x0 + det.width;
            int y1 = y0 + det.height;
            
            // Count HSV matching pixels
            int hsv_pixel_count = 0;
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    if (d_hsv_mask[y * mask_pitch + x]) {
                        hsv_pixel_count++;
                    }
                }
            }
            
            // Apply HSV filtering logic
            if (remove_hsv_matches) {
                if (hsv_pixel_count >= min_hsv_pixels) {
                    should_keep = false;
                }
            } else {
                if (hsv_pixel_count < min_hsv_pixels) {
                    should_keep = false;
                }
            }
        }
        
        // If detection passes all filters, add to output
        if (should_keep) {
            int output_idx = atomicAdd(output_count, 1);
            if (output_idx < max_output_detections) {
                output_detections[output_idx] = det;
            }
        }
    }
}

// Host function implementation
cudaError_t filterDetectionsByClassIdGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
    int* d_output_count,
    const unsigned char* d_ignored_class_ids,
    int max_check_id,
    const unsigned char* d_hsv_mask,
    int mask_pitch,
    int min_hsv_pixels,
    bool remove_hsv_matches,
    int max_output_detections,
    cudaStream_t stream
) {
    if (num_input_detections <= 0) return cudaSuccess;
    
    // Initialize output count to 0
    cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_input_detections + block_size - 1) / block_size;
    
    filterDetectionsByClassIdKernel<<<grid_size, block_size, 0, stream>>>(
        d_input_detections,
        num_input_detections,
        d_output_detections,
        d_output_count,
        d_ignored_class_ids,
        max_check_id,
        d_hsv_mask,
        mask_pitch,
        min_hsv_pixels,
        remove_hsv_matches,
        max_output_detections
    );
    
    return cudaGetLastError();
}