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

// Optimized HSV filtering with parallel reduction
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
    // Shared memory for warp-level reductions
    __shared__ int warp_counts[8]; // 256 threads = 8 warps
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < num_input_detections; idx += stride) {
        const Detection det = input_detections[idx];
        
        // Early rejection based on class ID
        if (det.classId >= 0 && det.classId < max_check_id && d_ignored_class_ids[det.classId]) {
            continue;
        }
        
        bool should_keep = true;
        
        // HSV filtering with parallel pixel counting
        if (d_hsv_mask != nullptr) {
            int x0 = det.x;
            int y0 = det.y;
            int x1 = x0 + det.width;
            int y1 = y0 + det.height;
            
            // Tile-based processing for better memory access
            const int TILE_SIZE = 32;
            int total_count = 0;
            
            for (int ty = y0; ty < y1; ty += TILE_SIZE) {
                for (int tx = x0; tx < x1; tx += TILE_SIZE) {
                    // Count pixels in this tile using warp-level parallelism
                    int local_count = 0;
                    int tid_in_warp = threadIdx.x & 31;
                    int warp_id = threadIdx.x / 32;
                    
                    // Each thread in warp processes different pixels
                    for (int offset = tid_in_warp; offset < TILE_SIZE * TILE_SIZE; offset += 32) {
                        int dy = offset / TILE_SIZE;
                        int dx = offset % TILE_SIZE;
                        int py = ty + dy;
                        int px = tx + dx;
                        
                        if (py < y1 && px < x1) {
                            local_count += d_hsv_mask[py * mask_pitch + px] ? 1 : 0;
                        }
                    }
                    
                    // Warp-level reduction
                    local_count = warpReduceSum(local_count);
                    
                    // First thread in warp accumulates result
                    if (tid_in_warp == 0) {
                        warp_counts[warp_id] = local_count;
                    }
                    __syncthreads();
                    
                    // Thread 0 sums all warp results
                    if (threadIdx.x == 0) {
                        for (int i = 0; i < 8; i++) {
                            total_count += warp_counts[i];
                        }
                    }
                    __syncthreads();
                    
                    // Broadcast result to all threads
                    total_count = __shfl_sync(0xffffffff, total_count, 0);
                    
                    // Early exit if threshold reached
                    if ((remove_hsv_matches && total_count >= min_hsv_pixels) ||
                        (!remove_hsv_matches && total_count >= min_hsv_pixels)) {
                        break;
                    }
                }
                
                if ((remove_hsv_matches && total_count >= min_hsv_pixels) ||
                    (!remove_hsv_matches && total_count >= min_hsv_pixels)) {
                    break;
                }
            }
            
            // Apply filtering logic
            if (remove_hsv_matches && total_count >= min_hsv_pixels) {
                should_keep = false;
            } else if (!remove_hsv_matches && total_count < min_hsv_pixels) {
                should_keep = false;
            }
        }
        
        // Write output if detection should be kept
        if (should_keep) {
            int write_idx = atomicAdd(output_count, 1);
            if (write_idx < max_output_detections) {
                output_detections[write_idx] = det;
            } else {
                atomicSub(output_count, 1);
            }
        }
    }
}

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
    cudaStream_t stream)
{
    if (num_input_detections <= 0) {
        
        return cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
    }

    
    cudaError_t err = cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[FilterGPU] Failed cudaMemsetAsync on output count: %s\n", cudaGetErrorString(err));
        return err;
    }

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
        max_output_detections);

    return cudaGetLastError();
} 