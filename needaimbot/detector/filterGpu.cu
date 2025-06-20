#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "filterGpu.h"
#include "postProcess.h" 


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
        bool should_keep = true;

        
        if (det.classId >= 0 && det.classId < max_check_id && d_ignored_class_ids[det.classId]) {
            continue; 
        }

        
        if (d_hsv_mask != nullptr) {
            int x0 = det.box.x;
            int y0 = det.box.y;
            int x1 = x0 + det.box.width;
            int y1 = y0 + det.box.height;
            int count = 0;
            for (int y = y0; y < y1; ++y) {
                const unsigned char* row = d_hsv_mask + y * mask_pitch;
                for (int x = x0; x < x1; ++x) {
                    if (row[x]) {
                        ++count;
                        if (remove_hsv_matches) {
                            
                            if (count >= min_hsv_pixels) {
                                goto skip_detection;
                            }
                        } else {
                            
                            if (count >= min_hsv_pixels) {
                                y = y1; 
                                break;
                            }
                        }
                    }
                }
            }
            
            if (!remove_hsv_matches && count < min_hsv_pixels) {
                goto skip_detection;
            }
        }

        
        int write_idx = atomicAdd(output_count, 1);
        if (write_idx < max_output_detections) {
            output_detections[write_idx] = det;
        } else {
            atomicSub(output_count, 1);
        }

        continue;
    skip_detection:
        continue;
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
