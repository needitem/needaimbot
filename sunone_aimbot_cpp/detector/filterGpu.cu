#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "filterGpu.h"
#include "postProcess.h" // Include Detection definition

__global__ void filterDetectionsByClassIdKernel(
    const Detection* input_detections,
    int num_input_detections,
    Detection* output_detections,
    int* output_count, // Assumed to be initialized to 0 on GPU
    int class_id_person, // Parameter remains but is unused in the modified kernel
    int class_id_head,   
    bool disable_headshot, 
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_input_detections) {
        const Detection& det = input_detections[idx];
        bool should_keep = true; // Default to keeping the detection

        // Only filter out 'head' class if disable_headshot is true
        if (disable_headshot && det.classId == class_id_head) {
            should_keep = false; // Do not keep if headshot is disabled and it's a head
        }

        if (should_keep) {
            // Atomically increment the output count and get the index to write to
            int write_idx = atomicAdd(output_count, 1);

            // Ensure we don't write past the allocated buffer size
            if (write_idx < max_output_detections) {
                output_detections[write_idx] = det;
            } else {
                 // Decrement count if we went over the limit
                 atomicSub(output_count, 1); 
            }
        }
    }
}

cudaError_t filterDetectionsByClassIdGpu(
    const Detection* d_input_detections,
    int num_input_detections,
    Detection* d_output_detections,
    int* d_output_count, // Remember to cudaMemset this to 0 before calling!
    int class_id_person,
    int class_id_head,
    bool disable_headshot,
    int max_output_detections,
    cudaStream_t stream)
{
    if (num_input_detections <= 0) {
        // No input detections, ensure output count is 0 (although it should already be)
        cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    // Ensure the output count is reset before the kernel launch
    cudaError_t err = cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[FilterGPU] Failed cudaMemsetAsync on output count: %s\n", cudaGetErrorString(err));
        return err;
    }


    const int block_size = 256;
    const int grid_size = (num_input_detections + block_size - 1) / block_size;

    filterDetectionsByClassIdKernel<<<grid_size, block_size, 0, stream>>>(
        d_input_detections,
        num_input_detections,
        d_output_detections,
        d_output_count,
        class_id_person,
        class_id_head,
        disable_headshot,
        max_output_detections);

    return cudaGetLastError();
} 