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
    const unsigned char* d_ignored_class_ids, // Changed from individual bools
    int max_check_id,                         // New parameter: size of d_ignored_class_ids
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_input_detections) {
        const Detection& det = input_detections[idx];
        bool should_keep = true; // Default to keeping the detection

        // --- Updated Filtering Logic ---
        // Check if the classId is within the bounds of the ignore array
        // and if the flag for this classId is set (1 means ignore)
        if (det.classId >= 0 && det.classId < max_check_id) {
            if (d_ignored_class_ids[det.classId]) { // 1 (true) means ignore this class
                should_keep = false;
            }
        }
        // Optional: else if classId is out of bounds, decide whether to keep or discard.
        // Current behavior: if classId is out of bounds of d_ignored_class_ids, it will be kept.

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
    const unsigned char* d_ignored_class_ids, // Changed from individual bools
    int max_check_id,                         // New parameter
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
        d_ignored_class_ids, // Pass the array
        max_check_id,        // Pass its size
        max_output_detections);

    return cudaGetLastError();
} 