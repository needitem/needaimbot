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
    bool ignore_class_0,
    bool ignore_class_1,
    bool ignore_class_2,
    bool ignore_class_3,
    bool ignore_class_4,
    bool ignore_class_5,
    bool ignore_class_6,
    bool ignore_class_7,
    bool ignore_class_8,
    bool ignore_class_9,
    bool ignore_class_10,
    int max_output_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_input_detections) {
        const Detection& det = input_detections[idx];
        bool should_keep = true; // Default to keeping the detection

        // --- Simplified Filtering Logic ---
        // Check individual class ignore flags
        if (ignore_class_0 && det.classId == 0) { should_keep = false; }
        else if (ignore_class_1 && det.classId == 1) { should_keep = false; }
        else if (ignore_class_2 && det.classId == 2) { should_keep = false; }
        else if (ignore_class_3 && det.classId == 3) { should_keep = false; }
        else if (ignore_class_4 && det.classId == 4) { should_keep = false; }
        else if (ignore_class_5 && det.classId == 5) { should_keep = false; }
        else if (ignore_class_6 && det.classId == 6) { should_keep = false; }
        else if (ignore_class_7 && det.classId == 7) { should_keep = false; } // Head filtering now solely based on this
        else if (ignore_class_8 && det.classId == 8) { should_keep = false; }
        else if (ignore_class_9 && det.classId == 9) { should_keep = false; }
        else if (ignore_class_10 && det.classId == 10) { should_keep = false; }

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
    bool ignore_class_0,
    bool ignore_class_1,
    bool ignore_class_2,
    bool ignore_class_3,
    bool ignore_class_4,
    bool ignore_class_5,
    bool ignore_class_6,
    bool ignore_class_7,
    bool ignore_class_8,
    bool ignore_class_9,
    bool ignore_class_10,
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
        ignore_class_0,
        ignore_class_1,
        ignore_class_2,
        ignore_class_3,
        ignore_class_4,
        ignore_class_5,
        ignore_class_6,
        ignore_class_7,
        ignore_class_8,
        ignore_class_9,
        ignore_class_10,
        max_output_detections);

    return cudaGetLastError();
} 