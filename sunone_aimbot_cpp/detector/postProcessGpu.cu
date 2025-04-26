#include <cuda_runtime.h>
#include <cuda_fp16.h> // For __half type and conversions
#include <device_launch_parameters.h>
#include <device_atomic_functions.h> // Include for atomicAdd
#include <vector>
#include <algorithm> // Include for std::min
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>       // For thrust::copy_if
#include <thrust/iterator/counting_iterator.h> // For thrust::counting_iterator
#include <thrust/gather.h>     // Added for thrust::gather

#include "postProcess.h"
#include <NvInferRuntimeCommon.h> // Include for nvinfer1::DataType

// CUDA kernel to calculate IoU between boxes
__global__ void calculateIoUKernel(
    const int* d_x1, const int* d_y1, const int* d_x2, const int* d_y2,
    const float* d_areas, float* d_iou_matrix,
    int num_boxes, float nms_threshold) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < num_boxes && idy < num_boxes && idx < idy) {
        int x1 = max(d_x1[idx], d_x1[idy]);
        int y1 = max(d_y1[idx], d_y1[idy]);
        int x2 = min(d_x2[idx], d_x2[idy]);
        int y2 = min(d_y2[idx], d_y2[idy]);
        
        // Check if boxes overlap
        float width = (float)max(0, x2 - x1);
        float height = (float)max(0, y2 - y1);
        
        if (width > 0 && height > 0) {
            float intersection_area = width * height;
            float union_area = d_areas[idx] + d_areas[idy] - intersection_area;
            float iou = intersection_area / union_area;
            
            // Store IoU value
            d_iou_matrix[idx * num_boxes + idy] = iou;
            d_iou_matrix[idy * num_boxes + idx] = iou; // Symmetric matrix
        } else {
            d_iou_matrix[idx * num_boxes + idy] = 0.0f;
            d_iou_matrix[idy * num_boxes + idx] = 0.0f;
        }
    }
}

// CUDA kernel to perform NMS suppression
__global__ void nmsKernel(
    bool* d_keep, const float* d_iou_matrix,
    const float* d_scores, const int* d_classIds,
    int num_boxes, float nms_threshold) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        if (!d_keep[idx]) return; // Already suppressed
        
        for (int i = 0; i < num_boxes; i++) {
            if (idx == i) continue; // Skip self
            
            // Check class ID match before suppressing
            if (d_classIds[idx] == d_classIds[i]) {
                // Suppress box i if it overlaps significantly with box idx AND box idx has a higher score
                if (d_scores[idx] > d_scores[i] && d_iou_matrix[idx * num_boxes + i] > nms_threshold) {
                    d_keep[i] = false; // Suppress box i only if classes match and IoU/score condition met
                }
            }
        }
    }
}

// Functor for Thrust copy_if
struct is_kept {
    const bool* d_keep_ptr;
    is_kept(const bool* ptr) : d_keep_ptr(ptr) {}
    __host__ __device__
    bool operator()(const int& i) const {
        return d_keep_ptr[i];
    }
};

// Kernel to extract data from Detection struct into separate arrays
__global__ void extractDataKernel(
    const Detection* d_input_detections, int n, 
    int* d_x1, int* d_y1, int* d_x2, int* d_y2, 
    float* d_areas, float* d_scores, int* d_classIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Access the box directly within the Detection struct
        // Note: Accessing cv::Rect members directly in a __global__ function 
        // might require ensuring cv::Rect is trivially copyable or using 
        // alternative data structures if it causes issues. 
        // Assuming cv::Rect is POD-like for GPU usage here.
        const Detection& det = d_input_detections[idx];
        d_x1[idx] = det.box.x;
        d_y1[idx] = det.box.y;
        d_x2[idx] = det.box.x + det.box.width;
        d_y2[idx] = det.box.y + det.box.height;
        // Calculate area safely, avoiding negative results from potential invalid boxes
        float width = max(0.0f, (float)det.box.width); 
        float height = max(0.0f, (float)det.box.height);
        d_areas[idx] = width * height; 
        d_scores[idx] = det.confidence;
        d_classIds[idx] = det.classId; // Extract classId
    }
}

// Modified NMSGpu function for direct GPU processing
void NMSGpu(
    const Detection* d_input_detections,
    int input_num_detections,
    Detection* d_output_detections,
    int* d_output_count_gpu,
    int max_output_detections,
    float nmsThreshold,
    cudaStream_t stream)
{
    // --- Declare all local variables at the beginning ---
    int* d_x1 = nullptr;
    int* d_y1 = nullptr;
    int* d_x2 = nullptr;
    int* d_y2 = nullptr;
    float* d_areas = nullptr;
    float* d_scores = nullptr;
    float* d_iou_matrix = nullptr;
    int* d_classIds = nullptr; // Add classIds pointer
    bool* d_keep = nullptr;
    int* d_indices = nullptr; // For copy_if
    cudaError_t err = cudaSuccess;
    const int block_size = 256; // Common block size
    int final_count = 0; // For Thrust result

    if (input_num_detections <= 0) {
        cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        return; // Exit early if no input
    }

    // Allocate memory
    size_t num_bytes_int = input_num_detections * sizeof(int);
    size_t num_bytes_float = input_num_detections * sizeof(float);
    size_t num_bytes_iou = (size_t)input_num_detections * input_num_detections * sizeof(float);
    size_t num_bytes_bool = input_num_detections * sizeof(bool);

    err = cudaMallocAsync(&d_x1, num_bytes_int, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_y1, num_bytes_int, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_x2, num_bytes_int, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_y2, num_bytes_int, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_areas, num_bytes_float, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_classIds, num_bytes_int, stream); if (err != cudaSuccess) goto cleanup; // Allocate memory for classIds
    err = cudaMallocAsync(&d_scores, num_bytes_float, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_iou_matrix, num_bytes_iou, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_keep, num_bytes_bool, stream); if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_indices, num_bytes_int, stream); if (err != cudaSuccess) goto cleanup;


    // --- Extract Data from Detection Struct ---
    {
        const int grid_extract = (input_num_detections + block_size - 1) / block_size;
        extractDataKernel<<<grid_extract, block_size, 0, stream>>>( 
            d_input_detections, input_num_detections,
            d_x1, d_y1, d_x2, d_y2,
            d_areas, d_scores, d_classIds // Pass classIds buffer
        );
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }


    // --- Initialize Keep Flags and IoU Matrix ---
    cudaMemsetAsync(d_keep, 1, num_bytes_bool, stream); // Set all to true (1)
    cudaMemsetAsync(d_iou_matrix, 0, num_bytes_iou, stream); // Zero out IoU matrix
    err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;

    // --- Calculate IoU Matrix ---
    {
        dim3 block_iou(16, 16); // 256 threads per block
        dim3 grid_iou((input_num_detections + block_iou.x - 1) / block_iou.x,
                       (input_num_detections + block_iou.y - 1) / block_iou.y);
        calculateIoUKernel<<<grid_iou, block_iou, 0, stream>>>( 
            d_x1, d_y1, d_x2, d_y2, d_areas, d_iou_matrix, input_num_detections, nmsThreshold
        );
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }

    // --- Perform NMS ---
    {
        const int grid_nms = (input_num_detections + block_size - 1) / block_size;
        nmsKernel<<<grid_nms, block_size, 0, stream>>>( 
            d_keep, d_iou_matrix, d_scores, d_classIds, input_num_detections, nmsThreshold // Pass classIds
        );
         err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }


    // --- Compact Results using Thrust ---
    try {
        thrust::device_ptr<const Detection> d_input_ptr(d_input_detections);
        thrust::device_ptr<Detection> d_output_ptr(d_output_detections);
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + input_num_detections;
        thrust::device_ptr<int> d_indices_ptr(d_indices);

        // Filter indices
        auto end_indices_iter = thrust::copy_if(
            thrust::cuda::par.on(stream),
            first, last,
            d_indices_ptr,
            is_kept(d_keep)
        );
        final_count = end_indices_iter - d_indices_ptr; // Assign to declared variable

        // Ensure final_count doesn't exceed output buffer size
        // Use std::min here - requires <algorithm> include if not already present
        // Alternatively, implement simple min: final_count = (final_count < max_output_detections) ? final_count : max_output_detections;
        final_count = std::min(final_count, max_output_detections); 

        // Gather data
        thrust::gather(
            thrust::cuda::par.on(stream),
            d_indices_ptr, d_indices_ptr + final_count,
            d_input_ptr,
            d_output_ptr
        );

        // Copy final count to output
        cudaMemcpyAsync(d_output_count_gpu, &final_count, sizeof(int), cudaMemcpyHostToDevice, stream);
         err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;

    } catch (const std::exception& e) {
        fprintf(stderr, "[Thrust Error] NMSGpu copy_if/gather: %s\n", e.what());
        err = cudaErrorUnknown; // Indicate error
        goto cleanup; // Go to cleanup
    }


cleanup: // Label for cleanup
    // Free temporary buffers
    cudaFreeAsync(d_x1, stream);
    cudaFreeAsync(d_y1, stream);
    cudaFreeAsync(d_x2, stream);
    cudaFreeAsync(d_y2, stream);
    cudaFreeAsync(d_areas, stream);
    cudaFreeAsync(d_scores, stream);
    cudaFreeAsync(d_iou_matrix, stream);
    cudaFreeAsync(d_classIds, stream); // Free classIds buffer
    cudaFreeAsync(d_keep, stream);
    cudaFreeAsync(d_indices, stream);

    // Check for errors during processing or cleanup
    cudaError_t lastErr = cudaGetLastError(); // Get error state after potential async frees
    if (err != cudaSuccess || lastErr != cudaSuccess) {
        cudaError_t errorToReport = (err != cudaSuccess) ? err : lastErr;
        fprintf(stderr, "[NMSGpu] CUDA error occurred: %s (%d)\n", cudaGetErrorString(errorToReport), errorToReport);
        // Set output count to 0 only if the error happened before successfully writing it
        if (err != cudaSuccess) { // Check if error happened before final memcpy
             cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        }
    }
}

// --- GPU Decoding Kernels ---

// Helper function to read value from raw output (handles float/half)
__device__ inline float readOutputValue(const void* buffer, nvinfer1::DataType type, size_t index) {
    if (type == nvinfer1::DataType::kFLOAT) {
        return reinterpret_cast<const float*>(buffer)[index];
    } else if (type == nvinfer1::DataType::kHALF) {
        return __half2float(reinterpret_cast<const __half*>(buffer)[index]);
    }
    // Add support for other types if necessary
    return 0.0f; // Should not happen with supported types
}

// Kernel to decode YOLOv10 output directly on GPU
__global__ void decodeYolo10GpuKernel(
    const void* d_raw_output,          // Raw output buffer (GPU, float* or half*)
    nvinfer1::DataType output_type,    // Data type of the raw output
    int64_t num_detections_raw,        // Number of raw detections (shape[1])
    int64_t stride,                    // Stride between detections (shape[2])
    int num_classes,                   // Number of classes
    float conf_threshold,              // Confidence threshold
    float img_scale,                   // Image scale factor
    Detection* d_decoded_detections,   // Output buffer for decoded detections (GPU)
    int* d_decoded_count,              // Output counter (GPU, atomic)
    int max_detections)                // Maximum number of detections allowed in output buffer
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections_raw) {
        size_t base_idx = idx * stride;

        float confidence = readOutputValue(d_raw_output, output_type, base_idx + 4);

        if (confidence > conf_threshold) {
            int classId = static_cast<int>(readOutputValue(d_raw_output, output_type, base_idx + 5));

            // Get box coordinates
            float cx = readOutputValue(d_raw_output, output_type, base_idx + 0);
            float cy = readOutputValue(d_raw_output, output_type, base_idx + 1);
            float x2 = readOutputValue(d_raw_output, output_type, base_idx + 2); // Assuming x2, y2 format
            float y2 = readOutputValue(d_raw_output, output_type, base_idx + 3);

            // Scale and calculate final box (adjust if format is cx,cy,w,h)
            int x = static_cast<int>(cx * img_scale);
            int y = static_cast<int>(cy * img_scale);
            int width = static_cast<int>((x2 - cx) * img_scale);
            int height = static_cast<int>((y2 - cy) * img_scale);

            // Basic sanity check
            if (width > 0 && height > 0) {
                // Use atomic operation to get the index to write to
                int write_idx = ::atomicAdd(d_decoded_count, 1);

                // Ensure we don't write past the buffer limit
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.box.x = x;
                    det.box.y = y;
                    det.box.width = width;
                    det.box.height = height;
                    det.confidence = confidence;
                    det.classId = classId;
                } else {
                    // Optional: Decrement count if we exceed max_detections to keep it accurate
                    // atomicSub(d_decoded_count, 1); // Be careful with races if multiple threads hit this
                }
            }
        }
    }
}


// Kernel to decode YOLOv11 (YOLOv8/9) output directly on GPU
__global__ void decodeYolo11GpuKernel(
    const void* d_raw_output,          // Raw output buffer (GPU, float* or half*)
    nvinfer1::DataType output_type,    // Data type of the raw output
    int64_t num_boxes_raw,             // Number of boxes (shape[2])
    int64_t num_rows,                  // Number of rows per box (shape[1])
    int num_classes,                   // Number of classes
    float conf_threshold,              // Confidence threshold
    float img_scale,                   // Image scale factor
    Detection* d_decoded_detections,   // Output buffer for decoded detections (GPU)
    int* d_decoded_count,              // Output counter (GPU, atomic)
    int max_detections)                // Maximum number of detections allowed in output buffer
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index of the box

    if (idx < num_boxes_raw) {
        size_t box_base_idx = idx; // Stride is num_boxes_raw for class scores

        // Find the class with the highest score
        float max_score = -1.0f;
        int max_class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            // Index calculation: row 4+c, column idx
            size_t score_idx = (4 + c) * num_boxes_raw + idx;
            float score = readOutputValue(d_raw_output, output_type, score_idx);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // Check against confidence threshold
        if (max_score > conf_threshold) {
            // Get box coordinates (cx, cy, w, h)
            float cx = readOutputValue(d_raw_output, output_type, 0 * num_boxes_raw + idx);
            float cy = readOutputValue(d_raw_output, output_type, 1 * num_boxes_raw + idx);
            float ow = readOutputValue(d_raw_output, output_type, 2 * num_boxes_raw + idx);
            float oh = readOutputValue(d_raw_output, output_type, 3 * num_boxes_raw + idx);

            // Basic sanity check
            if (ow > 0 && oh > 0) {
                // Calculate final box (x, y, w, h)
                const float half_ow = 0.5f * ow;
                const float half_oh = 0.5f * oh;
                int x = static_cast<int>((cx - half_ow) * img_scale);
                int y = static_cast<int>((cy - half_oh) * img_scale);
                int width = static_cast<int>(ow * img_scale);
                int height = static_cast<int>(oh * img_scale);


                 // Use atomic operation to get the index to write to
                int write_idx = ::atomicAdd(d_decoded_count, 1);

                // Ensure we don't write past the buffer limit
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.box.x = x;
                    det.box.y = y;
                    det.box.width = width;
                    det.box.height = height;
                    det.confidence = max_score;
                    det.classId = max_class_id;
                } else {
                    // Optional: Decrement count if we exceed max_detections
                    // atomicSub(d_decoded_count, 1);
                }
            }
        }
    }
}


// --- GPU Decoding Wrapper Functions ---

cudaError_t decodeYolo10Gpu(
    const void* d_raw_output,
    nvinfer1::DataType output_type,
    const std::vector<int64_t>& shape,
    int num_classes,
    float conf_threshold,
    float img_scale,
    Detection* d_decoded_detections,
    int* d_decoded_count, // Needs to be initialized to 0 before kernel launch
    int max_detections,
    cudaStream_t stream)
{
    if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo10Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    int64_t num_detections_raw = shape[1];
    int64_t stride = shape[2]; // columns per detection

    if (num_detections_raw <= 0 || stride == 0) {
        // Ensure counter is 0 if no work
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess; // No data to process
    }

    // Ensure counter is reset before launching kernel
    // cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream); // Do this in the caller (Detector class)

    const int block_size = 256;
    const int grid_size = (num_detections_raw + block_size - 1) / block_size;

    decodeYolo10GpuKernel<<<grid_size, block_size, 0, stream>>>( 
        d_raw_output, output_type, num_detections_raw, stride, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections);

    return cudaGetLastError();
}


cudaError_t decodeYolo11Gpu(
    const void* d_raw_output,
    nvinfer1::DataType output_type,
    const std::vector<int64_t>& shape,
    int num_classes,
    float conf_threshold,
    float img_scale,
    Detection* d_decoded_detections,
    int* d_decoded_count, // Needs to be initialized to 0 before kernel launch
    int max_detections,
    cudaStream_t stream)
{
     if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo11Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    int64_t num_rows = shape[1]; // Should be 4 + num_classes
    int64_t num_boxes_raw = shape[2];

     if (num_boxes_raw <= 0 || num_rows == 0) {
         // Ensure counter is 0 if no work
         cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
         return cudaSuccess; // No data to process
     }

    // cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream); // Do this in the caller (Detector class)

    const int block_size = 256;
    const int grid_size = (num_boxes_raw + block_size - 1) / block_size;

    decodeYolo11GpuKernel<<<grid_size, block_size, 0, stream>>>( 
        d_raw_output, output_type, num_boxes_raw, num_rows, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections);

    return cudaGetLastError();
}

/* Original NMSGpu implementation (operates on std::vector)
void NMSGpu(std::vector<Detection>& detections, float nmsThreshold, cudaStream_t stream) {
    if (detections.empty()) return;
    
    int num_boxes = detections.size();
    
    // Sort detections by confidence (highest first)
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    
    // Prepare host data
    thrust::host_vector<int> h_x1(num_boxes);
    thrust::host_vector<int> h_y1(num_boxes);
    thrust::host_vector<int> h_x2(num_boxes);
    thrust::host_vector<int> h_y2(num_boxes);
    thrust::host_vector<float> h_areas(num_boxes);
    thrust::host_vector<float> h_scores(num_boxes);
    
    for (int i = 0; i < num_boxes; i++) {
        const cv::Rect& box = detections[i].box;
        h_x1[i] = box.x;
        h_y1[i] = box.y;
        h_x2[i] = box.x + box.width;
        h_y2[i] = box.y + box.height;
        h_areas[i] = box.area();
        h_scores[i] = detections[i].confidence;
    }
    
    // Copy data to device
    thrust::device_vector<int> d_x1 = h_x1;
    thrust::device_vector<int> d_y1 = h_y1;
    thrust::device_vector<int> d_x2 = h_x2;
    thrust::device_vector<int> d_y2 = h_y2;
    thrust::device_vector<float> d_areas = h_areas;
    thrust::device_vector<float> d_scores = h_scores;
    thrust::device_vector<float> d_iou_matrix(num_boxes * num_boxes, 0.0f);
    thrust::device_vector<bool> d_keep(num_boxes, true);
    
    // Configure kernel execution
    dim3 block_size(16, 16);
    dim3 grid_size((num_boxes + block_size.x - 1) / block_size.x, 
                  (num_boxes + block_size.y - 1) / block_size.y);
    
    // Calculate IoU matrix
    calculateIoUKernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_x1.data()),
        thrust::raw_pointer_cast(d_y1.data()),
        thrust::raw_pointer_cast(d_x2.data()),
        thrust::raw_pointer_cast(d_y2.data()),
        thrust::raw_pointer_cast(d_areas.data()),
        thrust::raw_pointer_cast(d_iou_matrix.data()),
        num_boxes,
        nmsThreshold
    );
    
    // Perform NMS
    nmsKernel<<<(num_boxes + 255) / 256, 256, 0, stream>>>(
        thrust::raw_pointer_cast(d_keep.data()),
        thrust::raw_pointer_cast(d_iou_matrix.data()),
        thrust::raw_pointer_cast(d_scores.data()),
        num_boxes,
        nmsThreshold
    );
    
    // Copy results back to host
    thrust::host_vector<bool> h_keep = d_keep;
    
    // Update detections list
    std::vector<Detection> result;
    result.reserve(num_boxes);
    
    for (int i = 0; i < num_boxes; i++) {
        if (h_keep[i]) {
            result.push_back(detections[i]);
        }
    }
    
    detections = std::move(result);
} 
*/ 