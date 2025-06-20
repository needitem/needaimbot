#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <device_launch_parameters.h>
#include <device_atomic_functions.h> 
#include <vector>
#include <algorithm> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>       
#include <thrust/iterator/counting_iterator.h> 
#include <thrust/gather.h>     

#include "postProcess.h"
#include <NvInferRuntimeCommon.h> 


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
        
        
        float width = (float)max(0, x2 - x1);
        float height = (float)max(0, y2 - y1);
        
        if (width > 0 && height > 0) {
            float intersection_area = width * height;
            float union_area = d_areas[idx] + d_areas[idy] - intersection_area;
            float iou = intersection_area / union_area;
            
            
            d_iou_matrix[idx * num_boxes + idy] = iou;
            d_iou_matrix[idy * num_boxes + idx] = iou; 
        } else {
            d_iou_matrix[idx * num_boxes + idy] = 0.0f;
            d_iou_matrix[idy * num_boxes + idx] = 0.0f;
        }
    }
}


__global__ void nmsKernel(
    bool* d_keep, const float* d_iou_matrix,
    const float* d_scores, const int* d_classIds,
    int num_boxes, float nms_threshold) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        if (!d_keep[idx]) return; 
        
        for (int i = 0; i < num_boxes; i++) {
            if (idx == i) continue; 
            
            
            if (d_classIds[idx] == d_classIds[i]) {
                
                if (d_scores[idx] > d_scores[i] && d_iou_matrix[idx * num_boxes + i] > nms_threshold) {
                    d_keep[i] = false; 
                }
            }
        }
    }
}


struct is_kept {
    const bool* d_keep_ptr;
    is_kept(const bool* ptr) : d_keep_ptr(ptr) {}
    __host__ __device__
    bool operator()(const int& i) const {
        return d_keep_ptr[i];
    }
};


__global__ void extractDataKernel(
    const Detection* d_input_detections, int n, 
    int* d_x1, int* d_y1, int* d_x2, int* d_y2, 
    float* d_areas, float* d_scores, int* d_classIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const Detection& det = d_input_detections[idx];
        d_x1[idx] = det.box.x;
        d_y1[idx] = det.box.y;
        d_x2[idx] = det.box.x + det.box.width;
        d_y2[idx] = det.box.y + det.box.height;
        
        float width = max(0.0f, (float)det.box.width); 
        float height = max(0.0f, (float)det.box.height);
        d_areas[idx] = width * height; 
        d_scores[idx] = det.confidence;
        d_classIds[idx] = det.classId; 
    }
}


void NMSGpu(
    const Detection* d_input_detections,
    int input_num_detections,
    Detection* d_output_detections,
    int* d_output_count_gpu,
    int max_output_detections,
    float nmsThreshold,
    
    int* d_x1,
    int* d_y1,
    int* d_x2,
    int* d_y2,
    float* d_areas,
    float* d_scores_nms,      
    int* d_classIds_nms,      
    float* d_iou_matrix,
    bool* d_keep,
    int* d_indices,
    cudaStream_t stream)
{
    
    cudaError_t err = cudaSuccess;
    const int block_size = 256; 
    int final_count = 0; 

    if (input_num_detections <= 0) {
        cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        return; 
    }

    
    

    
    {
        const int grid_extract = (input_num_detections + block_size - 1) / block_size;
        extractDataKernel<<<grid_extract, block_size, 0, stream>>>( 
            d_input_detections, input_num_detections,
            d_x1, d_y1, d_x2, d_y2,
            d_areas, d_scores_nms, d_classIds_nms 
        );
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }


    
    
    
    cudaMemsetAsync(d_keep, 1, input_num_detections * sizeof(bool), stream); 
    cudaMemsetAsync(d_iou_matrix, 0, (size_t)input_num_detections * input_num_detections * sizeof(float), stream);
    err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;

    
    {
        dim3 block_iou(16, 16); 
        dim3 grid_iou((input_num_detections + block_iou.x - 1) / block_iou.x,
                       (input_num_detections + block_iou.y - 1) / block_iou.y);
        calculateIoUKernel<<<grid_iou, block_iou, 0, stream>>>( 
            d_x1, d_y1, d_x2, d_y2, d_areas, d_iou_matrix, input_num_detections, nmsThreshold
        );
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }

    
    {
        const int grid_nms = (input_num_detections + block_size - 1) / block_size;
        nmsKernel<<<grid_nms, block_size, 0, stream>>>( 
            d_keep, d_iou_matrix, d_scores_nms, d_classIds_nms, input_num_detections, nmsThreshold 
        );
         err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
    }


    
    try {
        thrust::device_ptr<const Detection> d_input_ptr(d_input_detections);
        thrust::device_ptr<Detection> d_output_ptr(d_output_detections);
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + input_num_detections;
        thrust::device_ptr<int> d_indices_ptr(d_indices);

        
        auto end_indices_iter = thrust::copy_if(
            thrust::cuda::par.on(stream),
            first, last,
            d_indices_ptr,
            is_kept(d_keep)
        );
        final_count = end_indices_iter - d_indices_ptr; 

        final_count = std::min(final_count, max_output_detections); 

        
        thrust::gather(
            thrust::cuda::par.on(stream),
            d_indices_ptr, d_indices_ptr + final_count,
            d_input_ptr,
            d_output_ptr
        );

        cudaMemcpyAsync(d_output_count_gpu, &final_count, sizeof(int), cudaMemcpyHostToDevice, stream);
         err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;

    } catch (const std::exception& e) {
        fprintf(stderr, "[Thrust Error] NMSGpu copy_if/gather: %s\n", e.what());
        err = cudaErrorUnknown; 
        goto cleanup; 
    }


cleanup: 
    
    

    cudaError_t lastErr = cudaGetLastError(); 
    if (err != cudaSuccess || lastErr != cudaSuccess) {
        cudaError_t errorToReport = (err != cudaSuccess) ? err : lastErr;
        fprintf(stderr, "[NMSGpu] CUDA error occurred: %s (%d)\n", cudaGetErrorString(errorToReport), errorToReport);
        if (err != cudaSuccess) { 
             cudaMemsetAsync(d_output_count_gpu, 0, sizeof(int), stream);
        }
    }
}




__device__ inline float readOutputValue(const void* buffer, nvinfer1::DataType type, size_t index) {
    if (type == nvinfer1::DataType::kFLOAT) {
        return reinterpret_cast<const float*>(buffer)[index];
    } else if (type == nvinfer1::DataType::kHALF) {
        return __half2float(reinterpret_cast<const __half*>(buffer)[index]);
    }
    
    return 0.0f; 
}


__global__ void decodeYolo10GpuKernel(
    const void* d_raw_output,          
    nvinfer1::DataType output_type,    
    int64_t num_detections_raw,        
    int64_t stride,                    
    int num_classes,                   
    float conf_threshold,              
    float img_scale,                   
    Detection* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_detections_raw) {
        size_t base_idx = idx * stride;

        float confidence = readOutputValue(d_raw_output, output_type, base_idx + 4);

        if (confidence > conf_threshold) {
            int classId = static_cast<int>(readOutputValue(d_raw_output, output_type, base_idx + 5));

            
            float cx = readOutputValue(d_raw_output, output_type, base_idx + 0);
            float cy = readOutputValue(d_raw_output, output_type, base_idx + 1);
            float x2 = readOutputValue(d_raw_output, output_type, base_idx + 2); 
            float y2 = readOutputValue(d_raw_output, output_type, base_idx + 3);

            
            int x = static_cast<int>(cx * img_scale);
            int y = static_cast<int>(cy * img_scale);
            int width = static_cast<int>((x2 - cx) * img_scale);
            int height = static_cast<int>((y2 - cy) * img_scale);

            
            if (width > 0 && height > 0) {
                
                int write_idx = ::atomicAdd(d_decoded_count, 1);

                
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.box.x = x;
                    det.box.y = y;
                    det.box.width = width;
                    det.box.height = height;
                    det.confidence = confidence;
                    det.classId = classId;
                } else {
                    
                    
                }
            }
        }
    }
}



__global__ void decodeYolo11GpuKernel(
    const void* d_raw_output,          
    nvinfer1::DataType output_type,    
    int64_t num_boxes_raw,             
    int64_t num_rows,                  
    int num_classes,                   
    float conf_threshold,              
    float img_scale,                   
    Detection* d_decoded_detections,   
    int* d_decoded_count,              
    int max_detections)                
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < num_boxes_raw) {
        size_t box_base_idx = idx; 

        
        float max_score = -1.0f;
        int max_class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            
            size_t score_idx = (4 + c) * num_boxes_raw + idx;
            float score = readOutputValue(d_raw_output, output_type, score_idx);
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        
        if (max_score > conf_threshold) {
            
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


                 
                int write_idx = ::atomicAdd(d_decoded_count, 1);

                
                if (write_idx < max_detections) {
                    Detection& det = d_decoded_detections[write_idx];
                    det.box.x = x;
                    det.box.y = y;
                    det.box.width = width;
                    det.box.height = height;
                    det.confidence = max_score;
                    det.classId = max_class_id;
                } else {
                    
                    
                }
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
    Detection* d_decoded_detections,
    int* d_decoded_count, 
    int max_detections,
    cudaStream_t stream)
{
    if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo10Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    int64_t num_detections_raw = shape[1];
    int64_t stride = shape[2]; 

    if (num_detections_raw <= 0 || stride == 0) {
        
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess; 
    }

    
    

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
    int* d_decoded_count, 
    int max_detections,
    cudaStream_t stream)
{
     if (shape.size() != 3) {
        fprintf(stderr, "[decodeYolo11Gpu] Error: Unexpected output shape size %zd\n", shape.size());
        return cudaErrorInvalidValue;
    }

    int64_t num_rows = shape[1]; 
    int64_t num_boxes_raw = shape[2];

     if (num_boxes_raw <= 0 || num_rows == 0) {
         
         cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
         return cudaSuccess; 
     }

    

    const int block_size = 256;
    const int grid_size = (num_boxes_raw + block_size - 1) / block_size;

    decodeYolo11GpuKernel<<<grid_size, block_size, 0, stream>>>( 
        d_raw_output, output_type, num_boxes_raw, num_rows, num_classes,
        conf_threshold, img_scale, d_decoded_detections, d_decoded_count, max_detections);

    return cudaGetLastError();
}

 
