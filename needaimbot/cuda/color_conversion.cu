#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_error_check.h"

__global__ void bgra2bgr_kernel(const uint8_t* __restrict__ src, 
                                uint8_t* __restrict__ dst, 
                                int width, int height, 
                                int src_pitch, int dst_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // 소스 및 대상 픽셀 위치 계산
    const uint8_t* src_pixel = src + y * src_pitch + x * 4;  // BGRA
    uint8_t* dst_pixel = dst + y * dst_pitch + x * 3;       // BGR
    
    // BGRA -> BGR 변환 (알파 채널 제거)
    dst_pixel[0] = src_pixel[0];  // B
    dst_pixel[1] = src_pixel[1];  // G  
    dst_pixel[2] = src_pixel[2];  // R
}

__global__ void bgr2bgra_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int width, int height,
                                int src_pitch, int dst_pitch,
                                uint8_t alpha = 255) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint8_t* src_pixel = src + y * src_pitch + x * 3;  // BGR
    uint8_t* dst_pixel = dst + y * dst_pitch + x * 4;        // BGRA
    
    // BGR -> BGRA 변환
    dst_pixel[0] = src_pixel[0];  // B
    dst_pixel[1] = src_pixel[1];  // G
    dst_pixel[2] = src_pixel[2];  // R
    dst_pixel[3] = alpha;         // A
}

// 최적화된 변환 함수 (2D 블록 사용)
extern "C" cudaError_t cuda_bgra2bgr(const uint8_t* src, uint8_t* dst, 
                                     int width, int height,
                                     int src_pitch, int dst_pitch,
                                     cudaStream_t stream) {
    dim3 block(32, 8);  // 256 스레드 per block
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    bgra2bgr_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, src_pitch, dst_pitch);
    
    return cudaGetLastError();
}

extern "C" cudaError_t cuda_bgr2bgra(const uint8_t* src, uint8_t* dst,
                                     int width, int height,
                                     int src_pitch, int dst_pitch,
                                     uint8_t alpha, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    bgr2bgra_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, src_pitch, dst_pitch, alpha);
    
    return cudaGetLastError();
}

__global__ void bgr2rgba_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int width, int height,
                                int src_pitch, int dst_pitch,
                                uint8_t alpha = 255) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint8_t* src_pixel = src + y * src_pitch + x * 3;  // BGR
    uint8_t* dst_pixel = dst + y * dst_pitch + x * 4;        // RGBA
    
    // BGR -> RGBA 변환 (색상 순서 변경)
    dst_pixel[0] = src_pixel[2];  // R (from B)
    dst_pixel[1] = src_pixel[1];  // G
    dst_pixel[2] = src_pixel[0];  // B (from R)
    dst_pixel[3] = alpha;         // A
}

extern "C" cudaError_t cuda_bgr2rgba(const uint8_t* src, uint8_t* dst,
                                     int width, int height,
                                     int src_pitch, int dst_pitch,
                                     uint8_t alpha, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    bgr2rgba_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, src_pitch, dst_pitch, alpha);
    
    return cudaGetLastError();
}