#pragma once

#include <cuda_runtime.h>

extern "C" {
    // BGRA to BGR conversion (removes alpha channel)
    cudaError_t cuda_bgra2bgr(const uint8_t* src, uint8_t* dst, 
                              int width, int height,
                              int src_pitch, int dst_pitch,
                              cudaStream_t stream = 0);
    
    // BGR to BGRA conversion (adds alpha channel)
    cudaError_t cuda_bgr2bgra(const uint8_t* src, uint8_t* dst,
                              int width, int height,
                              int src_pitch, int dst_pitch,
                              uint8_t alpha = 255,
                              cudaStream_t stream = 0);
    
    // BGR to RGBA conversion (adds alpha channel and swaps R/B)
    cudaError_t cuda_bgr2rgba(const uint8_t* src, uint8_t* dst,
                              int width, int height,
                              int src_pitch, int dst_pitch,
                              uint8_t alpha = 255,
                              cudaStream_t stream = 0);
}