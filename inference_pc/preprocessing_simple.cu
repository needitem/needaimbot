#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Preprocess RGB to normalized FP16 (NCHW format)
__global__ void preprocessKernelFP16(const uint8_t* rgb, __half* output,
                                      int srcWidth, int srcHeight,
                                      int dstWidth, int dstHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight) return;

    // Simple nearest-neighbor resize
    int srcX = x * srcWidth / dstWidth;
    int srcY = y * srcHeight / dstHeight;

    int srcIdx = (srcY * srcWidth + srcX) * 3;
    int dstIdx = y * dstWidth + x;
    int planeSize = dstWidth * dstHeight;

    // RGB normalized FP16, NCHW format
    output[0 * planeSize + dstIdx] = __float2half(rgb[srcIdx + 0] / 255.0f);  // R
    output[1 * planeSize + dstIdx] = __float2half(rgb[srcIdx + 1] / 255.0f);  // G
    output[2 * planeSize + dstIdx] = __float2half(rgb[srcIdx + 2] / 255.0f);  // B
}

// Host wrapper function for FP16
extern "C" void launchPreprocessKernel(const uint8_t* rgb, void* output,
                                       int srcWidth, int srcHeight,
                                       int dstWidth, int dstHeight,
                                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dstWidth + 15) / 16, (dstHeight + 15) / 16);
    preprocessKernelFP16<<<grid, block, 0, stream>>>(rgb, (__half*)output, srcWidth, srcHeight, dstWidth, dstHeight);
}
