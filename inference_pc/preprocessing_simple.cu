#include <cuda_runtime.h>
#include <cstdint>

// Preprocess RGB to normalized float (NCHW format)
__global__ void preprocessKernel(const uint8_t* rgb, float* output,
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

    // RGB -> normalized float, NCHW format
    output[0 * planeSize + dstIdx] = rgb[srcIdx + 0] / 255.0f;  // R
    output[1 * planeSize + dstIdx] = rgb[srcIdx + 1] / 255.0f;  // G
    output[2 * planeSize + dstIdx] = rgb[srcIdx + 2] / 255.0f;  // B
}

// Host wrapper function
extern "C" void launchPreprocessKernel(const uint8_t* rgb, float* output,
                                       int srcWidth, int srcHeight,
                                       int dstWidth, int dstHeight,
                                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dstWidth + 15) / 16, (dstHeight + 15) / 16);
    preprocessKernel<<<grid, block, 0, stream>>>(rgb, output, srcWidth, srcHeight, dstWidth, dstHeight);
}
