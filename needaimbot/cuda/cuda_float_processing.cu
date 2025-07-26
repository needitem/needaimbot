#include "cuda_float_processing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace CudaFloatProcessing {

// Kernel to convert uint8 to float with scaling
__global__ void convertToFloatKernel(const uint8_t* src, float* dst, 
                                    int width, int height, int channels,
                                    size_t srcStep, size_t dstStepFloats,
                                    float scale, float shift) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; ++c) {
        uint8_t value = src[y * srcStep + x * channels + c];
        dst[y * dstStepFloats + x * channels + c] = value * scale + shift;
    }
}

void convertToFloat(const SimpleCudaMat& src, SimpleCudaMatFloat& dst, float scale, float shift, cudaStream_t stream) {
    if (src.empty()) return;
    
    dst.create(src.rows(), src.cols(), src.channels());
    
    dim3 blockSize(16, 16);
    dim3 gridSize((src.cols() + blockSize.x - 1) / blockSize.x,
                  (src.rows() + blockSize.y - 1) / blockSize.y);
    
    convertToFloatKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(),
        src.cols(), src.rows(), src.channels(),
        src.step(), dst.step(),
        scale, shift
    );
}

// Kernel to split float channels
__global__ void splitFloatKernel(const float* src, float* dst0, float* dst1, float* dst2,
                                int width, int height, int channels,
                                size_t srcStepFloats, size_t dstStepFloats) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const float* srcPixel = src + y * srcStepFloats + x * channels;
    size_t dstOffset = y * dstStepFloats + x;
    
    dst0[dstOffset] = srcPixel[0];
    if (channels > 1 && dst1) dst1[dstOffset] = srcPixel[1];
    if (channels > 2 && dst2) dst2[dstOffset] = srcPixel[2];
}

void splitFloat(const SimpleCudaMatFloat& src, SimpleCudaMatFloat* channels, cudaStream_t stream) {
    if (src.empty() || !channels) return;
    
    int numChannels = src.channels();
    for (int i = 0; i < numChannels; ++i) {
        channels[i].create(src.rows(), src.cols(), 1);
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((src.cols() + blockSize.x - 1) / blockSize.x,
                  (src.rows() + blockSize.y - 1) / blockSize.y);
    
    float* dst0 = channels[0].data();
    float* dst1 = numChannels > 1 ? channels[1].data() : nullptr;
    float* dst2 = numChannels > 2 ? channels[2].data() : nullptr;
    
    splitFloatKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst0, dst1, dst2,
        src.cols(), src.rows(), numChannels,
        src.step(), channels[0].step()
    );
}

} // namespace CudaFloatProcessing