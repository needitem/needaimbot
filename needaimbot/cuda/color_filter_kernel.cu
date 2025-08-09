#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../postprocess/postProcess.h"

// RGB 범위 필터링 커널
__global__ void rgbRangeFilter_kernel(
    const uint8_t* __restrict__ image,
    uint8_t* __restrict__ mask,
    int width, int height, int pitch,
    uint8_t minR, uint8_t maxR,
    uint8_t minG, uint8_t maxG,
    uint8_t minB, uint8_t maxB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = y * pitch + x * 3;
    const uint8_t b = image[idx + 0];
    const uint8_t g = image[idx + 1];
    const uint8_t r = image[idx + 2];
    
    // RGB 범위 체크 (매우 빠름)
    const bool inRange = (r >= minR && r <= maxR &&
                         g >= minG && g <= maxG &&
                         b >= minB && b <= maxB);
    
    mask[y * width + x] = inRange ? 255 : 0;
}


// Target 필터링을 위한 효율적인 커널
__global__ void filterTargetsByColorMask_kernel(
    const Target* __restrict__ detections,
    const uint8_t* __restrict__ colorMask,
    bool* __restrict__ validFlags,
    int numDetections,
    int maskWidth, int maskHeight,
    int minPixels,
    bool removeOnMatch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;
    
    const Target& det = detections[idx];
    
    // Bounding box 내의 마스크 픽셀 카운트
    int pixelCount = 0;
    const int x1 = max(0, det.x);
    const int y1 = max(0, det.y);
    const int x2 = min(maskWidth - 1, det.x + det.width);
    const int y2 = min(maskHeight - 1, det.y + det.height);
    
    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            if (colorMask[y * maskWidth + x] > 0) {
                pixelCount++;
            }
        }
    }
    
    const bool hasEnoughPixels = (pixelCount >= minPixels);
    validFlags[idx] = removeOnMatch ? !hasEnoughPixels : hasEnoughPixels;
}

// C++ 인터페이스
extern "C" {
    void launchRGBRangeFilter(
        const uint8_t* image, uint8_t* mask,
        int width, int height, int pitch,
        uint8_t minR, uint8_t maxR,
        uint8_t minG, uint8_t maxG,
        uint8_t minB, uint8_t maxB,
        cudaStream_t stream)
    {
        dim3 blockSize(32, 8);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        rgbRangeFilter_kernel<<<gridSize, blockSize, 0, stream>>>(
            image, mask, width, height, pitch, minR, maxR, minG, maxG, minB, maxB
        );
    }
}