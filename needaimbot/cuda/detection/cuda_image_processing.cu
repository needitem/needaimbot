#include "cuda_image_processing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace CudaImageProcessing {

// Resize kernel with bilinear interpolation
__global__ void resizeKernel(const uint8_t* src, uint8_t* dst, 
                            int srcWidth, int srcHeight, int srcStep,
                            int dstWidth, int dstHeight, int dstStep, 
                            int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;
    
    float srcX = x * scaleX;
    float srcY = y * scaleY;
    
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);
    
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    for (int c = 0; c < channels; ++c) {
        float p00 = src[y0 * srcStep + x0 * channels + c];
        float p01 = src[y0 * srcStep + x1 * channels + c];
        float p10 = src[y1 * srcStep + x0 * channels + c];
        float p11 = src[y1 * srcStep + x1 * channels + c];
        
        float value = p00 * (1 - fx) * (1 - fy) +
                     p01 * fx * (1 - fy) +
                     p10 * (1 - fx) * fy +
                     p11 * fx * fy;
        
        dst[y * dstStep + x * channels + c] = (uint8_t)fminf(fmaxf(value, 0.0f), 255.0f);
    }
}

void resize(const SimpleCudaMat& src, SimpleCudaMat& dst, int dstWidth, int dstHeight, cudaStream_t stream) {
    if (src.empty()) return;
    
    dst.create(dstHeight, dstWidth, src.channels());
    
    dim3 blockSize(16, 16);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, 
                  (dstHeight + blockSize.y - 1) / blockSize.y);
    
    resizeKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(),
        src.cols(), src.rows(), static_cast<int>(src.step()),
        dstWidth, dstHeight, static_cast<int>(dst.step()),
        src.channels()
    );
}

// BGRA to BGR conversion kernel
__global__ void bgra2bgrKernel(const uint8_t* bgra, uint8_t* bgr, int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* srcPixel = bgra + y * srcStep + x * 4;
    uint8_t* dstPixel = bgr + y * dstStep + x * 3;
    
    dstPixel[0] = srcPixel[0]; // B
    dstPixel[1] = srcPixel[1]; // G
    dstPixel[2] = srcPixel[2]; // R
}

// BGRA to RGB conversion kernel (swap R and B channels)
__global__ void bgra2rgbKernel(const uint8_t* bgra, uint8_t* rgb, int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* srcPixel = bgra + y * srcStep + x * 4;
    uint8_t* dstPixel = rgb + y * dstStep + x * 3;
    
    dstPixel[0] = srcPixel[2]; // R (from B position)
    dstPixel[1] = srcPixel[1]; // G 
    dstPixel[2] = srcPixel[0]; // B (from R position)
}

void bgra2bgr(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 4) return;
    
    dst.create(src.rows(), src.cols(), 3);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    bgra2bgrKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

void bgra2rgb(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 4) return;
    
    dst.create(src.rows(), src.cols(), 3);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    bgra2rgbKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

// BGR to BGRA conversion kernel
__global__ void bgr2bgraKernel(const uint8_t* bgr, uint8_t* bgra, int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* srcPixel = bgr + y * srcStep + x * 3;
    uint8_t* dstPixel = bgra + y * dstStep + x * 4;
    
    dstPixel[0] = srcPixel[0]; // B
    dstPixel[1] = srcPixel[1]; // G
    dstPixel[2] = srcPixel[2]; // R
    dstPixel[3] = 255;         // A
}

void bgr2bgra(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 3) return;
    
    dst.create(src.rows(), src.cols(), 4);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    bgr2bgraKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

// BGR to HSV conversion kernel
__device__ void bgr2hsvPixel(uint8_t b, uint8_t g, uint8_t r, uint8_t& h, uint8_t& s, uint8_t& v) {
    float bf = b / 255.0f;
    float gf = g / 255.0f;
    float rf = r / 255.0f;
    
    float maxVal = fmaxf(rf, fmaxf(gf, bf));
    float minVal = fminf(rf, fminf(gf, bf));
    float delta = maxVal - minVal;
    
    // Value
    v = (uint8_t)(maxVal * 255);
    
    // Saturation
    if (maxVal > 0) {
        s = (uint8_t)((delta / maxVal) * 255);
    } else {
        s = 0;
    }
    
    // Hue
    float hue = 0;
    if (delta > 0) {
        if (maxVal == rf) {
            hue = 60 * ((gf - bf) / delta);
        } else if (maxVal == gf) {
            hue = 60 * (2 + (bf - rf) / delta);
        } else {
            hue = 60 * (4 + (rf - gf) / delta);
        }
        
        if (hue < 0) hue += 360;
        h = (uint8_t)(hue / 2); // OpenCV uses 0-180 range for hue
    } else {
        h = 0;
    }
}

__global__ void bgr2hsvKernel(const uint8_t* bgr, uint8_t* hsv, int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* srcPixel = bgr + y * srcStep + x * 3;
    uint8_t* dstPixel = hsv + y * dstStep + x * 3;
    
    bgr2hsvPixel(srcPixel[0], srcPixel[1], srcPixel[2], 
                 dstPixel[0], dstPixel[1], dstPixel[2]);
}

void bgr2hsv(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 3) return;
    
    dst.create(src.rows(), src.cols(), 3);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    bgr2hsvKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

// BGR to Gray conversion kernel
__global__ void bgr2grayKernel(const uint8_t* bgr, uint8_t* gray, int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* srcPixel = bgr + y * srcStep + x * 3;
    uint8_t* dstPixel = gray + y * dstStep + x;
    
    // Standard grayscale conversion weights
    float grayValue = 0.114f * srcPixel[0] + 0.587f * srcPixel[1] + 0.299f * srcPixel[2];
    *dstPixel = (uint8_t)fminf(fmaxf(grayValue, 0.0f), 255.0f);
}

void bgr2gray(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 3) return;
    
    dst.create(src.rows(), src.cols(), 1);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    bgr2grayKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

// Gray to BGR conversion kernel
__global__ void gray2bgrKernel(const uint8_t* gray, uint8_t* bgr, int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    uint8_t grayValue = gray[y * srcStep + x];
    uint8_t* dstPixel = bgr + y * dstStep + x * 3;
    
    dstPixel[0] = grayValue; // B
    dstPixel[1] = grayValue; // G
    dstPixel[2] = grayValue; // R
}

void gray2bgr(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 1) return;
    
    dst.create(src.rows(), src.cols(), 3);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    gray2bgrKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

// Convert to float with scaling
__global__ void convertToFloatKernel(const uint8_t* src, float* dst, int pixels, float scale, float shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    dst[idx] = src[idx] * scale + shift;
}

void convertTo(const SimpleCudaMat& src, SimpleCudaMat& dst, float scale, float shift, cudaStream_t stream) {
    if (src.empty()) return;
    
    // Note: This is a simplified version that assumes dst is already allocated as float
    // For now, we'll need the caller to handle float buffer allocation
    // This is a limitation we'll need to address based on usage patterns
}

// Apply mask kernel
__global__ void applyMaskKernel(const uint8_t* src, uint8_t* dst, const uint8_t* mask, 
                               int pixels, int channels, int srcStep, int dstStep, int maskStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    uint8_t maskValue = mask[y * maskStep + x];
    
    for (int c = 0; c < channels; ++c) {
        int offset = y * srcStep + x * channels + c;
        dst[offset] = maskValue > 0 ? src[offset] : 0;
    }
}

void applyMask(const SimpleCudaMat& src, SimpleCudaMat& dst, const SimpleCudaMat& mask, cudaStream_t stream) {
    if (src.empty() || mask.empty()) return;
    if (mask.channels() != 1) return;
    if (src.rows() != mask.rows() || src.cols() != mask.cols()) return;
    
    dst.create(src.rows(), src.cols(), src.channels());
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    applyMaskKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(), mask.data(), 
        pixels, src.channels(), static_cast<int>(src.step()), static_cast<int>(dst.step()), static_cast<int>(mask.step()), src.cols()
    );
}

// HSV in-range check kernel
__global__ void inRangeHSVKernel(const uint8_t* hsv, uint8_t* mask, 
                                 float hLow, float sLow, float vLow,
                                 float hHigh, float sHigh, float vHigh,
                                 int pixels, int srcStep, int dstStep, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* pixel = hsv + y * srcStep + x * 3;
    uint8_t h = pixel[0];
    uint8_t s = pixel[1];
    uint8_t v = pixel[2];
    
    bool inRange = (h >= hLow && h <= hHigh) &&
                   (s >= sLow && s <= sHigh) &&
                   (v >= vLow && v <= vHigh);
    
    mask[y * dstStep + x] = inRange ? 255 : 0;
}

void inRange(const SimpleCudaMat& src, const float* lowerBound, const float* upperBound, SimpleCudaMat& dst, cudaStream_t stream) {
    if (src.empty() || src.channels() != 3) return;
    
    dst.create(src.rows(), src.cols(), 1);
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    inRangeHSVKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst.data(),
        lowerBound[0], lowerBound[1], lowerBound[2],
        upperBound[0], upperBound[1], upperBound[2],
        pixels, static_cast<int>(src.step()), static_cast<int>(dst.step()), src.cols()
    );
}

// Split channels kernel
__global__ void splitKernel(const uint8_t* src, uint8_t* dst0, uint8_t* dst1, uint8_t* dst2,
                           int pixels, int srcStep, int dstStep, int width, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    const uint8_t* srcPixel = src + y * srcStep + x * channels;
    int dstOffset = y * dstStep + x;
    
    dst0[dstOffset] = srcPixel[0];
    if (channels > 1 && dst1) dst1[dstOffset] = srcPixel[1];
    if (channels > 2 && dst2) dst2[dstOffset] = srcPixel[2];
}

void split(const SimpleCudaMat& src, SimpleCudaMat* channels, cudaStream_t stream) {
    if (src.empty() || !channels) return;
    
    int numChannels = src.channels();
    for (int i = 0; i < numChannels; ++i) {
        channels[i].create(src.rows(), src.cols(), 1);
    }
    
    int pixels = src.rows() * src.cols();
    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    uint8_t* dst0 = channels[0].data();
    uint8_t* dst1 = numChannels > 1 ? channels[1].data() : nullptr;
    uint8_t* dst2 = numChannels > 2 ? channels[2].data() : nullptr;
    
    splitKernel<<<gridSize, blockSize, 0, stream>>>(
        src.data(), dst0, dst1, dst2,
        pixels, static_cast<int>(src.step()), static_cast<int>(channels[0].step()), src.cols(), numChannels
    );
}

} // namespace CudaImageProcessing