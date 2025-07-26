#pragma once

#include <cuda_runtime.h>
#include "simple_cuda_mat.h"

namespace CudaImageProcessing {

// Image resizing with bilinear interpolation
void resize(const SimpleCudaMat& src, SimpleCudaMat& dst, int dstWidth, int dstHeight, cudaStream_t stream = 0);

// Color conversion functions
void bgra2bgr(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream = 0);
void bgr2bgra(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream = 0);
void bgr2hsv(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream = 0);
void bgr2gray(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream = 0);
void gray2bgr(const SimpleCudaMat& src, SimpleCudaMat& dst, cudaStream_t stream = 0);

// Format conversion
void convertTo(const SimpleCudaMat& src, SimpleCudaMat& dst, float scale, float shift, cudaStream_t stream = 0);

// Apply mask
void applyMask(const SimpleCudaMat& src, SimpleCudaMat& dst, const SimpleCudaMat& mask, cudaStream_t stream = 0);

// In-range check for HSV
void inRange(const SimpleCudaMat& src, const float* lowerBound, const float* upperBound, SimpleCudaMat& dst, cudaStream_t stream = 0);

// Split channels
void split(const SimpleCudaMat& src, SimpleCudaMat* channels, cudaStream_t stream = 0);

// Image copy with type conversion
void copyMakeBorder(const SimpleCudaMat& src, SimpleCudaMat& dst, int top, int bottom, int left, int right, cudaStream_t stream = 0);

} // namespace CudaImageProcessing