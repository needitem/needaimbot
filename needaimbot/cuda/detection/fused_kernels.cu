#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "postProcess.h"

namespace cg = cooperative_groups;

// ============================================================================
// ULTRA-OPTIMIZED FUSED KERNELS FOR MAXIMUM PERFORMANCE
// ============================================================================

// Forward declare IoU calculation
__device__ inline float calculateIoU(const Target& a, const Target& b);

// Texture object is now passed as parameter (CUDA 11+ style)
// texture<float4, cudaTextureType2D, cudaReadModeNormalizedFloat> texCapture; // Deprecated

// Helper function for IoU calculation
__device__ inline float calculateIoU(const Target& a, const Target& b) {
    float x1 = fmaxf(a.x, b.x);
    float y1 = fmaxf(a.y, b.y);
    float x2 = fminf(a.x + a.width, b.x + b.width);
    float y2 = fminf(a.y + a.height, b.y + b.height);
    
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float union_area = areaA + areaB - intersection;
    
    return intersection / fmaxf(union_area, 1e-6f);
}