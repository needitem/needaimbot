#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "../postprocess/postProcess.h"

namespace cg = cooperative_groups;

// ============================================================================
// ULTRA-OPTIMIZED FUSED KERNELS FOR MAXIMUM PERFORMANCE
// ============================================================================

// Texture reference for hardware bilinear interpolation
texture<float4, cudaTextureType2D, cudaReadModeNormalizedFloat> texCapture;

// 1. Fused Capture + Resize + Normalize + Format Conversion
// Uses texture memory, shared memory, vectorized loads, and warp shuffles
__global__ void __launch_bounds__(256, 4) 
fusedCapturePreprocessOptimized(
    cudaTextureObject_t texInput,
    float* __restrict__ output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    float scaleX, float scaleY,
    float3 mean, float3 invStd,
    bool swapRB)
{
    // Use shared memory for intermediate results
    __shared__ float3 tileData[16][17]; // Avoid bank conflicts
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // Hardware-accelerated bilinear sampling
    float u = (x + 0.5f) * scaleX / srcWidth;
    float v = (y + 0.5f) * scaleY / srcHeight;
    
    // Single texture fetch with hardware interpolation
    float4 pixel = tex2D<float4>(texInput, u, v);
    
    // Vectorized normalization using FMA
    float3 normalized;
    normalized.x = fmaf(pixel.x, invStd.x, -mean.x * invStd.x);
    normalized.y = fmaf(pixel.y, invStd.y, -mean.y * invStd.y);
    normalized.z = fmaf(pixel.z, invStd.z, -mean.z * invStd.z);
    
    // Handle channel swapping
    if (swapRB) {
        float temp = normalized.x;
        normalized.x = normalized.z;
        normalized.z = temp;
    }
    
    // Store to shared memory for coalesced global writes
    tileData[ty][tx] = normalized;
    __syncthreads();
    
    // Coalesced write to global memory in CHW format
    const int pixelIdx = y * dstWidth + x;
    const int planeSize = dstHeight * dstWidth;
    
    output[0 * planeSize + pixelIdx] = tileData[ty][tx].x; // R
    output[1 * planeSize + pixelIdx] = tileData[ty][tx].y; // G
    output[2 * planeSize + pixelIdx] = tileData[ty][tx].z; // B
}

// 2. Fused Decode + NMS + Filtering in single kernel
// Eliminates multiple kernel launches and intermediate buffers
__global__ void __launch_bounds__(256, 4)
fusedDecodeNMSFilter(
    const float* __restrict__ modelOutput,
    Target* __restrict__ finalDetections,
    int* __restrict__ numDetections,
    int numBoxes, int numClasses,
    float confThreshold, float nmsThreshold,
    const bool* __restrict__ classFilter,
    int maxDetections)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Shared memory for intermediate detections
    extern __shared__ char sharedMem[];
    Target* sharedDetections = (Target*)sharedMem;
    float* sharedScores = (float*)&sharedDetections[256];
    
    const int tid = threadIdx.x;
    const int boxId = blockIdx.x * blockDim.x + tid;
    
    if (boxId >= numBoxes) return;
    
    // Decode bounding box
    const float* boxData = &modelOutput[boxId * (4 + numClasses)];
    float4 bbox = *((float4*)boxData);
    
    // Find best class using warp-level reduction
    float maxConf = 0.0f;
    int bestClass = -1;
    
    for (int c = 0; c < numClasses; c += 32) {
        int classId = c + warp.thread_rank();
        if (classId < numClasses) {
            float conf = boxData[4 + classId];
            if (conf > maxConf && (!classFilter || classFilter[classId])) {
                maxConf = conf;
                bestClass = classId;
            }
        }
    }
    
    // Warp-level reduction to find best class across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        float otherConf = warp.shfl_down(maxConf, offset);
        int otherClass = warp.shfl_down(bestClass, offset);
        if (otherConf > maxConf) {
            maxConf = otherConf;
            bestClass = otherClass;
        }
    }
    
    // First thread in warp writes result
    if (warp.thread_rank() == 0 && maxConf > confThreshold) {
        int idx = atomicAdd(numDetections, 1);
        if (idx < maxDetections) {
            Target det;
            det.x = bbox.x;
            det.y = bbox.y;
            det.width = bbox.z - bbox.x;
            det.height = bbox.w - bbox.y;
            det.confidence = maxConf;
            det.classId = bestClass;
            
            sharedDetections[tid] = det;
            sharedScores[tid] = maxConf;
        }
    }
    
    block.sync();
    
    // Collaborative NMS using shared memory
    if (tid < *numDetections) {
        Target myDet = sharedDetections[tid];
        bool keep = true;
        
        for (int i = 0; i < tid; ++i) {
            if (sharedScores[i] > sharedScores[tid]) {
                Target otherDet = sharedDetections[i];
                float iou = calculateIoU(myDet, otherDet);
                if (iou > nmsThreshold) {
                    keep = false;
                    break;
                }
            }
        }
        
        if (keep) {
            int finalIdx = atomicAdd(&numDetections[1], 1); // Use second element for final count
            if (finalIdx < maxDetections) {
                finalDetections[finalIdx] = myDet;
            }
        }
    }
}

// 3. Fused Target Selection + Movement Calculation
// Combines target selection, distance calculation, and movement in one kernel
__global__ void __launch_bounds__(128, 4)
fusedTargetMovement(
    const Target* __restrict__ detections,
    int numDetections,
    float2 crosshair,
    float2* __restrict__ movement,
    int* __restrict__ selectedTarget,
    float fovScale, float smoothing)
{
    auto block = cg::this_thread_block();
    
    // Shared memory for reduction
    __shared__ float minDistances[128];
    __shared__ int bestIndices[128];
    
    const int tid = threadIdx.x;
    const int detId = blockIdx.x * blockDim.x + tid;
    
    float minDist = FLT_MAX;
    int bestIdx = -1;
    
    // Each thread processes one detection
    if (detId < numDetections) {
        Target det = detections[detId];
        
        // Calculate center point
        float2 center;
        center.x = det.x + det.width * 0.5f;
        center.y = det.y + det.height * 0.5f;
        
        // Distance to crosshair
        float2 diff;
        diff.x = center.x - crosshair.x;
        diff.y = center.y - crosshair.y;
        
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        
        // Apply FOV scaling
        dist *= fovScale;
        
        minDist = dist;
        bestIdx = detId;
    }
    
    minDistances[tid] = minDist;
    bestIndices[tid] = bestIdx;
    block.sync();
    
    // Block-level reduction to find closest target
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (minDistances[tid + stride] < minDistances[tid]) {
                minDistances[tid] = minDistances[tid + stride];
                bestIndices[tid] = bestIndices[tid + stride];
            }
        }
        block.sync();
    }
    
    // First thread calculates movement and writes result
    if (tid == 0 && bestIndices[0] >= 0) {
        Target target = detections[bestIndices[0]];
        
        float2 targetCenter;
        targetCenter.x = target.x + target.width * 0.5f;
        targetCenter.y = target.y + target.height * 0.5f;
        
        float2 move;
        move.x = (targetCenter.x - crosshair.x) * smoothing;
        move.y = (targetCenter.y - crosshair.y) * smoothing;
        
        *movement = move;
        *selectedTarget = bestIndices[0];
    }
}

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

// 4. Super-fused end-to-end kernel (experimental)
// Processes entire pipeline in single kernel launch
template<int BLOCK_SIZE = 256>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
superFusedPipeline(
    cudaTextureObject_t inputTexture,
    const float* __restrict__ modelWeights,
    float2* __restrict__ mouseMovement,
    PipelineParams params)
{
    // This would require careful orchestration of:
    // 1. Preprocessing in first warp group
    // 2. Inference simulation in middle warp groups
    // 3. Postprocessing in last warp group
    // All using persistent threads and shared memory
    
    // Implementation would be model-specific
}

// Launch helper for optimized pipeline
extern "C" void launchOptimizedPipeline(
    cudaTextureObject_t captureTexture,
    float* preprocessBuffer,
    float* modelOutput,
    Target* detections,
    float2* movement,
    int srcW, int srcH,
    int dstW, int dstH,
    int numBoxes, int numClasses,
    float confThresh, float nmsThresh,
    cudaStream_t stream)
{
    // Launch fused capture preprocessing
    dim3 preprocBlock(16, 16);
    dim3 preprocGrid((dstW + 15) / 16, (dstH + 15) / 16);
    
    float3 mean = make_float3(0.5f, 0.5f, 0.5f);
    float3 invStd = make_float3(2.0f, 2.0f, 2.0f);
    
    fusedCapturePreprocessOptimized<<<preprocGrid, preprocBlock, 0, stream>>>(
        captureTexture, preprocessBuffer,
        srcW, srcH, dstW, dstH,
        (float)srcW / dstW, (float)srcH / dstH,
        mean, invStd, false
    );
    
    // Model inference happens here (TensorRT)
    
    // Launch fused decode+NMS+filter
    dim3 decodeBlock(256);
    dim3 decodeGrid((numBoxes + 255) / 256);
    size_t sharedSize = sizeof(Target) * 256 + sizeof(float) * 256;
    
    int numDet[2] = {0, 0}; // First for initial, second for final count
    cudaMemsetAsync(numDet, 0, sizeof(numDet), stream);
    
    fusedDecodeNMSFilter<<<decodeGrid, decodeBlock, sharedSize, stream>>>(
        modelOutput, detections, numDet,
        numBoxes, numClasses,
        confThresh, nmsThresh,
        nullptr, 1000
    );
    
    // Launch fused target selection + movement
    fusedTargetMovement<<<1, 128, 0, stream>>>(
        detections, numDet[1],
        make_float2(dstW/2, dstH/2),
        movement, nullptr,
        1.0f, 0.5f
    );
}