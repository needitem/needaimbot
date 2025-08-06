#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include "postProcess.h"
#include "../detector/detector.h"  // For MouseMovement struct

// GPU-persistent target selection with direct mouse movement calculation
// This eliminates CPU transfers completely

__constant__ float g_centerX;
__constant__ float g_centerY;
__constant__ float g_scopeMultiplier;
__constant__ float g_headYOffset;
__constant__ float g_bodyYOffset;
__constant__ int g_headClassId;

// Initialize constant memory
extern "C" __host__ void updateTargetSelectionConstants(
    float centerX, float centerY, float scopeMultiplier,
    float headYOffset, float bodyYOffset, int headClassId,
    cudaStream_t stream)
{
    cudaMemcpyToSymbolAsync(g_centerX, &centerX, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_centerY, &centerY, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_scopeMultiplier, &scopeMultiplier, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_headYOffset, &headYOffset, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_bodyYOffset, &bodyYOffset, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_headClassId, &headClassId, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
}

// Optimized kernel for target selection and movement calculation in one pass
__global__ void selectTargetAndCalculateMovementKernel(
    const Detection* detections,
    int numDetections,
    MouseMovement* movement,
    int* bestIdx,
    float maxDistance)
{
    __shared__ float sharedMinDist[256];
    __shared__ int sharedBestIdx[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float minDist = FLT_MAX;
    int localBestIdx = -1;
    
    // Each thread processes multiple detections for better efficiency
    for (int idx = gid; idx < numDetections; idx += gridDim.x * blockDim.x) {
        const Detection& det = detections[idx];
        
        // Calculate target center with offset based on class
        float targetCenterX = det.x + det.width * 0.5f;
        float targetCenterY;
        
        if (det.classId == g_headClassId) {
            targetCenterY = det.y + det.height * g_headYOffset;
        } else {
            targetCenterY = det.y + det.height * g_bodyYOffset;
        }
        
        // Calculate distance from crosshair
        float dx = targetCenterX - g_centerX;
        float dy = targetCenterY - g_centerY;
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Apply scope check
        if (dist < maxDistance && dist < minDist) {
            minDist = dist;
            localBestIdx = idx;
        }
    }
    
    // Store in shared memory
    sharedMinDist[tid] = minDist;
    sharedBestIdx[tid] = localBestIdx;
    __syncthreads();
    
    // Parallel reduction to find minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sharedMinDist[tid + s] < sharedMinDist[tid]) {
                sharedMinDist[tid] = sharedMinDist[tid + s];
                sharedBestIdx[tid] = sharedBestIdx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result and calculates movement
    if (tid == 0) {
        if (sharedBestIdx[0] >= 0) {
            *bestIdx = sharedBestIdx[0];
            const Detection& bestDet = detections[sharedBestIdx[0]];
            
            // Calculate target center with offset
            float targetCenterX = bestDet.x + bestDet.width * 0.5f;
            float targetCenterY;
            
            if (bestDet.classId == g_headClassId) {
                targetCenterY = bestDet.y + bestDet.height * g_headYOffset;
            } else {
                targetCenterY = bestDet.y + bestDet.height * g_bodyYOffset;
            }
            
            // Calculate movement delta
            movement->dx = (targetCenterX - g_centerX) / g_scopeMultiplier;
            movement->dy = (targetCenterY - g_centerY) / g_scopeMultiplier;
            movement->hasTarget = true;
            movement->targetDistance = sharedMinDist[0];
        } else {
            movement->dx = 0.0f;
            movement->dy = 0.0f;
            movement->hasTarget = false;
            movement->targetDistance = FLT_MAX;
        }
    }
}

// Fused kernel for NMS + target selection
__global__ void fusedNMSAndTargetSelectionKernel(
    Detection* detections,
    int* numDetections,
    MouseMovement* movement,
    float nmsThreshold,
    float maxTargetDistance)
{
    extern __shared__ char sharedMem[];
    bool* keep = (bool*)sharedMem;
    
    int tid = threadIdx.x;
    int numDets = *numDetections;
    
    // Initialize keep array
    for (int i = tid; i < numDets; i += blockDim.x) {
        keep[i] = true;
    }
    __syncthreads();
    
    // Parallel NMS
    for (int i = tid; i < numDets; i += blockDim.x) {
        if (!keep[i]) continue;
        
        Detection& boxA = detections[i];
        
        for (int j = i + 1; j < numDets; j++) {
            if (!keep[j]) continue;
            
            Detection& boxB = detections[j];
            
            // Same class check
            if (boxA.classId != boxB.classId) continue;
            
            // Calculate IoU
            float x1 = fmaxf(boxA.x, boxB.x);
            float y1 = fmaxf(boxA.y, boxB.y);
            float x2 = fminf(boxA.x + boxA.width, boxB.x + boxB.width);
            float y2 = fminf(boxA.y + boxA.height, boxB.y + boxB.height);
            
            float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
            float areaA = boxA.width * boxA.height;
            float areaB = boxB.width * boxB.height;
            float unionArea = areaA + areaB - intersection;
            
            float iou = intersection / unionArea;
            
            if (iou > nmsThreshold) {
                // Keep higher confidence box
                if (boxA.confidence < boxB.confidence) {
                    keep[i] = false;
                } else {
                    keep[j] = false;
                }
            }
        }
    }
    __syncthreads();
    
    // Find best target from kept detections
    float minDist = FLT_MAX;
    int bestIdx = -1;
    
    for (int i = tid; i < numDets; i += blockDim.x) {
        if (!keep[i]) continue;
        
        const Detection& det = detections[i];
        
        float targetCenterX = det.x + det.width * 0.5f;
        float targetCenterY = (det.classId == g_headClassId) ? 
            det.y + det.height * g_headYOffset : 
            det.y + det.height * g_bodyYOffset;
        
        float dx = targetCenterX - g_centerX;
        float dy = targetCenterY - g_centerY;
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (dist < maxTargetDistance && dist < minDist) {
            minDist = dist;
            bestIdx = i;
        }
    }
    
    // Write movement command
    if (tid == 0 && bestIdx >= 0) {
        const Detection& bestDet = detections[bestIdx];
        
        float targetCenterX = bestDet.x + bestDet.width * 0.5f;
        float targetCenterY = (bestDet.classId == g_headClassId) ? 
            bestDet.y + bestDet.height * g_headYOffset : 
            bestDet.y + bestDet.height * g_bodyYOffset;
        
        movement->dx = (targetCenterX - g_centerX) / g_scopeMultiplier;
        movement->dy = (targetCenterY - g_centerY) / g_scopeMultiplier;
        movement->hasTarget = true;
        movement->targetDistance = minDist;
    } else if (tid == 0) {
        movement->dx = 0.0f;
        movement->dy = 0.0f;
        movement->hasTarget = false;
        movement->targetDistance = FLT_MAX;
    }
}

// Export functions - remove extern "C" for C++ linkage
void selectTargetAndCalculateMovementGpu(
    const Detection* d_detections,
    int numDetections,
    MouseMovement* d_movement,
    int* d_bestIdx,
    float maxDistance,
    cudaStream_t stream)
{
    if (numDetections <= 0) return;
    
    int blockSize = 256;
    int gridSize = (numDetections + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 32); // Limit grid size
    
    selectTargetAndCalculateMovementKernel<<<gridSize, blockSize, 0, stream>>>(
        d_detections, numDetections, d_movement, d_bestIdx, maxDistance
    );
}

void fusedNMSAndTargetSelection(
    Detection* d_detections,
    int* d_numDetections,
    MouseMovement* d_movement,
    float nmsThreshold,
    float maxTargetDistance,
    cudaStream_t stream)
{
    int numDets;
    cudaMemcpyAsync(&numDets, d_numDetections, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (numDets <= 0) return;
    
    int blockSize = 256;
    size_t sharedMemSize = numDets * sizeof(bool);
    
    fusedNMSAndTargetSelectionKernel<<<1, blockSize, sharedMemSize, stream>>>(
        d_detections, d_numDetections, d_movement, nmsThreshold, maxTargetDistance
    );
}