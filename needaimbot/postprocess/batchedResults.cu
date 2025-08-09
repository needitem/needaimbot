#include <cuda_runtime.h>
#include "../detector/detector.h"
#include "postProcess.h"

// CUDA 커널: 모든 결과를 하나의 구조체로 통합
__global__ void prepareBatchedResultsKernel(
    const Target* finalDetections,
    const int* finalCount,
    const int* bestIndex,
    Detector::BatchedResults* batchedResults)
{
    if (threadIdx.x == 0) {
        // 먼저 모든 필드를 초기화
        batchedResults->finalCount = 0;
        batchedResults->bestIndex = -1;
        batchedResults->hasTarget = false;
        memset(&batchedResults->bestTarget, 0, sizeof(Target));
        
        // 실제 값 설정
        if (finalCount != nullptr) {
            batchedResults->finalCount = *finalCount;
        }
        
        // 유효한 타겟이 있으면 복사
        if (bestIndex != nullptr && finalCount != nullptr && 
            *bestIndex >= 0 && *bestIndex < *finalCount && *finalCount > 0) {
            batchedResults->bestIndex = *bestIndex;
            batchedResults->bestTarget = finalDetections[*bestIndex];
            batchedResults->hasTarget = true;
        }
    }
}

extern "C" cudaError_t prepareBatchedResultsGpu(
    const Target* finalDetections,
    const int* finalCount,
    const int* bestIndex,
    void* batchedResults,
    cudaStream_t stream)
{
    prepareBatchedResultsKernel<<<1, 32, 0, stream>>>(
        finalDetections,
        finalCount,
        bestIndex,
        (Detector::BatchedResults*)batchedResults
    );
    
    return cudaGetLastError();
}