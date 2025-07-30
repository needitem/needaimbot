#ifndef BATCHED_RESULTS_H
#define BATCHED_RESULTS_H

#include <cuda_runtime.h>

struct Detection;

extern "C" cudaError_t prepareBatchedResultsGpu(
    const Detection* finalDetections,
    const int* finalCount,
    const int* bestIndex,
    void* batchedResults,
    cudaStream_t stream);

#endif // BATCHED_RESULTS_H