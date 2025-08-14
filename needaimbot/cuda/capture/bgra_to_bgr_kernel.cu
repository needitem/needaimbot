#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized BGRA to BGR conversion kernel
__global__ void convertBGRAtoBGR_kernel(
    const uint8_t* __restrict__ src, 
    uint8_t* __restrict__ dst,
    int width, int height,
    int srcPitch, int dstPitch) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Coalesced memory access
    const int srcIdx = y * srcPitch + x * 4;
    const int dstIdx = y * dstPitch + x * 3;
    
    // Load BGRA pixel (4 bytes)
    uchar4 bgra = *reinterpret_cast<const uchar4*>(src + srcIdx);
    
    // Store BGR pixel (3 bytes)
    dst[dstIdx + 0] = bgra.x;  // B
    dst[dstIdx + 1] = bgra.y;  // G
    dst[dstIdx + 2] = bgra.z;  // R
}

// Optimized kernel using shared memory for better performance
__global__ void convertBGRAtoBGR_shared_kernel(
    const uint8_t* __restrict__ src, 
    uint8_t* __restrict__ dst,
    int width, int height,
    int srcPitch, int dstPitch) 
{
    extern __shared__ uchar4 sharedBGRA[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    
    // Load to shared memory
    if (x < width && y < height) {
        const int srcIdx = y * srcPitch + x * 4;
        sharedBGRA[ty * blockDim.x + tx] = *reinterpret_cast<const uchar4*>(src + srcIdx);
    }
    __syncthreads();
    
    // Write from shared memory
    if (x < width && y < height) {
        const int dstIdx = y * dstPitch + x * 3;
        const uchar4& bgra = sharedBGRA[ty * blockDim.x + tx];
        dst[dstIdx + 0] = bgra.x;  // B
        dst[dstIdx + 1] = bgra.y;  // G
        dst[dstIdx + 2] = bgra.z;  // R
    }
}

extern "C" void launchBGRAtoBGRConversion(
    const uint8_t* src, uint8_t* dst,
    int width, int height,
    int srcPitch, int dstPitch,
    cudaStream_t stream)
{
    dim3 blockSize(32, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    convertBGRAtoBGR_kernel<<<gridSize, blockSize, 0, stream>>>(
        src, dst, width, height, srcPitch, dstPitch
    );
}