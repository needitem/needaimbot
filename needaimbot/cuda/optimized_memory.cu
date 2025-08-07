#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include "../cuda/simple_cuda_mat.h"

// Optimized batched 2D memory copy for multiple buffers
class BatchedMemcpy2D {
private:
    cudaStream_t stream_;
    std::vector<cudaMemcpy3DParms> copyParams_;
    
public:
    BatchedMemcpy2D(cudaStream_t stream = 0) : stream_(stream) {
        copyParams_.reserve(16); // Pre-allocate for typical batch size
    }
    
    void addTransfer(void* dst, size_t dpitch, 
                     const void* src, size_t spitch,
                     size_t width, size_t height,
                     cudaMemcpyKind kind) {
        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr((void*)src, spitch, width, height);
        params.dstPtr = make_cudaPitchedPtr(dst, dpitch, width, height);
        params.extent = make_cudaExtent(width, height, 1);
        params.kind = kind;
        copyParams_.push_back(params);
    }
    
    cudaError_t execute() {
        if (copyParams_.empty()) return cudaSuccess;
        
        // Execute all transfers in a single batch
        for (const auto& params : copyParams_) {
            cudaError_t err = cudaMemcpy3DAsync(&params, stream_);
            if (err != cudaSuccess) return err;
        }
        
        copyParams_.clear();
        return cudaSuccess;
    }
};

// Optimized pinned memory pool for fast host-device transfers
class PinnedMemoryPool {
private:
    struct Buffer {
        void* ptr;
        size_t size;
        bool inUse;
    };
    
    std::vector<Buffer> buffers_;
    size_t totalAllocated_ = 0;
    const size_t maxPoolSize_ = 1024 * 1024 * 512; // 512MB max
    
public:
    void* allocate(size_t size) {
        // Try to find a free buffer of sufficient size
        for (auto& buf : buffers_) {
            if (!buf.inUse && buf.size >= size) {
                buf.inUse = true;
                return buf.ptr;
            }
        }
        
        // Allocate new pinned memory if within pool limit
        if (totalAllocated_ + size <= maxPoolSize_) {
            void* ptr;
            if (cudaHostAlloc(&ptr, size, cudaHostAllocDefault) == cudaSuccess) {
                buffers_.push_back({ptr, size, true});
                totalAllocated_ += size;
                return ptr;
            }
        }
        
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        for (auto& buf : buffers_) {
            if (buf.ptr == ptr) {
                buf.inUse = false;
                return;
            }
        }
    }
    
    ~PinnedMemoryPool() {
        for (const auto& buf : buffers_) {
            cudaFreeHost(buf.ptr);
        }
    }
};

// Global pinned memory pool instance
static PinnedMemoryPool g_pinnedPool;

// Optimized upload function using batched transfers and pinned memory
extern "C" cudaError_t uploadOptimized(
    const void* hostData, size_t hostPitch,
    SimpleCudaMat& gpuMat,
    cudaStream_t stream = 0)
{
    if (!hostData || gpuMat.empty()) {
        return cudaErrorInvalidValue;
    }
    
    size_t width = gpuMat.cols() * gpuMat.channels();
    size_t height = gpuMat.rows();
    
    // Use pinned memory for faster transfers
    void* pinnedBuffer = g_pinnedPool.allocate(hostPitch * height);
    if (!pinnedBuffer) {
        // Fallback to regular transfer
        return cudaMemcpy2DAsync(
            gpuMat.data(), gpuMat.step(),
            hostData, hostPitch,
            width, height,
            cudaMemcpyHostToDevice, stream
        );
    }
    
    // Copy to pinned memory first (can be done async on CPU)
    memcpy(pinnedBuffer, hostData, hostPitch * height);
    
    // Fast transfer from pinned to device memory
    cudaError_t err = cudaMemcpy2DAsync(
        gpuMat.data(), gpuMat.step(),
        pinnedBuffer, hostPitch,
        width, height,
        cudaMemcpyHostToDevice, stream
    );
    
    // Schedule deallocation after transfer completes
    if (stream) {
        cudaLaunchHostFunc(stream, 
            [](void* userData) {
                g_pinnedPool.deallocate(userData);
            }, pinnedBuffer);
    } else {
        g_pinnedPool.deallocate(pinnedBuffer);
    }
    
    return err;
}

// Optimized multi-buffer upload using batched transfers
extern "C" cudaError_t uploadMultipleOptimized(
    const std::vector<std::pair<const void*, size_t>>& hostBuffers,
    std::vector<SimpleCudaMat>& gpuMats,
    cudaStream_t stream = 0)
{
    if (hostBuffers.size() != gpuMats.size()) {
        return cudaErrorInvalidValue;
    }
    
    BatchedMemcpy2D batch(stream);
    
    for (size_t i = 0; i < hostBuffers.size(); ++i) {
        const void* hostData = hostBuffers[i].first;
        size_t hostPitch = hostBuffers[i].second;
        SimpleCudaMat& gpuMat = gpuMats[i];
        
        if (hostData && !gpuMat.empty()) {
            batch.addTransfer(
                gpuMat.data(), gpuMat.step(),
                hostData, hostPitch,
                gpuMat.cols() * gpuMat.channels(), gpuMat.rows(),
                cudaMemcpyHostToDevice
            );
        }
    }
    
    return batch.execute();
}

// Unified memory prefetching optimization
extern "C" cudaError_t optimizeUnifiedMemory(
    void* unifiedPtr, size_t size, int device = 0)
{
    cudaError_t err;
    
    // Set preferred location to GPU for performance
    err = cudaMemAdvise(unifiedPtr, size, 
                        cudaMemAdviseSetPreferredLocation, device);
    if (err != cudaSuccess) return err;
    
    // Allow direct GPU access without page faults
    err = cudaMemAdvise(unifiedPtr, size,
                        cudaMemAdviseSetAccessedBy, device);
    if (err != cudaSuccess) return err;
    
    // Prefetch to GPU for immediate use
    err = cudaMemPrefetchAsync(unifiedPtr, size, device, 0);
    
    return err;
}

// Optimized texture memory setup for bilinear interpolation
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;

extern "C" cudaError_t setupTextureMemory(
    cudaArray_t array, int width, int height)
{
    // Configure texture reference for optimized sampling
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModeLinear;  // Hardware bilinear interpolation
    texRef.normalized = false;
    
    // Bind array to texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    return cudaBindTextureToArray(texRef, array, channelDesc);
}

// Optimized resize kernel using texture memory
__global__ void resizeTextureKernel(
    float4* output, int outWidth, int outHeight,
    float scaleX, float scaleY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outWidth || y >= outHeight) return;
    
    // Hardware-accelerated bilinear sampling
    float srcX = x * scaleX + 0.5f;
    float srcY = y * scaleY + 0.5f;
    
    float4 pixel = tex2D(texRef, srcX, srcY);
    output[y * outWidth + x] = pixel;
}