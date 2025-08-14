#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <mutex>
#include <vector>
#include <atomic>
#ifdef _WIN32
// Prevent winsock.h from being included
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#endif
#include "simple_cuda_mat.h"
#include "detection/postProcess.h"

// Unified pipeline optimization with CUDA Graphs, kernel fusion, and zero-copy memory

// ============================================================================
// KERNEL FUSION: Capture + Preprocessing in single kernel
// ============================================================================

// Fused kernel for BGRA to BGR conversion + resize + normalize in one pass
__global__ void fusedCapturePreprocessKernel(
    const uchar4* __restrict__ input,   // BGRA input from capture
    float* __restrict__ output,         // Normalized float output for inference
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    float scaleX, float scaleY,
    float normMean, float normStd,
    bool swapRB)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstWidth || dstY >= dstHeight) return;
    
    // Calculate source coordinates with bilinear interpolation
    float srcXf = dstX * scaleX;
    float srcYf = dstY * scaleY;
    
    int srcX0 = __float2int_rd(srcXf);
    int srcY0 = __float2int_rd(srcYf);
    int srcX1 = min(srcX0 + 1, srcWidth - 1);
    int srcY1 = min(srcY0 + 1, srcHeight - 1);
    
    float fx = srcXf - srcX0;
    float fy = srcYf - srcY0;
    
    // Read 4 pixels for bilinear interpolation
    uchar4 p00 = input[srcY0 * srcWidth + srcX0];
    uchar4 p01 = input[srcY0 * srcWidth + srcX1];
    uchar4 p10 = input[srcY1 * srcWidth + srcX0];
    uchar4 p11 = input[srcY1 * srcWidth + srcX1];
    
    // Bilinear interpolation for each channel
    float b = (1-fx)*(1-fy)*p00.x + fx*(1-fy)*p01.x + (1-fx)*fy*p10.x + fx*fy*p11.x;
    float g = (1-fx)*(1-fy)*p00.y + fx*(1-fy)*p01.y + (1-fx)*fy*p10.y + fx*fy*p11.y;
    float r = (1-fx)*(1-fy)*p00.z + fx*(1-fy)*p01.z + (1-fx)*fy*p10.z + fx*fy*p11.z;
    
    // Swap R and B if needed
    if (swapRB) {
        float temp = r;
        r = b;
        b = temp;
    }
    
    // Normalize and write to CHW format
    int pixelIdx = dstY * dstWidth + dstX;
    int channelStride = dstWidth * dstHeight;
    
    output[0 * channelStride + pixelIdx] = (r / 255.0f - normMean) / normStd;  // R channel
    output[1 * channelStride + pixelIdx] = (g / 255.0f - normMean) / normStd;  // G channel
    output[2 * channelStride + pixelIdx] = (b / 255.0f - normMean) / normStd;  // B channel
}

// ============================================================================
// UNIFIED MEMORY MANAGEMENT
// ============================================================================

class UnifiedMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;
        int deviceId;
    };
    
    std::vector<MemoryBlock> blocks;
    std::mutex poolMutex;
    size_t totalAllocated = 0;
    const size_t MAX_POOL_SIZE = 2ULL * 1024 * 1024 * 1024; // 2GB
    
public:
    void* allocate(size_t size, int deviceId = 0) {
        std::lock_guard<std::mutex> lock(poolMutex);
        
        // Try to find existing block
        for (auto& block : blocks) {
            if (!block.inUse && block.size >= size && block.deviceId == deviceId) {
                block.inUse = true;
                return block.ptr;
            }
        }
        
        // Allocate new unified memory
        void* ptr = nullptr;
        cudaSetDevice(deviceId);
        
        if (cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) == cudaSuccess) {
            // Prefetch to GPU for better performance
            cudaMemPrefetchAsync(ptr, size, deviceId, 0);
            
            blocks.push_back({ptr, size, true, deviceId});
            totalAllocated += size;
            return ptr;
        }
        
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(poolMutex);
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.inUse = false;
                return;
            }
        }
    }
    
    ~UnifiedMemoryPool() {
        for (auto& block : blocks) {
            cudaFree(block.ptr);
        }
    }
};

// ============================================================================
// TRIPLE BUFFERING SYSTEM
// ============================================================================

template<typename T>
class TripleBuffer {
private:
    struct Buffer {
        T* data;
        cudaEvent_t ready;
        std::atomic<bool> isReady{false};
    };
    
    Buffer buffers[3];
    std::atomic<int> captureIdx{0};
    std::atomic<int> inferenceIdx{1};
    std::atomic<int> displayIdx{2};
    size_t bufferSize;
    
public:
    TripleBuffer(size_t size) : bufferSize(size) {
        for (int i = 0; i < 3; i++) {
            cudaMalloc(&buffers[i].data, size * sizeof(T));
            cudaEventCreateWithFlags(&buffers[i].ready, cudaEventDisableTiming);
        }
    }
    
    ~TripleBuffer() {
        for (int i = 0; i < 3; i++) {
            cudaFree(buffers[i].data);
            cudaEventDestroy(buffers[i].ready);
        }
    }
    
    T* getCaptureBuffer() {
        return buffers[captureIdx].data;
    }
    
    T* getInferenceBuffer() {
        int idx = inferenceIdx.load();
        if (buffers[idx].isReady) {
            return buffers[idx].data;
        }
        return nullptr;
    }
    
    T* getDisplayBuffer() {
        return buffers[displayIdx].data;
    }
    
    void swapCaptureBuffer(cudaStream_t stream) {
        int currentCapture = captureIdx.load();
        cudaEventRecord(buffers[currentCapture].ready, stream);
        buffers[currentCapture].isReady = true;
        
        // Rotate indices: capture -> inference -> display -> capture
        int nextCapture = displayIdx.load();
        int nextInference = currentCapture;
        int nextDisplay = inferenceIdx.load();
        
        captureIdx = nextCapture;
        inferenceIdx = nextInference;
        displayIdx = nextDisplay;
    }
    
    void waitForInference(cudaStream_t stream) {
        int idx = inferenceIdx.load();
        if (buffers[idx].isReady) {
            cudaStreamWaitEvent(stream, buffers[idx].ready, 0);
            buffers[idx].isReady = false;
        }
    }
};

// ============================================================================
// MULTI-STREAM PIPELINE COORDINATOR
// ============================================================================

class PipelineCoordinator {
private:
    // Stream priorities
    static constexpr int CAPTURE_PRIORITY = -2;     // Highest
    static constexpr int INFERENCE_PRIORITY = -1;   // High
    static constexpr int POSTPROCESS_PRIORITY = 0;  // Normal
    static constexpr int DISPLAY_PRIORITY = 1;      // Low
    
    // Streams
    cudaStream_t captureStream;
    cudaStream_t preprocessStream;
    cudaStream_t inferenceStream;
    cudaStream_t postprocessStream;
    cudaStream_t targetSelectionStream;
    cudaStream_t displayStream;
    
    // Events for synchronization
    cudaEvent_t captureComplete;
    cudaEvent_t preprocessComplete;
    cudaEvent_t inferenceComplete;
    cudaEvent_t postprocessComplete;
    cudaEvent_t targetSelected;
    
    // CUDA Graph handles
    cudaGraph_t fullPipelineGraph;
    cudaGraphExec_t fullPipelineGraphExec;
    bool graphCreated = false;
    
public:
    PipelineCoordinator() {
        // Create prioritized streams
        cudaStreamCreateWithPriority(&captureStream, cudaStreamNonBlocking, CAPTURE_PRIORITY);
        cudaStreamCreateWithPriority(&preprocessStream, cudaStreamNonBlocking, INFERENCE_PRIORITY);
        cudaStreamCreateWithPriority(&inferenceStream, cudaStreamNonBlocking, INFERENCE_PRIORITY);
        cudaStreamCreateWithPriority(&postprocessStream, cudaStreamNonBlocking, POSTPROCESS_PRIORITY);
        cudaStreamCreateWithPriority(&targetSelectionStream, cudaStreamNonBlocking, INFERENCE_PRIORITY);
        cudaStreamCreateWithPriority(&displayStream, cudaStreamNonBlocking, DISPLAY_PRIORITY);
        
        // Create events
        cudaEventCreateWithFlags(&captureComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&preprocessComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&inferenceComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&postprocessComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&targetSelected, cudaEventDisableTiming);
    }
    
    ~PipelineCoordinator() {
        if (graphCreated) {
            cudaGraphExecDestroy(fullPipelineGraphExec);
            cudaGraphDestroy(fullPipelineGraph);
        }
        
        cudaStreamDestroy(captureStream);
        cudaStreamDestroy(preprocessStream);
        cudaStreamDestroy(inferenceStream);
        cudaStreamDestroy(postprocessStream);
        cudaStreamDestroy(targetSelectionStream);
        cudaStreamDestroy(displayStream);
        
        cudaEventDestroy(captureComplete);
        cudaEventDestroy(preprocessComplete);
        cudaEventDestroy(inferenceComplete);
        cudaEventDestroy(postprocessComplete);
        cudaEventDestroy(targetSelected);
    }
    
    // Create CUDA Graph for entire pipeline
    void createPipelineGraph(
        void* captureBuffer,
        void* preprocessBuffer,
        void* inferenceInput,
        void* inferenceOutput,
        void* detections,
        void* movement,
        int width, int height,
        int modelWidth, int modelHeight)
    {
        // Begin graph capture
        cudaStreamBeginCapture(inferenceStream, cudaStreamCaptureModeGlobal);
        
        // 1. Preprocessing (fused kernel)
        dim3 blockSize(32, 32);
        dim3 gridSize((modelWidth + blockSize.x - 1) / blockSize.x,
                      (modelHeight + blockSize.y - 1) / blockSize.y);
        
        fusedCapturePreprocessKernel<<<gridSize, blockSize, 0, inferenceStream>>>(
            (uchar4*)captureBuffer,
            (float*)preprocessBuffer,
            width, height,
            modelWidth, modelHeight,
            (float)width / modelWidth,
            (float)height / modelHeight,
            0.5f, 0.5f,  // normalization parameters
            false        // swap RB
        );
        
        // 2. Record preprocessing complete
        cudaEventRecord(preprocessComplete, inferenceStream);
        
        // 3. Inference will be added by TensorRT
        // ...
        
        // 4. Record inference complete
        cudaEventRecord(inferenceComplete, inferenceStream);
        
        // 5. Postprocessing and target selection (fused)
        // This would be the fused NMS + target selection kernel
        
        // End graph capture
        cudaStreamEndCapture(inferenceStream, &fullPipelineGraph);
        
        // Create executable graph
        cudaGraphInstantiate(&fullPipelineGraphExec, fullPipelineGraph, nullptr, nullptr, 0);
        graphCreated = true;
    }
    
    // Execute the full pipeline graph
    void executePipeline() {
        if (graphCreated) {
            cudaGraphLaunch(fullPipelineGraphExec, inferenceStream);
        }
    }
    
    // Getters for streams
    cudaStream_t getCaptureStream() { return captureStream; }
    cudaStream_t getPreprocessStream() { return preprocessStream; }
    cudaStream_t getInferenceStream() { return inferenceStream; }
    cudaStream_t getPostprocessStream() { return postprocessStream; }
    cudaStream_t getTargetSelectionStream() { return targetSelectionStream; }
    cudaStream_t getDisplayStream() { return displayStream; }
};

// ============================================================================
// ZERO-COPY TEXTURE MEMORY FOR WINDOWS GRAPHICS CAPTURE
// ============================================================================

class ZeroCopyTextureCapture {
private:
    cudaGraphicsResource_t cudaResource = nullptr;
    cudaArray_t cudaArray = nullptr;
    cudaTextureObject_t texObject = 0;
    
public:
    bool registerD3D11Texture(void* d3dTexture) {
#ifdef _WIN32
        // Register D3D11 texture with CUDA
        cudaError_t err = cudaGraphicsD3D11RegisterResource(
            &cudaResource,
            static_cast<ID3D11Texture2D*>(d3dTexture),
            cudaGraphicsRegisterFlagsNone
        );
#else
        cudaError_t err = cudaErrorNotSupported;
#endif
        
        if (err != cudaSuccess) return false;
        
        // Map the resource
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);
        
        // Create texture object for zero-copy access
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;
        
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = false;
        
        cudaCreateTextureObject(&texObject, &resDesc, &texDesc, nullptr);
        
        return true;
    }
    
    cudaTextureObject_t getTextureObject() { return texObject; }
    
    ~ZeroCopyTextureCapture() {
        if (texObject) cudaDestroyTextureObject(texObject);
        if (cudaResource) {
            cudaGraphicsUnmapResources(1, &cudaResource, 0);
            cudaGraphicsUnregisterResource(cudaResource);
        }
    }
};

// ============================================================================
// OPTIMIZED CAPTURE-INFERENCE-MOVEMENT PIPELINE
// ============================================================================

extern "C" {
    
// Initialize the optimized pipeline
void* initializeOptimizedPipeline(
    int captureWidth, int captureHeight,
    int modelWidth, int modelHeight)
{
    PipelineCoordinator* pipeline = new PipelineCoordinator();
    
    // Pre-allocate all buffers with unified memory
    static UnifiedMemoryPool memPool;
    
    size_t captureSize = captureWidth * captureHeight * 4;  // BGRA
    size_t preprocessSize = modelWidth * modelHeight * 3 * sizeof(float);  // RGB float
    
    void* captureBuffer = memPool.allocate(captureSize);
    void* preprocessBuffer = memPool.allocate(preprocessSize);
    
    // Create triple buffers
    static TripleBuffer<unsigned char> captureTripleBuffer(captureSize);
    static TripleBuffer<float> preprocessTripleBuffer(preprocessSize);
    
    return pipeline;
}

// Execute one frame through the pipeline
void executeOptimizedPipeline(
    void* pipelineHandle,
    void* captureData,
    void* inferenceEngine,
    void* mouseMovement,
    float centerX, float centerY,
    float scopeMultiplier)
{
    PipelineCoordinator* pipeline = (PipelineCoordinator*)pipelineHandle;
    
    // Update constants for target selection
    extern void updateTargetSelectionConstants(
        float, float, float, float, float, int, cudaStream_t);
    
    updateTargetSelectionConstants(
        centerX, centerY, scopeMultiplier,
        0.3f, 0.5f, 0,  // head/body offsets and class ID
        pipeline->getTargetSelectionStream()
    );
    
    // Execute the full pipeline graph
    pipeline->executePipeline();
}

// Cleanup
void destroyOptimizedPipeline(void* pipelineHandle) {
    delete (PipelineCoordinator*)pipelineHandle;
}

}