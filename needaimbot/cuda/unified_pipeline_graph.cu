#include "unified_pipeline_graph.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include <NvInfer.h>
#include <vector>
#include <memory>
#include <atomic>
#include <float.h>
#include "../detector/detector.h"

// Structures used by kernels
struct Detection {
    float x, y, w, h;
    float confidence;
    int classId;
};

struct TargetInfo {
    float x, y;
    float width, height;
    float confidence;
    bool isValid;
};

// ======== CUDA Kernels ========

static __global__ void preprocessCaptureKernel(
    uchar4* input,
    float* output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // Calculate source coordinates
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;
    
    int srcX = (int)(x * scaleX);
    int srcY = (int)(y * scaleY);
    
    // Clamp to bounds
    srcX = min(max(srcX, 0), srcWidth - 1);
    srcY = min(max(srcY, 0), srcHeight - 1);
    
    // Read pixel
    uchar4 pixel = input[srcY * srcWidth + srcX];
    
    // Convert BGRA to RGB and normalize
    int dstIdx = (y * dstWidth + x) * 3;
    output[dstIdx + 0] = pixel.z / 255.0f;  // R
    output[dstIdx + 1] = pixel.y / 255.0f;  // G
    output[dstIdx + 2] = pixel.x / 255.0f;  // B
}

static __global__ void nmsPostprocessKernel(
    float* input,
    Detection* output,
    int* numDetections,
    int numBoxes,
    float confThreshold,
    float nmsThreshold,
    int maxDetections)
{
    // Simplified NMS implementation
    // In production, use a more sophisticated NMS algorithm
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoxes) return;
    
    // Each thread processes one detection
    float* box = input + tid * 85;
    
    // Find best class
    float maxConf = 0.0f;
    int bestClass = -1;
    
    for (int c = 0; c < 80; c++) {
        float conf = box[5 + c];
        if (conf > maxConf) {
            maxConf = conf;
            bestClass = c;
        }
    }
    
    // Check confidence threshold
    float objectness = box[4];
    float finalConf = objectness * maxConf;
    
    if (finalConf > confThreshold && *numDetections < maxDetections) {
        int idx = atomicAdd(numDetections, 1);
        if (idx < maxDetections) {
            Detection& det = output[idx];
            det.x = box[0];
            det.y = box[1];
            det.w = box[2];
            det.h = box[3];
            det.confidence = finalConf;
            det.classId = bestClass;
        }
    }
}

static __global__ void selectBestTargetKernel(
    Detection* detections,
    int* numDetections,
    TargetInfo* targetInfo,
    float centerX, float centerY,
    float fov)
{
    int tid = threadIdx.x;
    int num = *numDetections;
    
    if (tid >= num) return;
    
    __shared__ float distances[256];
    __shared__ int indices[256];
    
    // Calculate distance from center for each detection
    if (tid < num) {
        Detection& det = detections[tid];
        float dx = det.x - centerX;
        float dy = det.y - centerY;
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Check if within FOV
        if (dist <= fov) {
            distances[tid] = dist;
            indices[tid] = tid;
        } else {
            distances[tid] = FLT_MAX;
            indices[tid] = -1;
        }
    } else {
        distances[tid] = FLT_MAX;
        indices[tid] = -1;
    }
    
    __syncthreads();
    
    // Parallel reduction to find minimum distance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (distances[tid + stride] < distances[tid]) {
                distances[tid] = distances[tid + stride];
                indices[tid] = indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        if (indices[0] >= 0 && indices[0] < num) {
            Detection& best = detections[indices[0]];
            targetInfo->x = best.x;
            targetInfo->y = best.y;
            targetInfo->width = best.w;
            targetInfo->height = best.h;
            targetInfo->confidence = best.confidence;
            targetInfo->isValid = true;
        } else {
            targetInfo->isValid = false;
        }
    }
}

static __global__ void calculatePIDMovementKernel(
    TargetInfo* targetInfo,
    float* output,  // [dx, dy]
    float centerX, float centerY,
    float kp, float ki, float kd,
    float smoothing)
{
    // Static variables for integral and previous error (persistent across calls)
    __shared__ float integralX, integralY;
    __shared__ float prevErrorX, prevErrorY;
    
    if (!targetInfo->isValid) {
        output[0] = 0.0f;
        output[1] = 0.0f;
        return;
    }
    
    // Calculate error
    float errorX = targetInfo->x - centerX;
    float errorY = targetInfo->y - centerY;
    
    // Update integral
    integralX += errorX * 0.016f;  // Assuming 60 FPS
    integralY += errorY * 0.016f;
    
    // Anti-windup
    integralX = fmaxf(-100.0f, fminf(100.0f, integralX));
    integralY = fmaxf(-100.0f, fminf(100.0f, integralY));
    
    // Calculate derivative
    float derivativeX = (errorX - prevErrorX) / 0.016f;
    float derivativeY = (errorY - prevErrorY) / 0.016f;
    
    // PID calculation
    float pidX = kp * errorX + ki * integralX + kd * derivativeX;
    float pidY = kp * errorY + ki * integralY + kd * derivativeY;
    
    // Apply smoothing
    output[0] = output[0] * (1.0f - smoothing) + pidX * smoothing;
    output[1] = output[1] * (1.0f - smoothing) + pidY * smoothing;
    
    // Clamp output
    output[0] = fmaxf(-100.0f, fminf(100.0f, output[0]));
    output[1] = fmaxf(-100.0f, fminf(100.0f, output[1]));
    
    // Update previous error
    prevErrorX = errorX;
    prevErrorY = errorY;
}

// Simple memory pool implementation
class MemoryPool {
private:
    uint8_t* m_buffer;
    size_t m_size;
    size_t m_offset;
    
public:
    MemoryPool(size_t size) : m_size(size), m_offset(0) {
        cudaMalloc(&m_buffer, size);
    }
    
    ~MemoryPool() {
        if (m_buffer) cudaFree(m_buffer);
    }
    
    void* allocate(size_t size) {
        if (m_offset + size > m_size) return nullptr;
        void* ptr = m_buffer + m_offset;
        m_offset += size;
        return ptr;
    }
    
    void reset() { m_offset = 0; }
};

// Simple PID controller
class GPUPIDController {
public:
    GPUPIDController() {}
    ~GPUPIDController() {}
    void initialize(float kp, float ki, float kd) {}
};

// Simple Kalman filter
class GPUKalmanFilter {
public:
    GPUKalmanFilter() {}
    ~GPUKalmanFilter() {}
    void initialize(int maxDetections) {}
    void predictAndUpdate(void* detections, int* numDetections, cudaStream_t stream, float predictionFrames) {}
};

// Implementation class for UnifiedPipelineGraph
class UnifiedPipelineGraph::Impl {
private:
    // CUDA Graph handles
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCreated = false;
    bool m_graphCapturing = false;
    
    // Stream priorities
    static constexpr int HIGH_PRIORITY = -1;
    static constexpr int NORMAL_PRIORITY = 0;
    
    // Streams
    cudaStream_t m_mainStream = nullptr;
    
    // Memory pools
    MemoryPool* m_memoryPool = nullptr;
    
    // Buffers
    void* d_captureBuffer = nullptr;      // 캡처된 원본 이미지
    void* d_preprocessBuffer = nullptr;    // 전처리된 이미지
    void* d_inferenceInput = nullptr;      // 추론 입력
    void* d_inferenceOutput = nullptr;     // 추론 출력
    void* d_detections = nullptr;          // NMS 후 탐지 결과
    void* d_targetInfo = nullptr;          // 선택된 타겟 정보
    float* d_pidOutput = nullptr;          // PID 제어 출력 (dx, dy)
    int* d_numDetections = nullptr;        // 탐지 개수
    
    // Dimensions
    int m_captureWidth = 0;
    int m_captureHeight = 0;
    int m_modelWidth = 0;
    int m_modelHeight = 0;
    int m_maxDetections = 100;
    
    // Components
    nvinfer1::IExecutionContext* m_trtContext = nullptr;
    GPUPIDController* m_pidController = nullptr;
    GPUKalmanFilter* m_kalmanFilter = nullptr;
    
    // Graph nodes for dynamic updates
    cudaGraphNode_t m_preprocessNode = nullptr;
    cudaGraphNode_t m_inferenceNode = nullptr;
    cudaGraphNode_t m_nmsNode = nullptr;
    cudaGraphNode_t m_targetSelectNode = nullptr;
    cudaGraphNode_t m_pidNode = nullptr;
    cudaGraphNode_t m_kalmanNode = nullptr;
    
    // Parameters that can be updated
    PipelineParams m_params;
    
public:
    Impl() {
        // Create high priority stream
        cudaStreamCreateWithPriority(&m_mainStream, cudaStreamNonBlocking, HIGH_PRIORITY);
        
        // Initialize components
        m_pidController = new GPUPIDController();
        m_kalmanFilter = new GPUKalmanFilter();
        m_memoryPool = new MemoryPool(512 * 1024 * 1024); // 512MB pool
    }
    
    ~Impl() {
        cleanup();
        
        if (m_mainStream) cudaStreamDestroy(m_mainStream);
        
        delete m_pidController;
        delete m_kalmanFilter;
        delete m_memoryPool;
    }
    
    // Initialize pipeline
    bool initialize(int captureWidth, int captureHeight, 
                   int modelWidth, int modelHeight,
                   nvinfer1::IExecutionContext* trtContext) {
        m_captureWidth = captureWidth;
        m_captureHeight = captureHeight;
        m_modelWidth = modelWidth;
        m_modelHeight = modelHeight;
        m_trtContext = trtContext;
        
        // Allocate buffers
        size_t captureSize = captureWidth * captureHeight * 4; // BGRA
        size_t preprocessSize = modelWidth * modelHeight * 3 * sizeof(float); // RGB float
        size_t inferenceOutputSize = modelWidth * modelHeight * 85 * sizeof(float); // YOLO output
        size_t detectionsSize = m_maxDetections * sizeof(Detection);
        
        d_captureBuffer = m_memoryPool->allocate(captureSize);
        d_preprocessBuffer = m_memoryPool->allocate(preprocessSize);
        d_inferenceInput = d_preprocessBuffer; // 전처리 출력이 추론 입력
        d_inferenceOutput = m_memoryPool->allocate(inferenceOutputSize);
        d_detections = m_memoryPool->allocate(detectionsSize);
        d_targetInfo = m_memoryPool->allocate(sizeof(TargetInfo));
        d_pidOutput = (float*)m_memoryPool->allocate(2 * sizeof(float)); // dx, dy
        d_numDetections = (int*)m_memoryPool->allocate(sizeof(int));
        
        // Initialize PID controller
        m_pidController->initialize(m_params.pidKp, m_params.pidKi, m_params.pidKd);
        
        // Initialize Kalman filter
        m_kalmanFilter->initialize(m_maxDetections);
        
        return true;
    }
    
    // Capture the entire pipeline as a CUDA Graph
    bool createGraph() {
        if (m_graphCreated) {
            return true;
        }
        
        // Begin graph capture
        cudaError_t err = cudaStreamBeginCapture(m_mainStream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            return false;
        }
        
        m_graphCapturing = true;
        
        // ============= 1. Preprocessing =============
        dim3 preprocessBlock(32, 32);
        dim3 preprocessGrid(
            (m_modelWidth + preprocessBlock.x - 1) / preprocessBlock.x,
            (m_modelHeight + preprocessBlock.y - 1) / preprocessBlock.y
        );
        
        preprocessCaptureKernel<<<preprocessGrid, preprocessBlock, 0, m_mainStream>>>(
            (uchar4*)d_captureBuffer,
            (float*)d_preprocessBuffer,
            m_captureWidth, m_captureHeight,
            m_modelWidth, m_modelHeight
        );
        
        // ============= 2. Inference (TensorRT) =============
        void* bindings[] = { d_inferenceInput, d_inferenceOutput };
        m_trtContext->enqueueV3(m_mainStream);
        
        // ============= 3. NMS Postprocessing =============
        dim3 nmsBlock(256);
        dim3 nmsGrid((m_modelWidth * m_modelHeight + nmsBlock.x - 1) / nmsBlock.x);
        
        nmsPostprocessKernel<<<nmsGrid, nmsBlock, 0, m_mainStream>>>(
            (float*)d_inferenceOutput,
            (Detection*)d_detections,
            d_numDetections,
            m_modelWidth * m_modelHeight,
            m_params.confThreshold,
            m_params.nmsThreshold,
            m_maxDetections
        );
        
        // ============= 4. Kalman Filter Tracking =============
        m_kalmanFilter->predictAndUpdate(
            (Detection*)d_detections,
            d_numDetections,
            m_mainStream,
            m_params.predictionFrames
        );
        
        // ============= 5. Target Selection =============
        selectBestTargetKernel<<<1, 256, 0, m_mainStream>>>(
            (Detection*)d_detections,
            d_numDetections,
            (TargetInfo*)d_targetInfo,
            m_captureWidth / 2.0f,  // Screen center X
            m_captureHeight / 2.0f,  // Screen center Y
            m_params.targetSelectionFov
        );
        
        // ============= 6. PID Control for Mouse Movement =============
        calculatePIDMovementKernel<<<1, 1, 0, m_mainStream>>>(
            (TargetInfo*)d_targetInfo,
            d_pidOutput,
            m_captureWidth / 2.0f,
            m_captureHeight / 2.0f,
            m_params.pidKp,
            m_params.pidKi,
            m_params.pidKd,
            m_params.smoothingFactor
        );
        
        // End graph capture
        err = cudaStreamEndCapture(m_mainStream, &m_graph);
        if (err != cudaSuccess) {
            m_graphCapturing = false;
            return false;
        }
        
        // Instantiate graph
        err = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
            m_graphCapturing = false;
            return false;
        }
        
        m_graphCreated = true;
        m_graphCapturing = false;
        
        return true;
    }
    
    // Execute the entire pipeline with single graph launch
    bool execute(void* captureData, float& dx, float& dy, int& targetCount) {
        if (!m_graphCreated) {
            if (!createGraph()) {
                return false;
            }
        }
        
        // Copy capture data to device (this is the only memcpy needed)
        cudaMemcpyAsync(d_captureBuffer, captureData, 
                       m_captureWidth * m_captureHeight * 4,
                       cudaMemcpyHostToDevice, m_mainStream);
        
        // Launch the entire pipeline graph
        cudaError_t err = cudaGraphLaunch(m_graphExec, m_mainStream);
        if (err != cudaSuccess) {
            return false;
        }
        
        // Copy results back (only PID output and detection count)
        float pidResult[2];
        int detectionCount;
        
        cudaMemcpyAsync(pidResult, d_pidOutput, 
                       2 * sizeof(float),
                       cudaMemcpyDeviceToHost, m_mainStream);
        
        cudaMemcpyAsync(&detectionCount, d_numDetections,
                       sizeof(int),
                       cudaMemcpyDeviceToHost, m_mainStream);
        
        // Synchronize to get results
        cudaStreamSynchronize(m_mainStream);
        
        dx = pidResult[0];
        dy = pidResult[1];
        targetCount = detectionCount;
        
        return true;
    }
    
    // Check if graph is ready
    bool isReady() const {
        return m_graphCreated;
    }
    
    // Update parameters without recreating graph
    bool updateParameters(const PipelineParams& params) {
        m_params = params;
        
        if (!m_graphCreated) {
            return true; // Will use new params when graph is created
        }
        
        // Try to update graph with new parameters
        // For kernel parameter updates, we need to use cudaGraphExecKernelNodeSetParams
        // This is more complex and requires storing node handles during graph creation
        
        // For now, recreate the graph with new parameters
        cleanup();
        return createGraph();
    }
    
    // Get pipeline statistics
    void getStatistics(float& fps, float& latency, int& graphNodes) {
        if (m_graphCreated && m_graph) {
            size_t numNodes = 0;
            cudaGraphGetNodes(m_graph, nullptr, &numNodes);
            graphNodes = (int)numNodes;
        } else {
            graphNodes = 0;
        }
        
        // FPS and latency would be calculated from timing events
        fps = 0.0f;
        latency = 0.0f;
    }
    
private:
    void cleanup() {
        if (m_graphExec) {
            cudaGraphExecDestroy(m_graphExec);
            m_graphExec = nullptr;
        }
        
        if (m_graph) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
        }
        
        m_graphCreated = false;
    }
};

// UnifiedPipelineGraph public interface implementation
UnifiedPipelineGraph::UnifiedPipelineGraph() {
    pImpl = new Impl();
}

UnifiedPipelineGraph::~UnifiedPipelineGraph() {
    delete pImpl;
}

bool UnifiedPipelineGraph::initialize(int captureWidth, int captureHeight,
                                     int modelWidth, int modelHeight,
                                     nvinfer1::IExecutionContext* trtContext) {
    return pImpl->initialize(captureWidth, captureHeight, modelWidth, modelHeight, trtContext);
}

bool UnifiedPipelineGraph::createGraph() {
    return pImpl->createGraph();
}

bool UnifiedPipelineGraph::execute(void* captureData, float& dx, float& dy, int& targetCount) {
    return pImpl->execute(captureData, dx, dy, targetCount);
}

bool UnifiedPipelineGraph::updateParameters(const PipelineParams& params) {
    return pImpl->updateParameters(params);
}

void UnifiedPipelineGraph::getStatistics(float& fps, float& latency, int& graphNodes) {
    pImpl->getStatistics(fps, latency, graphNodes);
}

bool UnifiedPipelineGraph::isReady() const {
    return pImpl->isReady();
}

// Export C interface for integration
extern "C" {
    UnifiedPipelineGraph* createUnifiedPipeline() {
        return new UnifiedPipelineGraph();
    }
    
    void destroyUnifiedPipeline(UnifiedPipelineGraph* pipeline) {
        delete pipeline;
    }
    
    bool initializeUnifiedPipeline(UnifiedPipelineGraph* pipeline,
                                  int captureWidth, int captureHeight,
                                  int modelWidth, int modelHeight,
                                  void* trtContext) {
        return pipeline->initialize(captureWidth, captureHeight, 
                                   modelWidth, modelHeight,
                                   (nvinfer1::IExecutionContext*)trtContext);
    }
    
    bool executeUnifiedPipeline(UnifiedPipelineGraph* pipeline,
                               void* captureData,
                               float* dx, float* dy,
                               int* targetCount) {
        return pipeline->execute(captureData, *dx, *dy, *targetCount);
    }
    
    bool updatePipelineParams(UnifiedPipelineGraph* pipeline,
                             float confThreshold,
                             float nmsThreshold,
                             float pidKp, float pidKi, float pidKd) {
        PipelineParams params;
        params.confThreshold = confThreshold;
        params.nmsThreshold = nmsThreshold;
        params.pidKp = pidKp;
        params.pidKi = pidKi;
        params.pidKd = pidKd;
        return pipeline->updateParameters(params);
    }
}