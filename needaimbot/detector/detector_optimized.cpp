// Optimized Detector with GPU Chaining and Double Buffering
// Reduces synchronization points from 5-6 to 1-2 per frame

#include "detector.h"
#include <chrono>

// External GPU chain function
extern "C" cudaError_t launchPostProcessingChain(
    const void* rawOutput,
    Target* finalTargets,
    int* finalCount,
    const unsigned char* d_allow_flags,
    void* trackerContext,
    cudaStream_t stream,
    Target* decodedTargets,
    int* decodedCount,
    Target* filteredTargets,
    int* filteredCount,
    Target* nmsTargets,
    int* nmsCount,
    Target* trackedTargets,
    int* trackedCount,
    int max_classes,
    float conf_threshold,
    float iou_threshold,
    float tracking_dt,
    bool enable_tracking);

// Double buffering structure for pipeline
struct PipelineBuffers {
    // GPU buffers
    CudaBuffer<Target> decodedTargets;
    CudaBuffer<Target> filteredTargets;
    CudaBuffer<Target> nmsTargets;
    CudaBuffer<Target> trackedTargets;
    CudaBuffer<Target> finalTargets;
    
    // Count buffers
    CudaBuffer<int> decodedCount;
    CudaBuffer<int> filteredCount;
    CudaBuffer<int> nmsCount;
    CudaBuffer<int> trackedCount;
    CudaBuffer<int> finalCount;
    
    // Host staging buffers
    std::unique_ptr<Target[]> hostTargets;
    int hostCount;
    
    // Events for synchronization
    cudaEvent_t processingComplete;
    
    // Frame metadata
    int frameId;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    void allocate(size_t maxDetections) {
        decodedTargets.allocate(maxDetections);
        filteredTargets.allocate(maxDetections);
        nmsTargets.allocate(maxDetections);
        trackedTargets.allocate(maxDetections);
        finalTargets.allocate(maxDetections);
        
        decodedCount.allocate(1);
        filteredCount.allocate(1);
        nmsCount.allocate(1);
        trackedCount.allocate(1);
        finalCount.allocate(1);
        
        hostTargets.reset(new Target[maxDetections]);
        hostCount = 0;
        
        cudaEventCreate(&processingComplete);
        frameId = -1;
    }
    
    void deallocate() {
        if (processingComplete) {
            cudaEventDestroy(processingComplete);
            processingComplete = nullptr;
        }
    }
    
    ~PipelineBuffers() {
        deallocate();
    }
};

class OptimizedDetector {
private:
    // Double buffering for pipeline
    PipelineBuffers m_buffers[2];
    int m_currentBuffer = 0;
    bool m_firstFrame = true;
    
    // Streams for parallel execution
    cudaStream_t m_preprocessStream;
    cudaStream_t m_inferenceStream;
    cudaStream_t m_postprocessStream;
    
    // GPU tracking context
    void* m_gpuTrackerContext = nullptr;
    
    // Configuration
    bool m_enableTracking;
    float m_confThreshold;
    float m_iouThreshold;
    
public:
    void initializePipeline() {
        // Allocate double buffers
        const size_t maxDetections = 300;
        m_buffers[0].allocate(maxDetections);
        m_buffers[1].allocate(maxDetections);
        
        // Create streams
        cudaStreamCreate(&m_preprocessStream);
        cudaStreamCreate(&m_inferenceStream);
        cudaStreamCreate(&m_postprocessStream);
    }
    
    void performOptimizedInference() {
        auto& ctx = AppContext::getInstance();
        
        while (m_running) {
            auto frameStart = std::chrono::high_resolution_clock::now();
            
            // Get current and previous buffer indices
            int currentIdx = m_currentBuffer;
            int previousIdx = 1 - m_currentBuffer;
            
            PipelineBuffers& current = m_buffers[currentIdx];
            PipelineBuffers& previous = m_buffers[previousIdx];
            
            // Stage 1: Process previous frame results on CPU (non-blocking)
            if (!m_firstFrame) {
                processPreviousFrameCPU(previous);
            }
            
            // Stage 2: Launch current frame GPU pipeline
            launchGPUPipeline(current);
            
            // Stage 3: Minimal synchronization - only wait for previous frame if needed
            if (!m_firstFrame) {
                // Wait for previous frame to complete
                cudaEventSynchronize(previous.processingComplete);
                
                // Now previous frame results are ready for CPU processing
                finalizeFrameResults(previous);
            }
            
            // Swap buffers for next iteration
            m_currentBuffer = previousIdx;
            m_firstFrame = false;
            
            // Frame timing
            auto frameEnd = std::chrono::high_resolution_clock::now();
            auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
            updatePerformanceMetrics(frameDuration.count());
        }
    }
    
private:
    void launchGPUPipeline(PipelineBuffers& buffers) {
        auto& ctx = AppContext::getInstance();
        
        // Record frame metadata
        buffers.frameId++;
        buffers.timestamp = std::chrono::high_resolution_clock::now();
        
        // Stage 1: Preprocessing (if needed)
        // preprocessFrame(m_preprocessStream);
        
        // Stage 2: Inference
        // runInference(m_inferenceStream);
        
        // Stage 3: Post-processing chain (fully async)
        launchPostProcessingChain(
            getRawOutputPtr(),           // Raw network output
            buffers.finalTargets.get(),  // Final results
            buffers.finalCount.get(),    // Final count
            getClassFilterFlags(),       // Class filter
            m_gpuTrackerContext,         // Tracking context
            m_postprocessStream,         // Stream
            // Intermediate buffers
            buffers.decodedTargets.get(),
            buffers.decodedCount.get(),
            buffers.filteredTargets.get(),
            buffers.filteredCount.get(),
            buffers.nmsTargets.get(),
            buffers.nmsCount.get(),
            buffers.trackedTargets.get(),
            buffers.trackedCount.get(),
            // Parameters
            MAX_CLASSES,
            m_confThreshold,
            m_iouThreshold,
            0.033f,  // 30 FPS
            m_enableTracking
        );
        
        // Stage 4: Async copy results to host (only what we need)
        cudaMemcpyAsync(&buffers.hostCount, 
                       buffers.finalCount.get(),
                       sizeof(int),
                       cudaMemcpyDeviceToHost,
                       m_postprocessStream);
        
        // Copy targets only if count > 0 (will check after sync)
        // This is deferred until after we check the count
        
        // Record completion event
        cudaEventRecord(buffers.processingComplete, m_postprocessStream);
    }
    
    void processPreviousFrameCPU(PipelineBuffers& buffers) {
        // This runs while GPU processes current frame
        // Non-critical CPU work like logging, statistics, etc.
        
        if (buffers.hostCount > 0) {
            // Update tracking history
            updateTrackingHistory(buffers.hostTargets.get(), buffers.hostCount);
            
            // Calculate statistics
            updateDetectionStatistics(buffers.hostCount);
            
            // Log if needed
            if (shouldLog()) {
                logDetections(buffers.frameId, buffers.hostTargets.get(), buffers.hostCount);
            }
        }
    }
    
    void finalizeFrameResults(PipelineBuffers& buffers) {
        // After sync, copy detection data if needed
        if (buffers.hostCount > 0) {
            cudaMemcpyAsync(buffers.hostTargets.get(),
                           buffers.finalTargets.get(),
                           buffers.hostCount * sizeof(Target),
                           cudaMemcpyDeviceToHost,
                           m_postprocessStream);
            cudaStreamSynchronize(m_postprocessStream);
            
            // Update shared detection results
            updateSharedResults(buffers.hostTargets.get(), buffers.hostCount);
        }
    }
    
    // Helper functions (stubs)
    void* getRawOutputPtr() { return nullptr; }
    unsigned char* getClassFilterFlags() { return nullptr; }
    void updateTrackingHistory(Target* targets, int count) {}
    void updateDetectionStatistics(int count) {}
    bool shouldLog() { return false; }
    void logDetections(int frameId, Target* targets, int count) {}
    void updateSharedResults(Target* targets, int count) {}
    void updatePerformanceMetrics(float ms) {}
};

// Modified performGpuPostProcessing for existing detector
void Detector::performGpuPostProcessingOptimized(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // Use the new GPU chaining approach
    static PipelineBuffers chainBuffers;
    static bool initialized = false;
    
    if (!initialized) {
        chainBuffers.allocate(300);
        initialized = true;
    }
    
    // Launch the entire chain without CPU synchronization
    cudaError_t err = launchPostProcessingChain(
        outputBindings[outputNames[0]],  // Raw output
        m_finalTargetsGpu.get(),         // Final targets
        m_finalTargetsCountGpu.get(),    // Final count
        m_d_allow_flags_gpu.get(),       // Class filter
        m_gpuTrackerContext,              // Tracker
        stream,
        // Intermediate buffers
        chainBuffers.decodedTargets.get(),
        chainBuffers.decodedCount.get(),
        chainBuffers.filteredTargets.get(),
        chainBuffers.filteredCount.get(),
        chainBuffers.nmsTargets.get(),
        chainBuffers.nmsCount.get(),
        chainBuffers.trackedTargets.get(),
        chainBuffers.trackedCount.get(),
        // Parameters
        MAX_CLASSES_FOR_FILTERING,
        ctx.config.confidence_threshold,
        ctx.config.nms_iou_threshold,
        0.033f,
        ctx.config.enable_tracking
    );
    
    if (err != cudaSuccess) {
        std::cerr << "[GPU Chain] Error: " << cudaGetErrorString(err) << std::endl;
        // Fallback to original implementation
        performGpuPostProcessing(stream);
    }
}