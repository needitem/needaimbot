// Patch for detector.cpp to enable async GPU chaining
// Apply these changes to reduce synchronization from 5-6 points to 1-2

// Add to detector.h:
/*
private:
    // Double buffering for async pipeline
    struct AsyncBuffers {
        CudaBuffer<int> countBuffer;
        cudaEvent_t completeEvent;
        int frameId;
    };
    AsyncBuffers m_asyncBuffers[2];
    int m_currentAsyncBuffer = 0;
    
    // Optimized post-processing
    void performGpuPostProcessingAsync(cudaStream_t stream, int bufferIdx);
*/

// Replace performGpuPostProcessing with this optimized version:
void Detector::performGpuPostProcessingAsync(cudaStream_t stream, int bufferIdx) {
    auto& ctx = AppContext::getInstance();
    
    if (outputNames.empty()) {
        std::cerr << "[PostProcess] No output names found" << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    const std::string& primaryOutputName = outputNames[0];
    void* d_rawOutputPtr = outputBindings[primaryOutputName];
    
    if (!d_rawOutputPtr) {
        std::cerr << "[PostProcess] Raw output GPU pointer is null" << std::endl;
        cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);
        return;
    }

    // Clear buffers once at start (async)
    cudaMemsetAsync(m_decodedCountGpu.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(m_classFilteredCountGpu.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(m_finalTargetsCountGpu.get(), 0, sizeof(int), stream);

    // Chain 1: Decode -> Filter -> NMS (no CPU sync between)
    
    // Decode detections
    dim3 decodeBlocks(32);
    dim3 decodeThreads(256);
    decodeDetectionsGpu<<<decodeBlocks, decodeThreads, 0, stream>>>(
        d_rawOutputPtr,
        m_decodedTargetsGpu.get(),
        m_decodedCountGpu.get(),
        ctx.config.confidence_threshold
    );
    
    // Filter by class (uses GPU count directly)
    dim3 filterBlocks(16);
    dim3 filterThreads(256);
    filterTargetsByClassIdGpuAsync<<<filterBlocks, filterThreads, 0, stream>>>(
        m_decodedTargetsGpu.get(),
        m_decodedCountGpu.get(),  // GPU pointer, not host value
        m_classFilteredTargetsGpu.get(),
        m_classFilteredCountGpu.get(),
        m_d_allow_flags_gpu.get(),
        MAX_CLASSES_FOR_FILTERING,
        300
    );
    
    // NMS (uses GPU count directly)
    runNmsGpuAsync<<<1, 256, 0, stream>>>(
        m_classFilteredTargetsGpu.get(),
        m_classFilteredCountGpu.get(),  // GPU pointer
        m_finalTargetsGpu.get(),
        m_finalTargetsCountGpu.get(),
        ctx.config.nms_iou_threshold,
        300
    );
    
    // Chain 2: Tracking (conditional but GPU-based)
    if (ctx.config.enable_tracking && m_gpuTrackerContext) {
        // Tracking kernel checks count internally
        trackingChainKernel<<<1, 256, 0, stream>>>(
            m_finalTargetsGpu.get(),
            m_finalTargetsCountGpu.get(),  // GPU pointer
            m_gpuTrackerContext,
            m_trackedTargetsGpu.get(),
            m_asyncBuffers[bufferIdx].countBuffer.get(),
            0.033f
        );
        
        // Swap tracked results to final
        cudaMemcpyAsync(m_finalTargetsGpu.get(),
                       m_trackedTargetsGpu.get(),
                       300 * sizeof(Target),
                       cudaMemcpyDeviceToDevice,
                       stream);
        cudaMemcpyAsync(m_finalTargetsCountGpu.get(),
                       m_asyncBuffers[bufferIdx].countBuffer.get(),
                       sizeof(int),
                       cudaMemcpyDeviceToDevice,
                       stream);
    }
    
    // Only copy count to host for decision making
    cudaMemcpyAsync(&m_finalTargetsCountHost,
                   m_finalTargetsCountGpu.get(),
                   sizeof(int),
                   cudaMemcpyDeviceToHost,
                   stream);
    
    // Record completion
    cudaEventRecord(m_asyncBuffers[bufferIdx].completeEvent, stream);
}

// Modified inference loop section:
void Detector::inferenceThreadOptimized() {
    // ... initialization code ...
    
    // Initialize async buffers
    for (int i = 0; i < 2; i++) {
        m_asyncBuffers[i].countBuffer.allocate(1);
        cudaEventCreate(&m_asyncBuffers[i].completeEvent);
        m_asyncBuffers[i].frameId = -1;
    }
    
    while (m_running) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        
        int currentBuffer = m_currentAsyncBuffer;
        int previousBuffer = 1 - m_currentAsyncBuffer;
        
        // Process previous frame results (CPU work while GPU runs)
        if (m_asyncBuffers[previousBuffer].frameId >= 0) {
            // Wait for previous frame completion
            cudaEventSynchronize(m_asyncBuffers[previousBuffer].completeEvent);
            
            // Now copy detection data if count > 0
            if (m_finalTargetsCountHost > 0) {
                cudaMemcpyAsync(m_finalTargetsHost.get(),
                               m_finalTargetsGpu.get(),
                               m_finalTargetsCountHost * sizeof(Target),
                               cudaMemcpyDeviceToHost,
                               postprocessStream);
                cudaStreamSynchronize(postprocessStream);
                
                // Process detections on CPU
                processDetectionsCPU();
            }
        }
        
        // Launch current frame processing (fully async)
        {
            // Preprocessing
            cudaStreamWaitEvent(preprocessStream, m_frameCaptured, cudaEventWaitExternal);
            preprocessFrame(preprocessStream);
            cudaEventRecord(m_preprocessDone, preprocessStream);
            
            // Inference
            cudaStreamWaitEvent(stream, m_preprocessDone, cudaEventWaitExternal);
            if (m_graphCaptured && m_inferenceGraphExec) {
                cudaGraphLaunch(m_inferenceGraphExec, stream);
            } else {
                context->enqueueV3(stream);
            }
            cudaEventRecord(m_inferenceDone, stream);
            
            // Post-processing (async chain)
            cudaStreamWaitEvent(postprocessStream, m_inferenceDone, cudaEventWaitExternal);
            performGpuPostProcessingAsync(postprocessStream, currentBuffer);
            
            m_asyncBuffers[currentBuffer].frameId = frameCounter++;
        }
        
        // Swap buffers
        m_currentAsyncBuffer = previousBuffer;
        
        // Performance metrics
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
        ctx.g_current_inference_time_ms.store(duration.count());
    }
    
    // Cleanup
    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(m_asyncBuffers[i].completeEvent);
    }
}

// Add these kernel implementations to a .cu file:
__global__ void filterTargetsByClassIdGpuAsync(
    const Target* decodedTargets,
    const int* d_numDecodedTargets,
    Target* filteredTargets,
    int* filteredCount,
    const unsigned char* d_allow_flags,
    int max_check_id,
    int max_output) {
    
    int numTargets = *d_numDecodedTargets;
    if (numTargets == 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *filteredCount = 0;
        }
        return;
    }
    
    // Filtering logic...
}

__global__ void runNmsGpuAsync(
    const Target* filteredTargets,
    const int* d_numFilteredTargets,
    Target* nmsTargets,
    int* nmsCount,
    float iou_threshold,
    int max_output) {
    
    int numTargets = *d_numFilteredTargets;
    if (numTargets == 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *nmsCount = 0;
        }
        return;
    }
    
    // NMS logic...
}

__global__ void trackingChainKernel(
    Target* targets,
    const int* d_targetCount,
    void* trackerContext,
    Target* trackedTargets,
    int* trackedCount,
    float dt) {
    
    int count = *d_targetCount;
    if (count == 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *trackedCount = 0;
        }
        return;
    }
    
    // Tracking logic...
}