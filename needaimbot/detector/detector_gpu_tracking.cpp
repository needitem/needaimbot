// GPU Tracking implementation for detector.cpp
// This replaces the CPU SORT tracker section

void Detector::runGPUTracking(cudaStream_t stream) {
    auto& ctx = AppContext::getInstance();
    
    // GPU Tracking System
    if (ctx.config.enable_tracking && m_finalDetectionsCountHost > 0 && m_gpuTrackerContext) {
        
        // Allocate tracked targets buffer if needed
        if (!m_trackedTargetsGpu.get()) {
            m_trackedTargetsGpu.allocate(Constants::MAX_DETECTIONS);
        }
        
        // Allocate count buffer
        CudaBuffer<int> trackedCountGpu(1);
        
        // Run GPU tracking directly on GPU memory
        updateGPUTrackerDirect(
            m_gpuTrackerContext,
            m_finalDetectionsGpu.get(),        // Input: detections already on GPU
            m_finalDetectionsCountHost,         // Number of detections
            m_trackedTargetsGpu.get(),          // Output: tracked targets on GPU
            trackedCountGpu.get(),               // Output: number of tracked targets
            stream,                              // CUDA stream
            0.033f                               // dt: ~30 FPS
        );
        
        // Replace final detections with tracked targets
        cudaMemcpyAsync(m_finalDetectionsGpu.get(), 
                       m_trackedTargetsGpu.get(),
                       Constants::MAX_DETECTIONS * sizeof(Target),
                       cudaMemcpyDeviceToDevice, 
                       stream);
        
        // Update count
        int tracked_count = 0;
        cudaMemcpyAsync(&tracked_count, 
                       trackedCountGpu.get(),
                       sizeof(int), 
                       cudaMemcpyDeviceToHost, 
                       stream);
        cudaStreamSynchronize(stream);
        
        // Update final count
        if (tracked_count > 0 && tracked_count <= Constants::MAX_DETECTIONS) {
            m_finalDetectionsCountHost = tracked_count;
            cudaMemcpyAsync(m_finalDetectionsCountGpu.get(), 
                           &m_finalDetectionsCountHost,
                           sizeof(int), 
                           cudaMemcpyHostToDevice, 
                           stream);
            
            // Copy to host for CPU-based operations if needed
            cudaMemcpyAsync(m_finalDetectionsHost.get(),
                           m_finalDetectionsGpu.get(),
                           m_finalDetectionsCountHost * sizeof(Target),
                           cudaMemcpyDeviceToHost,
                           stream);
            
            // Update CPU tracking cache for other uses
            {
                std::lock_guard<std::mutex> lock(m_trackingMutex);
                m_trackedObjects.clear();
                m_trackedObjects.reserve(tracked_count);
                
                // After sync, copy tracked objects
                cudaStreamSynchronize(stream);
                for (int i = 0; i < tracked_count; i++) {
                    m_trackedObjects.push_back(m_finalDetectionsHost[i]);
                }
            }
            
            std::cout << "[GPU Tracker] Tracked " << tracked_count << " targets" << std::endl;
        }
    } else if (ctx.config.enable_tracking && m_finalDetectionsCountHost > 0) {
        std::cout << "[GPU Tracker] Tracking enabled but GPU tracker not initialized" << std::endl;
    }
}