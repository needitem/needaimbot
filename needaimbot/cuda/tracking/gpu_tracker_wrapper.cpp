#include "gpu_tracker.h"
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// Maximum number of tracks to return
#define MAX_OUTPUT_TRACKS 128

// C++ Wrapper Implementation
GPUTracker::GPUTracker(int max_age, int min_hits, float iou_threshold) 
    : num_tracks_(0), use_cuda_graph_(false) {
    
    // Initialize GPU tracking context
    ctx_ = initGPUTracker(max_age, min_hits, iou_threshold);
    
    // Allocate host memory for output
    h_output_tracks_ = new GPUTrackedObject[MAX_OUTPUT_TRACKS];
}

GPUTracker::~GPUTracker() {
    // Cleanup tracking context
    destroyGPUTracker(ctx_);
    
    // Free host memory
    delete[] h_output_tracks_;
}

void GPUTracker::update(const std::vector<Target>& detections,
                        std::vector<GPUTrackedObject>& tracked_objects,
                        cudaStream_t stream,
                        float dt) {
    
    int num_detections = static_cast<int>(detections.size());
    
    // Standard kernel launch mode
    updateGPUTracker(ctx_, 
                    detections.data(), 
                    num_detections,
                    h_output_tracks_, 
                    &num_tracks_,
                    stream, 
                    dt);
    
    // Wait for GPU to complete
    cudaStreamSynchronize(stream);
    
    // Copy active tracks to output vector
    tracked_objects.clear();
    for (int i = 0; i < num_tracks_ && i < MAX_OUTPUT_TRACKS; i++) {
        if (h_output_tracks_[i].active) {
            tracked_objects.push_back(h_output_tracks_[i]);
        }
    }
}

void GPUTracker::reset() {
    // Reset tracking context
    destroyGPUTracker(ctx_);
    ctx_ = initGPUTracker(30, 3, 0.3f);  // Use default parameters
    num_tracks_ = 0;
}