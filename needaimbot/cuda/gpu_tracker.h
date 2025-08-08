#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "../core/Target.h"

// GPU Tracked Object structure - aligned with unified Target
struct GPUTrackedObject {
    float x, y, width, height;  // Changed w,h to width,height for consistency
    float center_x, center_y;
    float velocity_x, velocity_y;
    float kalman_state[8];  // [cx, cy, w, h, vx, vy, vw, vh]
    float confidence;
    int classId;  // Changed class_id to classId for consistency
    int track_id;
    int age;
    int hits;
    int time_since_update;
    bool active;
};

// Forward declaration of GPU tracking context
struct GPUTrackingContext;

// C++ Interface for GPU Tracker
class GPUTracker {
public:
    GPUTracker(int max_age = 30, int min_hits = 3, float iou_threshold = 0.3f);
    ~GPUTracker();
    
    // Update tracking with new detections
    void update(const std::vector<Target>& detections,
                std::vector<GPUTrackedObject>& tracked_objects,
                cudaStream_t stream = 0,
                float dt = 0.033f);
    
    // Reset tracker
    void reset();
    
    // Get tracker statistics
    int getNumTracks() const { return num_tracks_; }
    
    // Enable/disable CUDA Graph mode
    void setCudaGraphMode(bool enable) { use_cuda_graph_ = enable; }
    
private:
    GPUTrackingContext* ctx_;
    GPUTrackedObject* h_output_tracks_;
    int num_tracks_;
    bool use_cuda_graph_;
};

// C Interface for compatibility
extern "C" {
    GPUTrackingContext* initGPUTracker(int max_age, int min_hits, float iou_threshold);
    
    void updateGPUTracker(
        GPUTrackingContext* ctx,
        const Target* h_detections,
        int num_detections,
        GPUTrackedObject* h_output_tracks,
        int* h_num_output_tracks,
        cudaStream_t stream,
        float dt);
    
    // Direct GPU-to-GPU tracking (no host copies)
    void updateGPUTrackerDirect(
        GPUTrackingContext* ctx,
        const Target* d_detections,      // GPU memory
        int num_detections,
        Target* d_output_tracks,          // GPU memory output
        int* d_num_output_tracks,
        cudaStream_t stream,
        float dt = 0.033f);
    
    void destroyGPUTracker(GPUTrackingContext* ctx);
}