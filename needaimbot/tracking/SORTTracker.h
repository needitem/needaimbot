#pragma once

#include "SimpleKalmanTracker.h"
#include "Hungarian.h"
#include "../core/Target.h"  // Use unified Target struct
#include <vector>
#include <memory>
#include <chrono>

// Use Target as TrackedObject and Detection for backwards compatibility
using TrackedObject = Target;
using Detection = Target;

class SORTTracker {
public:
    SORTTracker(int max_age = 5, int min_hits = 3, float iou_threshold = 0.3f);
    ~SORTTracker();
    
    // Update tracker with new detections
    std::vector<TrackedObject> update(const std::vector<Detection>& detections);
    
    // Get current tracked objects
    const std::vector<TrackedObject>& getTrackedObjects() const { return tracked_objects_; }
    
    // Reset all tracking
    void reset();
    
    // Set parameters
    void setMaxAge(int max_age) { max_age_ = max_age; }
    void setMinHits(int min_hits) { min_hits_ = min_hits; }
    void setIOUThreshold(float threshold) { iou_threshold_ = threshold; }
    
private:
    // Calculate IoU between two rectangles
    float calculateIOU(float x1, float y1, float w1, float h1,
                      float x2, float y2, float w2, float h2);
    
    // Associate detections to tracked objects
    void associateDetectionsToTrackers(const std::vector<Detection>& detections,
                                      std::vector<TrackedObject>& trackers,
                                      std::vector<std::pair<int, int>>& matched,
                                      std::vector<int>& unmatched_dets,
                                      std::vector<int>& unmatched_trks);
    
    // Parameters
    int max_age_;        // Maximum frames to keep track without detection
    int min_hits_;       // Minimum hits before track is confirmed  
    float iou_threshold_; // IOU threshold for matching
    
    // Track metadata (not part of POD Target)
    struct TrackMetadata {
        int age = 0;
        int time_since_update = 0;
        std::shared_ptr<SimpleKalmanTracker> kalman_tracker;
        std::chrono::high_resolution_clock::time_point detection_timestamp;
    };
    
    // Tracked objects
    std::vector<TrackedObject> tracked_objects_;
    std::vector<TrackMetadata> track_metadata_;  // Parallel array for metadata
    
    // For FPS-independent velocity calculation
    std::chrono::steady_clock::time_point last_update_time_;
    
    // Next available track ID
    static int next_id_;
};