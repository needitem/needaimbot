#ifndef BYTE_TRACKER_H
#define BYTE_TRACKER_H

#include <vector>
#include <memory>
#include <set>
#include "../modules/eigen/include/Eigen/Dense"
#include "KalmanTracker.h"
#include "../core/Target.h"

/**
 * ByteTracker: An improved tracking algorithm that handles low-confidence detections
 * to maintain tracking during occlusions.
 * 
 * Key improvements over SORT:
 * - Two-stage association: high confidence first, then low confidence
 * - Better occlusion handling by utilizing low-confidence detections
 * - Maintains track continuity even when confidence temporarily drops
 */
class ByteTracker {
public:
    // Track states
    enum class TrackState {
        NEW,        // Just created
        TRACKED,    // Successfully tracked
        LOST,       // Lost but may recover
        REMOVED     // To be removed
    };

    // Track information
    struct Track {
        int track_id;
        KalmanTracker kalman_tracker;
        TrackState state;
        int time_since_update;
        int hit_count;
        int start_frame;
        Target target;  // Using existing Target structure
        
        Track(const Target& t, int id) 
            : track_id(id), kalman_tracker(cv::Rect2f(t.x, t.y, t.width, t.height)), 
              state(TrackState::NEW), time_since_update(0), hit_count(1), 
              start_frame(0), target(t) {
            target.id = id;
        }
    };


public:
    ByteTracker();
    ~ByteTracker() = default;

    /**
     * Update tracker with new detections
     * @param detections All detections from current frame
     * @param frame_id Current frame number
     * @return Vector of tracked objects with stable IDs
     */
    std::vector<Target> update(const std::vector<Target>& detections, int frame_id);

    // Configuration setters
    void setTrackThresh(float thresh) { track_thresh_ = thresh; }
    void setHighThresh(float thresh) { high_thresh_ = thresh; }
    void setMatchThresh(float thresh) { match_thresh_ = thresh; }
    void setMaxTimeLost(int frames) { max_time_lost_ = frames; }
    void setMinHits(int hits) { min_hits_ = hits; }

    // Get current tracks
    std::vector<Track> getTracks() const { return tracks_; }

private:
    // Helper functions
    std::vector<std::vector<float>> calculateIOUMatrix(
        const std::vector<Track*>& tracks,
        const std::vector<Target>& detections);
    
    std::vector<std::pair<int, int>> hungarianMatching(
        const std::vector<std::vector<float>>& cost_matrix);
    
    void associateDetectionsToTracks(
        std::vector<Track*>& tracks,
        const std::vector<Target>& detections,
        float thresh,
        std::vector<int>& matched_tracks,
        std::vector<int>& matched_detections,
        std::vector<int>& unmatched_tracks,
        std::vector<int>& unmatched_detections);
    
    Track* createNewTrack(const Target& det);
    void updateTrack(Track* track, const Target& det);
    void removeDeletedTracks();

private:
    // Configuration parameters
    float track_thresh_ = 0.5f;    // Threshold for starting a track
    float high_thresh_ = 0.6f;     // High confidence threshold
    float match_thresh_ = 0.8f;    // IOU matching threshold
    int max_time_lost_ = 30;       // Max frames to keep lost track
    int min_hits_ = 3;              // Min hits to confirm track
    
    // Internal state
    std::vector<Track> tracks_;
    int next_id_ = 1;
    int frame_id_ = 0;
    
    // Track pools for efficient management
    std::vector<Track*> tracked_tracks_;
    std::vector<Track*> lost_tracks_;
    std::vector<Track*> removed_tracks_;
};

#endif // BYTE_TRACKER_H