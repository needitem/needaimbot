#include "SORTTracker.h"
#include <iostream>
#include <algorithm>

int SORTTracker::next_id_ = 0;

SORTTracker::SORTTracker(int max_age, int min_hits, float iou_threshold)
    : max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold) {
    last_update_time_ = std::chrono::steady_clock::now();
}

SORTTracker::~SORTTracker() {
}

std::vector<TrackedObject> SORTTracker::update(const std::vector<Detection>& detections) {
    // Ensure metadata array is same size as tracked objects
    if (track_metadata_.size() != tracked_objects_.size()) {
        track_metadata_.resize(tracked_objects_.size());
    }
    
    auto current_time = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(current_time - last_update_time_).count();
    last_update_time_ = current_time;
    
    // Predict new locations of existing trackers
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        auto& track = tracked_objects_[i];
        auto& metadata = track_metadata_[i];
        
        if (!metadata.kalman_tracker) {
            std::cerr << "[SORTTracker] ERROR: Null kalman_tracker for track ID " << track.id << std::endl;
            continue;
        }
        
        try {
            SimpleRect predicted_state = metadata.kalman_tracker->predict();
            track.x = predicted_state.x;
            track.y = predicted_state.y;
            track.width = std::max(1.0f, predicted_state.width);
            track.height = std::max(1.0f, predicted_state.height);
            track.updateCenter();
            metadata.time_since_update++;
            metadata.age++;
        } catch (const std::exception& e) {
            std::cerr << "[SORTTracker] Exception in predict for track ID " << track.id << ": " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SORTTracker] Unknown exception in predict for track ID " << track.id << std::endl;
        }
    }
    
    // Associate detections to trackers
    std::vector<std::pair<int, int>> matched;
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_trks;
    
    try {
        associateDetectionsToTrackers(detections, tracked_objects_, 
                                     matched, unmatched_dets, unmatched_trks);
    } catch (const std::exception& e) {
        std::cerr << "[SORTTracker::update] Exception in associateDetectionsToTrackers: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "[SORTTracker::update] Unknown exception in associateDetectionsToTrackers" << std::endl;
        throw;
    }
    
    // Update matched trackers with assigned detections
    for (const auto& match : matched) {
        int det_idx = match.first;
        int trk_idx = match.second;
        
        // Safety checks
        if (det_idx < 0 || det_idx >= static_cast<int>(detections.size()) ||
            trk_idx < 0 || trk_idx >= static_cast<int>(tracked_objects_.size()) ||
            trk_idx >= static_cast<int>(track_metadata_.size())) {
            std::cerr << "[SORTTracker] ERROR: Invalid match indices" << std::endl;
            continue;
        }
        
        // Convert Detection to SimpleRect
        float det_x = static_cast<float>(detections[det_idx].x);
        float det_y = static_cast<float>(detections[det_idx].y);
        float det_w = static_cast<float>(detections[det_idx].width);
        float det_h = static_cast<float>(detections[det_idx].height);
        SimpleRect det_rect(det_x, det_y, det_w, det_h);
        
        // Store previous center for velocity calculation
        float prev_center_x = tracked_objects_[trk_idx].center_x;
        float prev_center_y = tracked_objects_[trk_idx].center_y;
        
        // Validate kalman tracker
        if (!track_metadata_[trk_idx].kalman_tracker) {
            std::cerr << "[SORTTracker] ERROR: Null kalman_tracker at index " << trk_idx << std::endl;
            continue;
        }
        
        // Update Kalman filter
        try {
            track_metadata_[trk_idx].kalman_tracker->update(det_rect);
            track_metadata_[trk_idx].time_since_update = 0;
            tracked_objects_[trk_idx].confidence = detections[det_idx].confidence;
            tracked_objects_[trk_idx].classId = detections[det_idx].classId;
            
            // Update bbox and center
            SimpleRect updated_state = track_metadata_[trk_idx].kalman_tracker->get_state();
            tracked_objects_[trk_idx].x = updated_state.x;
            tracked_objects_[trk_idx].y = updated_state.y;
            tracked_objects_[trk_idx].width = std::max(1.0f, updated_state.width);
            tracked_objects_[trk_idx].height = std::max(1.0f, updated_state.height);
            tracked_objects_[trk_idx].updateCenter();
        } catch (const std::exception& e) {
            std::cerr << "[SORTTracker] Exception in update: " << e.what() << std::endl;
        }
        
        // Calculate velocity (pixels per second)
        if (dt > 0) {
            tracked_objects_[trk_idx].velocity_x = (tracked_objects_[trk_idx].center_x - prev_center_x) / dt;
            tracked_objects_[trk_idx].velocity_y = (tracked_objects_[trk_idx].center_y - prev_center_y) / dt;
        }
        
        // Update timestamp
        track_metadata_[trk_idx].detection_timestamp = std::chrono::high_resolution_clock::now();
    }
    
    // Create new trackers for unmatched detections
    const size_t MAX_TRACKED_OBJECTS = 50;
    
    for (int idx : unmatched_dets) {
        if (tracked_objects_.size() >= MAX_TRACKED_OBJECTS) {
            break;
        }
        
        float det_x = static_cast<float>(detections[idx].x);
        float det_y = static_cast<float>(detections[idx].y);
        float det_w = static_cast<float>(detections[idx].width);
        float det_h = static_cast<float>(detections[idx].height);
        SimpleRect det_rect(det_x, det_y, det_w, det_h);
        
        TrackedObject new_track;
        new_track.id = next_id_++;
        new_track.x = det_rect.x;
        new_track.y = det_rect.y;
        new_track.width = det_rect.width;
        new_track.height = det_rect.height;
        new_track.updateCenter();
        new_track.velocity_x = 0;
        new_track.velocity_y = 0;
        new_track.confidence = detections[idx].confidence;
        new_track.classId = detections[idx].classId;
        
        tracked_objects_.push_back(new_track);
        
        // Add metadata for new track
        TrackMetadata metadata;
        metadata.kalman_tracker = std::make_shared<SimpleKalmanTracker>(det_rect);
        metadata.age = 0;
        metadata.time_since_update = 0;
        metadata.detection_timestamp = std::chrono::high_resolution_clock::now();
        track_metadata_.push_back(metadata);
    }
    
    // Remove dead tracks
    size_t i = 0;
    while (i < tracked_objects_.size()) {
        if (track_metadata_[i].time_since_update > max_age_) {
            tracked_objects_.erase(tracked_objects_.begin() + i);
            track_metadata_.erase(track_metadata_.begin() + i);
        } else {
            i++;
        }
    }
    
    // Return only confirmed tracks (with enough hits)
    std::vector<TrackedObject> confirmed_tracks;
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        if (track_metadata_[i].kalman_tracker &&
            (track_metadata_[i].kalman_tracker->m_hit_streak >= min_hits_ || 
             track_metadata_[i].age <= min_hits_)) {
            confirmed_tracks.push_back(tracked_objects_[i]);
        }
    }
    
    return confirmed_tracks;
}

void SORTTracker::reset() {
    tracked_objects_.clear();
    track_metadata_.clear();
    next_id_ = 0;
    SimpleKalmanTracker::kf_count = 0;
}

float SORTTracker::calculateIOU(float x1, float y1, float w1, float h1,
                                float x2, float y2, float w2, float h2) {
    float xi1 = std::max(x1, x2);
    float yi1 = std::max(y1, y2);
    float xi2 = std::min(x1 + w1, x2 + w2);
    float yi2 = std::min(y1 + h1, y2 + h2);
    
    float intersection = std::max(0.0f, xi2 - xi1) * std::max(0.0f, yi2 - yi1);
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - intersection;
    
    if (union_area == 0) return 0;
    return intersection / union_area;
}

void SORTTracker::associateDetectionsToTrackers(const std::vector<Detection>& detections,
                                               std::vector<TrackedObject>& trackers,
                                               std::vector<std::pair<int, int>>& matched,
                                               std::vector<int>& unmatched_dets,
                                               std::vector<int>& unmatched_trks) {
    matched.clear();
    unmatched_dets.clear();
    unmatched_trks.clear();
    
    if (trackers.empty()) {
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_dets.push_back(static_cast<int>(i));
        }
        return;
    }
    
    if (detections.empty()) {
        for (size_t i = 0; i < trackers.size(); i++) {
            unmatched_trks.push_back(static_cast<int>(i));
        }
        return;
    }
    
    // Calculate cost matrix (negative IOU for Hungarian algorithm)
    std::vector<std::vector<double>> cost_matrix(detections.size(), std::vector<double>(trackers.size()));
    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < trackers.size(); j++) {
            float iou = calculateIOU(detections[i].x, detections[i].y, 
                                    detections[i].width, detections[i].height,
                                    trackers[j].x, trackers[j].y,
                                    trackers[j].width, trackers[j].height);
            cost_matrix[i][j] = 1.0 - iou;
        }
    }
    
    // Hungarian algorithm
    std::vector<int> assignment;
    HungarianAlgorithm hungarian;
    hungarian.Solve(cost_matrix, assignment);
    
    // Process assignments
    for (size_t det_idx = 0; det_idx < assignment.size(); det_idx++) {
        int trk_idx = assignment[det_idx];
        if (trk_idx == -1) {
            unmatched_dets.push_back(static_cast<int>(det_idx));
        } else {
            float iou = 1.0f - cost_matrix[det_idx][trk_idx];
            if (iou < iou_threshold_) {
                unmatched_dets.push_back(static_cast<int>(det_idx));
                unmatched_trks.push_back(trk_idx);
            } else {
                matched.push_back({static_cast<int>(det_idx), trk_idx});
            }
        }
    }
    
    // Find unmatched trackers
    for (size_t trk_idx = 0; trk_idx < trackers.size(); trk_idx++) {
        bool found = false;
        for (const auto& match : matched) {
            if (match.second == static_cast<int>(trk_idx)) {
                found = true;
                break;
            }
        }
        if (!found) {
            bool already_unmatched = false;
            for (int unmatched_trk : unmatched_trks) {
                if (unmatched_trk == static_cast<int>(trk_idx)) {
                    already_unmatched = true;
                    break;
                }
            }
            if (!already_unmatched) {
                unmatched_trks.push_back(static_cast<int>(trk_idx));
            }
        }
    }
}