#include "SORTTracker.h"
#include <algorithm>

int SORTTracker::next_id_ = 0;

SORTTracker::SORTTracker(int max_age, int min_hits, float iou_threshold)
    : max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold) {
    last_update_time_ = std::chrono::steady_clock::now();
}

SORTTracker::~SORTTracker() {
    tracked_objects_.clear();
}

std::vector<TrackedObject> SORTTracker::update(const std::vector<Detection>& detections) {
    
    auto current_time = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(current_time - last_update_time_).count();
    last_update_time_ = current_time;
    
    // Predict new locations of existing trackers
    for (auto& track : tracked_objects_) {
        if (!track.kalman_tracker) {
            std::cerr << "[SORTTracker] ERROR: Null kalman_tracker for track ID " << track.id << std::endl;
            continue;
        }
        
        try {
            SimpleRect predicted_state = track.kalman_tracker->predict();
            track.x = predicted_state.x;
            track.y = predicted_state.y;
            track.width = std::max(1.0f, predicted_state.width);  // Ensure positive width
            track.height = std::max(1.0f, predicted_state.height); // Ensure positive height
            track.updateCenter();
            track.time_since_update++;
            track.age++;
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
            trk_idx < 0 || trk_idx >= static_cast<int>(tracked_objects_.size())) {
            std::cerr << "[SORTTracker] ERROR: Invalid match indices (det_idx=" << det_idx 
                      << ", trk_idx=" << trk_idx << ")" << std::endl;
            continue;
        }
        
        // Convert Detection to SimpleRect (x, y, width, height)
        float det_x = static_cast<float>(detections[det_idx].x);
        float det_y = static_cast<float>(detections[det_idx].y);
        float det_w = static_cast<float>(detections[det_idx].width);
        float det_h = static_cast<float>(detections[det_idx].height);
        SimpleRect det_rect(det_x, det_y, det_w, det_h);
        
        // Store previous center for velocity calculation
        float prev_center_x = tracked_objects_[trk_idx].center_x;
        float prev_center_y = tracked_objects_[trk_idx].center_y;
        
        // Validate kalman tracker
        if (!tracked_objects_[trk_idx].kalman_tracker) {
            std::cerr << "[SORTTracker] ERROR: Null kalman_tracker at index " << trk_idx << std::endl;
            continue;
        }
        
        // Update Kalman filter
        try {
            tracked_objects_[trk_idx].kalman_tracker->update(det_rect);
            tracked_objects_[trk_idx].time_since_update = 0;
            tracked_objects_[trk_idx].confidence = detections[det_idx].confidence;
            tracked_objects_[trk_idx].classId = detections[det_idx].classId;
            
            // Update bbox and center
            SimpleRect updated_state = tracked_objects_[trk_idx].kalman_tracker->get_state();
            tracked_objects_[trk_idx].x = updated_state.x;
            tracked_objects_[trk_idx].y = updated_state.y;
            tracked_objects_[trk_idx].width = std::max(1.0f, updated_state.width);  // Ensure positive width
            tracked_objects_[trk_idx].height = std::max(1.0f, updated_state.height); // Ensure positive height
            tracked_objects_[trk_idx].updateCenter();
        } catch (const std::exception& e) {
            std::cerr << "[SORTTracker] Exception in update for track index " << trk_idx << ": " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SORTTracker] Unknown exception in update for track index " << trk_idx << std::endl;
        }
        
        // Calculate velocity (pixels per second)
        if (dt > 0) {
            tracked_objects_[trk_idx].velocity_x = (tracked_objects_[trk_idx].center_x - prev_center_x) / dt;
            tracked_objects_[trk_idx].velocity_y = (tracked_objects_[trk_idx].center_y - prev_center_y) / dt;
        }
    }
    
    // Create new trackers for unmatched detections (with limit to prevent memory issues)
    const size_t MAX_TRACKED_OBJECTS = 50;  // Limit to prevent excessive memory usage
    
    
    for (int idx : unmatched_dets) {
        // Check if we've reached the maximum number of tracked objects
        if (tracked_objects_.size() >= MAX_TRACKED_OBJECTS) {
            // Maximum tracked objects limit reached
            break;
        }
        
        float det_x = static_cast<float>(detections[idx].x);
        float det_y = static_cast<float>(detections[idx].y);
        float det_w = static_cast<float>(detections[idx].width);
        float det_h = static_cast<float>(detections[idx].height);
        SimpleRect det_rect(det_x, det_y, det_w, det_h);
        
        TrackedObject new_track;
        new_track.id = next_id_++;
        new_track.kalman_tracker = std::make_shared<SimpleKalmanTracker>(det_rect);
        new_track.x = det_rect.x;
        new_track.y = det_rect.y;
        new_track.width = det_rect.width;
        new_track.height = det_rect.height;
        new_track.updateCenter();
        new_track.velocity_x = 0;
        new_track.velocity_y = 0;
        new_track.confidence = detections[idx].confidence;
        new_track.classId = detections[idx].classId;
        new_track.age = 0;
        new_track.time_since_update = 0;
        new_track.detection_timestamp = std::chrono::high_resolution_clock::now();
        
        tracked_objects_.push_back(new_track);
    }
    
    // Remove dead tracks
    size_t tracks_before_removal = tracked_objects_.size();
    auto it = tracked_objects_.begin();
    while (it != tracked_objects_.end()) {
        if (it->time_since_update > max_age_) {
            it = tracked_objects_.erase(it);
        } else {
            ++it;
        }
    }
    size_t tracks_removed = tracks_before_removal - tracked_objects_.size();
    if (tracks_removed > 0) {
    }
    
    // Return only confirmed tracks (with enough hits)
    std::vector<TrackedObject> confirmed_tracks;
    for (const auto& track : tracked_objects_) {
        if (track.kalman_tracker->m_hit_streak >= min_hits_ || track.age <= min_hits_) {
            confirmed_tracks.push_back(track);
        }
    }
    
    return confirmed_tracks;
}

void SORTTracker::reset() {
    tracked_objects_.clear();
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
    
    // Create cost matrix
    std::vector<std::vector<double>> cost_matrix(detections.size(), 
                                                 std::vector<double>(trackers.size()));
    
    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < trackers.size(); j++) {
            try {
                // Boundary check
                if (i >= detections.size() || j >= trackers.size()) {
                    std::cerr << "[SORTTracker] ERROR: Index out of bounds (i=" << i << ", j=" << j << ")" << std::endl;
                    cost_matrix[i][j] = 1.0;  // Max cost (no match)
                    continue;
                }
                
                float det_x = static_cast<float>(detections[i].x);
                float det_y = static_cast<float>(detections[i].y);
                float det_w = static_cast<float>(detections[i].width);
                float det_h = static_cast<float>(detections[i].height);
                
                // Validate dimensions
                if (det_w <= 0 || det_h <= 0 || trackers[j].width <= 0 || trackers[j].height <= 0) {
                    std::cerr << "[SORTTracker] WARNING: Invalid dimensions detected" << std::endl;
                    cost_matrix[i][j] = 1.0;  // Max cost (no match)
                    continue;
                }
                
                float iou = calculateIOU(det_x, det_y, det_w, det_h,
                                        trackers[j].x, trackers[j].y, 
                                        trackers[j].width, trackers[j].height);
                cost_matrix[i][j] = 1.0 - iou;  // Convert IOU to cost
                
                // Debug: print IOU values for first few comparisons (with safety check)
                if (i == 0 && j < 3) {  // Reduced to 3 for safety
                }
            } catch (const std::exception& e) {
                std::cerr << "[SORTTracker] Exception in cost matrix calculation: " << e.what() << std::endl;
                cost_matrix[i][j] = 1.0;  // Max cost (no match)
            } catch (...) {
                std::cerr << "[SORTTracker] Unknown exception in cost matrix calculation" << std::endl;
                cost_matrix[i][j] = 1.0;  // Max cost (no match)
            }
        }
    }
    
    // Hungarian algorithm
    
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    
    try {
        hungarian.Solve(cost_matrix, assignment);
    } catch (const std::exception& e) {
        std::cerr << "[SORTTracker] Exception in Hungarian algorithm: " << e.what() << std::endl;
        // Fall back to simple greedy assignment
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_dets.push_back(static_cast<int>(i));
        }
        for (size_t i = 0; i < trackers.size(); i++) {
            unmatched_trks.push_back(static_cast<int>(i));
        }
        return;
    } catch (...) {
        std::cerr << "[SORTTracker] Unknown exception in Hungarian algorithm" << std::endl;
        // Fall back to simple greedy assignment
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_dets.push_back(static_cast<int>(i));
        }
        for (size_t i = 0; i < trackers.size(); i++) {
            unmatched_trks.push_back(static_cast<int>(i));
        }
        return;
    }
    
    // Filter out matched with low IOU
    for (size_t i = 0; i < detections.size(); i++) {
        try {
            // Safety check for assignment array
            if (i >= assignment.size()) {
                std::cerr << "[SORTTracker] ERROR: Assignment index out of bounds (i=" << i << ", size=" << assignment.size() << ")" << std::endl;
                unmatched_dets.push_back(static_cast<int>(i));
                continue;
            }
            
            if (assignment[i] >= 0 && assignment[i] < static_cast<int>(trackers.size())) {
                // Additional boundary check for cost_matrix
                if (i < cost_matrix.size() && assignment[i] < static_cast<int>(cost_matrix[i].size())) {
                    float iou = 1.0f - static_cast<float>(cost_matrix[i][assignment[i]]);
                    if (iou >= iou_threshold_) {
                        matched.push_back(std::make_pair(static_cast<int>(i), assignment[i]));
                    } else {
                        unmatched_dets.push_back(static_cast<int>(i));
                    }
                } else {
                    std::cerr << "[SORTTracker] ERROR: Cost matrix index out of bounds" << std::endl;
                    unmatched_dets.push_back(static_cast<int>(i));
                }
            } else {
                unmatched_dets.push_back(static_cast<int>(i));
            }
        } catch (const std::exception& e) {
            std::cerr << "[SORTTracker] Exception in filtering matched: " << e.what() << std::endl;
            unmatched_dets.push_back(static_cast<int>(i));
        } catch (...) {
            std::cerr << "[SORTTracker] Unknown exception in filtering matched" << std::endl;
            unmatched_dets.push_back(static_cast<int>(i));
        }
    }
    
    // Find unmatched trackers
    std::vector<bool> tracker_matched(trackers.size(), false);
    for (const auto& match : matched) {
        tracker_matched[match.second] = true;
    }
    
    for (size_t i = 0; i < trackers.size(); i++) {
        if (!tracker_matched[i]) {
            unmatched_trks.push_back(static_cast<int>(i));
        }
    }
}