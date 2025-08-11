#include "ByteTracker.h"
#include "Hungarian.h"
#include <algorithm>
#include <iostream>
#include <set>

ByteTracker::ByteTracker() {
    tracks_.reserve(100);
    tracked_tracks_.reserve(100);
    lost_tracks_.reserve(100);
    removed_tracks_.reserve(100);
}

std::vector<Target> ByteTracker::update(const std::vector<Target>& detections, int frame_id) {
    frame_id_ = frame_id;
    
    // Clear track pools
    tracked_tracks_.clear();
    lost_tracks_.clear();
    removed_tracks_.clear();
    
    // Classify existing tracks
    for (auto& track : tracks_) {
        if (track.state == TrackState::TRACKED) {
            tracked_tracks_.push_back(&track);
        } else if (track.state == TrackState::LOST) {
            lost_tracks_.push_back(&track);
        }
    }
    
    // Predict all tracks
    for (auto& track : tracks_) {
        track.kalman_tracker.predict();
    }
    
    // Separate detections by confidence
    std::vector<Target> high_detections;
    std::vector<Target> low_detections;
    
    for (const auto& det : detections) {
        if (det.confidence >= high_thresh_) {
            high_detections.push_back(det);
        } else if (det.confidence >= track_thresh_) {
            low_detections.push_back(det);
        }
    }
    
    // First association: tracked tracks with high confidence detections
    std::vector<int> matched_tracks, matched_dets, unmatched_tracks, unmatched_dets;
    associateDetectionsToTracks(tracked_tracks_, high_detections, match_thresh_,
                               matched_tracks, matched_dets, unmatched_tracks, unmatched_dets);
    
    // Update matched tracks
    for (size_t i = 0; i < matched_tracks.size(); ++i) {
        updateTrack(tracked_tracks_[matched_tracks[i]], high_detections[matched_dets[i]]);
    }
    
    // Second association: unmatched tracks with low confidence detections
    std::vector<Track*> unmatched_tracked;
    for (int idx : unmatched_tracks) {
        unmatched_tracked.push_back(tracked_tracks_[idx]);
    }
    
    std::vector<int> matched_tracks2, matched_dets2, unmatched_tracks2, unmatched_dets2;
    associateDetectionsToTracks(unmatched_tracked, low_detections, 0.5f,
                               matched_tracks2, matched_dets2, unmatched_tracks2, unmatched_dets2);
    
    // Update tracks matched with low confidence
    for (size_t i = 0; i < matched_tracks2.size(); ++i) {
        updateTrack(unmatched_tracked[matched_tracks2[i]], low_detections[matched_dets2[i]]);
    }
    
    // Mark unmatched tracked tracks as lost
    for (int idx : unmatched_tracks2) {
        auto* track = unmatched_tracked[idx];
        track->time_since_update++;
        if (track->state == TrackState::TRACKED && track->time_since_update > 1) {
            track->state = TrackState::LOST;
        }
    }
    
    // Third association: lost tracks with unmatched high confidence detections
    std::vector<Target> remaining_high_dets;
    for (int idx : unmatched_dets) {
        remaining_high_dets.push_back(high_detections[idx]);
    }
    
    std::vector<int> matched_tracks3, matched_dets3, unmatched_tracks3, unmatched_dets3;
    associateDetectionsToTracks(lost_tracks_, remaining_high_dets, match_thresh_,
                               matched_tracks3, matched_dets3, unmatched_tracks3, unmatched_dets3);
    
    // Re-activate matched lost tracks
    for (size_t i = 0; i < matched_tracks3.size(); ++i) {
        auto* track = lost_tracks_[matched_tracks3[i]];
        updateTrack(track, remaining_high_dets[matched_dets3[i]]);
        track->state = TrackState::TRACKED;
    }
    
    // Create new tracks from remaining high confidence detections
    for (int idx : unmatched_dets3) {
        const auto& det = remaining_high_dets[idx];
        if (det.confidence >= track_thresh_) {
            tracks_.emplace_back(det, next_id_++);
            tracks_.back().start_frame = frame_id;
        }
    }
    
    // Remove tracks that have been lost for too long
    for (auto& track : tracks_) {
        if (track.state == TrackState::LOST && track.time_since_update > max_time_lost_) {
            track.state = TrackState::REMOVED;
        }
    }
    
    // Clean up removed tracks
    removeDeletedTracks();
    
    // Prepare output
    std::vector<Target> result;
    for (const auto& track : tracks_) {
        if (track.state == TrackState::TRACKED && track.hit_count >= min_hits_) {
            Target t = track.target;
            t.id = track.track_id;
            result.push_back(t);
        }
    }
    
    return result;
}

std::vector<std::vector<float>> ByteTracker::calculateIOUMatrix(
    const std::vector<Track*>& tracks,
    const std::vector<Target>& detections) {
    
    size_t num_tracks = tracks.size();
    size_t num_dets = detections.size();
    std::vector<std::vector<float>> iou_matrix(num_tracks, std::vector<float>(num_dets, 0.0f));
    
    for (size_t i = 0; i < num_tracks; ++i) {
        cv::Rect2f track_bbox = tracks[i]->kalman_tracker.getPredictedBBox();
        
        for (size_t j = 0; j < num_dets; ++j) {
            cv::Rect2f det_bbox(detections[j].x, detections[j].y, 
                               detections[j].width, detections[j].height);
            
            // Calculate intersection
            float x1 = std::max(track_bbox.x, det_bbox.x);
            float y1 = std::max(track_bbox.y, det_bbox.y);
            float x2 = std::min(track_bbox.x + track_bbox.width, det_bbox.x + det_bbox.width);
            float y2 = std::min(track_bbox.y + track_bbox.height, det_bbox.y + det_bbox.height);
            
            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            
            // Calculate union
            float area1 = track_bbox.width * track_bbox.height;
            float area2 = det_bbox.width * det_bbox.height;
            float union_area = area1 + area2 - intersection;
            
            // Calculate IOU
            if (union_area > 0) {
                iou_matrix[i][j] = intersection / union_area;
            }
        }
    }
    
    return iou_matrix;
}

void ByteTracker::associateDetectionsToTracks(
    std::vector<Track*>& tracks,
    const std::vector<Target>& detections,
    float thresh,
    std::vector<int>& matched_tracks,
    std::vector<int>& matched_detections,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_detections) {
    
    matched_tracks.clear();
    matched_detections.clear();
    unmatched_tracks.clear();
    unmatched_detections.clear();
    
    if (tracks.empty() || detections.empty()) {
        for (size_t i = 0; i < tracks.size(); ++i) {
            unmatched_tracks.push_back(i);
        }
        for (size_t i = 0; i < detections.size(); ++i) {
            unmatched_detections.push_back(i);
        }
        return;
    }
    
    // Calculate IOU matrix
    auto iou_matrix = calculateIOUMatrix(tracks, detections);
    
    // Convert to cost matrix for Hungarian algorithm (1 - IOU)
    std::vector<std::vector<double>> cost_matrix(tracks.size(), std::vector<double>(detections.size()));
    for (size_t i = 0; i < tracks.size(); ++i) {
        for (size_t j = 0; j < detections.size(); ++j) {
            cost_matrix[i][j] = 1.0 - iou_matrix[i][j];
        }
    }
    
    // Run Hungarian algorithm
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(cost_matrix, assignment);
    
    // Process assignments
    std::set<int> matched_det_set;
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] >= 0 && iou_matrix[i][assignment[i]] >= thresh) {
            matched_tracks.push_back(i);
            matched_detections.push_back(assignment[i]);
            matched_det_set.insert(assignment[i]);
        } else {
            unmatched_tracks.push_back(i);
        }
    }
    
    // Find unmatched detections
    for (size_t j = 0; j < detections.size(); ++j) {
        if (matched_det_set.find(j) == matched_det_set.end()) {
            unmatched_detections.push_back(j);
        }
    }
}

void ByteTracker::updateTrack(Track* track, const Target& det) {
    cv::Rect2f bbox(det.x, det.y, det.width, det.height);
    track->kalman_tracker.update(bbox);
    track->target = det;
    track->target.id = track->track_id;
    track->time_since_update = 0;
    track->hit_count++;
}

void ByteTracker::removeDeletedTracks() {
    auto it = std::remove_if(tracks_.begin(), tracks_.end(),
        [](const Track& track) { return track.state == TrackState::REMOVED; });
    tracks_.erase(it, tracks_.end());
}