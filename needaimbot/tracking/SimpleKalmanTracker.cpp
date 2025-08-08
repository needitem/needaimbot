#include "SimpleKalmanTracker.h"
#include <cmath>
#include <algorithm>
#include <iostream>

int SimpleKalmanTracker::kf_count = 0;

SimpleKalmanTracker::SimpleKalmanTracker() {
    init_kf(SimpleRect());
    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = kf_count;
}

SimpleKalmanTracker::SimpleKalmanTracker(const SimpleRect& initRect) {
    init_kf(initRect);
    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = kf_count;
    kf_count++;
    
    // Safety check for reasonable kf_count
    if (kf_count > 10000) {
        std::cerr << "[SimpleKalmanTracker] WARNING: kf_count is very high (" << kf_count << "), possible memory leak" << std::endl;
    }
}

SimpleKalmanTracker::~SimpleKalmanTracker() {
    m_history.clear();
}

void SimpleKalmanTracker::init_kf(const SimpleRect& initRect) {
    // Validate input rect
    if (initRect.width <= 0 || initRect.height <= 0) {
        std::cerr << "[SimpleKalmanTracker] WARNING: Invalid rect dimensions (" 
                  << initRect.width << "x" << initRect.height << ")" << std::endl;
    }
    
    // Initialize state vector [x, y, width, height, vx, vy, vw, vh]
    state.resize(8);
    state[0] = initRect.x + initRect.width / 2;  // center x
    state[1] = initRect.y + initRect.height / 2; // center y
    state[2] = std::max(1.0f, initRect.width);  // Ensure positive width
    state[3] = std::max(1.0f, initRect.height); // Ensure positive height
    state[4] = 0; // vx
    state[5] = 0; // vy
    state[6] = 0; // vw
    state[7] = 0; // vh
    
    // Initialize covariance matrix (simplified)
    covariance.resize(64, 0);
    for (int i = 0; i < 8; i++) {
        covariance[i * 8 + i] = 1000.0f; // diagonal elements
    }
    
    current_rect = initRect;
    
    // Filter parameters
    process_noise = 0.1f;
    measurement_noise = 1.0f;
    dt = 1.0f;
}

SimpleRect SimpleKalmanTracker::predict() {
    // Validate state before prediction
    if (state.size() < 8) {
        std::cerr << "[SimpleKalmanTracker] ERROR: Invalid state size (" << state.size() << ")" << std::endl;
        return current_rect;
    }
    
    // Simple prediction: x = x + vx * dt
    state[0] += state[4] * dt;
    state[1] += state[5] * dt;
    state[2] += state[6] * dt;
    state[3] += state[7] * dt;
    
    // Ensure positive dimensions
    state[2] = std::max(1.0f, state[2]);
    state[3] = std::max(1.0f, state[3]);
    
    // Update age and time since update
    m_age++;
    m_time_since_update++;
    
    // Convert back to rectangle
    current_rect.x = state[0] - state[2] / 2;
    current_rect.y = state[1] - state[3] / 2;
    current_rect.width = state[2];
    current_rect.height = state[3];
    
    // Store in history
    m_history.push_back(current_rect);
    if (m_history.size() > 30) {
        m_history.erase(m_history.begin());
    }
    
    return current_rect;
}

void SimpleKalmanTracker::update(const SimpleRect& measurement) {
    // Validate measurement
    if (measurement.width <= 0 || measurement.height <= 0) {
        std::cerr << "[SimpleKalmanTracker] WARNING: Invalid measurement dimensions (" 
                  << measurement.width << "x" << measurement.height << ")" << std::endl;
        return;
    }
    
    // Validate state
    if (state.size() < 8) {
        std::cerr << "[SimpleKalmanTracker] ERROR: Invalid state size (" << state.size() << ")" << std::endl;
        return;
    }
    
    // Simple update with measurement
    float center_x = measurement.x + measurement.width / 2;
    float center_y = measurement.y + measurement.height / 2;
    
    // Calculate velocity (simple difference)
    if (m_hits > 0) {
        state[4] = (center_x - state[0]) * 0.5f; // smooth velocity update
        state[5] = (center_y - state[1]) * 0.5f;
        state[6] = (measurement.width - state[2]) * 0.5f;
        state[7] = (measurement.height - state[3]) * 0.5f;
    }
    
    // Update state with measurement (simple weighted average)
    float alpha = 0.7f; // measurement weight
    state[0] = alpha * center_x + (1 - alpha) * state[0];
    state[1] = alpha * center_y + (1 - alpha) * state[1];
    state[2] = alpha * measurement.width + (1 - alpha) * state[2];
    state[3] = alpha * measurement.height + (1 - alpha) * state[3];
    
    // Update current rect
    current_rect = measurement;
    
    // Update tracking statistics
    m_hits++;
    m_hit_streak++;
    m_time_since_update = 0;
}