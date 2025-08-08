#pragma once

#include <vector>
#include <memory>

// Simple rectangle structure to replace OpenCV's Rect
struct SimpleRect {
    float x, y, width, height;
    
    SimpleRect() : x(0), y(0), width(0), height(0) {}
    SimpleRect(float x_, float y_, float w_, float h_) : x(x_), y(y_), width(w_), height(h_) {}
};

// Simple Kalman filter implementation for tracking
class SimpleKalmanTracker {
public:
    SimpleKalmanTracker();
    SimpleKalmanTracker(const SimpleRect& initRect);
    ~SimpleKalmanTracker();
    
    SimpleRect predict();
    void update(const SimpleRect& measurement);
    
    SimpleRect get_state() const { return current_rect; }
    
    static int kf_count;
    
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;
    
private:
    void init_kf(const SimpleRect& initRect);
    
    // State: [x, y, width, height, vx, vy, vw, vh]
    std::vector<float> state;
    std::vector<float> covariance;
    
    SimpleRect current_rect;
    
    // Simple Kalman filter parameters
    float process_noise;
    float measurement_noise;
    float dt;
    
    std::vector<SimpleRect> m_history;
};