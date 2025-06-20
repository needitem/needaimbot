#include "VelocityPredictor.h"
#include <stdexcept>
#include <algorithm>

VelocityPredictor::VelocityPredictor() 
    : prediction_time_seconds_(0.016f), 
      last_position_({0.0f, 0.0f}),
      current_velocity_({0.0f, 0.0f}),
      has_previous_update_(false)
{
    
    last_timestamp_ = std::chrono::steady_clock::now(); 
}

void VelocityPredictor::configure(float prediction_ms) {
    if (prediction_ms < 0.0f) {
        
        
        prediction_time_seconds_ = 0.0f;
    } else {
        prediction_time_seconds_ = prediction_ms / 1000.0f; 
    }
}

void VelocityPredictor::update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) {
    if (has_previous_update_) {
        float dt_seconds = std::chrono::duration<float>(timestamp - last_timestamp_).count();
        // clamp dt to reasonable range to avoid spikes
        dt_seconds = std::clamp(dt_seconds, 1e-6f, 0.1f);

        if (dt_seconds > 1e-6f) {
            // compute raw velocity
            float raw_vx = (position.x - last_position_.x) / dt_seconds;
            float raw_vy = (position.y - last_position_.y) / dt_seconds;
            // smooth velocity to reduce noise (50% smoothing)
            current_velocity_.x = 0.5f * raw_vx + 0.5f * current_velocity_.x;
            current_velocity_.y = 0.5f * raw_vy + 0.5f * current_velocity_.y;
        }
    } else {
        
        current_velocity_ = {0.0f, 0.0f};
        has_previous_update_ = true;
    }

    last_position_ = position;
    last_timestamp_ = timestamp;
}

Point2D VelocityPredictor::predict() const {
    Point2D predicted_position;
    predicted_position.x = last_position_.x + current_velocity_.x * prediction_time_seconds_;
    predicted_position.y = last_position_.y + current_velocity_.y * prediction_time_seconds_;
    return predicted_position;
}

void VelocityPredictor::reset() {
    
    last_position_ = {0.0f, 0.0f};
    current_velocity_ = {0.0f, 0.0f};
    last_timestamp_ = std::chrono::steady_clock::now(); 
    has_previous_update_ = false;
    
} 
