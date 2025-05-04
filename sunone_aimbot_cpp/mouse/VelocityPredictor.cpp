#include "VelocityPredictor.h"
#include <stdexcept> // For std::runtime_error

VelocityPredictor::VelocityPredictor() 
    : prediction_time_seconds_(0.016f), // Default to ~16ms
      last_position_({0.0f, 0.0f}),
      current_velocity_({0.0f, 0.0f}),
      has_previous_update_(false)
{
    // Initialize last_timestamp_ to a valid point
    last_timestamp_ = std::chrono::steady_clock::now(); 
}

void VelocityPredictor::configure(float prediction_ms) {
    if (prediction_ms < 0.0f) {
        // Handle error: prediction time cannot be negative
        // Optionally, throw an exception or log an error
        prediction_time_seconds_ = 0.0f;
    } else {
        prediction_time_seconds_ = prediction_ms / 1000.0f; // Convert ms to seconds
    }
}

void VelocityPredictor::update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) {
    if (has_previous_update_) {
        auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - last_timestamp_);
        float dt_seconds = static_cast<float>(time_diff.count()) / 1e6f; // Time difference in seconds

        if (dt_seconds > 1e-6) { // Avoid division by zero or near-zero
            current_velocity_.x = (position.x - last_position_.x) / dt_seconds;
            current_velocity_.y = (position.y - last_position_.y) / dt_seconds;
        } else {
            // If time difference is too small, assume velocity is unchanged or zero
            // current_velocity_ = {0.0f, 0.0f}; // Or keep previous velocity
        }
    } else {
        // First update, cannot calculate velocity yet
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
    // Reset internal state to initial conditions
    last_position_ = {0.0f, 0.0f};
    current_velocity_ = {0.0f, 0.0f};
    last_timestamp_ = std::chrono::steady_clock::now(); // Reset time
    has_previous_update_ = false;
    // prediction_time_seconds_ remains as configured
} 