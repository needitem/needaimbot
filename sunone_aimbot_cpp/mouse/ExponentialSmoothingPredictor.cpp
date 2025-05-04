#include "ExponentialSmoothingPredictor.h"
#include <stdexcept>
#include <cmath> // For std::abs

ExponentialSmoothingPredictor::ExponentialSmoothingPredictor()
    : alpha_(0.5f), // Default smoothing factor
      prediction_time_seconds_(0.016f), // Default prediction time
      smoothed_position_({0.0f, 0.0f}),
      smoothed_velocity_({0.0f, 0.0f}), // For Holt's method (optional trend smoothing)
      last_timestamp_(),
      is_initialized_(false)
{
}

void ExponentialSmoothingPredictor::configure(float alpha, float prediction_ms) {
    if (alpha < 0.01f || alpha > 1.0f) {
        throw std::invalid_argument("Alpha must be between 0.01 and 1.0 for exponential smoothing.");
        // Or clamp the value: alpha_ = std::max(0.01f, std::min(1.0f, alpha));
    }
    alpha_ = alpha;

    if (prediction_ms < 0.0f) {
        prediction_time_seconds_ = 0.0f;
    } else {
        prediction_time_seconds_ = prediction_ms / 1000.0f;
    }
}

void ExponentialSmoothingPredictor::update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) {
    if (!is_initialized_) {
        // First update: initialize smoothed values directly
        smoothed_position_ = position;
        smoothed_velocity_ = {0.0f, 0.0f}; // Assume zero initial velocity
        last_timestamp_ = timestamp;
        is_initialized_ = true;
        return;
    }

    // Calculate time difference
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - last_timestamp_);
    float dt_seconds = static_cast<float>(time_diff.count()) / 1e6f;

    if (dt_seconds < 1e-6) {
        // If time difference is too small, skip update to avoid instability
        return;
    }

    // --- Simple Exponential Smoothing (Level only) ---
    // smoothed_position_.x = alpha_ * position.x + (1.0f - alpha_) * smoothed_position_.x;
    // smoothed_position_.y = alpha_ * position.y + (1.0f - alpha_) * smoothed_position_.y;

    // --- Holt's Method (Level and Trend/Velocity) --- 
    // This version is generally better for prediction as it incorporates velocity.
    // Requires another smoothing factor, beta (let's assume beta = alpha for simplicity now)
    float beta = alpha_; // Or add a separate beta parameter to configure()
    
    Point2D previous_smoothed_position = smoothed_position_;
    Point2D previous_smoothed_velocity = smoothed_velocity_;

    // Update level (smoothed position)
    smoothed_position_.x = alpha_ * position.x + (1.0f - alpha_) * (previous_smoothed_position.x + previous_smoothed_velocity.x * dt_seconds);
    smoothed_position_.y = alpha_ * position.y + (1.0f - alpha_) * (previous_smoothed_position.y + previous_smoothed_velocity.y * dt_seconds);

    // Update trend (smoothed velocity)
    smoothed_velocity_.x = beta * (smoothed_position_.x - previous_smoothed_position.x) / dt_seconds + (1.0f - beta) * previous_smoothed_velocity.x;
    smoothed_velocity_.y = beta * (smoothed_position_.y - previous_smoothed_position.y) / dt_seconds + (1.0f - beta) * previous_smoothed_velocity.y;

    last_timestamp_ = timestamp;
}

Point2D ExponentialSmoothingPredictor::predict() const {
    if (!is_initialized_) {
        return {0.0f, 0.0f}; // Return zero if not initialized
    }

    // Predict using the latest smoothed position and velocity
    Point2D predicted_position;
    predicted_position.x = smoothed_position_.x + smoothed_velocity_.x * prediction_time_seconds_;
    predicted_position.y = smoothed_position_.y + smoothed_velocity_.y * prediction_time_seconds_;

    return predicted_position;
}

void ExponentialSmoothingPredictor::reset() {
    // Reset initialization flag and smoothed values
    is_initialized_ = false;
    smoothed_position_ = {0.0f, 0.0f};
    smoothed_velocity_ = {0.0f, 0.0f};
    // last_timestamp_ will be set on the first update after reset
    // Configured parameters (alpha_, prediction_time_seconds_) remain unchanged
} 