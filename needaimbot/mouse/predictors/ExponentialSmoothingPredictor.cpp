#include "ExponentialSmoothingPredictor.h"
#include <stdexcept>
#include <cmath> 

ExponentialSmoothingPredictor::ExponentialSmoothingPredictor()
    : alpha_(0.5f), 
      prediction_time_seconds_(0.016f), 
      smoothed_position_({0.0f, 0.0f}),
      smoothed_velocity_({0.0f, 0.0f}), 
      last_timestamp_(),
      is_initialized_(false),
      beta_(0.5f)
{
}

void ExponentialSmoothingPredictor::configure(float alpha, float beta, float prediction_ms) {
    if (alpha < 0.01f || alpha > 1.0f || beta < 0.01f || beta > 1.0f) {
        throw std::invalid_argument("Alpha and Beta must be between 0.01 and 1.0 for exponential smoothing.");
    }
    alpha_ = alpha;
    beta_ = beta;

    if (prediction_ms < 0.0f) {
        prediction_time_seconds_ = 0.0f;
    } else {
        prediction_time_seconds_ = prediction_ms / 1000.0f;
    }
}

void ExponentialSmoothingPredictor::update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) {
    if (!is_initialized_) {
        smoothed_position_ = position;
        smoothed_velocity_ = {0.0f, 0.0f}; 
        last_timestamp_ = timestamp;
        is_initialized_ = true;
        return;
    }

    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - last_timestamp_);
    float dt_seconds = static_cast<float>(time_diff.count()) / 1e6f;

    if (dt_seconds < 1e-6) {
        return;
    }

    float beta_val = beta_;

    Point2D previous_smoothed_position = smoothed_position_;
    Point2D previous_smoothed_velocity = smoothed_velocity_;

    smoothed_position_.x = alpha_ * position.x + (1.0f - alpha_) * (previous_smoothed_position.x + previous_smoothed_velocity.x * dt_seconds);
    smoothed_position_.y = alpha_ * position.y + (1.0f - alpha_) * (previous_smoothed_position.y + previous_smoothed_velocity.y * dt_seconds);

    smoothed_velocity_.x = beta_val * (smoothed_position_.x - previous_smoothed_position.x) / dt_seconds + (1.0f - beta_val) * previous_smoothed_velocity.x;
    smoothed_velocity_.y = beta_val * (smoothed_position_.y - previous_smoothed_position.y) / dt_seconds + (1.0f - beta_val) * previous_smoothed_velocity.y;

    last_timestamp_ = timestamp;
}

Point2D ExponentialSmoothingPredictor::predict() const {
    if (!is_initialized_) {
        return {0.0f, 0.0f}; 
    }

    Point2D predicted_position;
    predicted_position.x = smoothed_position_.x + smoothed_velocity_.x * prediction_time_seconds_;
    predicted_position.y = smoothed_position_.y + smoothed_velocity_.y * prediction_time_seconds_;

    return predicted_position;
}

void ExponentialSmoothingPredictor::reset() {
    is_initialized_ = false;
    smoothed_position_ = {0.0f, 0.0f};
    smoothed_velocity_ = {0.0f, 0.0f};
    
} 
