#include "LinearRegressionPredictor.h"
#include <stdexcept>
#include <numeric> // For std::accumulate

LinearRegressionPredictor::LinearRegressionPredictor()
    : num_points_to_use_(10), // Default number of points
      prediction_time_seconds_(0.016f) // Default prediction time
{
}

void LinearRegressionPredictor::configure(int num_past_points, float prediction_ms) {
    if (num_past_points < 2) {
        // Need at least 2 points to fit a line
        throw std::invalid_argument("Number of past points must be at least 2 for linear regression.");
        // Or set to a default minimum, e.g., num_points_to_use_ = 2;
    }
    num_points_to_use_ = num_past_points;

    if (prediction_ms < 0.0f) {
        prediction_time_seconds_ = 0.0f;
    } else {
        prediction_time_seconds_ = prediction_ms / 1000.0f;
    }
}

void LinearRegressionPredictor::update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) {
    history_.push_back({position, timestamp});

    // Keep only the last N points
    if (history_.size() > static_cast<size_t>(num_points_to_use_)) {
        history_.pop_front();
    }
}

Point2D LinearRegressionPredictor::predict() const {
    if (history_.size() < 2) {
        // Not enough data to perform regression
        return history_.empty() ? Point2D{0.0f, 0.0f} : history_.back().position;
    }

    // Perform linear regression separately for X and Y coordinates against time
    // Reference time: time of the first point in the current history window
    auto first_time = history_.front().timestamp;

    float sum_t = 0.0f, sum_t2 = 0.0f, sum_tx = 0.0f, sum_ty = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
    size_t n = history_.size();

    for (const auto& entry : history_) {
        float t = std::chrono::duration_cast<std::chrono::microseconds>(entry.timestamp - first_time).count() / 1e6f;
        sum_t += t;
        sum_t2 += t * t;
        sum_x += entry.position.x;
        sum_y += entry.position.y;
        sum_tx += t * entry.position.x;
        sum_ty += t * entry.position.y;
    }

    float denominator = n * sum_t2 - sum_t * sum_t;

    Point2D predicted_position;
    auto last_entry = history_.back();
    float last_t = std::chrono::duration_cast<std::chrono::microseconds>(last_entry.timestamp - first_time).count() / 1e6f;
    float predict_t = last_t + prediction_time_seconds_;

    if (std::abs(denominator) > 1e-6) { // Avoid division by zero
        // Calculate slope (velocity) and intercept for X
        float slope_x = (n * sum_tx - sum_t * sum_x) / denominator;
        float intercept_x = (sum_x - slope_x * sum_t) / n;
        predicted_position.x = intercept_x + slope_x * predict_t;

        // Calculate slope (velocity) and intercept for Y
        float slope_y = (n * sum_ty - sum_t * sum_y) / denominator;
        float intercept_y = (sum_y - slope_y * sum_t) / n;
        predicted_position.y = intercept_y + slope_y * predict_t;
    } else {
        // If denominator is too small (e.g., all timestamps are the same),
        // return the last known position
        predicted_position = last_entry.position;
    }

    return predicted_position;
}

void LinearRegressionPredictor::reset() {
    // Clear the history of data points
    history_.clear();
    // Configured parameters (num_points_to_use_, prediction_time_seconds_) remain unchanged
} 