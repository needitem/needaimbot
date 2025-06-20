#include "LinearRegressionPredictor.h"
#include <stdexcept>
#include <numeric>
#include <chrono>
#include <cmath>

LinearRegressionPredictor::LinearRegressionPredictor()
    : num_points_to_use_(10), 
      prediction_time_seconds_(0.016f)
{
    sum_t_ = sum_t2_ = sum_tx_ = sum_ty_ = sum_x_ = sum_y_ = 0.0f;
}

void LinearRegressionPredictor::configure(int num_past_points, float prediction_ms) {
    if (num_past_points < 2) {
        throw std::invalid_argument("Number of past points must be at least 2 for linear regression.");
    }
    num_points_to_use_ = num_past_points;

    if (prediction_ms < 0.0f) {
        prediction_time_seconds_ = 0.0f;
    } else {
        prediction_time_seconds_ = prediction_ms / 1000.0f;
    }
}

void LinearRegressionPredictor::update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) {
    float t = std::chrono::duration<float>(timestamp.time_since_epoch()).count();
    HistoryEntry entry{position, t};
    history_.push_back(entry);
    sum_t_ += t;
    sum_t2_ += t * t;
    sum_x_ += position.x;
    sum_y_ += position.y;
    sum_tx_ += t * position.x;
    sum_ty_ += t * position.y;

    if (history_.size() > static_cast<size_t>(num_points_to_use_)) {
        const auto& old = history_.front();
        sum_t_ -= old.t;
        sum_t2_ -= old.t * old.t;
        sum_x_ -= old.position.x;
        sum_y_ -= old.position.y;
        sum_tx_ -= old.t * old.position.x;
        sum_ty_ -= old.t * old.position.y;
        history_.pop_front();
    }
}

Point2D LinearRegressionPredictor::predict() const {
    size_t n = history_.size();
    if (n < 2) {
        return history_.empty() ? Point2D{0.0f, 0.0f} : history_.back().position;
    }
    float last_t = history_.back().t;
    float predict_t = last_t + prediction_time_seconds_;

    float denominator = n * sum_t2_ - sum_t_ * sum_t_;
    Point2D predicted_position;
    if (std::abs(denominator) > 1e-6f) {
        float slope_x = (n * sum_tx_ - sum_t_ * sum_x_) / denominator;
        float intercept_x = (sum_x_ - slope_x * sum_t_) / n;
        predicted_position.x = intercept_x + slope_x * predict_t;
        float slope_y = (n * sum_ty_ - sum_t_ * sum_y_) / denominator;
        float intercept_y = (sum_y_ - slope_y * sum_t_) / n;
        predicted_position.y = intercept_y + slope_y * predict_t;
    } else {
        predicted_position = history_.back().position;
    }
    return predicted_position;
}

void LinearRegressionPredictor::reset() {
    history_.clear();
    sum_t_ = sum_t2_ = sum_tx_ = sum_ty_ = sum_x_ = sum_y_ = 0.0f;
} 
