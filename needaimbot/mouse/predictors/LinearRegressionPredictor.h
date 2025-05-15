#ifndef LINEAR_REGRESSION_PREDICTOR_H
#define LINEAR_REGRESSION_PREDICTOR_H

#include "IPredictor.h" // Include the interface definition
#include <vector>
#include <deque>
#include <chrono>

// Point2D is defined in IPredictor.h

class LinearRegressionPredictor : public IPredictor { // Inherit from IPredictor
public:
    LinearRegressionPredictor();
    ~LinearRegressionPredictor() override = default; // Override virtual destructor

    /**
     * @brief Configures the predictor.
     * Note: Specific to LinearRegressionPredictor, called after creation.
     * @param num_past_points Number of historical points (>= 2) to use for regression.
     * @param prediction_ms Time ahead to predict in milliseconds.
     */
    void configure(int num_past_points, float prediction_ms);

    // Override interface methods
    void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    struct HistoryEntry {
        Point2D position;
        std::chrono::steady_clock::time_point timestamp;
    };

    int num_points_to_use_;
    float prediction_time_seconds_;
    std::deque<HistoryEntry> history_; // Using deque to efficiently add/remove from both ends
};

#endif // LINEAR_REGRESSION_PREDICTOR_H 