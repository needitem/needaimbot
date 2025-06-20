#ifndef LINEAR_REGRESSION_PREDICTOR_H
#define LINEAR_REGRESSION_PREDICTOR_H

#include "IPredictor.h" 
#include <vector>
#include <deque>
#include <chrono>



class LinearRegressionPredictor : public IPredictor { 
public:
    LinearRegressionPredictor();
    ~LinearRegressionPredictor() override = default; 

    
    void configure(int num_past_points, float prediction_ms);

    
    void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    struct HistoryEntry {
        Point2D position;
        float t; // timestamp in seconds since epoch
    };

    int num_points_to_use_;
    float prediction_time_seconds_;
    std::deque<HistoryEntry> history_; 
    // incremental sums for O(1) regression
    float sum_t_;
    float sum_t2_;
    float sum_tx_;
    float sum_ty_;
    float sum_x_;
    float sum_y_;
};

#endif 
