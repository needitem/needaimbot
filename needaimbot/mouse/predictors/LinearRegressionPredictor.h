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
        std::chrono::steady_clock::time_point timestamp;
    };

    int num_points_to_use_;
    float prediction_time_seconds_;
    std::deque<HistoryEntry> history_; 
};

#endif 
