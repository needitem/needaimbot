#ifndef EXPONENTIAL_SMOOTHING_PREDICTOR_H
#define EXPONENTIAL_SMOOTHING_PREDICTOR_H

#include "IPredictor.h" 
#include <chrono>



class ExponentialSmoothingPredictor : public IPredictor { 
public:
    ExponentialSmoothingPredictor();
    ~ExponentialSmoothingPredictor() override = default; 

    
    void configure(float alpha, float beta, float prediction_ms);

    
    void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    float alpha_; 
    float beta_; 
    
    float prediction_time_seconds_;
    
    Point2D smoothed_position_;
    Point2D smoothed_velocity_; 
    std::chrono::steady_clock::time_point last_timestamp_;
    bool is_initialized_; 
};

#endif 
