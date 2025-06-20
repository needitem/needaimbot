#ifndef VELOCITY_PREDICTOR_H
#define VELOCITY_PREDICTOR_H

#include "IPredictor.h" 
#include <chrono> 



class VelocityPredictor : public IPredictor { 
public:
    VelocityPredictor();
    ~VelocityPredictor() override = default; 

    
    void configure(float prediction_ms);

    
    void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    float prediction_time_seconds_;
    Point2D last_position_;
    Point2D current_velocity_;
    std::chrono::steady_clock::time_point last_timestamp_;
    bool has_previous_update_; 
};

#endif 
