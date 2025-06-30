#ifndef PREDICTIVE_CONTROLLER_2D_H
#define PREDICTIVE_CONTROLLER_2D_H

#include "KalmanTracker2D.h"
#include "PIDController2D.h"
#include <chrono>

class PredictiveController2D
{
private:
    KalmanTracker2D kalman_tracker;
    PIDController2D pid_controller;
    
    // Prediction settings
    float prediction_time_ms;
    float velocity_scale;
    bool use_prediction;
    
    // Adaptive parameters
    float high_velocity_threshold;
    float low_velocity_threshold;
    
    std::chrono::steady_clock::time_point last_update_time;
    
public:
    PredictiveController2D(
        float kp_x, float ki_x, float kd_x, 
        float kp_y, float ki_y, float kd_y, 
        float prediction_time_ms = 50.0f,
        float process_noise = 10.0f,
        float measurement_noise = 5.0f
    );
    
    // Main calculation function
    Eigen::Vector2f calculate(const Eigen::Vector2f& target_position, const Eigen::Vector2f& current_crosshair);
    
    // Reset both controllers
    void reset();
    
    // Update parameters
    void updatePIDParameters(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);
    void updateKalmanParameters(float process_noise, float measurement_noise);
    void updatePredictionParameters(float prediction_time_ms, float velocity_scale, bool use_prediction);
    
    // Get current velocity and prediction info
    Eigen::Vector2f getCurrentVelocity() const;
    Eigen::Vector2f getPredictedPosition() const;
    
    // Enable/disable prediction
    void setPredictionEnabled(bool enabled) { use_prediction = enabled; }
    bool isPredictionEnabled() const { return use_prediction; }
};

#endif