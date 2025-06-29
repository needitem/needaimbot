#include "PredictiveController2D.h"
#include <cmath>
#include <algorithm>

PredictiveController2D::PredictiveController2D(
    float kp_x, float ki_x, float kd_x, 
    float kp_y, float ki_y, float kd_y, 
    float derivative_smoothing_factor,
    float prediction_time_ms,
    float process_noise,
    float measurement_noise
) : kalman_tracker(process_noise, measurement_noise),
    pid_controller(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, derivative_smoothing_factor),
    prediction_time_ms(prediction_time_ms),
    velocity_scale(1.0f),
    use_prediction(true),
    high_velocity_threshold(500.0f), // pixels/second
    low_velocity_threshold(100.0f)   // pixels/second
{
    last_update_time = std::chrono::steady_clock::now();
}

Eigen::Vector2f PredictiveController2D::calculate(const Eigen::Vector2f& target_position, const Eigen::Vector2f& current_crosshair)
{
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float, std::milli>(now - last_update_time).count();
    last_update_time = now;
    
    // Update Kalman tracker with target position
    Eigen::Vector2f filtered_position = kalman_tracker.update(target_position);
    
    Eigen::Vector2f aim_target = filtered_position;
    
    if (use_prediction && kalman_tracker.isInitialized()) {
        // Get current velocity
        Eigen::Vector2f velocity = kalman_tracker.getVelocity();
        float velocity_magnitude = velocity.norm();
        
        // Adaptive prediction time based on velocity
        float adaptive_prediction_time = prediction_time_ms;
        
        if (velocity_magnitude > high_velocity_threshold) {
            // For fast targets, use longer prediction time
            adaptive_prediction_time = prediction_time_ms * 1.5f;
        } else if (velocity_magnitude < low_velocity_threshold) {
            // For slow targets, use shorter prediction time
            adaptive_prediction_time = prediction_time_ms * 0.5f;
        }
        
        // Get predicted position
        Eigen::Vector2f predicted_position = kalman_tracker.predict(adaptive_prediction_time);
        
        // Blend filtered and predicted positions based on velocity confidence
        float prediction_weight = std::min(velocity_magnitude / high_velocity_threshold, 1.0f);
        prediction_weight = std::pow(prediction_weight, 0.5f); // Square root for smoother transition
        
        aim_target = filtered_position * (1.0f - prediction_weight) + predicted_position * prediction_weight;
    }
    
    // Calculate error for PID controller
    Eigen::Vector2f error = aim_target - current_crosshair;
    
    // Use PID controller to calculate control output
    Eigen::Vector2f control_output = pid_controller.calculate(error);
    
    return control_output;
}

void PredictiveController2D::reset()
{
    kalman_tracker.reset();
    pid_controller.reset();
    last_update_time = std::chrono::steady_clock::now();
}

void PredictiveController2D::updatePIDParameters(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y, float derivative_smoothing_factor)
{
    pid_controller.updateSeparatedParameters(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, derivative_smoothing_factor);
}

void PredictiveController2D::updateKalmanParameters(float process_noise, float measurement_noise)
{
    kalman_tracker.updateParameters(process_noise, measurement_noise);
}

void PredictiveController2D::updatePredictionParameters(float prediction_time_ms, float velocity_scale, bool use_prediction)
{
    this->prediction_time_ms = prediction_time_ms;
    this->velocity_scale = velocity_scale;
    this->use_prediction = use_prediction;
}

Eigen::Vector2f PredictiveController2D::getCurrentVelocity() const
{
    return kalman_tracker.getVelocity();
}

Eigen::Vector2f PredictiveController2D::getPredictedPosition() const
{
    if (!kalman_tracker.isInitialized()) {
        return Eigen::Vector2f::Zero();
    }
    return kalman_tracker.predict(prediction_time_ms);
}