#ifndef TARGET_KALMAN_FILTER_H
#define TARGET_KALMAN_FILTER_H

#include <Eigen/Dense>
#include <chrono>

class TargetKalmanFilter {
public:
    TargetKalmanFilter();
    
    // Initialize filter with initial position
    void initialize(float x, float y);
    
    // Update filter with new measurement and get predicted position
    Eigen::Vector2f predict(float measured_x, float measured_y, float prediction_time_ms);
    
    // Reset the filter
    void reset();
    
    // Set noise parameters
    void setProcessNoise(float position_noise, float velocity_noise, float acceleration_noise);
    void setMeasurementNoise(float noise);
    
    // Get current state estimates
    Eigen::Vector2f getPosition() const;
    Eigen::Vector2f getVelocity() const;
    Eigen::Vector2f getAcceleration() const;
    
    // Get prediction confidence (0-1)
    float getConfidence() const;
    
private:
    // State vector: [x, y, vx, vy, ax, ay]
    Eigen::VectorXf state_;
    
    // State covariance matrix
    Eigen::MatrixXf P_;
    
    // Process noise covariance
    Eigen::MatrixXf Q_;
    
    // Measurement noise covariance
    Eigen::MatrixXf R_;
    
    // State transition matrix
    Eigen::MatrixXf F_;
    
    // Measurement matrix
    Eigen::MatrixXf H_;
    
    // Identity matrix
    Eigen::MatrixXf I_;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_update_time_;
    bool initialized_;
    
    // Confidence tracking
    int consecutive_updates_;
    float innovation_magnitude_;
    
    // Update matrices with time delta
    void updateMatrices(float dt);
    
    // Calculate innovation (prediction error)
    Eigen::Vector2f calculateInnovation(float measured_x, float measured_y);
};

#endif // TARGET_KALMAN_FILTER_H