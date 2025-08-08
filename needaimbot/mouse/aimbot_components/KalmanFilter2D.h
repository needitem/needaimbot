#pragma once

#include "../../modules/eigen/include/Eigen/Dense"
#include <chrono>

class KalmanFilter2D {
public:
    // State vector: [x, y, vx, vy]
    // x, y: position
    // vx, vy: velocity
    
    KalmanFilter2D();
    
    // Initialize filter with tunable parameters
    void initialize(float process_noise = 1.0f, 
                   float measurement_noise = 10.0f,
                   float dt = 0.016f);  // 60 FPS default
    
    // Update the filter with a new measurement
    Eigen::Vector2f update(const Eigen::Vector2f& measurement);
    
    // Get predicted position with lookahead time
    Eigen::Vector2f getPredictedPosition(float lookahead_time = 0.0f) const;
    
    // Get current velocity estimate
    Eigen::Vector2f getVelocity() const;
    
    // Reset the filter state
    void reset();
    
    // Set/Get parameters
    void setProcessNoise(float noise) { process_noise_ = noise; updateNoiseMatrices(); }
    void setMeasurementNoise(float noise) { measurement_noise_ = noise; updateNoiseMatrices(); }
    void setDeltaTime(float dt) { dt_ = dt; updateSystemMatrices(); }
    
    float getProcessNoise() const { return process_noise_; }
    float getMeasurementNoise() const { return measurement_noise_; }
    float getDeltaTime() const { return dt_; }
    
    // Check if filter is initialized with at least one measurement
    bool isInitialized() const { return initialized_; }
    
private:
    // State vector [x, y, vx, vy]
    Eigen::Vector4f state_;
    
    // Error covariance matrix
    Eigen::Matrix4f P_;
    
    // State transition matrix
    Eigen::Matrix4f F_;
    
    // Measurement matrix (we only measure position)
    Eigen::Matrix<float, 2, 4> H_;
    
    // Process noise covariance
    Eigen::Matrix4f Q_;
    
    // Measurement noise covariance
    Eigen::Matrix2f R_;
    
    // Parameters
    float process_noise_;
    float measurement_noise_;
    float dt_;
    
    bool initialized_;
    
    // Update matrices based on parameters
    void updateSystemMatrices();
    void updateNoiseMatrices();
    
    // For adaptive dt calculation
    std::chrono::steady_clock::time_point last_update_time_;
};