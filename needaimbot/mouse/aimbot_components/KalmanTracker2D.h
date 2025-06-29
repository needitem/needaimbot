#ifndef KALMAN_TRACKER_2D_H
#define KALMAN_TRACKER_2D_H

#include <chrono>
#include "../modules/eigen/include/Eigen/Dense"

class KalmanTracker2D
{
private:
    // State vector: [x, y, vx, vy] - position and velocity
    Eigen::Vector4f state;
    
    // Covariance matrix
    Eigen::Matrix4f P;
    
    // State transition matrix
    Eigen::Matrix4f F;
    
    // Process noise covariance
    Eigen::Matrix4f Q;
    
    // Measurement matrix (we only observe position)
    Eigen::Matrix<float, 2, 4> H;
    
    // Measurement noise covariance
    Eigen::Matrix2f R;
    
    std::chrono::steady_clock::time_point last_time_point;
    bool initialized;
    
    // Prediction parameters
    float process_noise;
    float measurement_noise;
    
public:
    KalmanTracker2D(float process_noise = 10.0f, float measurement_noise = 5.0f);
    
    // Update with new measurement
    Eigen::Vector2f update(const Eigen::Vector2f& measurement);
    
    // Get predicted position for future time
    Eigen::Vector2f predict(float future_time_ms = 0.0f);
    
    // Get current velocity
    Eigen::Vector2f getVelocity() const;
    
    // Reset tracker
    void reset();
    
    // Update noise parameters
    void updateParameters(float process_noise, float measurement_noise);
    
    // Check if tracker is initialized
    bool isInitialized() const { return initialized; }
};

#endif