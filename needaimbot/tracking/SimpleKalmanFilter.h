#ifndef SIMPLE_KALMAN_FILTER_H
#define SIMPLE_KALMAN_FILTER_H

#include "../modules/eigen/include/Eigen/Dense"

/**
 * Simple Kalman Filter for 2D bounding box tracking
 * States: [x, y, w, h, vx, vy, vw, vh] (position, size, and their velocities)
 */
class SimpleKalmanFilter {
public:
    SimpleKalmanFilter();
    SimpleKalmanFilter(float x, float y, float w, float h);
    
    // Predict next state
    void predict();
    
    // Update with measurement
    void update(float x, float y, float w, float h);
    
    // Get current state as bounding box
    void getState(float& x, float& y, float& w, float& h) const;
    
    // Get predicted state
    void getPrediction(float& x, float& y, float& w, float& h) const;

private:
    // State vector: [x, y, w, h, vx, vy, vw, vh]
    Eigen::VectorXf x_;  // State
    Eigen::MatrixXf F_;  // State transition matrix
    Eigen::MatrixXf H_;  // Measurement matrix
    Eigen::MatrixXf P_;  // Error covariance
    Eigen::MatrixXf Q_;  // Process noise covariance
    Eigen::MatrixXf R_;  // Measurement noise covariance
    
    void initializeMatrices();
};

#endif // SIMPLE_KALMAN_FILTER_H