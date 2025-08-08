#include "KalmanFilter2D.h"
#include <iostream>

KalmanFilter2D::KalmanFilter2D() 
    : process_noise_(1.0f)
    , measurement_noise_(10.0f)
    , dt_(0.016f)  // 60 FPS default
    , initialized_(false) {
    
    // Initialize state vector (position and velocity)
    state_ = Eigen::Vector4f::Zero();
    
    // Initialize covariance matrix with high uncertainty
    P_ = Eigen::Matrix4f::Identity() * 1000.0f;
    
    // Initialize measurement matrix (we only measure position)
    H_ = Eigen::Matrix<float, 2, 4>::Zero();
    H_(0, 0) = 1.0f;  // x position
    H_(1, 1) = 1.0f;  // y position
    
    updateSystemMatrices();
    updateNoiseMatrices();
    
    last_update_time_ = std::chrono::steady_clock::now();
}

void KalmanFilter2D::initialize(float process_noise, float measurement_noise, float dt) {
    process_noise_ = process_noise;
    measurement_noise_ = measurement_noise;
    dt_ = dt;
    
    updateSystemMatrices();
    updateNoiseMatrices();
}

void KalmanFilter2D::updateSystemMatrices() {
    // State transition matrix
    // [1  0  dt  0]
    // [0  1  0  dt]
    // [0  0  1   0]
    // [0  0  0   1]
    F_ = Eigen::Matrix4f::Identity();
    F_(0, 2) = dt_;  // x += vx * dt
    F_(1, 3) = dt_;  // y += vy * dt
}

void KalmanFilter2D::updateNoiseMatrices() {
    // Process noise covariance
    // Assumes constant velocity model with some acceleration noise
    Q_ = Eigen::Matrix4f::Zero();
    float dt2 = dt_ * dt_;
    float dt3 = dt2 * dt_;
    float dt4 = dt3 * dt_;
    
    // Position variance increases with dt^4
    Q_(0, 0) = dt4 / 4.0f * process_noise_;
    Q_(1, 1) = dt4 / 4.0f * process_noise_;
    
    // Position-velocity covariance
    Q_(0, 2) = dt3 / 2.0f * process_noise_;
    Q_(1, 3) = dt3 / 2.0f * process_noise_;
    Q_(2, 0) = dt3 / 2.0f * process_noise_;
    Q_(3, 1) = dt3 / 2.0f * process_noise_;
    
    // Velocity variance increases with dt^2
    Q_(2, 2) = dt2 * process_noise_;
    Q_(3, 3) = dt2 * process_noise_;
    
    // Measurement noise covariance
    R_ = Eigen::Matrix2f::Identity() * measurement_noise_;
}

Eigen::Vector2f KalmanFilter2D::update(const Eigen::Vector2f& measurement) {
    // Calculate adaptive dt based on actual time elapsed
    auto now = std::chrono::steady_clock::now();
    float actual_dt = std::chrono::duration<float>(now - last_update_time_).count();
    last_update_time_ = now;
    
    // Update dt if significantly different (> 20% change)
    if (std::abs(actual_dt - dt_) / dt_ > 0.2f && actual_dt > 0.001f && actual_dt < 1.0f) {
        dt_ = actual_dt;
        updateSystemMatrices();
        updateNoiseMatrices();
    }
    
    if (!initialized_) {
        // First measurement - initialize state
        state_(0) = measurement(0);  // x
        state_(1) = measurement(1);  // y
        state_(2) = 0.0f;  // vx
        state_(3) = 0.0f;  // vy
        initialized_ = true;
        return measurement;
    }
    
    // Prediction step
    // x_k|k-1 = F * x_k-1|k-1
    Eigen::Vector4f predicted_state = F_ * state_;
    
    // P_k|k-1 = F * P_k-1|k-1 * F^T + Q
    Eigen::Matrix4f predicted_P = F_ * P_ * F_.transpose() + Q_;
    
    // Innovation (measurement residual)
    // y_k = z_k - H * x_k|k-1
    Eigen::Vector2f innovation = measurement - H_ * predicted_state;
    
    // Innovation covariance
    // S = H * P_k|k-1 * H^T + R
    Eigen::Matrix2f S = H_ * predicted_P * H_.transpose() + R_;
    
    // Kalman gain
    // K = P_k|k-1 * H^T * S^-1
    Eigen::Matrix<float, 4, 2> K = predicted_P * H_.transpose() * S.inverse();
    
    // Update step
    // x_k|k = x_k|k-1 + K * y_k
    state_ = predicted_state + K * innovation;
    
    // P_k|k = (I - K * H) * P_k|k-1
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    P_ = (I - K * H_) * predicted_P;
    
    // Return filtered position
    return Eigen::Vector2f(state_(0), state_(1));
}

Eigen::Vector2f KalmanFilter2D::getPredictedPosition(float lookahead_time) const {
    if (!initialized_) {
        return Eigen::Vector2f::Zero();
    }
    
    // Predict future position based on current state and velocity
    float pred_x = state_(0) + state_(2) * lookahead_time;
    float pred_y = state_(1) + state_(3) * lookahead_time;
    
    return Eigen::Vector2f(pred_x, pred_y);
}

Eigen::Vector2f KalmanFilter2D::getVelocity() const {
    if (!initialized_) {
        return Eigen::Vector2f::Zero();
    }
    
    return Eigen::Vector2f(state_(2), state_(3));
}

void KalmanFilter2D::reset() {
    initialized_ = false;
    state_ = Eigen::Vector4f::Zero();
    P_ = Eigen::Matrix4f::Identity() * 1000.0f;
    last_update_time_ = std::chrono::steady_clock::now();
}