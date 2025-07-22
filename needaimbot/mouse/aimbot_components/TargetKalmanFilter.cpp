#include "TargetKalmanFilter.h"
#include <algorithm>
#include <cmath>

TargetKalmanFilter::TargetKalmanFilter() 
    : state_(6), P_(6, 6), Q_(6, 6), R_(2, 2), F_(6, 6), H_(2, 6), I_(6, 6),
      initialized_(false), consecutive_updates_(0), innovation_magnitude_(0.0f) {
    
    // Initialize state vector
    state_.setZero();
    
    // Initialize covariance matrix with high uncertainty
    P_.setIdentity();
    P_ *= 1000.0f;
    
    // Process noise (tuned for typical target movement)
    Q_.setZero();
    setProcessNoise(0.1f, 1.0f, 10.0f); // position, velocity, acceleration noise
    
    // Measurement noise
    R_.setIdentity();
    R_ *= 5.0f; // pixels squared
    
    // Measurement matrix (we only observe position)
    H_.setZero();
    H_(0, 0) = 1.0f; // x position
    H_(1, 1) = 1.0f; // y position
    
    // Identity matrix
    I_.setIdentity();
}

void TargetKalmanFilter::initialize(float x, float y) {
    state_.setZero();
    state_(0) = x;
    state_(1) = y;
    
    // Reset covariance
    P_.setIdentity();
    P_.block<2, 2>(0, 0) *= 10.0f;   // position uncertainty
    P_.block<2, 2>(2, 2) *= 100.0f;  // velocity uncertainty  
    P_.block<2, 2>(4, 4) *= 1000.0f; // acceleration uncertainty
    
    initialized_ = true;
    consecutive_updates_ = 0;
    last_update_time_ = std::chrono::high_resolution_clock::now();
}

void TargetKalmanFilter::updateMatrices(float dt) {
    // State transition matrix for constant acceleration model
    F_.setIdentity();
    
    // Position updates from velocity
    F_(0, 2) = dt;
    F_(1, 3) = dt;
    
    // Position updates from acceleration
    F_(0, 4) = 0.5f * dt * dt;
    F_(1, 5) = 0.5f * dt * dt;
    
    // Velocity updates from acceleration
    F_(2, 4) = dt;
    F_(3, 5) = dt;
    
    // Update process noise based on time delta
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt2 * dt2;
    
    Q_.setZero();
    
    // Position noise
    Q_(0, 0) = dt4 / 4.0f * 10.0f;
    Q_(1, 1) = dt4 / 4.0f * 10.0f;
    
    // Velocity noise
    Q_(2, 2) = dt2 * 1.0f;
    Q_(3, 3) = dt2 * 1.0f;
    
    // Acceleration noise
    Q_(4, 4) = 10.0f;
    Q_(5, 5) = 10.0f;
    
    // Cross-correlations
    Q_(0, 2) = Q_(2, 0) = dt3 / 2.0f * 1.0f;
    Q_(1, 3) = Q_(3, 1) = dt3 / 2.0f * 1.0f;
    Q_(0, 4) = Q_(4, 0) = dt2 / 2.0f * 10.0f;
    Q_(1, 5) = Q_(5, 1) = dt2 / 2.0f * 10.0f;
    Q_(2, 4) = Q_(4, 2) = dt * 10.0f;
    Q_(3, 5) = Q_(5, 3) = dt * 10.0f;
}

Eigen::Vector2f TargetKalmanFilter::calculateInnovation(float measured_x, float measured_y) {
    Eigen::Vector2f measurement(measured_x, measured_y);
    Eigen::Vector2f predicted_measurement = H_ * state_;
    return measurement - predicted_measurement;
}

Eigen::Vector2f TargetKalmanFilter::predict(float measured_x, float measured_y, float prediction_time_ms) {
    if (!initialized_) {
        initialize(measured_x, measured_y);
        return Eigen::Vector2f(measured_x, measured_y);
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - last_update_time_).count();
    last_update_time_ = now;
    
    // Clamp dt to reasonable values
    dt = std::clamp(dt, 0.001f, 0.1f);
    
    // Update state transition and process noise matrices
    updateMatrices(dt);
    
    // Prediction step
    Eigen::VectorXf predicted_state = F_ * state_;
    Eigen::MatrixXf predicted_P = F_ * P_ * F_.transpose() + Q_;
    
    // Measurement update step
    Eigen::Vector2f innovation = calculateInnovation(measured_x, measured_y);
    innovation_magnitude_ = innovation.norm();
    
    // Calculate Kalman gain
    Eigen::MatrixXf S = H_ * predicted_P * H_.transpose() + R_;
    Eigen::MatrixXf K = predicted_P * H_.transpose() * S.inverse();
    
    // Update state and covariance
    state_ = predicted_state + K * innovation;
    P_ = (I_ - K * H_) * predicted_P;
    
    // Limit velocity and acceleration to reasonable values
    const float MAX_VELOCITY = 2000.0f; // pixels/second
    const float MAX_ACCELERATION = 5000.0f; // pixels/second^2
    
    state_(2) = std::clamp(state_(2), -MAX_VELOCITY, MAX_VELOCITY);
    state_(3) = std::clamp(state_(3), -MAX_VELOCITY, MAX_VELOCITY);
    state_(4) = std::clamp(state_(4), -MAX_ACCELERATION, MAX_ACCELERATION);
    state_(5) = std::clamp(state_(5), -MAX_ACCELERATION, MAX_ACCELERATION);
    
    consecutive_updates_++;
    
    // Predict future position
    float pred_time = prediction_time_ms / 1000.0f;
    
    // Use full kinematic equation: pos = current_pos + vel * t + 0.5 * accel * t^2
    float predicted_x = state_(0) + state_(2) * pred_time + 0.5f * state_(4) * pred_time * pred_time;
    float predicted_y = state_(1) + state_(3) * pred_time + 0.5f * state_(5) * pred_time * pred_time;
    
    return Eigen::Vector2f(predicted_x, predicted_y);
}

void TargetKalmanFilter::reset() {
    initialized_ = false;
    consecutive_updates_ = 0;
    state_.setZero();
    P_.setIdentity();
    P_ *= 1000.0f;
}

void TargetKalmanFilter::setProcessNoise(float position_noise, float velocity_noise, float acceleration_noise) {
    // Process noise will be scaled by dt in updateMatrices
    // These are base values
}

void TargetKalmanFilter::setMeasurementNoise(float noise) {
    R_.setIdentity();
    R_ *= noise * noise;
}

Eigen::Vector2f TargetKalmanFilter::getPosition() const {
    return Eigen::Vector2f(state_(0), state_(1));
}

Eigen::Vector2f TargetKalmanFilter::getVelocity() const {
    return Eigen::Vector2f(state_(2), state_(3));
}

Eigen::Vector2f TargetKalmanFilter::getAcceleration() const {
    return Eigen::Vector2f(state_(4), state_(5));
}

float TargetKalmanFilter::getConfidence() const {
    if (consecutive_updates_ < 3) {
        return 0.2f; // Low confidence initially
    }
    
    // Base confidence from update count
    float update_confidence = std::min(1.0f, consecutive_updates_ / 10.0f);
    
    // Reduce confidence if innovation is large (poor prediction)
    float innovation_penalty = std::min(1.0f, innovation_magnitude_ / 50.0f);
    float innovation_confidence = 1.0f - innovation_penalty * 0.5f;
    
    // Reduce confidence if uncertainty is high
    float position_uncertainty = std::sqrt(P_(0, 0) + P_(1, 1));
    float uncertainty_penalty = std::min(1.0f, position_uncertainty / 100.0f);
    float uncertainty_confidence = 1.0f - uncertainty_penalty * 0.3f;
    
    return update_confidence * innovation_confidence * uncertainty_confidence;
}