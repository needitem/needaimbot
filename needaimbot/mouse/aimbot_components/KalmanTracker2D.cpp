#include "KalmanTracker2D.h"
#include <cmath>

KalmanTracker2D::KalmanTracker2D(float process_noise, float measurement_noise)
    : process_noise(process_noise), measurement_noise(measurement_noise), initialized(false)
{
    // Initialize state vector [x, y, vx, vy]
    state = Eigen::Vector4f::Zero();
    
    // Initial covariance matrix (high uncertainty)
    P = Eigen::Matrix4f::Identity() * 1000.0f;
    
    // State transition matrix (constant velocity model)
    F = Eigen::Matrix4f::Identity();
    // F will be updated with dt in each iteration
    
    // Measurement matrix (we observe only position)
    H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.0f; // x position
    H(1, 1) = 1.0f; // y position
    
    // Measurement noise covariance
    R = Eigen::Matrix2f::Identity() * (measurement_noise * measurement_noise);
    
    last_time_point = std::chrono::steady_clock::now();
}

Eigen::Vector2f KalmanTracker2D::update(const Eigen::Vector2f& measurement)
{
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float, std::milli>(now - last_time_point).count() * 0.001f;
    last_time_point = now;
    
    // Clamp dt to reasonable values
    dt = std::clamp(dt, 0.001f, 0.1f);
    
    if (!initialized) {
        // Initialize state with first measurement
        state(0) = measurement.x();
        state(1) = measurement.y();
        state(2) = 0.0f; // initial velocity = 0
        state(3) = 0.0f;
        initialized = true;
    } else {
        // Update state transition matrix with current dt
        F(0, 2) = dt; // x = x + vx * dt
        F(1, 3) = dt; // y = y + vy * dt
        
        // Process noise covariance (acceleration uncertainty)
        float dt2 = dt * dt;
        float dt3 = dt2 * dt;
        float dt4 = dt3 * dt;
        
        Q = Eigen::Matrix4f::Zero();
        Q(0, 0) = dt4 * process_noise / 4.0f;   // position noise x
        Q(0, 2) = dt3 * process_noise / 2.0f;   // position-velocity covariance x
        Q(1, 1) = dt4 * process_noise / 4.0f;   // position noise y
        Q(1, 3) = dt3 * process_noise / 2.0f;   // position-velocity covariance y
        Q(2, 0) = dt3 * process_noise / 2.0f;   // velocity-position covariance x
        Q(2, 2) = dt2 * process_noise;          // velocity noise x
        Q(3, 1) = dt3 * process_noise / 2.0f;   // velocity-position covariance y
        Q(3, 3) = dt2 * process_noise;          // velocity noise y
        
        // Prediction step
        state = F * state;
        P = F * P * F.transpose() + Q;
        
        // Update step
        Eigen::Vector2f y = measurement - H * state; // Innovation
        Eigen::Matrix2f S = H * P * H.transpose() + R; // Innovation covariance
        Eigen::Matrix<float, 4, 2> K = P * H.transpose() * S.inverse(); // Kalman gain
        
        state = state + K * y;
        P = (Eigen::Matrix4f::Identity() - K * H) * P;
    }
    
    return Eigen::Vector2f(state(0), state(1));
}

Eigen::Vector2f KalmanTracker2D::predict(float future_time_ms)
{
    if (!initialized) {
        return Eigen::Vector2f::Zero();
    }
    
    float dt = future_time_ms * 0.001f; // Convert ms to seconds
    
    // Predict future position: position + velocity * time
    float predicted_x = state(0) + state(2) * dt;
    float predicted_y = state(1) + state(3) * dt;
    
    return Eigen::Vector2f(predicted_x, predicted_y);
}

Eigen::Vector2f KalmanTracker2D::getVelocity() const
{
    if (!initialized) {
        return Eigen::Vector2f::Zero();
    }
    
    return Eigen::Vector2f(state(2), state(3));
}

void KalmanTracker2D::reset()
{
    state = Eigen::Vector4f::Zero();
    P = Eigen::Matrix4f::Identity() * 1000.0f;
    initialized = false;
    last_time_point = std::chrono::steady_clock::now();
}

void KalmanTracker2D::updateParameters(float process_noise, float measurement_noise)
{
    this->process_noise = process_noise;
    this->measurement_noise = measurement_noise;
    R = Eigen::Matrix2f::Identity() * (measurement_noise * measurement_noise);
}