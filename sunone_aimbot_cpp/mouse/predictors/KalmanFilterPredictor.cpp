#include "KalmanFilterPredictor.h"
#include <stdexcept>

// Constructor: Initialize state and matrices
KalmanFilterPredictor::KalmanFilterPredictor()
    : is_initialized_(false),
      state_(4, 1),          // State vector [x, y, vx, vy]
      covariance_(4, 4),     // Covariance matrix P
      process_noise_(4, 4),  // Process noise covariance Q
      measurement_noise_(2, 2),// Measurement noise covariance R
      measurement_matrix_(2, 4),// Measurement matrix H
      transition_matrix_(4, 4) // State transition matrix F
{
    // Initialize matrices to identity or zero as appropriate
    state_.setZero();
    covariance_.setIdentity(); // Initial uncertainty (P)
    process_noise_.setIdentity(); // Placeholder for Q
    measurement_noise_.setIdentity(); // Placeholder for R
    measurement_matrix_.setZero();
    transition_matrix_.setIdentity(); // Placeholder for F

    // Setup H: We only measure position (x, y)
    measurement_matrix_(0, 0) = 1.0f;
    measurement_matrix_(1, 1) = 1.0f;

    // Default noise values (should be configured)
    configure(0.1f, 1.0f, 1.0f, 16.0f); // Default Q, R, P factor, prediction time
}

// Configuration: Set noise parameters and prediction time
void KalmanFilterPredictor::configure(float q_factor, float r_factor, float p_factor, float prediction_ms) {
    // Store factors
    configured_q_factor_ = q_factor;
    configured_r_factor_ = r_factor;
    configured_p_factor_ = p_factor;

    // Q: Process noise - Uncertainty in the model (how much velocity can change)
    // Simplified: Assume noise affects acceleration, which integrates to velocity/position
    // This is a common simplification, more sophisticated models exist.
    float dt = 0.016f; // Assume a nominal dt for Q scaling, refinement needed
    float dt2 = dt * dt;
    float dt3_over_2 = dt2 * dt / 2.0f;
    float dt4_over_4 = dt2 * dt2 / 4.0f;

    process_noise_ << dt4_over_4, 0, dt3_over_2, 0,
                       0, dt4_over_4, 0, dt3_over_2,
                       dt3_over_2, 0, dt2, 0,
                       0, dt3_over_2, 0, dt2;
    process_noise_ *= q_factor; // Scale by user-defined factor

    // R: Measurement noise - Uncertainty in the detection (x, y)
    measurement_noise_ = Eigen::MatrixXf::Identity(2, 2) * r_factor;

    // P: Initial Covariance - If re-configuring, reset initial uncertainty
    // If the filter is already initialized, maybe don't reset P?
    // For now, let's reset it based on p_factor if configuring after init.
    if (!is_initialized_) {
         covariance_ = Eigen::MatrixXf::Identity(4, 4) * configured_p_factor_; // Use stored factor
    } // else: potentially adjust P differently if filter is running
    
    prediction_time_seconds_ = prediction_ms / 1000.0f;

    // Note: last_timestamp_ is only updated in the update step
}

// Update: Incorporate a new measurement
void KalmanFilterPredictor::update(const Point2D& measurement, std::chrono::steady_clock::time_point timestamp) {
    if (!is_initialized_) {
        // Initialize state with first measurement
        state_(0) = measurement.x;
        state_(1) = measurement.y;
        state_(2) = 0.0f; // Assume zero initial velocity
        state_(3) = 0.0f;
        last_timestamp_ = timestamp;
        is_initialized_ = true;
        return;
    }

    // --- Prediction Step --- 
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - last_timestamp_);
    float dt = static_cast<float>(time_diff.count()) / 1e6f; // Time delta in seconds
    if (dt < 1e-6f) dt = 1e-6f; // Avoid zero dt

    // Update State Transition Matrix F based on dt
    transition_matrix_ << 1, 0, dt, 0,
                          0, 1, 0, dt,
                          0, 0, 1, 0,
                          0, 0, 0, 1;
    
    // Predict state: x_pred = F * x
    state_ = transition_matrix_ * state_;
    // Predict covariance: P_pred = F * P * F^T + Q
    // Need to recalculate Q based on the actual dt for better accuracy
    // Re-calculating Q based on actual dt (similar to configure)
    float dt2 = dt * dt;
    float dt3_over_2 = dt2 * dt / 2.0f;
    float dt4_over_4 = dt2 * dt2 / 4.0f;
    Eigen::Matrix4f current_q;
    current_q << dt4_over_4, 0, dt3_over_2, 0,
                 0, dt4_over_4, 0, dt3_over_2,
                 dt3_over_2, 0, dt2, 0,
                 0, dt3_over_2, 0, dt2;
    current_q *= process_noise_(0,0) / (0.016f*0.016f*0.016f*0.016f/4.0f); // Scale based on configured q_factor
    
    covariance_ = transition_matrix_ * covariance_ * transition_matrix_.transpose() + current_q; 

    // --- Measurement Update Step --- 
    Eigen::VectorXf measurement_vec(2);
    measurement_vec << measurement.x, measurement.y;

    // Measurement residual (innovation): y = z - H * x_pred
    Eigen::VectorXf y = measurement_vec - measurement_matrix_ * state_;
    
    // Residual covariance: S = H * P_pred * H^T + R
    Eigen::MatrixXf S = measurement_matrix_ * covariance_ * measurement_matrix_.transpose() + measurement_noise_;
    
    // Optimal Kalman gain: K = P_pred * H^T * S^-1
    Eigen::MatrixXf K = covariance_ * measurement_matrix_.transpose() * S.inverse();
    
    // Update state estimate: x = x_pred + K * y
    state_ = state_ + K * y;
    
    // Update covariance estimate: P = (I - K * H) * P_pred
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(4, 4);
    covariance_ = (I - K * measurement_matrix_) * covariance_;

    last_timestamp_ = timestamp;
}

// Predict: Estimate state at a future time
Point2D KalmanFilterPredictor::predict() const {
    if (!is_initialized_) {
        return {0.0f, 0.0f};
    }

    // Predict future state based on current state and prediction time
    Eigen::MatrixXf predict_F(4, 4);
    float dt = prediction_time_seconds_;
    predict_F << 1, 0, dt, 0,
                 0, 1, 0, dt,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

    Eigen::VectorXf predicted_state = predict_F * state_;

    return {predicted_state(0), predicted_state(1)}; // Return predicted x, y
}

void KalmanFilterPredictor::reset() {
    is_initialized_ = false;
    state_.setZero(); // Reset state vector
    // Reset covariance to initial uncertainty using the configured p_factor
    covariance_ = Eigen::MatrixXf::Identity(4, 4) * configured_p_factor_;
    // last_timestamp_ will be set on the first update after reset
    // Q and R matrices remain based on configured factors
    // F matrix is calculated dynamically in update()
} 