#include "KalmanFilterPredictor.h"
#include <stdexcept>


KalmanFilterPredictor::KalmanFilterPredictor()
    : is_initialized_(false),
      state_(4, 1),          
      covariance_(4, 4),     
      process_noise_(4, 4),  
      measurement_noise_(2, 2),
      measurement_matrix_(2, 4),
      transition_matrix_(4, 4) 
{
    
    state_.setZero();
    covariance_.setIdentity(); 
    process_noise_.setIdentity(); 
    measurement_noise_.setIdentity(); 
    measurement_matrix_.setZero();
    transition_matrix_.setIdentity(); 

    
    measurement_matrix_(0, 0) = 1.0f;
    measurement_matrix_(1, 1) = 1.0f;

    
    configure(0.1f, 1.0f, 1.0f, 16.0f); 
}


void KalmanFilterPredictor::configure(float q_factor, float r_factor, float p_factor, float prediction_ms) {
    
    configured_q_factor_ = q_factor;
    configured_r_factor_ = r_factor;
    configured_p_factor_ = p_factor;

    
    
    
    float dt = 0.016f; 
    float dt2 = dt * dt;
    float dt3_over_2 = dt2 * dt / 2.0f;
    float dt4_over_4 = dt2 * dt2 / 4.0f;

    process_noise_ << dt4_over_4, 0, dt3_over_2, 0,
                       0, dt4_over_4, 0, dt3_over_2,
                       dt3_over_2, 0, dt2, 0,
                       0, dt3_over_2, 0, dt2;
    process_noise_ *= q_factor; 

    
    measurement_noise_ = Eigen::MatrixXf::Identity(2, 2) * r_factor;

    
    
    
    if (!is_initialized_) {
         covariance_ = Eigen::MatrixXf::Identity(4, 4) * configured_p_factor_; 
    } 
    
    prediction_time_seconds_ = prediction_ms / 1000.0f;

    
}


void KalmanFilterPredictor::update(const Point2D& measurement, std::chrono::steady_clock::time_point timestamp) {
    if (!is_initialized_) {
        
        state_(0) = measurement.x;
        state_(1) = measurement.y;
        state_(2) = 0.0f; 
        state_(3) = 0.0f;
        last_timestamp_ = timestamp;
        is_initialized_ = true;
        return;
    }

    
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - last_timestamp_);
    float dt = static_cast<float>(time_diff.count()) / 1e6f; 
    if (dt < 1e-6f) dt = 1e-6f; 

    
    transition_matrix_ << 1, 0, dt, 0,
                          0, 1, 0, dt,
                          0, 0, 1, 0,
                          0, 0, 0, 1;
    
    
    state_ = transition_matrix_ * state_;
    
    
    
    float dt2 = dt * dt;
    float dt3_over_2 = dt2 * dt / 2.0f;
    float dt4_over_4 = dt2 * dt2 / 4.0f;
    Eigen::Matrix4f current_q;
    current_q << dt4_over_4, 0, dt3_over_2, 0,
                 0, dt4_over_4, 0, dt3_over_2,
                 dt3_over_2, 0, dt2, 0,
                 0, dt3_over_2, 0, dt2;
    current_q *= process_noise_(0,0) / (0.016f*0.016f*0.016f*0.016f/4.0f); 
    
    covariance_ = transition_matrix_ * covariance_ * transition_matrix_.transpose() + current_q; 

    
    Eigen::VectorXf measurement_vec(2);
    measurement_vec << measurement.x, measurement.y;

    
    Eigen::VectorXf y = measurement_vec - measurement_matrix_ * state_;
    
    
    Eigen::MatrixXf S = measurement_matrix_ * covariance_ * measurement_matrix_.transpose() + measurement_noise_;
    
    
    Eigen::MatrixXf K = covariance_ * measurement_matrix_.transpose() * S.inverse();
    
    
    state_ = state_ + K * y;
    
    
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(4, 4);
    covariance_ = (I - K * measurement_matrix_) * covariance_;

    last_timestamp_ = timestamp;
}


Point2D KalmanFilterPredictor::predict() const {
    if (!is_initialized_) {
        return {0.0f, 0.0f};
    }

    
    Eigen::MatrixXf predict_F(4, 4);
    float dt = prediction_time_seconds_;
    predict_F << 1, 0, dt, 0,
                 0, 1, 0, dt,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

    Eigen::VectorXf predicted_state = predict_F * state_;

    return {predicted_state(0), predicted_state(1)}; 
}

void KalmanFilterPredictor::reset() {
    is_initialized_ = false;
    state_.setZero(); 
    
    covariance_ = Eigen::MatrixXf::Identity(4, 4) * configured_p_factor_;
    
    
    
} 
