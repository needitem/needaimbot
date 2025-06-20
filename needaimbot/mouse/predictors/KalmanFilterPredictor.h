#ifndef KALMAN_FILTER_PREDICTOR_H
#define KALMAN_FILTER_PREDICTOR_H

#include "IPredictor.h" 
#include <chrono>
#include "../modules/eigen/include/Eigen/Dense" 



class KalmanFilterPredictor : public IPredictor { 
public:
    KalmanFilterPredictor();
    ~KalmanFilterPredictor() override = default; 

    
    void configure(float q_factor, float r_factor, float p_factor, float prediction_ms);

    
    void update(const Point2D& measurement, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    bool is_initialized_;
    float prediction_time_seconds_;
    std::chrono::steady_clock::time_point last_timestamp_;

    
    Eigen::VectorXf state_;             
    Eigen::MatrixXf covariance_;        
    Eigen::MatrixXf process_noise_;     
    Eigen::MatrixXf measurement_noise_; 
    Eigen::MatrixXf measurement_matrix_;
    Eigen::MatrixXf transition_matrix_; 

    
    float configured_q_factor_;
    float configured_r_factor_;
    float configured_p_factor_;
};

#endif 
