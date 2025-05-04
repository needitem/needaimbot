#ifndef KALMAN_FILTER_PREDICTOR_H
#define KALMAN_FILTER_PREDICTOR_H

#include "IPredictor.h" // Include the interface definition
#include <chrono>
#include "../modules/eigen/include/Eigen/Dense" // Include Eigen library (relative path)

// Point2D is defined in IPredictor.h

class KalmanFilterPredictor : public IPredictor { // Inherit from IPredictor
public:
    KalmanFilterPredictor();
    ~KalmanFilterPredictor() override = default; // Override virtual destructor

    /**
     * @brief Configures the Kalman filter parameters.
     * Note: Specific to KalmanFilterPredictor, called after creation.
     * @param q_factor Factor scaling the process noise covariance (Q).
     * @param r_factor Factor scaling the measurement noise covariance (R).
     * @param p_factor Factor scaling the initial state covariance (P).
     * @param prediction_ms Time ahead to predict in milliseconds.
     */
    void configure(float q_factor, float r_factor, float p_factor, float prediction_ms);

    // Override interface methods
    void update(const Point2D& measurement, std::chrono::steady_clock::time_point timestamp) override;
    Point2D predict() const override;
    void reset() override;

private:
    bool is_initialized_;
    float prediction_time_seconds_;
    std::chrono::steady_clock::time_point last_timestamp_;

    // Kalman filter matrices and state vector using Eigen
    Eigen::VectorXf state_;             // State estimate [x, y, vx, vy]^T (4x1)
    Eigen::MatrixXf covariance_;        // State covariance matrix P (4x4)
    Eigen::MatrixXf process_noise_;     // Process noise covariance Q (4x4) - Stores scaled Q based on q_factor
    Eigen::MatrixXf measurement_noise_; // Measurement noise covariance R (2x2) - Stores scaled R based on r_factor
    Eigen::MatrixXf measurement_matrix_;// Measurement matrix H (2x4)
    Eigen::MatrixXf transition_matrix_; // State transition matrix F (4x4) - Updated based on dt in update()

    // Store configuration factors to potentially reset P or re-calculate Q/R if needed
    float configured_q_factor_;
    float configured_r_factor_;
    float configured_p_factor_;
};

#endif // KALMAN_FILTER_PREDICTOR_H 