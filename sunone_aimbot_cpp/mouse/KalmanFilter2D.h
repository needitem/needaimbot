#ifndef KALMAN_FILTER_2D_H
#define KALMAN_FILTER_2D_H

// Use the relative path consistent with mouse.h
#include "../modules/eigen/include/Eigen/Dense"

// Constants for Kalman filter - Consider making these configurable or members if needed
constexpr float VEL_NOISE_FACTOR = 2.5f;
constexpr float ACC_NOISE_FACTOR = 4.0f;

class KalmanFilter2D
{
public:
    KalmanFilter2D(float process_noise_q, float measurement_noise_r);

    void predict(float dt);
    void update(const Eigen::Vector2f &measurement);
    void reset();
    void updateParameters(float process_noise_q, float measurement_noise_r);

    const Eigen::Matrix<float, 6, 1> &getState() const { return x; }

private:
    void initializeMatrices(float process_noise_q, float measurement_noise_r);

    // State vector [x, y, vx, vy, ax, ay]
    Eigen::Matrix<float, 6, 1> x;

    // State transition matrix
    Eigen::Matrix<float, 6, 6> A;

    // Measurement matrix
    Eigen::Matrix<float, 2, 6> H;

    // Process noise covariance matrix
    Eigen::Matrix<float, 6, 6> Q;

    // Measurement noise covariance matrix
    Eigen::Matrix2f R;

    // Estimate error covariance matrix
    Eigen::Matrix<float, 6, 6> P;
};

#endif // KALMAN_FILTER_2D_H 