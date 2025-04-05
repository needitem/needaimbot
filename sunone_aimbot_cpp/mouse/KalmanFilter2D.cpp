#include "KalmanFilter2D.h"
#include <cmath> // For std::sqrt
#include <algorithm> // For std::clamp (if needed later, although predict uses Eigen's clamp)

// Constants defined in header, remove from here if truly global
// constexpr float VEL_NOISE_FACTOR = 2.5f; 
// ACC_NOISE_FACTOR is removed as acceleration is no longer part of the state

void KalmanFilter2D::initializeMatrices(float process_noise_q, float measurement_noise_r)
{
    // Changed Q to 4x4
    Q = Eigen::Matrix<float, 4, 4>::Identity() * process_noise_q;
    
    // Scale noise for velocity components only
    Q(2, 2) = process_noise_q * VEL_NOISE_FACTOR;
    Q(3, 3) = process_noise_q * VEL_NOISE_FACTOR;
    // Removed acceleration noise scaling
    
    R = Eigen::Matrix2f::Identity() * measurement_noise_r;
}

KalmanFilter2D::KalmanFilter2D(float process_noise_q, float measurement_noise_r)
{
    // Initialize state transition matrix (A) - Changed to 4x4
    A = Eigen::Matrix<float, 4, 4>::Identity();

    // Initialize measurement matrix (H) - Changed to 2x4
    H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.0f; // Measure x position
    H(1, 1) = 1.0f; // Measure y position

    // Initialize noise and covariance matrices
    initializeMatrices(process_noise_q, measurement_noise_r);
    // Changed P to 4x4
    P = Eigen::Matrix<float, 4, 4>::Identity(); // Initial estimate error covariance
    // Changed x to 4x1
    x = Eigen::Matrix<float, 4, 1>::Zero();      // Initial state estimate (at origin, zero velocity)
}

void KalmanFilter2D::predict(float dt)
{
    // Update state transition matrix A based on time delta dt
    // Assumes constant velocity model now
    A(0, 2) = dt;             // x = x + vx*dt
    // Removed A(0, 4)
    A(1, 3) = dt;             // y = y + vy*dt
    // Removed A(1, 5)
    // Removed A(2, 4)
    // Removed A(3, 5)

    // Predict next state
    x = A * x;
    // Predict next estimate error covariance
    P = A * P * A.transpose() + Q;
}

void KalmanFilter2D::update(const Eigen::Vector2f &measurement)
{
    // Calculate Kalman gain (K)
    Eigen::Matrix2f S = H * P * H.transpose() + R; // Innovation covariance (remains 2x2)

    // Directly calculate 2x2 inverse for S
    float detS = S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0);
    Eigen::Matrix2f S_inv;
    // Avoid division by zero or near-zero determinant
    if (std::abs(detS) < 1e-6f) {
        // Handle singularity: Skip update
        return; 
    }
    float invDetS = 1.0f / detS;
    S_inv(0, 0) =  S(1, 1) * invDetS;
    S_inv(0, 1) = -S(0, 1) * invDetS;
    S_inv(1, 0) = -S(1, 0) * invDetS;
    S_inv(1, 1) =  S(0, 0) * invDetS;

    // Kalman gain K is now 4x2
    Eigen::Matrix<float, 4, 2> K = P * H.transpose() * S_inv;

    // Update state estimate with measurement
    Eigen::Vector2f y = measurement - H * x; // Measurement residual (innovation)
    x = x + K * y;

    // Update estimate error covariance - P is 4x4, K is 4x2, H is 2x4
    // Identity matrix needs to be 4x4
    P = (Eigen::Matrix<float, 4, 4>::Identity() - K * H) * P;
}

void KalmanFilter2D::reset()
{
    // Reset state estimate to zero - x is 4x1
    x = Eigen::Matrix<float, 4, 1>::Zero();
    // Reset estimate error covariance to identity (high uncertainty) - P is 4x4
    P = Eigen::Matrix<float, 4, 4>::Identity();
}

void KalmanFilter2D::updateParameters(float process_noise_q, float measurement_noise_r)
{
    // Re-initialize noise matrices Q and R with new parameters
    initializeMatrices(process_noise_q, measurement_noise_r);
} 