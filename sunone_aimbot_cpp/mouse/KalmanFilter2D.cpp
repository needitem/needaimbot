#include "KalmanFilter2D.h"
#include <cmath> // For std::sqrt
#include <algorithm> // For std::clamp (if needed later, although predict uses Eigen's clamp)

// Constants defined in header, remove from here if truly global
// constexpr float VEL_NOISE_FACTOR = 2.5f; 
// constexpr float ACC_NOISE_FACTOR = 4.0f;

void KalmanFilter2D::initializeMatrices(float process_noise_q, float measurement_noise_r)
{
    Q = Eigen::Matrix<float, 6, 6>::Identity() * process_noise_q;
    
    // Scale noise for velocity and acceleration components
    Q(2, 2) = process_noise_q * VEL_NOISE_FACTOR;
    Q(3, 3) = process_noise_q * VEL_NOISE_FACTOR;
    Q(4, 4) = process_noise_q * ACC_NOISE_FACTOR;
    Q(5, 5) = process_noise_q * ACC_NOISE_FACTOR;
    
    R = Eigen::Matrix2f::Identity() * measurement_noise_r;
}

KalmanFilter2D::KalmanFilter2D(float process_noise_q, float measurement_noise_r)
{
    // Initialize state transition matrix (A)
    A = Eigen::Matrix<float, 6, 6>::Identity();

    // Initialize measurement matrix (H)
    H = Eigen::Matrix<float, 2, 6>::Zero();
    H(0, 0) = 1.0f; // Measure x position
    H(1, 1) = 1.0f; // Measure y position

    // Initialize noise and covariance matrices
    initializeMatrices(process_noise_q, measurement_noise_r);
    P = Eigen::Matrix<float, 6, 6>::Identity(); // Initial estimate error covariance
    x = Eigen::Matrix<float, 6, 1>::Zero();      // Initial state estimate (at origin, zero velocity/accel)
}

void KalmanFilter2D::predict(float dt)
{
    // Update state transition matrix A based on time delta dt
    // Assumes constant acceleration model
    A(0, 2) = dt;             // x = x + vx*dt
    A(0, 4) = 0.5f * dt * dt; // x = x + 0.5*ax*dt^2
    A(1, 3) = dt;             // y = y + vy*dt
    A(1, 5) = 0.5f * dt * dt; // y = y + 0.5*ay*dt^2
    A(2, 4) = dt;             // vx = vx + ax*dt
    A(3, 5) = dt;             // vy = vy + ay*dt

    // Predict next state
    x = A * x;
    // Predict next estimate error covariance
    P = A * P * A.transpose() + Q;
}

void KalmanFilter2D::update(const Eigen::Vector2f &measurement)
{
    // Calculate Kalman gain (K)
    Eigen::Matrix2f S = H * P * H.transpose() + R; // Innovation covariance

    // Directly calculate 2x2 inverse for S = [[s00, s01], [s10, s11]]
    float detS = S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0);
    Eigen::Matrix2f S_inv;
    // Avoid division by zero or near-zero determinant
    if (std::abs(detS) < 1e-6f) {
        // Handle singularity: e.g., use identity, pseudo-inverse, or skip update
        // Using identity might be problematic. Skipping update or using pseudo-inverse might be safer.
        // For now, let's skip the update if determinant is too small.
        // std::cerr << "[Kalman] Warning: Skipping update due to near-zero determinant." << std::endl;
        return; 
        // Or: S_inv = Eigen::Matrix2f::Identity(); // Less ideal
    }
    float invDetS = 1.0f / detS;
    S_inv(0, 0) =  S(1, 1) * invDetS;
    S_inv(0, 1) = -S(0, 1) * invDetS;
    S_inv(1, 0) = -S(1, 0) * invDetS;
    S_inv(1, 1) =  S(0, 0) * invDetS;

    Eigen::Matrix<float, 6, 2> K = P * H.transpose() * S_inv; // Kalman gain using calculated inverse

    // Update state estimate with measurement
    Eigen::Vector2f y = measurement - H * x; // Measurement residual (innovation)
    x = x + K * y;

    // Update estimate error covariance
    P = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * P;
}

void KalmanFilter2D::reset()
{
    // Reset state estimate to zero
    x = Eigen::Matrix<float, 6, 1>::Zero();
    // Reset estimate error covariance to identity (high uncertainty)
    P = Eigen::Matrix<float, 6, 6>::Identity();
}

void KalmanFilter2D::updateParameters(float process_noise_q, float measurement_noise_r)
{
    // Re-initialize noise matrices Q and R with new parameters
    initializeMatrices(process_noise_q, measurement_noise_r);
} 