#include "SplineKalmanController.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SplineKalmanController::SplineKalmanController()
    : spline_segments_(20)
    , spline_tension_(0.5f)
    , spline_continuity_(0.5f)
    , spline_bias_(0.0f)
    , process_noise_(0.1f)
    , measurement_noise_(1.0f)
    , rng_(std::random_device{}())
    , noise_dist_(0.0f, 0.5f) {
    
    // Initialize Kalman filter state
    state_ = Eigen::Vector4f::Zero();
    covariance_ = Eigen::Matrix4f::Identity() * 100.0f;
    
    // Process covariance matrix
    process_covariance_ = Eigen::Matrix4f::Identity() * process_noise_;
    
    // Measurement matrix (we observe position only)
    measurement_matrix_ = Eigen::Matrix<float, 2, 4>::Zero();
    measurement_matrix_(0, 0) = 1.0f;  // x position
    measurement_matrix_(1, 1) = 1.0f;  // y position
    
    // Measurement covariance
    measurement_covariance_ = Eigen::Matrix2f::Identity() * measurement_noise_;
}

std::vector<std::pair<int, int>> SplineKalmanController::calculatePath(
    float start_x, float start_y, float end_x, float end_y) {
    
    std::vector<std::pair<int, int>> path;
    
    // Generate control points for spline
    auto control_points = generateSplineControlPoints(start_x, start_y, end_x, end_y);
    
    // Generate spline path
    auto spline_path = catmullRomSpline(control_points);
    
    // Apply Kalman filtering to smooth the path
    for (const auto& point : spline_path) {
        // Predict next state
        kalmanPredict(0.016f);  // Assume 60 FPS
        
        // Add measurement noise for realism
        Eigen::Vector2f noisy_measurement = point;
        noisy_measurement(0) += noise_dist_(rng_) * 0.3f;
        noisy_measurement(1) += noise_dist_(rng_) * 0.3f;
        
        // Update with measurement
        kalmanUpdate(noisy_measurement);
        
        // Extract filtered position
        int x = static_cast<int>(std::round(state_(0)));
        int y = static_cast<int>(std::round(state_(1)));
        
        // Only add point if it's different from the last one
        if (path.empty() || path.back().first != x || path.back().second != y) {
            path.push_back({x, y});
        }
    }
    
    return path;
}

std::vector<Eigen::Vector2f> SplineKalmanController::generateSplineControlPoints(
    float start_x, float start_y, float end_x, float end_y) {
    
    std::vector<Eigen::Vector2f> control_points;
    
    // Add extra control point before start for smooth curve
    float dx = end_x - start_x;
    float dy = end_y - start_y;
    control_points.push_back(Eigen::Vector2f(start_x - dx * 0.1f, start_y - dy * 0.1f));
    
    // Start point
    control_points.push_back(Eigen::Vector2f(start_x, start_y));
    
    // Generate intermediate control points with slight randomness
    int num_intermediate = 2 + (rng_() % 2);  // 2 or 3 intermediate points
    for (int i = 1; i <= num_intermediate; ++i) {
        float t = static_cast<float>(i) / (num_intermediate + 1);
        
        // Linear interpolation with perpendicular offset
        float mid_x = start_x + t * dx;
        float mid_y = start_y + t * dy;
        
        // Add perpendicular offset for curve
        float perpendicular_x = -dy * 0.15f * std::sin(t * M_PI);
        float perpendicular_y = dx * 0.15f * std::sin(t * M_PI);
        
        // Add slight randomness
        perpendicular_x += noise_dist_(rng_) * 0.05f * std::abs(dx);
        perpendicular_y += noise_dist_(rng_) * 0.05f * std::abs(dy);
        
        control_points.push_back(Eigen::Vector2f(mid_x + perpendicular_x, mid_y + perpendicular_y));
    }
    
    // End point
    control_points.push_back(Eigen::Vector2f(end_x, end_y));
    
    // Add extra control point after end for smooth curve
    control_points.push_back(Eigen::Vector2f(end_x + dx * 0.1f, end_y + dy * 0.1f));
    
    return control_points;
}

std::vector<Eigen::Vector2f> SplineKalmanController::catmullRomSpline(
    const std::vector<Eigen::Vector2f>& control_points) {
    
    std::vector<Eigen::Vector2f> spline_points;
    
    if (control_points.size() < 4) {
        return spline_points;
    }
    
    // Generate points along the spline
    for (size_t i = 1; i < control_points.size() - 2; ++i) {
        const auto& p0 = control_points[i - 1];
        const auto& p1 = control_points[i];
        const auto& p2 = control_points[i + 1];
        const auto& p3 = control_points[i + 2];
        
        int segments = (i == 1 || i == control_points.size() - 3) ? 
                      spline_segments_ : spline_segments_ / 2;
        
        for (int j = 0; j < segments; ++j) {
            float t = static_cast<float>(j) / segments;
            
            // Hermite interpolation for x and y
            float x = hermiteInterpolate(p0.x(), p1.x(), p2.x(), p3.x(), t, 
                                        spline_tension_, spline_bias_, spline_continuity_);
            float y = hermiteInterpolate(p0.y(), p1.y(), p2.y(), p3.y(), t, 
                                        spline_tension_, spline_bias_, spline_continuity_);
            
            spline_points.push_back(Eigen::Vector2f(x, y));
        }
    }
    
    // Add the final point
    spline_points.push_back(control_points[control_points.size() - 2]);
    
    return spline_points;
}

void SplineKalmanController::kalmanPredict(float dt) {
    // State transition matrix
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F(0, 2) = dt;  // x += vx * dt
    F(1, 3) = dt;  // y += vy * dt
    
    // Predict state
    state_ = F * state_;
    
    // Predict covariance
    covariance_ = F * covariance_ * F.transpose() + process_covariance_;
}

void SplineKalmanController::kalmanUpdate(const Eigen::Vector2f& measurement) {
    // Innovation
    Eigen::Vector2f y = measurement - measurement_matrix_ * state_;
    
    // Innovation covariance
    Eigen::Matrix2f S = measurement_matrix_ * covariance_ * measurement_matrix_.transpose() + measurement_covariance_;
    
    // Kalman gain
    Eigen::Matrix<float, 4, 2> K = covariance_ * measurement_matrix_.transpose() * S.inverse();
    
    // Update state
    state_ = state_ + K * y;
    
    // Update covariance
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    covariance_ = (I - K * measurement_matrix_) * covariance_;
}

float SplineKalmanController::hermiteInterpolate(float y0, float y1, float y2, float y3, 
                                                float mu, float tension, float bias, float continuity) {
    float m0, m1, mu2, mu3;
    float a0, a1, a2, a3;

    mu2 = mu * mu;
    mu3 = mu2 * mu;
    
    m0 = (y1 - y0) * (1 + bias) * (1 - tension) / 2;
    m0 += (y2 - y1) * (1 - bias) * (1 - tension) / 2;
    m1 = (y2 - y1) * (1 + bias) * (1 - tension) / 2;
    m1 += (y3 - y2) * (1 - bias) * (1 - tension) / 2;
    
    float c0 = (1 - continuity) * m0 + continuity * (y2 - y0) / 2;
    float c1 = (1 - continuity) * m1 + continuity * (y3 - y1) / 2;
    
    a0 = 2 * mu3 - 3 * mu2 + 1;
    a1 = mu3 - 2 * mu2 + mu;
    a2 = mu3 - mu2;
    a3 = -2 * mu3 + 3 * mu2;

    return (a0 * y1 + a1 * c0 + a2 * c1 + a3 * y2);
}