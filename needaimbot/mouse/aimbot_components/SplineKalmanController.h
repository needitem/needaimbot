#ifndef SPLINE_KALMAN_CONTROLLER_H
#define SPLINE_KALMAN_CONTROLLER_H

#include <vector>
#include <array>
#include <random>
#include "../modules/eigen/include/Eigen/Core"
#include "../modules/eigen/include/Eigen/Dense"

class SplineKalmanController {
public:
    SplineKalmanController();
    ~SplineKalmanController() = default;

    // Main function to calculate mouse movement
    std::vector<std::pair<int, int>> calculatePath(float start_x, float start_y, 
                                                   float end_x, float end_y);

    // Configuration parameters
    void setSplineSegments(int segments) { spline_segments_ = segments; }
    void setKalmanProcessNoise(float noise) { process_noise_ = noise; }
    void setKalmanMeasurementNoise(float noise) { measurement_noise_ = noise; }
    void setSplineTension(float tension) { spline_tension_ = tension; }
    void setSplineContinuity(float continuity) { spline_continuity_ = continuity; }
    void setSplineBias(float bias) { spline_bias_ = bias; }
    
    // Get configuration
    int getSplineSegments() const { return spline_segments_; }
    float getKalmanProcessNoise() const { return process_noise_; }
    float getKalmanMeasurementNoise() const { return measurement_noise_; }
    float getSplineTension() const { return spline_tension_; }
    float getSplineContinuity() const { return spline_continuity_; }
    float getSplineBias() const { return spline_bias_; }

private:
    // Spline interpolation parameters
    int spline_segments_;
    float spline_tension_;
    float spline_continuity_;
    float spline_bias_;
    
    // Kalman filter state
    Eigen::Vector4f state_;  // [x, y, vx, vy]
    Eigen::Matrix4f covariance_;
    Eigen::Matrix4f process_covariance_;
    Eigen::Matrix<float, 2, 4> measurement_matrix_;
    Eigen::Matrix2f measurement_covariance_;
    
    // Kalman filter parameters
    float process_noise_;
    float measurement_noise_;
    
    // Random number generator for natural movement
    std::mt19937 rng_;
    std::normal_distribution<float> noise_dist_;
    
    // Helper functions
    std::vector<Eigen::Vector2f> generateSplineControlPoints(float start_x, float start_y,
                                                            float end_x, float end_y);
    std::vector<Eigen::Vector2f> catmullRomSpline(const std::vector<Eigen::Vector2f>& control_points);
    void kalmanPredict(float dt);
    void kalmanUpdate(const Eigen::Vector2f& measurement);
    float hermiteInterpolate(float y0, float y1, float y2, float y3, float mu, 
                            float tension, float bias, float continuity);
};

#endif // SPLINE_KALMAN_CONTROLLER_H