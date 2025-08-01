#ifndef BEZIER_CURVE_CONTROLLER_H
#define BEZIER_CURVE_CONTROLLER_H

#include "../../modules/eigen/include/Eigen/Dense"
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <mutex>

class BezierCurveController {
public:
    BezierCurveController(float speed = 50.0f, float curve_factor = 0.2f);
    
    // Main calculation function - returns the movement commands for this frame
    std::vector<Eigen::Vector2f> calculatePath(const Eigen::Vector2f& error);
    
    // Reset the controller state
    void reset();
    
    // Update parameters
    void updateParameters(float speed, float curve_factor);
    
    // Get/Set speed
    float getSpeed() const { return speed_; }
    void setSpeed(float speed) { speed_ = speed; }
    
    // Get/Set curve factor
    float getCurveFactor() const { return curve_factor_; }
    void setCurveFactor(float factor) { curve_factor_ = factor; }

private:
    float speed_;           // Number of steps in the bezier curve
    float curve_factor_;    // How much curve to add (0.0 - 1.0)
    
    // Thread-safe random number generation
    mutable std::mutex rng_mutex_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_;
    
    // Reusable buffer to avoid frequent allocations
    std::vector<Eigen::Vector2f> movement_buffer_;
    
    // Cache for frequently used config values
    struct ConfigCache {
        float bezier_min_steps;
        float bezier_step_multiplier;
        float bezier_curve_offset_scale;
        float bezier_max_curve_offset;
        float bezier_control1_min;
        float bezier_control1_range;
        float bezier_control2_min;
        float bezier_control2_range;
        float bezier_s_curve_probability;
        float bezier_s_curve_offset1;
        float bezier_s_curve_offset2;
        float bezier_single_offset1;
        float bezier_single_offset2;
        float bezier_min_movement;
        bool needs_update = true;
    } config_cache_;
    
    // Precomputed ease-in-out lookup table
    static constexpr int EASE_LUT_SIZE = 256;
    std::array<float, EASE_LUT_SIZE> ease_lut_;
    
    // Helper functions
    inline float easeInOut(float t) const;
    inline float easeInOutLUT(float t) const;
    
    // Optimized bezier calculation using forward differencing
    void calculateBezierForwardDiff(const Eigen::Vector2f& p0, const Eigen::Vector2f& p1,
                                    const Eigen::Vector2f& p2, const Eigen::Vector2f& p3,
                                    int steps, std::vector<Eigen::Vector2f>& points);
    
    // Calculate a single point on the bezier curve (fallback method)
    inline Eigen::Vector2f calculateBezierPoint(float t, const Eigen::Vector2f& start, 
                                         const Eigen::Vector2f& control1, 
                                         const Eigen::Vector2f& control2, 
                                         const Eigen::Vector2f& end) const;
    
    // Update config cache
    void updateConfigCache();
    
    // Generate random float with thread safety
    float getRandomFloat();
    
    // Initialize lookup tables
    void initializeLUT();
};

#endif // BEZIER_CURVE_CONTROLLER_H