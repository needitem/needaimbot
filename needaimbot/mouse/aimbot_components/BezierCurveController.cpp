#include "BezierCurveController.h"
#include "../../AppContext.h"
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

BezierCurveController::BezierCurveController(float speed, float curve_factor)
    : speed_(speed)
    , curve_factor_(curve_factor)
    , gen_(std::random_device{}())
    , dist_(0.0f, 1.0f)
{
    movement_buffer_.reserve(256);  // Pre-allocate reasonable size
    initializeLUT();
    reset();
}

void BezierCurveController::initializeLUT()
{
    // Precompute ease-in-out values for faster lookup
    for (int i = 0; i < EASE_LUT_SIZE; ++i) {
        float t = static_cast<float>(i) / (EASE_LUT_SIZE - 1);
        ease_lut_[i] = easeInOut(t);
    }
}

void BezierCurveController::updateConfigCache()
{
    auto& ctx = AppContext::getInstance();
    config_cache_.bezier_min_steps = ctx.config.bezier_min_steps;
    config_cache_.bezier_step_multiplier = ctx.config.bezier_step_multiplier;
    config_cache_.bezier_curve_offset_scale = ctx.config.bezier_curve_offset_scale;
    config_cache_.bezier_max_curve_offset = ctx.config.bezier_max_curve_offset;
    config_cache_.bezier_control1_min = ctx.config.bezier_control1_min;
    config_cache_.bezier_control1_range = ctx.config.bezier_control1_range;
    config_cache_.bezier_control2_min = ctx.config.bezier_control2_min;
    config_cache_.bezier_control2_range = ctx.config.bezier_control2_range;
    config_cache_.bezier_s_curve_probability = ctx.config.bezier_s_curve_probability;
    config_cache_.bezier_s_curve_offset1 = ctx.config.bezier_s_curve_offset1;
    config_cache_.bezier_s_curve_offset2 = ctx.config.bezier_s_curve_offset2;
    config_cache_.bezier_single_offset1 = ctx.config.bezier_single_offset1;
    config_cache_.bezier_single_offset2 = ctx.config.bezier_single_offset2;
    config_cache_.bezier_min_movement = ctx.config.bezier_min_movement;
    config_cache_.needs_update = false;
}

float BezierCurveController::getRandomFloat()
{
    std::lock_guard<std::mutex> lock(rng_mutex_);
    return dist_(gen_);
}

std::vector<Eigen::Vector2f> BezierCurveController::calculatePath(const Eigen::Vector2f& error)
{
    // Clear and reuse buffer instead of creating new vector
    movement_buffer_.clear();
    
    // If error is too small, don't move
    float error_magnitude = error.norm();
    if (error_magnitude < 1.0f) {
        return movement_buffer_;
    }
    
    // Update config cache if needed
    if (config_cache_.needs_update) {
        updateConfigCache();
    }
    
    // Dynamic step calculation based on distance
    int steps = std::max(static_cast<int>(config_cache_.bezier_min_steps), 
                         std::min(static_cast<int>(speed_), 
                         static_cast<int>(error_magnitude / config_cache_.bezier_step_multiplier)));
    
    // Ensure buffer has enough capacity
    if (movement_buffer_.capacity() < static_cast<size_t>(steps)) {
        movement_buffer_.reserve(steps * 2);  // Reserve extra to avoid frequent reallocations
    }
    
    // Start position is always (0, 0) relative to current position
    Eigen::Vector2f start(0.0f, 0.0f);
    Eigen::Vector2f end = error;
    
    // Calculate curve offset based on distance
    float curve_offset = std::min(error_magnitude * curve_factor_ * config_cache_.bezier_curve_offset_scale, 
                                  config_cache_.bezier_max_curve_offset);
    
    // Generate control points - batch random number generation
    float rand1 = getRandomFloat();
    float rand2 = getRandomFloat();
    float rand3 = getRandomFloat();
    
    float control1_factor = config_cache_.bezier_control1_min + rand1 * config_cache_.bezier_control1_range;
    float control2_factor = config_cache_.bezier_control2_min + rand2 * config_cache_.bezier_control2_range;
    
    Eigen::Vector2f control1 = start + error * control1_factor;
    Eigen::Vector2f control2 = start + error * control2_factor;
    
    // Decide curve type
    bool use_s_curve = (rand3 < config_cache_.bezier_s_curve_probability);
    
    // Pre-calculate absolute error values once
    float abs_error_x = std::abs(error.x());
    float abs_error_y = std::abs(error.y());
    bool horizontal_dominant = abs_error_x > abs_error_y;
    
    // Generate another random for curve side
    float rand4 = getRandomFloat();
    
    if (use_s_curve) {
        // S-curve: control points on opposite sides
        float side1 = (rand4 < 0.5f) ? -1.0f : 1.0f;
        float side2 = -side1;
        
        // Pre-calculate offset values
        float offset1 = curve_offset * side1 * config_cache_.bezier_s_curve_offset1;
        float offset2 = curve_offset * side2 * config_cache_.bezier_s_curve_offset2;
        
        if (horizontal_dominant) {
            control1.y() += offset1;
            control2.y() += offset2;
        } else {
            control1.x() += offset1;
            control2.x() += offset2;
        }
    } else {
        // Single side curve
        float side = (rand4 < 0.5f) ? -1.0f : 1.0f;
        float offset1 = curve_offset * side * config_cache_.bezier_single_offset1;
        float offset2 = curve_offset * side * config_cache_.bezier_single_offset2;
        
        if (horizontal_dominant) {
            control1.y() += offset1;
            control2.y() += offset2;
        } else {
            control1.x() += offset1;
            control2.x() += offset2;
        }
    }
    
    // Use optimized forward differencing for bezier calculation
    calculateBezierForwardDiff(start, control1, control2, end, steps, movement_buffer_);
    
    return movement_buffer_;
}

void BezierCurveController::calculateBezierForwardDiff(const Eigen::Vector2f& p0, const Eigen::Vector2f& p1,
                                                        const Eigen::Vector2f& p2, const Eigen::Vector2f& p3,
                                                        int steps, std::vector<Eigen::Vector2f>& points)
{
    // Forward differencing for cubic bezier - much faster than repeated evaluation
    float dt = 1.0f / steps;
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    
    // Calculate forward differences
    Eigen::Vector2f a = -p0 + 3.0f * p1 - 3.0f * p2 + p3;
    Eigen::Vector2f b = 3.0f * p0 - 6.0f * p1 + 3.0f * p2;
    Eigen::Vector2f c = -3.0f * p0 + 3.0f * p1;
    Eigen::Vector2f d = p0;
    
    // Initial values
    Eigen::Vector2f f = a * dt3 + b * dt2 + c * dt;
    Eigen::Vector2f df = 6.0f * a * dt3 + 2.0f * b * dt2;
    Eigen::Vector2f ddf = 6.0f * a * dt3;
    
    Eigen::Vector2f current_pos = p0;
    Eigen::Vector2f previous_pos = p0;
    
    const float MIN_MOVEMENT_THRESHOLD = config_cache_.bezier_min_movement;
    
    for (int i = 1; i <= steps; i++) {
        // Apply easing using LUT
        float t = static_cast<float>(i) / steps;
        float progress = easeInOutLUT(t);
        
        // Interpolate between linear and bezier based on easing
        Eigen::Vector2f bezier_pos = current_pos + f;
        Eigen::Vector2f linear_pos = p0 + (p3 - p0) * t;
        Eigen::Vector2f final_pos = linear_pos + (bezier_pos - linear_pos) * progress;
        
        // Calculate movement delta
        Eigen::Vector2f delta = final_pos - previous_pos;
        
        // Only add movement if it's above threshold or it's the last step
        if (delta.norm() > MIN_MOVEMENT_THRESHOLD || i == steps) {
            points.push_back(delta);
            previous_pos = final_pos;
        }
        
        // Update forward differences
        f += df;
        df += ddf;
        current_pos = bezier_pos;
    }
}

void BezierCurveController::reset()
{
    // Clear buffer to free memory if it's too large
    if (movement_buffer_.capacity() > 512) {
        movement_buffer_.clear();
        movement_buffer_.shrink_to_fit();
        movement_buffer_.reserve(256);
    }
    config_cache_.needs_update = true;
}

void BezierCurveController::updateParameters(float speed, float curve_factor)
{
    speed_ = speed;
    curve_factor_ = curve_factor;
}

float BezierCurveController::easeInOut(float t) const
{
    // More aggressive ease-in-out for faster initial movement
    if (t < 0.3f) {
        // Faster start
        return 3.33f * t * t;
    } else if (t < 0.7f) {
        // Linear middle section for consistent speed
        return 0.3f + (t - 0.3f) * 1.0f;
    } else {
        // Smooth deceleration at the end
        float adj_t = (t - 0.7f) / 0.3f;
        return 0.7f + 0.3f * (1.0f - (1.0f - adj_t) * (1.0f - adj_t));
    }
}

float BezierCurveController::easeInOutLUT(float t) const
{
    // Use lookup table for faster easing calculation
    int index = static_cast<int>(t * (EASE_LUT_SIZE - 1));
    index = std::min(std::max(index, 0), EASE_LUT_SIZE - 1);
    return ease_lut_[index];
}

Eigen::Vector2f BezierCurveController::calculateBezierPoint(float t, const Eigen::Vector2f& start, 
                                                           const Eigen::Vector2f& control1, 
                                                           const Eigen::Vector2f& control2, 
                                                           const Eigen::Vector2f& end) const
{
    // Optimized bezier point calculation using vector operations
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float t2 = t * t;
    
    float c0 = mt2 * mt;
    float c1 = 3.0f * mt2 * t;
    float c2 = 3.0f * mt * t2;
    float c3 = t2 * t;
    
    // Direct vector calculation with potential for SIMD optimization
    return c0 * start + c1 * control1 + c2 * control2 + c3 * end;
}