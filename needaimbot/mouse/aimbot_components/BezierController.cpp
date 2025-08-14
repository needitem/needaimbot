#include "BezierController.h"
#include <cmath>
#include <algorithm>

#define M_PI 3.14159265358979323846f
BezierController::BezierController()
    : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      dist(-1.0f, 1.0f)
{
    reset();
    last_time_point = std::chrono::steady_clock::now();
}

LA::Vector2f BezierController::generateRandomOffset(float max_magnitude)
{
    if (max_magnitude <= 0.0f) return LA::Vector2f::Zero();
    
    float angle = dist(rng) * M_PI;  // Random angle
    float magnitude = (dist(rng) * 0.5f + 0.5f) * max_magnitude;  // 0.5 to 1.0 of max
    
    return LA::Vector2f(std::cos(angle) * magnitude, std::sin(angle) * magnitude);
}

LA::Vector2f BezierController::calculateBezierPoint(float t) const
{
    float one_minus_t = 1.0f - t;
    float one_minus_t_sq = one_minus_t * one_minus_t;
    float one_minus_t_cu = one_minus_t_sq * one_minus_t;
    float t_sq = t * t;
    float t_cu = t_sq * t;
    
    // Cubic Bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    return one_minus_t_cu * p0 + 
           3.0f * one_minus_t_sq * t * p1 + 
           3.0f * one_minus_t * t_sq * p2 + 
           t_cu * p3;
}

void BezierController::setTarget(const LA::Vector2f& current_mouse_pos, 
                                 const LA::Vector2f& target_pos,
                                 const LA::Vector2f& mouse_velocity)
{
    // Set start and end points
    p0 = current_mouse_pos;
    p3 = target_pos;
    current_target_pos = target_pos;
    last_mouse_velocity = mouse_velocity;
    
    LA::Vector2f vector_to_target = p3 - p0;
    float distance = vector_to_target.norm();
    
    // Avoid division by zero for very close targets
    if (distance < 0.1f) {
        current_t = 1.0f;  // Mark as complete
        return;
    }
    
    // Calculate control point P1 (initial direction)
    p1 = p0 + vector_to_target * params.curve_tension_1;
    
    // Add velocity influence if enabled
    if (params.velocity_influence > 0.0f && mouse_velocity.norm() > 0.1f) {
        p1 += mouse_velocity * params.velocity_influence;
    }
    
    // Add random offset for human-like movement
    p1 += generateRandomOffset(params.random_offset);
    
    // Calculate control point P2 (approach to target)
    p2 = p3 - vector_to_target * params.curve_tension_2;
    p2 += generateRandomOffset(params.random_offset * 0.5f);  // Less randomness near target
    
    // Calculate duration based on distance
    total_duration = std::max(params.base_movement_time, distance / params.speed_factor);
    
    // Reset curve traversal
    current_t = 0.0f;
    curve_start_time = std::chrono::steady_clock::now();
    last_calculated_pos = current_mouse_pos;
}

LA::Vector2f BezierController::calculate(const LA::Vector2f& error)
{
    // If error is too small, no movement needed
    if (error.norm() < 0.5f) {
        return LA::Vector2f::Zero();
    }
    
    // Simple direct movement - using proportional control
    // This provides smoother and more reliable aiming than complex Bezier curves
    float speed_multiplier = 0.15f;  // Move 15% of the error per frame
    LA::Vector2f movement = error * speed_multiplier;
    
    return movement;
}

void BezierController::reset()
{
    p0 = p1 = p2 = p3 = LA::Vector2f::Zero();
    current_t = 1.0f;  // Mark as complete
    total_duration = 0.0f;
    last_calculated_pos = LA::Vector2f::Zero();
    current_target_pos = LA::Vector2f::Zero();
    last_mouse_velocity = LA::Vector2f::Zero();
    curve_start_time = std::chrono::steady_clock::now();
}

bool BezierController::needsRecalculation(const LA::Vector2f& new_target_pos) const
{
    float distance = (new_target_pos - current_target_pos).norm();
    return distance > params.recalc_threshold;
}