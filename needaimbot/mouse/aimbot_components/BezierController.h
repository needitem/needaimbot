#ifndef BEZIER_CONTROLLER_H
#define BEZIER_CONTROLLER_H

#include <chrono>
#include <random>
#include "../../math/LinearAlgebra.h"

class BezierController
{
public:
    // Parameters that can be adjusted in UI
    struct Parameters {
        float curve_tension_1 = 0.3f;      // Controls P1's influence (0.1 - 1.0)
        float curve_tension_2 = 0.15f;     // Controls P2's influence (0.0 - 0.5)
        float base_movement_time = 0.08f;  // Minimum time for any movement (seconds)
        float speed_factor = 800.0f;       // Pixels per second
        float random_offset = 5.0f;        // Max pixel offset for randomness (0-20)
        float recalc_threshold = 10.0f;    // Distance target must move to trigger recalc
        float velocity_influence = 0.2f;   // How much current velocity influences P1
    };

private:
    // Bezier control points
    LA::Vector2f p0, p1, p2, p3;
    
    // Current state
    float current_t = 1.0f;  // Current parameter t along curve (0.0 to 1.0)
    float total_duration = 0.0f;  // Total time to traverse current curve
    LA::Vector2f last_calculated_pos;
    LA::Vector2f current_target_pos;
    LA::Vector2f last_mouse_velocity;
    
    // Timing
    std::chrono::steady_clock::time_point last_time_point;
    std::chrono::steady_clock::time_point curve_start_time;
    
    // Parameters
    Parameters params;
    
    // Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
    // Helper functions
    LA::Vector2f generateRandomOffset(float max_magnitude);
    LA::Vector2f calculateBezierPoint(float t) const;

public:
    BezierController();
    
    // Set new target and recalculate curve
    void setTarget(const LA::Vector2f& current_mouse_pos, 
                   const LA::Vector2f& target_pos,
                   const LA::Vector2f& mouse_velocity = LA::Vector2f::Zero());
    
    // Calculate next mouse movement for this frame
    LA::Vector2f calculate(const LA::Vector2f& error);  // Keep similar interface to PID
    
    // Reset controller
    void reset();
    
    // Check if we need to recalculate due to target movement
    bool needsRecalculation(const LA::Vector2f& new_target_pos) const;
    
    // Parameter access
    Parameters& getParameters() { return params; }
    const Parameters& getParameters() const { return params; }
    
    // Check if movement is complete
    bool isComplete() const { return current_t >= 1.0f; }
};

#endif // BEZIER_CONTROLLER_H