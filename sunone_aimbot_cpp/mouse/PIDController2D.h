#ifndef PID_CONTROLLER_2D_H
#define PID_CONTROLLER_2D_H

#include <chrono>
#include "../modules/eigen/include/Eigen/Dense" // Assuming Eigen is needed for Vector2f

class PIDController2D
{
private:
    // Base PID gains
    float kp_x, kp_y;  // Proportional gain
    float ki_x, ki_y;  // Integral gain
    float kd_x, kd_y;  // Derivative gain

    // State variables
    Eigen::Vector2f prev_error;      // Previous error (for derivative term)
    Eigen::Vector2f integral;        // Accumulated error (for integral term)
    Eigen::Vector2f derivative;      // Change rate (derivative term)
    Eigen::Vector2f prev_derivative; // Previous derivative (for derivative filtering)
    std::chrono::steady_clock::time_point last_time_point;  // Previous calculation time (for dt calculation)

public:
    // New constructor with separated X/Y gains
    PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);

    Eigen::Vector2f calculate(const Eigen::Vector2f &error);
    void reset();  // Controller reset (used when starting to aim at a new target)

    // X/Y separated gain update function
    void updateSeparatedParameters(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);
};

#endif // PID_CONTROLLER_2D_H
