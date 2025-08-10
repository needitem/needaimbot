#ifndef PID_CONTROLLER_2D_H
#define PID_CONTROLLER_2D_H

#include <chrono>
#include "../modules/eigen/include/Eigen/Dense" 
#include "../../AppContext.h"

class PIDController2D
{
private:
    
    float kp_x, kp_y;  
    float ki_x, ki_y;  
    float kd_x, kd_y;  
 

    
    Eigen::Vector2f prev_error;      
    Eigen::Vector2f integral;        
    Eigen::Vector2f derivative;      
    std::chrono::steady_clock::time_point last_time_point;  

    // Recent error deltas for small-window robust derivative (median-of-three)
    float recent_delta_x[3] = {0.0f, 0.0f, 0.0f};
    float recent_delta_y[3] = {0.0f, 0.0f, 0.0f};
    int delta_index = 0;

    // Warmup frames after reset or target change to suppress derivative kick
    int warmup_frames_remaining = 0;

    // Exponential moving average for derivative (low-pass filtered)
    float filtered_deriv_x = 0.0f;
    float filtered_deriv_y = 0.0f;

public:
    
    PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);

    Eigen::Vector2f calculate(const Eigen::Vector2f &error);
    void reset();  

    
    void updateSeparatedParameters(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y);
    
    // Getters for gains
    float getKpX() const { return kp_x; }
    float getKpY() const { return kp_y; }
};

#endif 

