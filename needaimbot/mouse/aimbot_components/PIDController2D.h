#ifndef PID_CONTROLLER_2D_H
#define PID_CONTROLLER_2D_H

#include <chrono>
#include "../modules/eigen/include/Eigen/Dense" 

class PIDController2D
{
private:
    
    float kp_x, kp_y;  
    float ki_x, ki_y;  
    float kd_x, kd_y;  
    float derivative_smoothing_factor; 

    
    Eigen::Vector2f prev_error;      
    Eigen::Vector2f integral;        
    Eigen::Vector2f derivative;      
    Eigen::Vector2f prev_derivative; 
    std::chrono::steady_clock::time_point last_time_point;  

public:
    
    PIDController2D(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y, float derivative_smoothing_factor);

    Eigen::Vector2f calculate(const Eigen::Vector2f &error);
    void reset();  

    
    void updateSeparatedParameters(float kp_x, float ki_x, float kd_x, float kp_y, float ki_y, float kd_y, float derivative_smoothing_factor);
};

#endif 

