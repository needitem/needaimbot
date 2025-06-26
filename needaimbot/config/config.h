#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>


struct ClassSetting {
    int id;
    std::string name;
    bool ignore;

    
    ClassSetting(int i = 0, std::string n = "", bool ign = false) : id(i), name(std::move(n)), ignore(ign) {}
};

struct Config; 

class Detector;
class GhubMouse; 
class SerialConnection;





class Config
{
public:
    
    Config(); 
    ~Config() = default;
    Config(const Config&) = default;
    Config& operator=(const Config&) = default;
    Config(Config&&) = default;
    Config& operator=(Config&&) = default;

    
    int detection_resolution;
    int capture_fps;
    int monitor_idx;
    bool circle_mask;
    bool capture_borders;
    bool capture_cursor;
    bool capture_use_cuda;

    int capture_timeout_ms = 5; 
    float target_fps; 

    
    float body_y_offset;
    float head_y_offset;
    float offset_step;  
    bool ignore_third_person;
    bool shooting_range_targets;
    bool auto_aim;

    
    bool easynorecoil;
    float easynorecoilstrength;
    float norecoil_step;  
    float norecoil_ms;    
    std::string input_method; 
    int easynorecoil_start_delay_ms; 
    int easynorecoil_end_delay_ms;   

    
    int active_scope_magnification; 
    float recoil_mult_2x;
    float recoil_mult_3x;
    float recoil_mult_4x;
    float recoil_mult_6x;

    
    double kp_x;
    double ki_x;
    double kd_x;
    double kp_y;
    double ki_y;
    double kd_y;

    
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_16_bit_mouse;
    bool arduino_enable_keys;

    
    std::string kmbox_ip;         
    std::string kmbox_port;       
    std::string kmbox_mac;    

    
    float bScope_multiplier;

    
    std::string ai_model;
    float confidence_threshold;
    float nms_threshold;
    float confidence_weight;
    float distance_weight;
    int max_detections;
    std::string postprocess;
    bool export_enable_fp8;
    bool export_enable_fp16;
    int onnx_input_resolution;

    
    int cuda_device_id = 0;

    
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_shoot;
    std::vector<std::string> button_zoom;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;
    std::vector<std::string> button_open_overlay;
    std::vector<std::string> button_disable_upward_aim;
    std::vector<std::string> button_auto_shoot;
    std::vector<std::string> button_silent_aim;

    
    int overlay_opacity;
    float overlay_ui_scale;

    
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool always_on_top;
    bool verbose;

    
    std::vector<ClassSetting> class_settings;
    std::string head_class_name = "Head"; 
                                         

    
    std::string prediction_algorithm;
    float velocity_prediction_ms;
    int lr_past_points;
    float es_alpha;
    float es_beta;
    double kalman_q;
    double kalman_r; 
    double kalman_p; 

    
    bool enable_target_locking;
    float target_locking_iou_threshold;
    int target_locking_max_lost_frames;

    
    bool enable_optical_flow;
    bool draw_optical_flow; 
    float optical_flow_alpha_cpu; 
    int draw_optical_flow_steps;    
    float optical_flow_magnitudeThreshold; 
    float staticFrameThreshold; 
    float fovX; 
    float fovY; 

    
    bool enable_hsv_filter;
    int hsv_lower_h;
    int hsv_lower_s;
    int hsv_lower_v;
    int hsv_upper_h;
    int hsv_upper_s;
    int hsv_upper_v;
    int min_hsv_pixels;
    bool remove_hsv_matches; 

    bool loadConfig(const std::string& filename = "config.ini");
    bool saveConfig(const std::string& filename = "config.ini");

    
    std::vector<std::string> listProfiles();
    bool saveProfile(const std::string& profileName);
    bool loadProfile(const std::string& profileName);
    bool deleteProfile(const std::string& profileName);
    void resetConfig();

    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
};

#endif 

