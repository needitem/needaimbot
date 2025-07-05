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

struct WeaponRecoilProfile {
    std::string weapon_name;
    float base_strength;
    float fire_rate_multiplier;
    float scope_mult_1x;
    float scope_mult_2x;
    float scope_mult_3x;
    float scope_mult_4x;
    float scope_mult_6x;
    float scope_mult_8x;
    int start_delay_ms;
    int end_delay_ms;
    float recoil_ms;
    
    WeaponRecoilProfile(std::string name = "Default", float strength = 3.0f, float fire_mult = 1.0f)
        : weapon_name(std::move(name)), base_strength(strength), fire_rate_multiplier(fire_mult),
          scope_mult_1x(0.8f), scope_mult_2x(1.0f), scope_mult_3x(1.2f), 
          scope_mult_4x(1.4f), scope_mult_6x(1.6f), scope_mult_8x(1.8f),
          start_delay_ms(0), end_delay_ms(0), recoil_ms(10.0f) {}
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
    
    bool use_1ms_capture;

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

    
    std::vector<WeaponRecoilProfile> weapon_profiles;
    int active_weapon_profile_index;
    std::string current_weapon_name;

    
    double kp_x;
    double ki_x;
    double kd_x;
    double kp_y;
    double ki_y;
    double kd_y;
    
    float pid_derivative_smoothing;
    float movement_smoothing;
    bool enable_adaptive_pid;

    
    bool use_predictive_controller;
    float prediction_time_ms;
    float kalman_process_noise;
    float kalman_measurement_noise;

    
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

    
    bool optical_flow_norecoil;
    float optical_flow_norecoil_strength;
    float optical_flow_norecoil_threshold;
    int optical_flow_norecoil_frames;

    
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

    
    void initializeDefaultWeaponProfiles();
    bool addWeaponProfile(const WeaponRecoilProfile& profile);
    bool removeWeaponProfile(const std::string& weapon_name);
    WeaponRecoilProfile* getWeaponProfile(const std::string& weapon_name);
    WeaponRecoilProfile* getCurrentWeaponProfile();
    bool setActiveWeaponProfile(const std::string& weapon_name);
    std::vector<std::string> getWeaponProfileNames() const;
    
    bool saveWeaponProfiles(const std::string& filename = "weapon_profiles.ini");
    bool loadWeaponProfiles(const std::string& filename = "weapon_profiles.ini");

    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
};

#endif 

