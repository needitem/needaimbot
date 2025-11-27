#ifndef NEEDAIMBOT_CONFIG_CONFIG_H
#define NEEDAIMBOT_CONFIG_CONFIG_H

#include <string>
#include <vector>
#include <mutex>
#include <atomic>



/**
 * @brief Configuration for object detection class filtering
 * 
 * Defines how specific object classes should be handled during detection,
 * including whether to allow certain classes or apply special processing.
 */
struct ClassSetting {
    int id;              ///< Unique class identifier
    std::string name;    ///< Human-readable class name
    bool allow;          ///< Whether to allow this class during detection

    /**
     * @brief Construct a new Class Setting object
     * @param i Class ID
     * @param n Class name
     * @param allow_class Whether to allow this class
     */
    ClassSetting(int i = 0, std::string n = "", bool allow_class = true) 
        : id(i), name(std::move(n)), allow(allow_class) {}
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

class Config; 

class Detector;
class GhubMouse; 
class SerialConnection;





/**
 * @brief Comprehensive configuration manager for the aimbot system
 * 
 * Manages all configuration settings including AI model parameters,
 * input/output settings, weapon profiles, and system preferences.
 * Supports profile management and persistent storage.
 */
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
    int monitor_idx;
    bool circle_mask;
    bool capture_borders;
    bool capture_cursor;
    // Capture backend: "DDA"
    std::string capture_method;

    float body_y_offset;
    float head_y_offset;
    float offset_step;

    bool auto_aim;
    bool auto_shoot;
    bool ignore_up_aim;


    float crosshair_offset_x;
    float crosshair_offset_y;
    
    // Separate offset for when aiming and shooting
    bool enable_aim_shoot_offset;
    float aim_shoot_offset_x;
    float aim_shoot_offset_y;

    std::string input_method;


    int active_scope_magnification;

    
    std::vector<WeaponRecoilProfile> weapon_profiles;
    int active_weapon_profile_index;
    std::string current_weapon_name;

    
    // Simple Movement Parameters

    
    
    

    
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_enable_keys;

    
    std::string kmbox_ip;         
    std::string kmbox_port;       
    std::string kmbox_mac;    


    // Makcu (2PC network) settings
    std::string makcu_port;       // legacy, unused in 2PC mode but kept for compatibility
    int makcu_baudrate;           // legacy, unused in 2PC mode but kept for compatibility
    std::string makcu_remote_ip;  // second PC IP for MakcuRelay
    int makcu_remote_port;        // UDP port for MakcuRelay

    
    float bScope_multiplier;


    std::string ai_model;
    float confidence_threshold;
    int max_detections;
    std::string postprocess;

    
    int cuda_device_id = 0;

    // Capture timing tuning
    // Scale factor applied to EMA of present interval when computing AcquireNextFrame timeout.
    // Typical range 0.55–0.65; default 0.60.
    float capture_timeout_scale = 0.60f;

    // Pipeline loop delay (ms) - prevents game FPS drops
    int pipeline_loop_delay_ms = 1;

    
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;
    std::vector<std::string> button_open_overlay;
    std::vector<std::string> button_disable_upward_aim;
    std::vector<std::string> button_auto_shoot;
    std::vector<std::string> button_single_shot;

    
    int overlay_opacity;
    float overlay_ui_scale;
    

    
    bool show_window;
    bool show_fps;
    
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool always_on_top;

    
    std::vector<ClassSetting> class_settings;
    std::string head_class_name = "Head"; 
                                         

    
     



    // GPU performance settings
    int persistent_cache_limit_mb = 32; // TensorRT persistent L2 cache size in MB (default: 32MB for RTX 40 series)
    bool use_cuda_graph = false; // Enable CUDA Graph optimization (faster but not compatible with all models)
    int graph_warmup_iterations = 3; // Number of warmup iterations for TensorRT/Graph setup
    
    
    // PID Controller settings
    float pid_kp_x = 0.5f;   // Proportional gain for X axis (0-1 typical, 1.0 = move by full error)
    float pid_kp_y = 0.5f;   // Proportional gain for Y axis (0-1 typical, 1.0 = move by full error)
    float pid_ki_x = 0.0f;   // Integral gain for X axis (0-0.1 typical, for tracking moving targets)
    float pid_ki_y = 0.0f;   // Integral gain for Y axis (0-0.1 typical, for tracking moving targets)
    float pid_kd_x = 0.3f;   // Derivative gain for X axis (0-1 typical, for oscillation dampening)
    float pid_kd_y = 0.3f;   // Derivative gain for Y axis (0-1 typical, for oscillation dampening)
    float pid_integral_max = 100.0f;  // Max integral windup limit (pixels)
    float pid_derivative_max = 50.0f; // Max derivative clamp (pixels, prevents oscillation from large movements)

    // Jitter filter (deadband) settings - per axis
    int deadband_enter_x = 2;  // pixels to enter settle zone on X
    int deadband_exit_x  = 5;  // pixels to exit settle zone on X
    int deadband_enter_y = 2;  // pixels to enter settle zone on Y
    int deadband_exit_y  = 5;  // pixels to exit settle zone on Y

    // Target selection stickiness (IoU threshold)
    float iou_stickiness_threshold = 0.30f; // prefer previous target if IoU > threshold

    // Color Filter settings
    bool color_filter_enabled = false;
    int color_filter_mode = 0;  // 0=RGB, 1=HSV
    int color_filter_r_min = 0;
    int color_filter_r_max = 255;
    int color_filter_g_min = 0;
    int color_filter_g_max = 255;
    int color_filter_b_min = 0;
    int color_filter_b_max = 255;
    int color_filter_h_min = 0;
    int color_filter_h_max = 179;
    int color_filter_s_min = 0;
    int color_filter_s_max = 255;
    int color_filter_v_min = 0;
    int color_filter_v_max = 255;
    float color_filter_mask_opacity = 0.2f; // 0.0 = full black, 1.0 = original color
    bool color_filter_pixel_enabled = false;
    int color_filter_pixel_mode = 0;      // 0=below threshold (<=), 1=above threshold (>=)
    int color_filter_pixel_threshold = 50; // pixel count threshold

    // Active profile management
    std::string active_profile_name = "Default";
    
    
    // Path utilities
    std::string getExecutableDir();
    std::string getConfigPath(const std::string& filename);
    
    bool loadConfig(const std::string& filename = "");
    bool saveConfig(const std::string& filename = "");

    
    std::vector<std::string> listProfiles();
    bool saveProfile(const std::string& profileName);
    bool loadProfile(const std::string& profileName);
    bool deleteProfile(const std::string& profileName);
    void resetConfig();
    
    // Active profile methods
    bool setActiveProfile(const std::string& profileName);
    std::string getActiveProfile() const { return active_profile_name; }
    bool saveActiveProfile(); // Save current settings to active profile
    bool isProfileModified() const; // Check if current settings differ from saved profile

    
    void initializeDefaultWeaponProfiles();
    bool addWeaponProfile(const WeaponRecoilProfile& profile);
    bool removeWeaponProfile(const std::string& weapon_name);
    WeaponRecoilProfile* getWeaponProfile(const std::string& weapon_name);
    WeaponRecoilProfile* getCurrentWeaponProfile();
    bool setActiveWeaponProfile(const std::string& weapon_name);
    std::vector<std::string> getWeaponProfileNames() const;

    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
};

#endif // NEEDAIMBOT_CONFIG_CONFIG_H

