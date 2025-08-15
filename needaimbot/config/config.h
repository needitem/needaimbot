#ifndef NEEDAIMBOT_CONFIG_CONFIG_H
#define NEEDAIMBOT_CONFIG_CONFIG_H

#include <string>
#include <vector>
#include <mutex>
#include <atomic>

// 설정값 캐싱 시스템
struct ConfigCache {
    float detection_resolution;
    float confidence_threshold;
    float nms_threshold;
    bool enable_aimbot;
    float mouse_sensitivity;
    int target_priority;
    uint64_t version; // atomic 제거하여 복사 가능하게 함
    
    ConfigCache() : version(0) {}
    
    // 복사 생성자 명시적 정의
    ConfigCache(const ConfigCache& other) 
        : detection_resolution(other.detection_resolution)
        , confidence_threshold(other.confidence_threshold)
        , nms_threshold(other.nms_threshold)
        , enable_aimbot(other.enable_aimbot)
        , mouse_sensitivity(other.mouse_sensitivity)
        , target_priority(other.target_priority)
        , version(other.version) {}
    
    // 대입 연산자 명시적 정의
    ConfigCache& operator=(const ConfigCache& other) {
        if (this != &other) {
            detection_resolution = other.detection_resolution;
            confidence_threshold = other.confidence_threshold;
            nms_threshold = other.nms_threshold;
            enable_aimbot = other.enable_aimbot;
            mouse_sensitivity = other.mouse_sensitivity;
            target_priority = other.target_priority;
            version = other.version;
        }
        return *this;
    }
};

class CachedConfig {
private:
    ConfigCache cache;
    std::atomic<uint64_t> current_version{0};
    mutable std::mutex cache_mutex;
    
public:
    void updateCache(float resolution, float confidence, float nms, bool aimbot, 
                    float sensitivity, int priority) {
        std::lock_guard<std::mutex> lock(cache_mutex); // 스레드 안전성 보장
        cache.detection_resolution = resolution;
        cache.confidence_threshold = confidence; 
        cache.nms_threshold = nms;
        cache.enable_aimbot = aimbot;
        cache.mouse_sensitivity = sensitivity;
        cache.target_priority = priority;
        cache.version = ++current_version;
    }
    
    ConfigCache getCache() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        return cache; // 스레드 안전한 복사
    }
    
    bool needsUpdate(uint64_t last_version) const {
        return last_version != current_version.load();
    }
    
    uint64_t getCurrentVersion() const {
        return current_version.load();
    }
};


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
    int capture_fps;
    int monitor_idx;
    bool circle_mask;
    bool capture_borders;
    bool capture_cursor;
    
 
    float target_fps;
    
    std::string capture_method; // "simple", "wingraphics", "virtual_camera", "ndi"
    
    // NDI capture settings
    std::string ndi_source_name;
    std::string ndi_network_url; 
    bool ndi_low_latency;
    
    float body_y_offset;
    float head_y_offset;
    float offset_step;  
    bool ignore_third_person;
    bool shooting_range_targets;
    bool auto_aim;
    bool enable_aimbot;
    bool enable_triggerbot;
    bool enable_rapidfire;
    int rapidfire_cps;  // Clicks per second for rapidfire

    
    float crosshair_offset_x;
    float crosshair_offset_y;
    
    // Separate offset for when aiming and shooting
    bool enable_aim_shoot_offset;
    float aim_shoot_offset_x;
    float aim_shoot_offset_y;
    
    // Target lock feature
    bool enable_target_lock;

    
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
    
    // Crouch recoil reduction
    bool crouch_recoil_enabled;
    float crouch_recoil_reduction; // -100% to +100%, default -50%

    
    std::vector<WeaponRecoilProfile> weapon_profiles;
    int active_weapon_profile_index;
    std::string current_weapon_name;

    
    // Simple Movement Parameters

    // ByteTracker parameters
    bool enable_tracking;       // Enable/disable target tracking
    float byte_track_thresh = 0.5f;     // Threshold for starting a track
    float byte_high_thresh = 0.6f;      // High confidence threshold
    float byte_match_thresh = 0.8f;     // IOU matching threshold
    int byte_max_time_lost = 30;        // Max frames to keep lost track
    
    // Legacy tracker parameters (for compatibility)
    int tracker_max_age = 30;            // Max frames without detection
    int tracker_min_hits = 3;            // Min hits to confirm track
    float tracker_iou_threshold = 0.3f;  // IOU threshold for matching
    
    // Kalman filter parameters (old - deprecated, use new settings below)
    float kalman_lookahead_time = 0.016f; // Deprecated - use kalman_dt instead
    
    

    
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_16_bit_mouse;
    bool arduino_enable_keys;

    
    std::string kmbox_ip;         
    std::string kmbox_port;       
    std::string kmbox_mac;    

    
    std::string makcu_port;
    int makcu_baudrate;

    
    float bScope_multiplier;

    
    std::string ai_model;
    float confidence_threshold;
    float nms_threshold;
    float confidence_weight;
    float distance_weight;
    float sticky_target_threshold;  // How much better a new target must be to switch (0.0-1.0)
    int max_detections;
    std::string postprocess;
    bool export_enable_fp8;
    bool export_enable_fp16;
    bool tensorrt_fp16;
    int onnx_input_resolution;

    
    int cuda_device_id = 0;

    
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;
    std::vector<std::string> button_open_overlay;
    std::vector<std::string> button_disable_upward_aim;
    std::vector<std::string> button_auto_shoot;

    
    int overlay_opacity;
    float overlay_ui_scale;

    
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool always_on_top;
    bool show_metrics;  // For performance metrics display

    
    std::vector<ClassSetting> class_settings;
    std::string head_class_name = "Head"; 
                                         

    
     



    // GPU performance settings
    int persistent_cache_limit_mb = 32; // TensorRT persistent L2 cache size in MB (default: 32MB for RTX 40 series)
    
    // RGB Color filter settings
    bool enable_color_filter;
    
    // RGB filter ranges
    int rgb_min_r;
    int rgb_max_r;
    int rgb_min_g;
    int rgb_max_g;
    int rgb_min_b;
    int rgb_max_b;
    
    // Filter settings
    int min_color_pixels;
    bool remove_color_matches; 

    // Kalman filter settings
    bool enable_kalman_filter = false;
    bool kalman_use_cuda_graph = true;
    float kalman_dt = 0.033f;              // Time delta (30 FPS default)
    float kalman_process_noise = 1.0f;     // Process noise scale
    float kalman_measurement_noise = 1.0f; // Measurement noise scale
    int kalman_min_hits = 3;               // Minimum hits before track is confirmed
    int kalman_max_age = 5;                // Maximum frames without detection
    
    // Mouse control settings
    float mouse_sensitivity = 1.0f;        // Mouse movement sensitivity (legacy - will be removed)
    float mouse_sensitivity_x = 1.0f;      // Mouse X-axis sensitivity
    float mouse_sensitivity_y = 1.0f;      // Mouse Y-axis sensitivity
    
    // Movement parameters separated for X and Y
    float movement_factor = 0.3f;          // Legacy - will be removed
    float movement_factor_x = 0.3f;        // X-axis movement factor (% of error per frame)
    float movement_factor_y = 0.3f;        // Y-axis movement factor (% of error per frame)
    
    float min_movement_threshold = 1.0f;   // Legacy - will be removed
    float min_movement_threshold_x = 1.0f; // X-axis minimum movement threshold (deadzone)
    float min_movement_threshold_y = 1.0f; // Y-axis minimum movement threshold (deadzone)

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

