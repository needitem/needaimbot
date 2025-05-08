#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// New structure for customizable class settings
struct ClassSetting {
    int id;
    std::string name;
    bool ignore;

    // Default constructor for emplace_back or default initialization
    ClassSetting(int i = 0, std::string n = "", bool ign = false) : id(i), name(std::move(n)), ignore(ign) {}
};

struct Config; // Forward declaration

class Detector;
class GhubMouse; // Forward declarations if needed by Config
class SerialConnection;

// Define KeyBind struct if it's not in an included header
// struct KeyBind { /* ... members ... */ }; 
// Assuming KeyBind is defined elsewhere or included

class Config
{
public:
    // Explicitly defaulted special members (Rule of Five/Zero)
    Config(); // Your user-defined constructor
    ~Config() = default;
    Config(const Config&) = default;
    Config& operator=(const Config&) = default;
    Config(Config&&) = default;
    Config& operator=(Config&&) = default;

    // Capture
    int detection_resolution;
    int capture_fps;
    int monitor_idx;
    bool circle_mask;
    bool capture_borders;
    bool capture_cursor;
    std::string virtual_camera_name;
    bool capture_use_cuda;

    // Target
    float body_y_offset;
    float head_y_offset;
    float offset_step;  // Step size for adjusting offsets
    bool ignore_third_person;
    bool shooting_range_targets;
    bool auto_aim;

    // Mouse
    bool easynorecoil;
    float easynorecoilstrength;
    float norecoil_step;  // Step size for adjusting norecoil strength
    float norecoil_ms;    // Millisecond delay for recoil control
    std::string input_method; // "WIN32", "GHUB", "ARDUINO"

    // Scope Recoil Control
    int active_scope_magnification; // 0=None, 2=2x, 3=3x, 4=4x, 6=6x
    float recoil_mult_2x;
    float recoil_mult_3x;
    float recoil_mult_4x;
    float recoil_mult_6x;

    // Separated X/Y PID Controllers
    double kp_x;
    double ki_x;
    double kd_x;
    double kp_y;
    double ki_y;
    double kd_y;

    // Arduino
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_16_bit_mouse;
    bool arduino_enable_keys;

    // KMBOX net params:
    std::string kmbox_ip;         
    std::string kmbox_port;       
    std::string kmbox_mac;    

    // Mouse shooting
    float bScope_multiplier;

    // AI
    std::string ai_model;
    float confidence_threshold;
    float nms_threshold;
    int max_detections;
    std::string postprocess;
    bool export_enable_fp8;
    bool export_enable_fp16;
    int onnx_input_resolution; 

    // CUDA
    bool use_pinned_memory;
    int cuda_device_id = 0;

    // Buttons
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_shoot;
    std::vector<std::string> button_zoom;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;
    std::vector<std::string> button_open_overlay;
    std::vector<std::string> button_disable_upward_aim;
    std::vector<std::string> button_auto_shoot;

    // Overlay
    int overlay_opacity;
    float overlay_ui_scale;

    // Debug
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool always_on_top;
    bool verbose;

    // --- New Class Settings --- 
    std::vector<ClassSetting> class_settings;
    std::string head_class_name = "Head"; // Default value initialized here for clarity
                                         // Actual default during file creation will be in config.cpp

    // --- Prediction Algorithm Settings --- 
    std::string prediction_algorithm = "None";
    float velocity_prediction_ms = 16.0f;
    int   lr_past_points = 10;
    float es_alpha = 0.5f;
    double kalman_q = 0.1;
    double kalman_r = 0.1;
    double kalman_p = 0.1;

    bool loadConfig(const std::string& filename = "config.ini");
    bool saveConfig(const std::string& filename = "config.ini");

    // Profile Management
    std::vector<std::string> listProfiles();
    bool saveProfile(const std::string& profileName);
    bool loadProfile(const std::string& profileName);
    bool deleteProfile(const std::string& profileName);
    void resetConfig();

    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
};

#endif // CONFIG_H
