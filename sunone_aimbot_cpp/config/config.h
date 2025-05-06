#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    // Capture
    // std::string capture_method; // "duplication_api", "winrt", "virtual_camera"
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

    // Target Stickiness
    // Removed: float sticky_bonus;
    // Removed: float sticky_iou_threshold;

    // Mouse
    bool easynorecoil;
    float easynorecoilstrength;
    float norecoil_step;  // Step size for adjusting norecoil strength
    float norecoil_ms;    // Millisecond delay for recoil control
    std::string input_method; // "WIN32", "GHUB", "ARDUINO"

    // Razer Input
    // std::string razer_dll_path; // Path to rzctl.dll <-- Removed

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
    // bool auto_shoot; // Removed
    float bScope_multiplier;

    // Kalman Filter settings
    float prediction_time_ms;
    float kalman_process_noise;
    float kalman_measurement_noise;
    bool enable_prediction; // Flag to enable/disable prediction

    // AI
    std::string ai_model;
    float confidence_threshold;
    float nms_threshold;
    int max_detections;
    std::string postprocess;
    bool export_enable_fp8;
    bool export_enable_fp16;
    int onnx_input_resolution; // Added for selectable ONNX input resolution (e.g., 160, 320, 640)

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
    std::vector<std::string> button_auto_shoot; // Added for separate auto-shoot hotkey

    // Overlay
    int overlay_opacity;
    float overlay_ui_scale;

    // Custom Classes
    int class_player;                  // 0
    int class_bot;                     // 1
    int class_weapon;                  // 2
    int class_outline;                 // 3
    int class_dead_body;               // 4
    int class_hideout_target_human;    // 5
    int class_hideout_target_balls;    // 6
    int class_head;                    // 7
    int class_smoke;                   // 8
    int class_fire;                    // 9
    int class_third_person;            // 10

    // Debug
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool always_on_top;
    bool verbose;

    bool ignore_class_0; // player
    bool ignore_class_1; // bot
    bool ignore_class_2; // weapon
    bool ignore_class_3; // outline
    bool ignore_class_4; // dead_body
    bool ignore_class_5; // hideout_target_human
    bool ignore_class_6; // hideout_target_balls
    bool ignore_class_7; // head (Note: May overlap with disable_headshot)
    bool ignore_class_8; // smoke
    bool ignore_class_9; // fire
    bool ignore_class_10; // third_person

    // --- Prediction Algorithm Settings --- 
    std::string prediction_algorithm = "None";    // None, Velocity Based, Linear Regression, Exponential Smoothing, Kalman Filter
    float velocity_prediction_ms = 16.0f;     // For Velocity Based
    int   lr_past_points = 10;              // For Linear Regression (min 2)
    float es_alpha = 0.5f;                  // For Exponential Smoothing (0.01 - 1.0)
    float kalman_q = 0.1f;                  // For Kalman Filter
    float kalman_r = 0.1f;
    float kalman_p = 0.1f;

    // PID Controller Settings (using double for precision)
    // double kp_x = 0.350;

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
