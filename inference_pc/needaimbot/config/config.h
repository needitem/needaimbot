#ifndef GPA_CONFIG_CONFIG_H
#define GPA_CONFIG_CONFIG_H

#include <string>
#include <vector>
#include <map>
#include "modules/json.hpp"

using json = nlohmann::json;

// Class setting for detection filtering
struct ClassSetting {
    int id = 0;
    std::string name;
    bool allow = true;

    ClassSetting() = default;
    ClassSetting(int i, std::string n, bool a) : id(i), name(std::move(n)), allow(a) {}
};

inline void to_json(json& j, const ClassSetting& c) {
    j = json{{"id", c.id}, {"name", c.name}, {"allow", c.allow}};
}
inline void from_json(const json& j, ClassSetting& c) {
    j.at("id").get_to(c.id);
    j.at("name").get_to(c.name);
    j.at("allow").get_to(c.allow);
}

// Input profile for stabilizer
struct InputProfile {
    std::string profile_name = "Default";
    float base_strength = 3.0f;
    float fire_rate_multiplier = 1.0f;
    float scope_mult_1x = 0.8f;
    float scope_mult_2x = 1.0f;
    float scope_mult_3x = 1.2f;
    float scope_mult_4x = 1.4f;
    float scope_mult_6x = 1.6f;
    float scope_mult_8x = 1.8f;
    int start_delay_ms = 0;
    int end_delay_ms = 0;
    float interval_ms = 10.0f;
};

inline void to_json(json& j, const InputProfile& p) {
    j = json{
        {"profile_name", p.profile_name}, {"base_strength", p.base_strength},
        {"fire_rate_multiplier", p.fire_rate_multiplier},
        {"scope_mult_1x", p.scope_mult_1x}, {"scope_mult_2x", p.scope_mult_2x},
        {"scope_mult_3x", p.scope_mult_3x}, {"scope_mult_4x", p.scope_mult_4x},
        {"scope_mult_6x", p.scope_mult_6x}, {"scope_mult_8x", p.scope_mult_8x},
        {"start_delay_ms", p.start_delay_ms}, {"end_delay_ms", p.end_delay_ms},
        {"interval_ms", p.interval_ms}
    };
}
inline void from_json(const json& j, InputProfile& p) {
    if (j.contains("profile_name")) j.at("profile_name").get_to(p.profile_name);
    if (j.contains("base_strength")) j.at("base_strength").get_to(p.base_strength);
    if (j.contains("fire_rate_multiplier")) j.at("fire_rate_multiplier").get_to(p.fire_rate_multiplier);
    if (j.contains("scope_mult_1x")) j.at("scope_mult_1x").get_to(p.scope_mult_1x);
    if (j.contains("scope_mult_2x")) j.at("scope_mult_2x").get_to(p.scope_mult_2x);
    if (j.contains("scope_mult_3x")) j.at("scope_mult_3x").get_to(p.scope_mult_3x);
    if (j.contains("scope_mult_4x")) j.at("scope_mult_4x").get_to(p.scope_mult_4x);
    if (j.contains("scope_mult_6x")) j.at("scope_mult_6x").get_to(p.scope_mult_6x);
    if (j.contains("scope_mult_8x")) j.at("scope_mult_8x").get_to(p.scope_mult_8x);
    if (j.contains("start_delay_ms")) j.at("start_delay_ms").get_to(p.start_delay_ms);
    if (j.contains("end_delay_ms")) j.at("end_delay_ms").get_to(p.end_delay_ms);
    if (j.contains("interval_ms")) j.at("interval_ms").get_to(p.interval_ms);
}

// Profile data - game/aim specific settings
struct ProfileData {
    // Capture
    int detection_resolution = 256;
    int monitor_idx = 0;
    bool circle_mask = false;
    bool capture_borders = false;
    bool capture_cursor = false;
    std::string capture_method = "DDA";
    float capture_timeout_scale = 0.60f;
    int pipeline_loop_delay_ms = 1;

    // Target
    float body_y_offset = 0.15f;
    float head_y_offset = 0.05f;
    float offset_step = 0.01f;
    bool auto_aim = false;
    bool auto_action = false;
    bool ignore_up_aim = false;
    float crosshair_offset_x = 0.0f;
    float crosshair_offset_y = 0.0f;
    bool enable_aim_shoot_offset = false;
    float aim_shoot_offset_x = 0.0f;
    float aim_shoot_offset_y = 0.0f;
    float iou_stickiness_threshold = 0.30f;

    // PID Controller
    float pid_kp_x = 0.5f;
    float pid_kp_y = 0.5f;
    float pid_ki_x = 0.0f;
    float pid_ki_y = 0.0f;
    float pid_kd_x = 0.3f;
    float pid_kd_y = 0.3f;
    float pid_integral_max = 100.0f;
    float pid_derivative_max = 50.0f;

    // Deadband
    int deadband_enter_x = 2;
    int deadband_exit_x = 5;
    int deadband_enter_y = 2;
    int deadband_exit_y = 5;

    // AI
    std::string ai_model = "model.engine";
    float confidence_threshold = 0.25f;
    int max_detections = 30;
    std::string postprocess = "yolo11";

    // Color Filter
    bool color_filter_enabled = false;
    int color_filter_mode = 0;
    int color_filter_r_min = 0, color_filter_r_max = 255;
    int color_filter_g_min = 0, color_filter_g_max = 255;
    int color_filter_b_min = 0, color_filter_b_max = 255;
    int color_filter_h_min = 0, color_filter_h_max = 179;
    int color_filter_s_min = 0, color_filter_s_max = 255;
    int color_filter_v_min = 0, color_filter_v_max = 255;
    float color_filter_mask_opacity = 0.2f;
    bool color_filter_target_enabled = false;
    int color_filter_target_mode = 0;
    int color_filter_comparison = 0;
    float color_filter_min_ratio = 0.1f;
    float color_filter_max_ratio = 1.0f;
    int color_filter_min_count = 10;
    int color_filter_max_count = 10000;

    // Classes
    std::string head_class_name = "Head";
    std::vector<ClassSetting> class_settings;

    // Input profiles (stabilizer)
    std::vector<InputProfile> input_profiles;
    int active_input_profile_index = 0;

    // Scope
    int active_scope_magnification = 0;
    float bScope_multiplier = 1.0f;
};

void to_json(json& j, const ProfileData& p);
void from_json(const json& j, ProfileData& p);

// Global settings - hardware/system settings shared across profiles
struct GlobalSettings {
    // Input method: "WIN32", "MAKCU", "MAKCU_NET", "KMBOX"
    std::string input_method = "WIN32";
    
    // Makcu serial settings (device connected directly to this PC)
    std::string makcu_port = "COM3";
    int makcu_baudrate = 4000000;
    
    // Makcu network relay settings (legacy - for relay to another PC)
    std::string makcu_remote_ip = "127.0.0.1";
    int makcu_remote_port = 5005;
    
    // KMBox network settings
    std::string kmbox_ip = "192.168.2.188";
    std::string kmbox_port = "16896";
    std::string kmbox_mac = "46405c53";
    
    // 2PC Network settings - UDP frame receiver from game PC
    bool use_network_capture = false;           // Use UDP frames instead of local DDA
    std::string game_pc_ip = "";                // Game PC IP (auto-detected if empty)
    int frame_recv_port = 5007;                 // Port to receive frames from game PC
    int mouse_state_port = 5006;                // Port to send mouse state to game PC
    int cuda_device_id = 0;
    int persistent_cache_limit_mb = 32;
    bool use_cuda_graph = false;
    int graph_warmup_iterations = 3;
    std::vector<std::string> button_targeting = {"RightMouseButton"};
    std::vector<std::string> button_exit = {"F2"};
    std::vector<std::string> button_pause = {"F3"};
    std::vector<std::string> button_reload_config = {"F4"};
    std::vector<std::string> button_open_overlay = {"Home"};
    std::vector<std::string> button_disable_upward_aim = {"None"};
    std::vector<std::string> button_auto_action = {"LeftMouseButton"};
    std::vector<std::string> button_single_shot = {"F8"};
    std::vector<std::string> button_stabilizer = {"None"};
    int overlay_opacity = 225;
    float overlay_ui_scale = 1.0f;
    bool show_window = true;
    bool show_fps = true;
    std::vector<std::string> screenshot_button = {"None"};
    int screenshot_delay = 500;
    bool always_on_top = false;
};

void to_json(json& j, const GlobalSettings& g);
void from_json(const json& j, GlobalSettings& g);

class Config {
public:
    Config();
    ~Config() = default;

    bool loadConfig(const std::string& filename = "");
    bool saveConfig(const std::string& filename = "");

    // Profile access
    ProfileData& profile() { return *current_profile; }
    const ProfileData& profile() const { return *current_profile; }
    GlobalSettings& global() { return global_settings; }
    const GlobalSettings& global() const { return global_settings; }

    // Profile management
    std::string getActiveProfileName() const { return active_profile_name; }
    std::vector<std::string> listProfiles() const;
    bool switchProfile(const std::string& name);
    bool createProfile(const std::string& name);
    bool deleteProfile(const std::string& name);
    bool duplicateProfile(const std::string& src, const std::string& dst);

    // Input profile management
    InputProfile* getCurrentInputProfile();
    bool setActiveInputProfile(const std::string& name);
    std::vector<std::string> getInputProfileNames() const;
    bool addInputProfile(const InputProfile& profile);
    bool removeInputProfile(const std::string& name);
    InputProfile* getInputProfile(const std::string& name);

    // Utilities
    std::string getExecutableDir();
    std::string getConfigPath(const std::string& filename);
    
    static std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",") {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) result += delimiter;
            result += vec[i];
        }
        return result;
    }

private:
    void initializeDefaults();
    void initializeDefaultClassSettings(ProfileData& p);
    void initializeDefaultInputProfiles(ProfileData& p);

    std::string active_profile_name = "Default";
    std::map<std::string, ProfileData> profiles;
    GlobalSettings global_settings;
    ProfileData* current_profile = nullptr;
};

#endif // GPA_CONFIG_CONFIG_H
