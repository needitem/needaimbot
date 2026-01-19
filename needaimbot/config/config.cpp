#include "../core/windows_headers.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "config.h"

// JSON serialization for ProfileData
void to_json(json& j, const ProfileData& p) {
    j = json{
        {"detection_resolution", p.detection_resolution},
        {"monitor_idx", p.monitor_idx},
        {"circle_mask", p.circle_mask},
        {"capture_borders", p.capture_borders},
        {"capture_cursor", p.capture_cursor},
        {"capture_method", p.capture_method},
        {"capture_timeout_scale", p.capture_timeout_scale},
        {"pipeline_loop_delay_ms", p.pipeline_loop_delay_ms},
        {"body_y_offset", p.body_y_offset},
        {"head_y_offset", p.head_y_offset},
        {"offset_step", p.offset_step},
        {"auto_aim", p.auto_aim},
        {"auto_action", p.auto_action},
        {"ignore_up_aim", p.ignore_up_aim},
        {"crosshair_offset_x", p.crosshair_offset_x},
        {"crosshair_offset_y", p.crosshair_offset_y},
        {"enable_aim_shoot_offset", p.enable_aim_shoot_offset},
        {"aim_shoot_offset_x", p.aim_shoot_offset_x},
        {"aim_shoot_offset_y", p.aim_shoot_offset_y},
        {"iou_stickiness_threshold", p.iou_stickiness_threshold},
        {"pid_kp_x", p.pid_kp_x}, {"pid_kp_y", p.pid_kp_y},
        {"pid_ki_x", p.pid_ki_x}, {"pid_ki_y", p.pid_ki_y},
        {"pid_kd_x", p.pid_kd_x}, {"pid_kd_y", p.pid_kd_y},
        {"pid_integral_max", p.pid_integral_max},
        {"pid_derivative_max", p.pid_derivative_max},
        {"deadband_enter_x", p.deadband_enter_x},
        {"deadband_exit_x", p.deadband_exit_x},
        {"deadband_enter_y", p.deadband_enter_y},
        {"deadband_exit_y", p.deadband_exit_y},
        {"ai_model", p.ai_model},
        {"confidence_threshold", p.confidence_threshold},
        {"max_detections", p.max_detections},
        {"postprocess", p.postprocess},
        {"color_filter_enabled", p.color_filter_enabled},
        {"color_filter_mode", p.color_filter_mode},
        {"color_filter_r_min", p.color_filter_r_min},
        {"color_filter_r_max", p.color_filter_r_max},
        {"color_filter_g_min", p.color_filter_g_min},
        {"color_filter_g_max", p.color_filter_g_max},
        {"color_filter_b_min", p.color_filter_b_min},
        {"color_filter_b_max", p.color_filter_b_max},
        {"color_filter_h_min", p.color_filter_h_min},
        {"color_filter_h_max", p.color_filter_h_max},
        {"color_filter_s_min", p.color_filter_s_min},
        {"color_filter_s_max", p.color_filter_s_max},
        {"color_filter_v_min", p.color_filter_v_min},
        {"color_filter_v_max", p.color_filter_v_max},
        {"color_filter_mask_opacity", p.color_filter_mask_opacity},
        {"color_filter_target_enabled", p.color_filter_target_enabled},
        {"color_filter_target_mode", p.color_filter_target_mode},
        {"color_filter_comparison", p.color_filter_comparison},
        {"color_filter_min_ratio", p.color_filter_min_ratio},
        {"color_filter_max_ratio", p.color_filter_max_ratio},
        {"color_filter_min_count", p.color_filter_min_count},
        {"color_filter_max_count", p.color_filter_max_count},
        {"head_class_name", p.head_class_name},
        {"class_settings", p.class_settings},
        {"input_profiles", p.input_profiles},
        {"active_input_profile_index", p.active_input_profile_index},
        {"active_scope_magnification", p.active_scope_magnification},
        {"bScope_multiplier", p.bScope_multiplier},
        {"noise_enabled", p.noise_enabled},
        {"noise_stddev_x", p.noise_stddev_x},
        {"noise_stddev_y", p.noise_stddev_y},
        {"depth_enabled", p.depth_enabled},
        {"depth_model_path", p.depth_model_path},
        {"depth_fps", p.depth_fps},
        {"depth_near_percent", p.depth_near_percent},
        {"depth_invert", p.depth_invert},
        {"show_capture_frame", p.show_capture_frame},
        {"capture_frame_r", p.capture_frame_r},
        {"capture_frame_g", p.capture_frame_g},
        {"capture_frame_b", p.capture_frame_b},
        {"capture_frame_a", p.capture_frame_a},
        {"capture_frame_thickness", p.capture_frame_thickness}
    };
}

void from_json(const json& j, ProfileData& p) {
    // Use obfuscated JSON keys only (no fallback to original names)
    #define GET_OBF(field, obf_key) if (j.contains(obf_key)) j.at(obf_key).get_to(p.field)
    #define GET_IF(field) if (j.contains(#field)) j.at(#field).get_to(p.field)
    
    GET_OBF(detection_resolution, "analysis_resolution");
    GET_IF(monitor_idx);
    GET_IF(circle_mask);
    GET_OBF(capture_borders, "acquire_borders");
    GET_OBF(capture_cursor, "acquire_cursor");
    GET_OBF(capture_method, "acquire_method");
    GET_OBF(capture_timeout_scale, "acquire_timeout_scale");
    GET_IF(pipeline_loop_delay_ms);
    GET_IF(body_y_offset);
    GET_IF(head_y_offset);
    GET_IF(offset_step);
    GET_OBF(auto_aim, "auto_fcs");
    GET_IF(auto_action);
    GET_OBF(ignore_up_aim, "ignore_up_fcs");
    GET_IF(crosshair_offset_x);
    GET_IF(crosshair_offset_y);
    GET_OBF(enable_aim_shoot_offset, "enable_fcs_fire_offset");
    GET_OBF(aim_shoot_offset_x, "fcs_fire_offset_x");
    GET_OBF(aim_shoot_offset_y, "fcs_fire_offset_y");
    GET_IF(iou_stickiness_threshold);
    GET_IF(pid_kp_x); GET_IF(pid_kp_y);
    GET_IF(pid_ki_x); GET_IF(pid_ki_y);
    GET_IF(pid_kd_x); GET_IF(pid_kd_y);
    GET_IF(pid_integral_max);
    GET_IF(pid_derivative_max);
    GET_IF(deadband_enter_x);
    GET_IF(deadband_exit_x);
    GET_IF(deadband_enter_y);
    GET_IF(deadband_exit_y);
    GET_OBF(ai_model, "ai_module");
    GET_IF(confidence_threshold);
    GET_OBF(max_detections, "max_results");
    GET_IF(postprocess);
    GET_IF(color_filter_enabled);
    GET_IF(color_filter_mode);
    GET_IF(color_filter_r_min);
    GET_IF(color_filter_r_max);
    GET_IF(color_filter_g_min);
    GET_IF(color_filter_g_max);
    GET_IF(color_filter_b_min);
    GET_IF(color_filter_b_max);
    GET_IF(color_filter_h_min);
    GET_IF(color_filter_h_max);
    GET_IF(color_filter_s_min);
    GET_IF(color_filter_s_max);
    GET_IF(color_filter_v_min);
    GET_IF(color_filter_v_max);
    GET_IF(color_filter_mask_opacity);
    GET_OBF(color_filter_target_enabled, "color_filter_point_enabled");
    GET_OBF(color_filter_target_mode, "color_filter_point_mode");
    GET_IF(color_filter_comparison);
    GET_IF(color_filter_min_ratio);
    GET_IF(color_filter_max_ratio);
    GET_IF(color_filter_min_count);
    GET_IF(color_filter_max_count);
    GET_IF(head_class_name);
    GET_IF(class_settings);
    GET_IF(input_profiles);
    GET_IF(active_input_profile_index);
    GET_IF(active_scope_magnification);
    GET_IF(bScope_multiplier);
    GET_IF(noise_enabled);
    GET_IF(noise_stddev_x);
    GET_IF(noise_stddev_y);
    GET_IF(depth_enabled);
    GET_IF(depth_model_path);
    GET_IF(depth_fps);
    GET_IF(depth_near_percent);
    GET_IF(depth_invert);
    GET_IF(show_capture_frame);
    GET_IF(capture_frame_r);
    GET_IF(capture_frame_g);
    GET_IF(capture_frame_b);
    GET_IF(capture_frame_a);
    GET_IF(capture_frame_thickness);
    #undef GET_IF
    #undef GET_OBF
}

void to_json(json& j, const GlobalSettings& g) {
    j = json{
        {"input_method", g.input_method},
        {"arduino_baudrate", g.arduino_baudrate},
        {"arduino_port", g.arduino_port},
        {"arduino_enable_keys", g.arduino_enable_keys},
        {"kmbox_ip", g.kmbox_ip},
        {"kmbox_port", g.kmbox_port},
        {"kmbox_mac", g.kmbox_mac},
        {"makcu_port", g.makcu_port},
        {"makcu_baudrate", g.makcu_baudrate},
        {"makcu_remote_ip", g.makcu_remote_ip},
        {"makcu_remote_port", g.makcu_remote_port},
        {"cuda_device_id", g.cuda_device_id},
        {"persistent_cache_limit_mb", g.persistent_cache_limit_mb},
        {"use_cuda_graph", g.use_cuda_graph},
        {"graph_warmup_iterations", g.graph_warmup_iterations},
        {"button_targeting", g.button_targeting},
        {"button_exit", g.button_exit},
        {"button_pause", g.button_pause},
        {"button_reload_config", g.button_reload_config},
        {"button_open_overlay", g.button_open_overlay},
        {"button_disable_upward_aim", g.button_disable_upward_aim},
        {"button_auto_action", g.button_auto_action},
        {"button_single_shot", g.button_single_shot},
        {"button_stabilizer", g.button_stabilizer},
        {"overlay_opacity", g.overlay_opacity},
        {"overlay_ui_scale", g.overlay_ui_scale},
        {"show_window", g.show_window},
        {"show_fps", g.show_fps},
        {"screenshot_button", g.screenshot_button},
        {"screenshot_delay", g.screenshot_delay},
        {"always_on_top", g.always_on_top}
    };
}

void from_json(const json& j, GlobalSettings& g) {
    // Use obfuscated JSON keys only (no fallback to original names)
    #define GET_OBF(field, obf_key) if (j.contains(obf_key)) j.at(obf_key).get_to(g.field)
    #define GET_IF(field) if (j.contains(#field)) j.at(#field).get_to(g.field)
    
    GET_IF(input_method);
    GET_OBF(arduino_baudrate, "mcu_baud");
    GET_OBF(arduino_port, "mcu_port");
    GET_OBF(arduino_enable_keys, "mcu_enable_keys");
    GET_OBF(kmbox_ip, "dev_addr");
    GET_OBF(kmbox_port, "dev_port");
    GET_OBF(kmbox_mac, "dev_hwid");
    GET_IF(makcu_port);
    GET_IF(makcu_baudrate);
    GET_IF(makcu_remote_ip);
    GET_IF(makcu_remote_port);
    GET_IF(cuda_device_id);
    GET_IF(persistent_cache_limit_mb);
    GET_IF(use_cuda_graph);
    GET_IF(graph_warmup_iterations);
    GET_OBF(button_targeting, "button_pointing");
    GET_IF(button_exit);
    GET_IF(button_pause);
    GET_IF(button_reload_config);
    GET_OBF(button_open_overlay, "button_open_layer");
    GET_OBF(button_disable_upward_aim, "button_disable_upward_fcs");
    GET_IF(button_auto_action);
    GET_IF(button_single_shot);
    GET_IF(button_stabilizer);
    GET_OBF(overlay_opacity, "layer_opacity");
    GET_OBF(overlay_ui_scale, "layer_ui_scale");
    GET_IF(show_window);
    GET_IF(show_fps);
    GET_OBF(screenshot_button, "snapshot_btn");
    GET_OBF(screenshot_delay, "snapshot_delay");
    GET_IF(always_on_top);
    #undef GET_IF
    #undef GET_OBF
}


Config::Config() {
    initializeDefaults();
}

void Config::initializeDefaults() {
    ProfileData defaultProfile;
    initializeDefaultClassSettings(defaultProfile);
    initializeDefaultInputProfiles(defaultProfile);
    profiles["Default"] = std::move(defaultProfile);
    active_profile_name = "Default";
    current_profile = &profiles["Default"];
}

void Config::initializeDefaultClassSettings(ProfileData& p) {
    p.class_settings.clear();
    p.class_settings.emplace_back(0, "Player", true);
    p.class_settings.emplace_back(1, "NPC", true);
    p.class_settings.emplace_back(2, "Item", false);
    p.class_settings.emplace_back(3, "Outline", false);
    p.class_settings.emplace_back(4, "Dead Body", false);
    p.class_settings.emplace_back(5, "Hideout Human", true);
    p.class_settings.emplace_back(6, "Hideout Balls", true);
    p.class_settings.emplace_back(7, "Head", true);
    p.class_settings.emplace_back(8, "Smoke", false);
    p.class_settings.emplace_back(9, "Fire", false);
    p.class_settings.emplace_back(10, "Third Person", false);
}

void Config::initializeDefaultInputProfiles(ProfileData& p) {
    p.input_profiles.clear();
    p.input_profiles.push_back(InputProfile{"Default", 3.0f, 1.0f});
    p.active_input_profile_index = 0;
}

std::string Config::getExecutableDir() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    return exePath.parent_path().string();
}

std::string Config::getConfigPath(const std::string& filename) {
    return getExecutableDir() + "/" + filename;
}

bool Config::loadConfig(const std::string& filename) {
    std::string configFile = filename.empty() ? getConfigPath("config.json") : filename;

    if (!std::filesystem::exists(configFile)) {
        std::cout << "[Config] Creating default config: " << configFile << std::endl;
        initializeDefaults();
        saveConfig(configFile);
        return true;
    }

    try {
        std::ifstream file(configFile);
        if (!file.is_open()) {
            std::cerr << "[Config] Failed to open: " << configFile << std::endl;
            return false;
        }

        json j;
        file >> j;

        if (j.contains("active_profile")) {
            active_profile_name = j["active_profile"].get<std::string>();
        }

        if (j.contains("global")) {
            global_settings = j["global"].get<GlobalSettings>();
        }

        profiles.clear();
        if (j.contains("profiles")) {
            for (auto& [name, data] : j["profiles"].items()) {
                profiles[name] = data.get<ProfileData>();
            }
        }

        // Ensure Default profile exists
        if (profiles.find("Default") == profiles.end()) {
            ProfileData defaultProfile;
            initializeDefaultClassSettings(defaultProfile);
            initializeDefaultInputProfiles(defaultProfile);
            profiles["Default"] = std::move(defaultProfile);
        }

        // Ensure active profile exists
        if (profiles.find(active_profile_name) == profiles.end()) {
            active_profile_name = "Default";
        }

        current_profile = &profiles[active_profile_name];

        std::cout << "[Config] Loaded: " << configFile << " (profile: " << active_profile_name << ")" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Config] Parse error: " << e.what() << std::endl;
        initializeDefaults();
        return false;
    }
}

bool Config::saveConfig(const std::string& filename) {
    std::string configFile = filename.empty() ? getConfigPath("config.json") : filename;

    try {
        json j;
        j["active_profile"] = active_profile_name;
        j["global"] = global_settings;

        for (const auto& [name, data] : profiles) {
            j["profiles"][name] = data;
        }

        std::ofstream file(configFile);
        if (!file.is_open()) {
            std::cerr << "[Config] Failed to write: " << configFile << std::endl;
            return false;
        }

        file << j.dump(2);
        std::cout << "[Config] Saved: " << configFile << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Config] Save error: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> Config::listProfiles() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : profiles) {
        names.push_back(name);
    }
    return names;
}

bool Config::switchProfile(const std::string& name) {
    auto it = profiles.find(name);
    if (it == profiles.end()) {
        std::cerr << "[Config] Profile not found: " << name << std::endl;
        return false;
    }

    active_profile_name = name;
    current_profile = &it->second;

    std::cout << "[Config] Switched to: " << name << std::endl;
    return true;
}

bool Config::createProfile(const std::string& name) {
    if (profiles.find(name) != profiles.end()) {
        return false;
    }

    ProfileData newProfile;
    initializeDefaultClassSettings(newProfile);
    initializeDefaultInputProfiles(newProfile);
    profiles[name] = std::move(newProfile);
    return true;
}

bool Config::deleteProfile(const std::string& name) {
    if (name == "Default") return false;

    auto it = profiles.find(name);
    if (it == profiles.end()) return false;

    profiles.erase(it);

    if (active_profile_name == name) {
        switchProfile("Default");
    }
    return true;
}

bool Config::duplicateProfile(const std::string& src, const std::string& dst) {
    auto it = profiles.find(src);
    if (it == profiles.end()) return false;
    if (profiles.find(dst) != profiles.end()) return false;

    profiles[dst] = it->second;
    return true;
}

InputProfile* Config::getCurrentInputProfile() {
    if (!current_profile || current_profile->input_profiles.empty()) return nullptr;

    int idx = current_profile->active_input_profile_index;
    if (idx < 0 || idx >= static_cast<int>(current_profile->input_profiles.size())) {
        idx = 0;
    }
    return &current_profile->input_profiles[idx];
}

bool Config::setActiveInputProfile(const std::string& name) {
    if (!current_profile) return false;

    for (size_t i = 0; i < current_profile->input_profiles.size(); ++i) {
        if (current_profile->input_profiles[i].profile_name == name) {
            current_profile->active_input_profile_index = static_cast<int>(i);
            return true;
        }
    }
    return false;
}

std::vector<std::string> Config::getInputProfileNames() const {
    std::vector<std::string> names;
    if (current_profile) {
        for (const auto& p : current_profile->input_profiles) {
            names.push_back(p.profile_name);
        }
    }
    return names;
}

bool Config::addInputProfile(const InputProfile& profile) {
    if (!current_profile) return false;
    
    // Check for duplicate name
    for (const auto& p : current_profile->input_profiles) {
        if (p.profile_name == profile.profile_name) {
            return false;
        }
    }
    
    current_profile->input_profiles.push_back(profile);
    return true;
}

bool Config::removeInputProfile(const std::string& name) {
    if (!current_profile) return false;
    if (current_profile->input_profiles.size() <= 1) return false; // Keep at least one
    
    for (auto it = current_profile->input_profiles.begin(); it != current_profile->input_profiles.end(); ++it) {
        if (it->profile_name == name) {
            current_profile->input_profiles.erase(it);
            if (current_profile->active_input_profile_index >= static_cast<int>(current_profile->input_profiles.size())) {
                current_profile->active_input_profile_index = 0;
            }
            return true;
        }
    }
    return false;
}

InputProfile* Config::getInputProfile(const std::string& name) {
    if (!current_profile) return nullptr;
    
    for (auto& p : current_profile->input_profiles) {
        if (p.profile_name == name) {
            return &p;
        }
    }
    return nullptr;
}
