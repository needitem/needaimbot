#include "../core/windows_headers.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <algorithm>
#include <shlobj.h>

#include "config.h"
#include "modules/SimpleIni.h"
#include "keyboard/keyboard_listener.h"

std::string Config::getExecutableDir() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    return exePath.parent_path().string();
}

std::string Config::getConfigPath(const std::string& filename) {
    return getExecutableDir() + "/" + filename;
}

std::vector<std::string> Config::splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
            item.erase(item.begin());
        while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
            item.pop_back();

        tokens.push_back(item);
    }
    return tokens;
}

std::string Config::joinStrings(const std::vector<std::string>& vec, const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (i != 0) oss << delimiter;
        oss << vec[i];
    }
    return oss.str();
}

bool Config::loadConfig(const std::string& filename)
{
    std::string configFile = filename.empty() ? getConfigPath("config.ini") : filename;
    bool is_main_config = (configFile == getConfigPath("config.ini"));
    if (!std::filesystem::exists(configFile))
    {

        
        detection_resolution = 256;  // Optimized resolution for faster inference
        monitor_idx = 0;
        circle_mask = false;  // Disabled for better performance
        capture_borders = false;  // Disabled for better performance
        capture_cursor = false;  // Disabled for better performance
 
        // NDI capture defaults
 

        
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        offset_step = 0.01f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;


        crosshair_offset_x = 0.0f;
        crosshair_offset_y = 0.0f;

        pd_kp_x = 0.4f;
        pd_kp_y = 0.4f;
        
        // Aim+shoot offset defaults
        enable_aim_shoot_offset = false;
        aim_shoot_offset_x = 0.0f;
        aim_shoot_offset_y = 0.0f;
        
        // Target lock defaults
        enable_target_lock = false;

        
        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        norecoil_ms = 10.0f;
        input_method = "WIN32";  // Default value, will be overridden by config file if present
        easynorecoil_start_delay_ms = 0;
        easynorecoil_end_delay_ms = 0;

        
        active_scope_magnification = 0;
        
        // Crouch recoil reduction
        crouch_recoil_enabled = true;
        crouch_recoil_reduction = -50.0f; // Default -50% reduction

        
        

        

        
        

        
        arduino_baudrate = 2000000;
        arduino_port = "COM0";
        arduino_enable_keys = false;

		
        kmbox_ip = "192.168.2.188";
        kmbox_port = "16896";
        kmbox_mac = "46405c53";
        
        
        makcu_port = "COM0";
        makcu_baudrate = 4000000;
        
        
        bScope_multiplier = 1.0f;

        
        ai_model = "sunxds_0.5.6.engine"; 
        confidence_threshold = 0.25f;  // Higher threshold = fewer detections = better performance
        // NMS removed - no longer needed
        confidence_weight = 0.0f;  // Ignore confidence completely
        distance_weight = 1.0f;  // Only use distance
        sticky_target_threshold = 0.0f; // Always switch to closest target
        max_detections = 30;  // Reduced from 100 for better performance
        postprocess = "yolo11";  // Changed from yolo12 to match common model format

        
        cuda_device_id = 0;

        
        button_targeting = splitString("RightMouseButton");
        button_exit = splitString("F2");
        button_pause = splitString("F3");
        button_reload_config = splitString("F4");
        button_open_overlay = splitString("Home");
        button_disable_upward_aim = splitString("None");
        button_auto_shoot = splitString("LeftMouseButton"); 
 

        
        overlay_opacity = 225;
        overlay_ui_scale = 1.0f;
        overlay_target_fps = 60.0f;  // Default 60 FPS for overlay

        
        head_class_name = "Head";
        class_settings.clear();
        class_settings.emplace_back(0, "Player", false);
        class_settings.emplace_back(1, "Bot", false);
        class_settings.emplace_back(2, "Weapon", true); 
        class_settings.emplace_back(3, "Outline", true);
        class_settings.emplace_back(4, "Dead Body", true);
        class_settings.emplace_back(5, "Hideout Human", false);
        class_settings.emplace_back(6, "Hideout Balls", false);
        class_settings.emplace_back(7, "Head", false); 
        class_settings.emplace_back(8, "Smoke", true);
        class_settings.emplace_back(9, "Fire", true);
        class_settings.emplace_back(10, "Third Person", true); 

        
        show_window = true;
        show_fps = true;
        window_name = "Debug";
        window_size = 80;
        screenshot_button = splitString("None");
        screenshot_delay = 500;
        always_on_top = true;


        // RGB Color filter defaults
        enable_color_filter = false;
        
        // RGB defaults (detect red enemies)
        rgb_min_r = 180;
        rgb_max_r = 255;
        rgb_min_g = 0;
        rgb_max_g = 80;
        rgb_min_b = 0;
        rgb_max_b = 80;
        
        // Filter settings
        min_color_pixels = 10;
        remove_color_matches = false;

        active_profile_name = "Default";

        if (is_main_config) {
            // Initialize default profile with baseline settings
            saveConfig(getConfigPath("Default.ini"));
            saveConfig(configFile);
        } else {
            saveConfig(configFile);
        }
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(configFile.c_str());
    if (rc < 0) {
        return false;
    }


    if (is_main_config) {
        const char* profile_value = ini.GetValue("Profile", "active_profile", "");
        std::string stored_profile = profile_value ? profile_value : "";
        std::string requested_profile = stored_profile.empty() ? "Default" : stored_profile;

        std::string profile_file = getConfigPath(requested_profile + ".ini");
        if (!std::filesystem::exists(profile_file) && requested_profile != "Default") {
            // Fallback to Default profile when requested profile is missing
            requested_profile = "Default";
            profile_file = getConfigPath(requested_profile + ".ini");
        }

        bool loaded = loadConfig(profile_file);
        if (loaded) {
            active_profile_name = requested_profile;
            if (requested_profile != stored_profile) {
                saveConfig(configFile);
            }
        }
        return loaded;
    }


    auto get_string_ini = [&](const char* section, const char* key, const char* defval) {
        const char* val = ini.GetValue(section, key, defval);
        return std::string(val ? val : "");
    };
    auto get_bool_ini = [&](const char* section, const char* key, bool defval) {
        return ini.GetBoolValue(section, key, defval);
    };
    auto get_long_ini = [&](const char* section, const char* key, long defval) {
        return (int)ini.GetLongValue(section, key, defval);
    };
    auto get_double_ini = [&](const char* section, const char* key, double defval) {
        return ini.GetDoubleValue(section, key, defval);
    };

    
    detection_resolution = get_long_ini("Capture", "detection_resolution", 320);
    monitor_idx = get_long_ini("Capture", "monitor_idx", 0);
    circle_mask = get_bool_ini("Capture", "circle_mask", true);
    capture_borders = get_bool_ini("Capture", "capture_borders", true);
    capture_cursor = get_bool_ini("Capture", "capture_cursor", true);
    // target_fps removed - no longer used
    

    // Batch load floats for better cache locality
    body_y_offset = static_cast<float>(get_double_ini("Target", "body_y_offset", 0.15));
    head_y_offset = static_cast<float>(get_double_ini("Target", "head_y_offset", 0.05));
    offset_step = static_cast<float>(get_double_ini("Target", "offset_step", 0.01));
    
    // Batch load booleans
    ignore_third_person = get_bool_ini("Target", "ignore_third_person", false);
    shooting_range_targets = get_bool_ini("Target", "shooting_range_targets", false);
    auto_aim = get_bool_ini("Target", "auto_aim", false);

    crosshair_offset_x = static_cast<float>(get_double_ini("Target", "crosshair_offset_x", 0.0));
    crosshair_offset_y = static_cast<float>(get_double_ini("Target", "crosshair_offset_y", 0.0));
    
    // Aim+shoot offset settings
    enable_aim_shoot_offset = get_bool_ini("Target", "enable_aim_shoot_offset", false);
    aim_shoot_offset_x = static_cast<float>(get_double_ini("Target", "aim_shoot_offset_x", 0.0));
    aim_shoot_offset_y = static_cast<float>(get_double_ini("Target", "aim_shoot_offset_y", 0.0));
    
    // Target lock settings
    enable_target_lock = get_bool_ini("Target", "enable_target_lock", false);

    easynorecoil = get_bool_ini("Mouse", "easynorecoil", false);
    easynorecoilstrength = static_cast<float>(get_double_ini("Mouse", "easynorecoilstrength", 0.0));
    norecoil_ms = static_cast<float>(get_double_ini("Mouse", "norecoil_ms", 10.0));
    
    // Load PD controller settings
    // Load PD controller settings - group by axis for cache
    pd_kp_x = static_cast<float>(get_double_ini("PDController", "pd_kp_x", 0.4));
    pd_kp_y = static_cast<float>(get_double_ini("PDController", "pd_kp_y", 0.4));
    
    input_method = get_string_ini("Mouse", "input_method", "WIN32");
    easynorecoil_start_delay_ms = get_long_ini("Mouse", "easynorecoil_start_delay_ms", 0);
    easynorecoil_end_delay_ms = get_long_ini("Mouse", "easynorecoil_end_delay_ms", 0);
    bScope_multiplier = static_cast<float>(get_double_ini("Mouse", "bScope_multiplier", 1.2));
    crouch_recoil_enabled = get_bool_ini("Mouse", "crouch_recoil_enabled", true);
    crouch_recoil_reduction = static_cast<float>(get_double_ini("Mouse", "crouch_recoil_reduction", -50.0));

    active_scope_magnification = get_long_ini("Recoil", "active_scope_magnification", 0);

    
    
    
    // Hybrid aim control settings

    arduino_baudrate = get_long_ini("Arduino", "arduino_baudrate", 2000000);
    arduino_port = get_string_ini("Arduino", "arduino_port", "COM0");
    arduino_enable_keys = get_bool_ini("Arduino", "arduino_enable_keys", false);

    kmbox_ip = get_string_ini("KMBOX", "ip", "192.168.2.188");
    kmbox_port = get_string_ini("KMBOX", "port", "16896");
    kmbox_mac = get_string_ini("KMBOX", "mac", "46405C53");

    makcu_port = get_string_ini("MAKCU", "makcu_port", "COM0");
    makcu_baudrate = get_long_ini("MAKCU", "makcu_baudrate", 4000000);

    ai_model = get_string_ini("AI", "ai_model", "sunxds_0.5.6.engine");
    confidence_threshold = static_cast<float>(get_double_ini("AI", "confidence_threshold", 0.25));
    // NMS removed - no longer needed
    confidence_weight = static_cast<float>(get_double_ini("AI", "confidence_weight", 0.0)); 
    distance_weight = static_cast<float>(get_double_ini("AI", "distance_weight", 1.0)); 
    sticky_target_threshold = static_cast<float>(get_double_ini("AI", "sticky_target_threshold", 0.0));
    max_detections = get_long_ini("AI", "max_detections", 30);
    postprocess = get_string_ini("AI", "postprocess", "yolo11");  // Changed default from yolo12 to yolo11

    cuda_device_id = get_long_ini("CUDA", "cuda_device_id", 0);
    
    // GPU performance settings
    persistent_cache_limit_mb = get_long_ini("GPU", "persistent_cache_limit_mb", 32);
    use_cuda_graph = get_bool_ini("GPU", "use_cuda_graph", false);

    button_targeting = splitString(get_string_ini("Buttons", "button_targeting", "RightMouseButton"));
    button_exit = splitString(get_string_ini("Buttons", "button_exit", "F2"));
    button_pause = splitString(get_string_ini("Buttons", "button_pause", "F3"));
    button_reload_config = splitString(get_string_ini("Buttons", "button_reload_config", "F4"));
    button_open_overlay = splitString(get_string_ini("Buttons", "button_open_overlay", "Home"));
    button_disable_upward_aim = splitString(get_string_ini("Buttons", "button_disable_upward_aim", "None"));
    button_auto_shoot = splitString(get_string_ini("Buttons", "button_auto_shoot", "LeftMouseButton"));
 

    overlay_opacity = get_long_ini("Overlay", "overlay_opacity", 225);
    overlay_ui_scale = static_cast<float>(get_double_ini("Overlay", "overlay_ui_scale", 1.0));
    overlay_target_fps = static_cast<float>(get_double_ini("Overlay", "overlay_target_fps", 60.0));

    show_window = get_bool_ini("Debug", "show_window", true);
    show_fps = get_bool_ini("Debug", "show_fps", true);
    window_name = get_string_ini("Debug", "window_name", "Debug");
    window_size = get_long_ini("Debug", "window_size", 80);
    screenshot_button = splitString(get_string_ini("Debug", "screenshot_button", "None"));
    screenshot_delay = get_long_ini("Debug", "screenshot_delay", 500);
    always_on_top = get_bool_ini("Debug", "always_on_top", true);

    // RGB Color filter settings
    enable_color_filter = get_bool_ini("ColorFilter", "enable_color_filter", false);
    
    // RGB settings
    rgb_min_r = get_long_ini("ColorFilter", "rgb_min_r", 180);
    rgb_max_r = get_long_ini("ColorFilter", "rgb_max_r", 255);
    rgb_min_g = get_long_ini("ColorFilter", "rgb_min_g", 0);
    rgb_max_g = get_long_ini("ColorFilter", "rgb_max_g", 80);
    rgb_min_b = get_long_ini("ColorFilter", "rgb_min_b", 0);
    rgb_max_b = get_long_ini("ColorFilter", "rgb_max_b", 80);
    
    // Filter settings
    min_color_pixels = get_long_ini("ColorFilter", "min_color_pixels", 10);
    remove_color_matches = get_bool_ini("ColorFilter", "remove_color_matches", false);

    
    head_class_name = get_string_ini("Classes", "HeadClassName", "Head");

    int classSettingsCount = ini.GetLongValue("ClassSettings", "Count", -1); 

    class_settings.clear(); 
    if (classSettingsCount != -1) { 
        for (int i = 0; i < classSettingsCount; ++i) {
            std::string id_key = "Class_" + std::to_string(i) + "_ID";
            std::string name_key = "Class_" + std::to_string(i) + "_Name";
            std::string allow_key = "Class_" + std::to_string(i) + "_Allow";
            
            int id_val = ini.GetLongValue("ClassSettings", id_key.c_str(), i);
            std::string name_val = ini.GetValue("ClassSettings", name_key.c_str(), ""); 
            bool allow_val = ini.GetBoolValue("ClassSettings", allow_key.c_str(), true);
            
            if (name_val.empty()) { 
                name_val = "Class " + std::to_string(id_val);
            }
            class_settings.emplace_back(id_val, name_val, allow_val);
        }
    } else { 
        bool temp_allows[11];
        temp_allows[0] = true;  // Player - allow by default
        temp_allows[1] = true;  // Bot - allow by default
        temp_allows[2] = false; // Weapon - don't allow by default
        temp_allows[3] = false; // Outline - don't allow by default
        temp_allows[4] = false; // Dead Body - don't allow by default
        temp_allows[5] = true;  // Hideout Human - allow by default
        temp_allows[6] = true;  // Hideout Balls - allow by default
        temp_allows[7] = true;  // Head - allow by default
        temp_allows[8] = false; // Smoke - don't allow by default
        temp_allows[9] = false; // Fire - don't allow by default
        temp_allows[10] = false; // Third Person - don't allow by default

        class_settings.emplace_back(0, "Player", temp_allows[0]);
        class_settings.emplace_back(1, "Bot", temp_allows[1]);
        class_settings.emplace_back(2, "Weapon", temp_allows[2]);
        class_settings.emplace_back(3, "Outline", temp_allows[3]);
        class_settings.emplace_back(4, "Dead Body", temp_allows[4]);
        class_settings.emplace_back(5, "Hideout Human", temp_allows[5]);
        class_settings.emplace_back(6, "Hideout Balls", temp_allows[6]);
        class_settings.emplace_back(7, "Head", temp_allows[7]);
        class_settings.emplace_back(8, "Smoke", temp_allows[8]);
        class_settings.emplace_back(9, "Fire", temp_allows[9]);
        class_settings.emplace_back(10, "Third Person", temp_allows[10]);
    }
    // Load weapon profiles from the config file
    weapon_profiles.clear();
    
    int weapon_count = get_long_ini("WeaponProfiles", "Count", 0);
    if (weapon_count > 0) {
        active_weapon_profile_index = get_long_ini("WeaponProfiles", "active_weapon_profile_index", 0);
        current_weapon_name = get_string_ini("WeaponProfiles", "current_weapon_name", "Default");

        for (int i = 0; i < weapon_count; ++i) {
            std::string section = "Weapon_" + std::to_string(i);
            
            WeaponRecoilProfile profile;
            profile.weapon_name = get_string_ini(section.c_str(), "weapon_name", "Default");
            profile.base_strength = static_cast<float>(get_double_ini(section.c_str(), "base_strength", 3.0));
            profile.fire_rate_multiplier = static_cast<float>(get_double_ini(section.c_str(), "fire_rate_multiplier", 1.0));
            profile.scope_mult_1x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_1x", 0.8));
            profile.scope_mult_2x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_2x", 1.0));
            profile.scope_mult_3x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_3x", 1.2));
            profile.scope_mult_4x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_4x", 1.4));
            profile.scope_mult_6x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_6x", 1.6));
            profile.scope_mult_8x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_8x", 1.8));
            profile.start_delay_ms = get_long_ini(section.c_str(), "start_delay_ms", 0);
            profile.end_delay_ms = get_long_ini(section.c_str(), "end_delay_ms", 0);
            profile.recoil_ms = static_cast<float>(get_double_ini(section.c_str(), "recoil_ms", 10.0));
            
            weapon_profiles.push_back(profile);
        }
    }
    
    // Initialize default weapon profiles if none were loaded
    if (weapon_profiles.empty()) {
        initializeDefaultWeaponProfiles();
    }
    return true;
}

bool Config::saveConfig(const std::string& filename)
{
    std::string configFile;
    bool is_main_config = false;

    if (filename.empty()) {
        std::string target_profile = active_profile_name.empty() ? "Default" : active_profile_name;
        configFile = getConfigPath(target_profile + ".ini");
    } else {
        configFile = filename;
        is_main_config = (configFile == getConfigPath("config.ini"));
    }
    std::ofstream file(configFile);
    if (!file.is_open())
    {
        std::cerr << "Error opening config for writing: " << configFile << std::endl;
        return false;
    }

    file << "# Config file generated by needaimbot\n";

    if (is_main_config) {
        file << "[Profile]\n";
        file << "active_profile = " << (active_profile_name.empty() ? "Default" : active_profile_name) << "\n";
        file.close();
        return true;
    }

    file << "[Capture]\n";
    file << "detection_resolution = " << detection_resolution << "\n";
    file << "monitor_idx = " << monitor_idx << "\n";
    file << "circle_mask = " << (circle_mask ? "true" : "false") << "\n";
    file << "capture_borders = " << (capture_borders ? "true" : "false") << "\n";
    file << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n";
    
    // target_fps removed - no longer saved
    file << "[Target]\n";
    file << std::fixed << std::setprecision(6);
    file << "body_y_offset = " << body_y_offset << "\n";
    file << "head_y_offset = " << head_y_offset << "\n";
    file << "offset_step = " << offset_step << "\n";
    file << "crosshair_offset_x = " << crosshair_offset_x << "\n";
    file << "crosshair_offset_y = " << crosshair_offset_y << "\n";
    file << "enable_aim_shoot_offset = " << (enable_aim_shoot_offset ? "true" : "false") << "\n";
    file << "aim_shoot_offset_x = " << aim_shoot_offset_x << "\n";
    file << "aim_shoot_offset_y = " << aim_shoot_offset_y << "\n";
    file << "enable_target_lock = " << (enable_target_lock ? "true" : "false") << "\n";
    file << std::noboolalpha;
    file << "ignore_third_person = " << (ignore_third_person ? "true" : "false") << "\n";
    file << "shooting_range_targets = " << (shooting_range_targets ? "true" : "false") << "\n";
    file << "auto_aim = " << (auto_aim ? "true" : "false") << "\n";
    file << "\n";


    file << "[Mouse]\n";
    file << "easynorecoil = " << (easynorecoil ? "true" : "false") << "\n";
    file << std::fixed << std::setprecision(6);
    file << "easynorecoilstrength = " << easynorecoilstrength << "\n";
    file << "norecoil_ms = " << norecoil_ms << "\n";
    
    file << std::noboolalpha;
    file << "input_method = " << input_method << "\n";
    file << std::fixed << std::setprecision(6);
    file << "easynorecoil_start_delay_ms = " << easynorecoil_start_delay_ms << "\n";
    file << "easynorecoil_end_delay_ms = " << easynorecoil_end_delay_ms << "\n";
    file << "bScope_multiplier = " << bScope_multiplier << "\n";
    file << "crouch_recoil_enabled = " << (crouch_recoil_enabled ? "true" : "false") << "\n";
    file << "crouch_recoil_reduction = " << crouch_recoil_reduction << "\n\n";

    file << "[PDController]\n";
    file << std::fixed << std::setprecision(6);
    file << "pd_kp_x = " << pd_kp_x << "\n";
    file << "pd_kp_y = " << pd_kp_y << "\n";

    file << "[Recoil]\n";
    file << "active_scope_magnification = " << active_scope_magnification << "\n\n";

    
    


    file << "[Arduino]\n";
    file << std::noboolalpha;
    file << "arduino_baudrate = " << arduino_baudrate << "\n";
    file << "arduino_port = " << arduino_port << "\n";
    file << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    file << "[KMBOX]\n";
    file << "ip = " << kmbox_ip << "\n";
    file << "port = " << kmbox_port << "\n";
    file << "mac = " << kmbox_mac << "\n\n";
    
    file << "[MAKCU]\n";
    file << "makcu_port = " << makcu_port << "\n";
    file << "makcu_baudrate = " << makcu_baudrate << "\n\n";
    
    file << "[AI]\n";
    file << "ai_model = " << ai_model << "\n";
    file << std::fixed << std::setprecision(6);
    file << "confidence_threshold = " << confidence_threshold << "\n";
    // NMS removed - no longer saved
    file << "confidence_weight = " << confidence_weight << "\n";
    file << "distance_weight = " << distance_weight << "\n";
    file << "sticky_target_threshold = " << sticky_target_threshold << "\n";
    file << std::noboolalpha;
    file << "max_detections = " << max_detections << "\n";
    file << "postprocess = " << postprocess << "\n\n";

    file << "[CUDA]\n";
    file << "cuda_device_id = " << cuda_device_id << "\n\n";
    
    file << "[GPU]\n";
    file << "persistent_cache_limit_mb = " << persistent_cache_limit_mb << "\n";
    file << "use_cuda_graph = " << (use_cuda_graph ? "true" : "false") << "\n\n";

    file << "[Buttons]\n";
    file << "button_targeting = " << joinStrings(button_targeting) << "\n";
    file << "button_exit = " << joinStrings(button_exit) << "\n";
    file << "button_pause = " << joinStrings(button_pause) << "\n";
    file << "button_reload_config = " << joinStrings(button_reload_config) << "\n";
    file << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n";
    file << "button_disable_upward_aim = " << joinStrings(button_disable_upward_aim) << "\n";
    file << "button_auto_shoot = " << joinStrings(button_auto_shoot) << "\n";
 

    file << "[Overlay]\n";
    file << "overlay_opacity = " << overlay_opacity << "\n";
    file << std::fixed << std::setprecision(6);
    file << "overlay_target_fps = " << overlay_target_fps << "\n";
    file << "overlay_ui_scale = " << overlay_ui_scale << "\n\n";

    file << "[Debug]\n";
    file << "show_window = " << (show_window ? "true" : "false") << "\n";
    file << "show_fps = " << (show_fps ? "true" : "false") << "\n";
    file << "window_name = " << window_name << "\n";
    file << "window_size = " << window_size << "\n";
    file << "screenshot_button = " << joinStrings(screenshot_button) << "\n";
    file << "screenshot_delay = " << screenshot_delay << "\n";
    file << "always_on_top = " << (always_on_top ? "true" : "false") << "\n\n";

    file << "[Classes]\n";
    file << "HeadClassName = " << head_class_name << "\n\n";

    file << "[ClassSettings]\n";
    file << "Count = " << class_settings.size() << "\n";
    for (size_t i = 0; i < class_settings.size(); ++i) {
        file << "Class_" << i << "_ID = " << class_settings[i].id << "\n";
        file << "Class_" << i << "_Name = " << class_settings[i].name << "\n";
        file << "Class_" << i << "_Allow = " << (class_settings[i].allow ? "true" : "false") << "\n";
    }
    file << "\n";


    // RGB ColorFilter section
    file << "[ColorFilter]\n";
    file << "enable_color_filter = " << (enable_color_filter ? "true" : "false") << "\n";
    file << "rgb_min_r = " << rgb_min_r << "\n";
    file << "rgb_max_r = " << rgb_max_r << "\n";
    file << "rgb_min_g = " << rgb_min_g << "\n";
    file << "rgb_max_g = " << rgb_max_g << "\n";
    file << "rgb_min_b = " << rgb_min_b << "\n";
    file << "rgb_max_b = " << rgb_max_b << "\n";
    file << "min_color_pixels = " << min_color_pixels << "\n";
    file << "remove_color_matches = " << (remove_color_matches ? "true" : "false") << "\n\n";
    
    // Save weapon profiles with each profile
    file << "[WeaponProfiles]\n";
    file << "Count = " << weapon_profiles.size() << "\n";
    file << "active_weapon_profile_index = " << active_weapon_profile_index << "\n";
    file << "current_weapon_name = " << current_weapon_name << "\n\n";

    for (size_t i = 0; i < weapon_profiles.size(); ++i) {
        const auto& profile = weapon_profiles[i];
        file << "[Weapon_" << i << "]\n";
        file << "weapon_name = " << profile.weapon_name << "\n";
        file << std::fixed << std::setprecision(6);
        file << "base_strength = " << profile.base_strength << "\n";
        file << "fire_rate_multiplier = " << profile.fire_rate_multiplier << "\n";
        file << "scope_mult_1x = " << profile.scope_mult_1x << "\n";
        file << "scope_mult_2x = " << profile.scope_mult_2x << "\n";
        file << "scope_mult_3x = " << profile.scope_mult_3x << "\n";
        file << "scope_mult_4x = " << profile.scope_mult_4x << "\n";
        file << "scope_mult_6x = " << profile.scope_mult_6x << "\n";
        file << "scope_mult_8x = " << profile.scope_mult_8x << "\n";
        file << "start_delay_ms = " << profile.start_delay_ms << "\n";
        file << "end_delay_ms = " << profile.end_delay_ms << "\n";
        file << "recoil_ms = " << profile.recoil_ms << "\n\n";
    }
    
    file.close();
    return true;
}

std::vector<std::string> Config::listProfiles() {
    std::vector<std::string> profiles;
    std::string config_dir = getExecutableDir(); 
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string extension = entry.path().extension().string();

                if (extension == ".ini" && filename != "config.ini") { 
                    std::string profileName = entry.path().stem().string(); 
                    profiles.push_back(profileName);
                }
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        // Error accessing profiles directory - ignore and return empty list
    }
    
    std::sort(profiles.begin(), profiles.end());
    return profiles;
}

bool Config::saveProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
        return false;
    }
    std::string filename = getConfigPath(profileName + ".ini");
    return saveConfig(filename); 
}

bool Config::loadProfile(const std::string& profileName) {
     if (profileName.empty()) {
        return false;
    }
    std::string filename = getConfigPath(profileName + ".ini");

    if (!std::filesystem::exists(filename)) {
         return false; 
    }
    return loadConfig(filename); 
}

bool Config::deleteProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
        return false;
    }
    std::string filename = getConfigPath(profileName + ".ini");

    try {
        if (std::filesystem::exists(filename)) {
            if (std::filesystem::remove(filename)) {
                return true;
            } else {
                return false;
            }
        } else {
            return false; 
        }
    } catch (const std::filesystem::filesystem_error&) {
        return false;
    }
}

void Config::resetConfig()
{
    
    
    loadConfig("__dummy_nonexistent_file_for_reset__.ini"); 
    
}

bool Config::setActiveProfile(const std::string& profileName) {
    // Early return for empty
    if (profileName.empty()) return false;
    
    // Load and save in one flow
    if (!loadProfile(profileName)) return false;
    
    active_profile_name = profileName;
    saveConfig(getConfigPath("config.ini"));
    return true;
}

bool Config::saveActiveProfile() {
    std::string profile_to_save = active_profile_name.empty() ? "Default" : active_profile_name;

    if (!saveProfile(profile_to_save)) {
        return false;
    }

    active_profile_name = profile_to_save;
    saveConfig(getConfigPath("config.ini"));
    return true;
}

bool Config::isProfileModified() const {
    // This would require storing a copy of the last loaded profile settings
    // For now, we'll return false - can be implemented later if needed
    return false;
}


Config::Config()
{
    // Ensure we use the correct path for config files
    std::string exePath = getExecutableDir();
    
    // Use empty string to trigger default path logic
    loadConfig("");
    // Weapon profiles are now loaded from the profile INI file itself
}

void Config::initializeDefaultWeaponProfiles()
{
    if (weapon_profiles.empty()) {
        weapon_profiles.push_back(WeaponRecoilProfile("Default", 3.0f, 1.0f));
        weapon_profiles.push_back(WeaponRecoilProfile("AK47", 4.5f, 1.2f));
        weapon_profiles.push_back(WeaponRecoilProfile("M4A4", 3.5f, 1.1f));
        weapon_profiles.push_back(WeaponRecoilProfile("AWP", 2.0f, 0.8f));
        weapon_profiles.push_back(WeaponRecoilProfile("MP5", 2.5f, 1.3f));
        
        active_weapon_profile_index = 0;
        current_weapon_name = "Default";
    }
}

bool Config::addWeaponProfile(const WeaponRecoilProfile& profile)
{
    for (const auto& existing : weapon_profiles) {
        if (existing.weapon_name == profile.weapon_name) {
            return false;
        }
    }
    weapon_profiles.push_back(profile);
    return true;
}

bool Config::removeWeaponProfile(const std::string& weapon_name)
{
    if (weapon_name == "Default") return false;
    
    auto it = std::find_if(weapon_profiles.begin(), weapon_profiles.end(),
        [&weapon_name](const WeaponRecoilProfile& profile) {
            return profile.weapon_name == weapon_name;
        });
    
    if (it != weapon_profiles.end()) {
        weapon_profiles.erase(it);
        if (current_weapon_name == weapon_name) {
            setActiveWeaponProfile("Default");
        }
        return true;
    }
    return false;
}

WeaponRecoilProfile* Config::getWeaponProfile(const std::string& weapon_name)
{
    for (auto& profile : weapon_profiles) {
        if (profile.weapon_name == weapon_name) {
            return &profile;
        }
    }
    return nullptr;
}

WeaponRecoilProfile* Config::getCurrentWeaponProfile()
{
    if (active_weapon_profile_index >= 0 && 
        active_weapon_profile_index < static_cast<int>(weapon_profiles.size())) {
        return &weapon_profiles[active_weapon_profile_index];
    }
    return nullptr;
}

bool Config::setActiveWeaponProfile(const std::string& weapon_name)
{
    for (size_t i = 0; i < weapon_profiles.size(); ++i) {
        if (weapon_profiles[i].weapon_name == weapon_name) {
            active_weapon_profile_index = static_cast<int>(i);
            current_weapon_name = weapon_name;
            return true;
        }
    }
    return false;
}

std::vector<std::string> Config::getWeaponProfileNames() const
{
    std::vector<std::string> names;
    for (const auto& profile : weapon_profiles) {
        names.push_back(profile.weapon_name);
    }
    return names;
}

// Deprecated - weapon profiles are now saved within saveConfig
/*
bool Config::saveWeaponProfiles(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening weapon profiles file for writing: " << filename << std::endl;
        return false;
    }

    file << "# Weapon Profiles for needaimbot\n";
    file << "[WeaponProfiles]\n";
    file << "Count = " << weapon_profiles.size() << "\n";
    file << "active_weapon_profile_index = " << active_weapon_profile_index << "\n";
    file << "current_weapon_name = " << current_weapon_name << "\n\n";

    for (size_t i = 0; i < weapon_profiles.size(); ++i) {
        const auto& profile = weapon_profiles[i];
        file << "[Weapon_" << i << "]\n";
        file << "weapon_name = " << profile.weapon_name << "\n";
        file << std::fixed << std::setprecision(6);
        file << "base_strength = " << profile.base_strength << "\n";
        file << "fire_rate_multiplier = " << profile.fire_rate_multiplier << "\n";
        file << "scope_mult_1x = " << profile.scope_mult_1x << "\n";
        file << "scope_mult_2x = " << profile.scope_mult_2x << "\n";
        file << "scope_mult_3x = " << profile.scope_mult_3x << "\n";
        file << "scope_mult_4x = " << profile.scope_mult_4x << "\n";
        file << "scope_mult_6x = " << profile.scope_mult_6x << "\n";
        file << "scope_mult_8x = " << profile.scope_mult_8x << "\n";
        file << "start_delay_ms = " << profile.start_delay_ms << "\n";
        file << "end_delay_ms = " << profile.end_delay_ms << "\n";
        file << "recoil_ms = " << profile.recoil_ms << "\n\n";
    }

    file.close();
    return true;
}
*/

// Deprecated - weapon profiles are now loaded within loadConfig
/*
bool Config::loadWeaponProfiles(const std::string& filename)
{
    if (!std::filesystem::exists(filename)) {
        return false;
    }

    CSimpleIniA ini;
    SI_Error rc = ini.LoadFile(filename.c_str());
    if (rc < 0) {
        return false;
    }

    auto get_string_ini = [&](const char* section, const char* key, const char* default_val) -> std::string {
        const char* value = ini.GetValue(section, key, default_val);
        return value ? std::string(value) : std::string(default_val);
    };

    auto get_double_ini = [&](const char* section, const char* key, double default_val) -> double {
        return ini.GetDoubleValue(section, key, default_val);
    };

    auto get_long_ini = [&](const char* section, const char* key, long default_val) -> long {
        return ini.GetLongValue(section, key, default_val);
    };

    weapon_profiles.clear();
    
    int count = get_long_ini("WeaponProfiles", "Count", 0);
    active_weapon_profile_index = get_long_ini("WeaponProfiles", "active_weapon_profile_index", 0);
    current_weapon_name = get_string_ini("WeaponProfiles", "current_weapon_name", "Default");

    for (int i = 0; i < count; ++i) {
        std::string section = "Weapon_" + std::to_string(i);
        
        WeaponRecoilProfile profile;
        profile.weapon_name = get_string_ini(section.c_str(), "weapon_name", "Default");
        profile.base_strength = static_cast<float>(get_double_ini(section.c_str(), "base_strength", 3.0));
        profile.fire_rate_multiplier = static_cast<float>(get_double_ini(section.c_str(), "fire_rate_multiplier", 1.0));
        profile.scope_mult_1x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_1x", 0.8));
        profile.scope_mult_2x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_2x", 1.0));
        profile.scope_mult_3x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_3x", 1.2));
        profile.scope_mult_4x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_4x", 1.4));
        profile.scope_mult_6x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_6x", 1.6));
        profile.scope_mult_8x = static_cast<float>(get_double_ini(section.c_str(), "scope_mult_8x", 1.8));
        profile.start_delay_ms = get_long_ini(section.c_str(), "start_delay_ms", 0);
        profile.end_delay_ms = get_long_ini(section.c_str(), "end_delay_ms", 0);
        profile.recoil_ms = static_cast<float>(get_double_ini(section.c_str(), "recoil_ms", 10.0));
        
        weapon_profiles.push_back(profile);
    }

    return true;
}
*/

