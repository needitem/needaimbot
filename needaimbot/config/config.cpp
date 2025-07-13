#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <algorithm>
#include <shlobj.h>

#include "config.h"
#include "config_validator.h"
#include "modules/SimpleIni.h"
#include "keyboard/keyboard_listener.h"

std::string Config::getExecutableDir() {
    std::filesystem::path exePath = std::filesystem::current_path();
    return exePath.string();
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
    if (!std::filesystem::exists(configFile))
    {

        
        detection_resolution = 320;
        capture_fps = 0;  // 0 = unlimited FPS
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
 
        target_fps = 120.0f;
        capture_method = "simple";
        target_game_name = ""; 

        
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        offset_step = 0.01f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;

        
        crosshair_offset_x = 0.0f;
        crosshair_offset_y = 0.0f;

        
        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        norecoil_step = 5.0f;
        norecoil_ms = 10.0f;
        input_method = "WIN32";  // Default value, will be overridden by config file if present
        easynorecoil_start_delay_ms = 0;
        easynorecoil_end_delay_ms = 0;

        
        active_scope_magnification = 0;
        recoil_mult_2x = 1.0f;
        recoil_mult_3x = 1.0f;
        recoil_mult_4x = 1.0f;
        recoil_mult_6x = 1.0f;

        
        

        
        use_predictive_controller = true;
        prediction_time_ms = 50.0f;
        kalman_process_noise = 10.0f;
        kalman_measurement_noise = 5.0f;
        
        // Sub-pixel and prediction defaults
        enable_subpixel_dithering = true;
        dither_strength = 0.3f;
        enable_velocity_history = true;
        velocity_history_size = 5;
        prediction_time_factor = 0.001f;

        
        kp_x = 0.5; 
        ki_x = 0.0;
        kd_x = 0.1;
        kp_y = 0.4; 
        ki_y = 0.0;
        kd_y = 0.15;
        
        pid_derivative_smoothing = 0.2f;
        enable_adaptive_pid = true;

        
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;

		
        kmbox_ip = "192.168.2.188";
        kmbox_port = "16896";
        kmbox_mac = "46405c53";
        
        
        bScope_multiplier = 1.0f;

        
        ai_model = "sunxds_0.5.6.engine"; 
        confidence_threshold = 0.15f;
        nms_threshold = 0.50f;
        confidence_weight = 0.65f; 
        distance_weight = 0.35f; 
        sticky_target_threshold = 0.8f; // Default: new target must be 20% better
        max_detections = 100;
        postprocess = "yolo10";
        export_enable_fp8 = false;
        export_enable_fp16 = true;
        tensorrt_fp16 = true;
        onnx_input_resolution = 640;

        
        cuda_device_id = 0;

        
        button_targeting = splitString("RightMouseButton");
        button_exit = splitString("F2");
        button_pause = splitString("F3");
        button_reload_config = splitString("F4");
        button_open_overlay = splitString("Home");
        button_disable_upward_aim = splitString("None");
        button_auto_shoot = splitString("None"); 
 

        
        overlay_opacity = 225;
        overlay_ui_scale = 1.0f;

        
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
        verbose = false;


        
        enable_hsv_filter = false;
        hsv_lower_h = 0;
        hsv_lower_s = 0;
        hsv_lower_v = 0;
        hsv_upper_h = 179;
        hsv_upper_s = 255;
        hsv_upper_v = 255;
        min_hsv_pixels = 10;
        remove_hsv_matches = false;

        saveConfig(configFile); 
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(configFile.c_str());
    if (rc < 0) {
        return false;
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
    capture_fps = get_long_ini("Capture", "capture_fps", 0);  // 0 = unlimited FPS
    monitor_idx = get_long_ini("Capture", "monitor_idx", 0);
    circle_mask = get_bool_ini("Capture", "circle_mask", true);
    capture_borders = get_bool_ini("Capture", "capture_borders", true);
    capture_cursor = get_bool_ini("Capture", "capture_cursor", true);
    target_fps = (float)get_double_ini("Capture", "target_fps", 120.0);
    capture_method = get_string_ini("Capture", "capture_method", "simple");
    target_game_name = get_string_ini("Capture", "target_game_name", "");

    body_y_offset = (float)get_double_ini("Target", "body_y_offset", 0.15);
    head_y_offset = (float)get_double_ini("Target", "head_y_offset", 0.05);
    offset_step = (float)get_double_ini("Target", "offset_step", 0.01);
    ignore_third_person = get_bool_ini("Target", "ignore_third_person", false);
    shooting_range_targets = get_bool_ini("Target", "shooting_range_targets", false);
    auto_aim = get_bool_ini("Target", "auto_aim", false);

    crosshair_offset_x = (float)get_double_ini("Target", "crosshair_offset_x", 0.0);
    crosshair_offset_y = (float)get_double_ini("Target", "crosshair_offset_y", 0.0);

    easynorecoil = get_bool_ini("Mouse", "easynorecoil", false);
    easynorecoilstrength = (float)get_double_ini("Mouse", "easynorecoilstrength", 0.0);
    norecoil_step = (float)get_double_ini("Mouse", "norecoil_step", 5.0);
    norecoil_ms = (float)get_double_ini("Mouse", "norecoil_ms", 10.0);
    input_method = get_string_ini("Mouse", "input_method", "WIN32");
    easynorecoil_start_delay_ms = get_long_ini("Mouse", "easynorecoil_start_delay_ms", 0);
    easynorecoil_end_delay_ms = get_long_ini("Mouse", "easynorecoil_end_delay_ms", 0);
    bScope_multiplier = (float)get_double_ini("Mouse", "bScope_multiplier", 1.2);

    active_scope_magnification = get_long_ini("Recoil", "active_scope_magnification", 0);
    recoil_mult_2x = (float)get_double_ini("Recoil", "recoil_mult_2x", 1.0);
    recoil_mult_3x = (float)get_double_ini("Recoil", "recoil_mult_3x", 1.0);
    recoil_mult_4x = (float)get_double_ini("Recoil", "recoil_mult_4x", 1.0);
    recoil_mult_6x = (float)get_double_ini("Recoil", "recoil_mult_6x", 1.0);

    
    
    kp_x = get_double_ini("PID", "kp_x", 0.5);
    ki_x = get_double_ini("PID", "ki_x", 0.0);
    kd_x = get_double_ini("PID", "kd_x", 0.1);
    kp_y = get_double_ini("PID", "kp_y", 0.4);
    ki_y = get_double_ini("PID", "ki_y", 0.0);
    kd_y = get_double_ini("PID", "kd_y", 0.15);
    
    pid_derivative_smoothing = (float)get_double_ini("PID", "pid_derivative_smoothing", 0.2);
    enable_adaptive_pid = get_bool_ini("PID", "enable_adaptive_pid", true);

    use_predictive_controller = get_bool_ini("PID", "use_predictive_controller", true);
    prediction_time_ms = (float)get_double_ini("PID", "prediction_time_ms", 50.0);
    kalman_process_noise = (float)get_double_ini("PID", "kalman_process_noise", 10.0);
    kalman_measurement_noise = (float)get_double_ini("PID", "kalman_measurement_noise", 5.0);
    
    // Sub-pixel and prediction settings
    enable_subpixel_dithering = get_bool_ini("PID", "enable_subpixel_dithering", true);
    dither_strength = (float)get_double_ini("PID", "dither_strength", 0.3);
    enable_velocity_history = get_bool_ini("PID", "enable_velocity_history", true);
    velocity_history_size = get_long_ini("PID", "velocity_history_size", 5);
    prediction_time_factor = (float)get_double_ini("PID", "prediction_time_factor", 0.001);
    
    // Hybrid aim control settings

    arduino_baudrate = get_long_ini("Arduino", "arduino_baudrate", 115200);
    arduino_port = get_string_ini("Arduino", "arduino_port", "COM0");
    arduino_16_bit_mouse = get_bool_ini("Arduino", "arduino_16_bit_mouse", false);
    arduino_enable_keys = get_bool_ini("Arduino", "arduino_enable_keys", false);

    kmbox_ip = get_string_ini("KMBOX", "ip", "192.168.2.188");
    kmbox_port = get_string_ini("KMBOX", "port", "16896");
    kmbox_mac = get_string_ini("KMBOX", "mac", "46405C53");

    ai_model = get_string_ini("AI", "ai_model", "sunxds_0.5.6.engine");
    confidence_threshold = (float)get_double_ini("AI", "confidence_threshold", 0.15);
    nms_threshold = (float)get_double_ini("AI", "nms_threshold", 0.50);
    confidence_weight = (float)get_double_ini("AI", "confidence_weight", 0.65); 
    distance_weight = (float)get_double_ini("AI", "distance_weight", 0.35); 
    sticky_target_threshold = (float)get_double_ini("AI", "sticky_target_threshold", 0.8);
    max_detections = get_long_ini("AI", "max_detections", 20);
    postprocess = get_string_ini("AI", "postprocess", "yolo10");
    export_enable_fp8 = get_bool_ini("AI", "export_enable_fp8", false);
    export_enable_fp16 = get_bool_ini("AI", "export_enable_fp16", true);
    tensorrt_fp16 = get_bool_ini("AI", "tensorrt_fp16", true);
    onnx_input_resolution = get_long_ini("AI", "onnx_input_resolution", 640);

    cuda_device_id = get_long_ini("CUDA", "cuda_device_id", 0);

    button_targeting = splitString(get_string_ini("Buttons", "button_targeting", "RightMouseButton"));
    button_exit = splitString(get_string_ini("Buttons", "button_exit", "F2"));
    button_pause = splitString(get_string_ini("Buttons", "button_pause", "F3"));
    button_reload_config = splitString(get_string_ini("Buttons", "button_reload_config", "F4"));
    button_open_overlay = splitString(get_string_ini("Buttons", "button_open_overlay", "Home"));
    button_disable_upward_aim = splitString(get_string_ini("Buttons", "button_disable_upward_aim", "None"));
    button_auto_shoot = splitString(get_string_ini("Buttons", "button_auto_shoot", "None"));
 

    overlay_opacity = get_long_ini("Overlay", "overlay_opacity", 225);
    overlay_ui_scale = (float)get_double_ini("Overlay", "overlay_ui_scale", 1.0);

    show_window = get_bool_ini("Debug", "show_window", true);
    show_fps = get_bool_ini("Debug", "show_fps", true);
    window_name = get_string_ini("Debug", "window_name", "Debug");
    window_size = get_long_ini("Debug", "window_size", 80);
    screenshot_button = splitString(get_string_ini("Debug", "screenshot_button", "None"));
    screenshot_delay = get_long_ini("Debug", "screenshot_delay", 500);
    always_on_top = get_bool_ini("Debug", "always_on_top", true);
    verbose = get_bool_ini("Debug", "verbose", false);

    

    
    enable_hsv_filter = get_bool_ini("HSVFilter", "enable_hsv_filter", false);
    hsv_lower_h = get_long_ini("HSVFilter", "hsv_lower_h", 0);
    hsv_lower_s = get_long_ini("HSVFilter", "hsv_lower_s", 0);
    hsv_lower_v = get_long_ini("HSVFilter", "hsv_lower_v", 0);
    hsv_upper_h = get_long_ini("HSVFilter", "hsv_upper_h", 179);
    hsv_upper_s = get_long_ini("HSVFilter", "hsv_upper_s", 255);
    hsv_upper_v = get_long_ini("HSVFilter", "hsv_upper_v", 255);
    min_hsv_pixels = get_long_ini("HSVFilter", "min_hsv_pixels", 10);
    remove_hsv_matches = get_bool_ini("HSVFilter", "remove_hsv_matches", false);

    
    head_class_name = get_string_ini("Classes", "HeadClassName", "Head");

    int classSettingsCount = ini.GetLongValue("ClassSettings", "Count", -1); 

    class_settings.clear(); 
    if (classSettingsCount != -1) { 
        for (int i = 0; i < classSettingsCount; ++i) {
            std::string id_key = "Class_" + std::to_string(i) + "_ID";
            std::string name_key = "Class_" + std::to_string(i) + "_Name";
            std::string ignore_key = "Class_" + std::to_string(i) + "_Ignore";
            
            int id_val = ini.GetLongValue("ClassSettings", id_key.c_str(), i);
            std::string name_val = ini.GetValue("ClassSettings", name_key.c_str(), ""); 
            bool ignore_val = ini.GetBoolValue("ClassSettings", ignore_key.c_str(), false);
            
            if (name_val.empty()) { 
                name_val = "Class " + std::to_string(id_val);
            }
            class_settings.emplace_back(id_val, name_val, ignore_val);
        }
    } else { 
        bool temp_ignores[11];
        temp_ignores[0] = ini.GetBoolValue("Ignore Classes", "ignore_class_0", false);
        temp_ignores[1] = ini.GetBoolValue("Ignore Classes", "ignore_class_1", false);
        temp_ignores[2] = ini.GetBoolValue("Ignore Classes", "ignore_class_2", true); 
        temp_ignores[3] = ini.GetBoolValue("Ignore Classes", "ignore_class_3", true);
        temp_ignores[4] = ini.GetBoolValue("Ignore Classes", "ignore_class_4", true);
        temp_ignores[5] = ini.GetBoolValue("Ignore Classes", "ignore_class_5", false);
        temp_ignores[6] = ini.GetBoolValue("Ignore Classes", "ignore_class_6", false);
        temp_ignores[7] = ini.GetBoolValue("Ignore Classes", "ignore_class_7", false); 
        temp_ignores[8] = ini.GetBoolValue("Ignore Classes", "ignore_class_8", true);
        temp_ignores[9] = ini.GetBoolValue("Ignore Classes", "ignore_class_9", true);
        temp_ignores[10] = ini.GetBoolValue("Ignore Classes", "ignore_class_10", true);

        class_settings.emplace_back(0, "Player", temp_ignores[0]);
        class_settings.emplace_back(1, "Bot", temp_ignores[1]);
        class_settings.emplace_back(2, "Weapon", temp_ignores[2]);
        class_settings.emplace_back(3, "Outline", temp_ignores[3]);
        class_settings.emplace_back(4, "Dead Body", temp_ignores[4]);
        class_settings.emplace_back(5, "Hideout Human", temp_ignores[5]);
        class_settings.emplace_back(6, "Hideout Balls", temp_ignores[6]);
        class_settings.emplace_back(7, "Head", temp_ignores[7]);
        class_settings.emplace_back(8, "Smoke", temp_ignores[8]);
        class_settings.emplace_back(9, "Fire", temp_ignores[9]);
        class_settings.emplace_back(10, "Third Person", temp_ignores[10]);
    }
    
    


    // Validate and correct all configuration values
    ConfigValidator::validateAndCorrect(*this);

    return true;
}

bool Config::saveConfig(const std::string& filename)
{
    std::string configFile = filename.empty() ? getConfigPath("config.ini") : filename;
    std::ofstream file(configFile);
    if (!file.is_open())
    {
        std::cerr << "Error opening config for writing: " << configFile << std::endl;
        return false;
    }

    file << "# Config file generated by needaimbot\n";

    file << "[Capture]\n";
    file << "detection_resolution = " << detection_resolution << "\n";
    file << "capture_fps = " << capture_fps << "\n";
    file << "monitor_idx = " << monitor_idx << "\n";
    file << "circle_mask = " << (circle_mask ? "true" : "false") << "\n";
    file << "capture_borders = " << (capture_borders ? "true" : "false") << "\n";
    file << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n";
    
    file << "target_fps = " << target_fps << "\n";
    file << "capture_method = " << capture_method << "\n";
    file << "target_game_name = " << target_game_name << "\n\n";

    file << "[Target]\n";
    file << std::fixed << std::setprecision(6);
    file << "body_y_offset = " << body_y_offset << "\n";
    file << "head_y_offset = " << head_y_offset << "\n";
    file << "offset_step = " << offset_step << "\n";
    file << "crosshair_offset_x = " << crosshair_offset_x << "\n";
    file << "crosshair_offset_y = " << crosshair_offset_y << "\n";
    file << std::noboolalpha;
    file << "ignore_third_person = " << (ignore_third_person ? "true" : "false") << "\n";
    file << "shooting_range_targets = " << (shooting_range_targets ? "true" : "false") << "\n";
    file << "auto_aim = " << (auto_aim ? "true" : "false") << "\n\n";

    file << "[Mouse]\n";
    file << "easynorecoil = " << (easynorecoil ? "true" : "false") << "\n";
    file << std::fixed << std::setprecision(6);
    file << "easynorecoilstrength = " << easynorecoilstrength << "\n";
    file << "norecoil_step = " << norecoil_step << "\n";
    file << "norecoil_ms = " << norecoil_ms << "\n";
    file << std::noboolalpha;
    file << "input_method = " << input_method << "\n";
    file << std::fixed << std::setprecision(6);
    file << "easynorecoil_start_delay_ms = " << easynorecoil_start_delay_ms << "\n";
    file << "easynorecoil_end_delay_ms = " << easynorecoil_end_delay_ms << "\n";
    file << "bScope_multiplier = " << bScope_multiplier << "\n\n";

    file << "[Recoil]\n";
    file << "active_scope_magnification = " << active_scope_magnification << "\n";
    file << std::fixed << std::setprecision(6);
    file << "recoil_mult_2x = " << recoil_mult_2x << "\n";
    file << "recoil_mult_3x = " << recoil_mult_3x << "\n";
    file << "recoil_mult_4x = " << recoil_mult_4x << "\n";
    file << "recoil_mult_6x = " << recoil_mult_6x << "\n\n";

    
    
    
    file << "[PID]\n";
    file << std::fixed << std::setprecision(6);
    file << "kp_x = " << kp_x << "\n";
    file << "ki_x = " << ki_x << "\n";
    file << "kd_x = " << kd_x << "\n";
    file << "kp_y = " << kp_y << "\n";
    file << "ki_y = " << ki_y << "\n";
    file << "kd_y = " << kd_y << "\n";
    file << "pid_derivative_smoothing = " << pid_derivative_smoothing << "\n";
    file << "enable_adaptive_pid = " << (enable_adaptive_pid ? "true" : "false") << "\n";
    file << "use_predictive_controller = " << (use_predictive_controller ? "true" : "false") << "\n";
    file << "prediction_time_ms = " << prediction_time_ms << "\n";
    file << "kalman_process_noise = " << kalman_process_noise << "\n";
    file << "kalman_measurement_noise = " << kalman_measurement_noise << "\n";
    file << "enable_subpixel_dithering = " << (enable_subpixel_dithering ? "true" : "false") << "\n";
    file << "dither_strength = " << dither_strength << "\n";
    file << "enable_velocity_history = " << (enable_velocity_history ? "true" : "false") << "\n";
    file << std::noboolalpha;
    file << "velocity_history_size = " << velocity_history_size << "\n";
    file << std::fixed << std::setprecision(6);
    file << "prediction_time_factor = " << prediction_time_factor << "\n\n";

    file << "[Arduino]\n";
    file << std::noboolalpha;
    file << "arduino_baudrate = " << arduino_baudrate << "\n";
    file << "arduino_port = " << arduino_port << "\n";
    file << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n";
    file << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    file << "[KMBOX]\n";
    file << "ip = " << kmbox_ip << "\n";
    file << "port = " << kmbox_port << "\n";
    file << "mac = " << kmbox_mac << "\n\n";
    
    file << "[AI]\n";
    file << "ai_model = " << ai_model << "\n";
    file << std::fixed << std::setprecision(6);
    file << "confidence_threshold = " << confidence_threshold << "\n";
    file << "nms_threshold = " << nms_threshold << "\n";
    file << "confidence_weight = " << confidence_weight << "\n";
    file << "distance_weight = " << distance_weight << "\n";
    file << "sticky_target_threshold = " << sticky_target_threshold << "\n";
    file << std::noboolalpha;
    file << "max_detections = " << max_detections << "\n";
    file << "postprocess = " << postprocess << "\n";
    file << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n";
    file << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n";
    file << "tensorrt_fp16 = " << (tensorrt_fp16 ? "true" : "false") << "\n";
    file << "onnx_input_resolution = " << onnx_input_resolution << "\n\n";

    file << "[CUDA]\n";
    file << "cuda_device_id = " << cuda_device_id << "\n\n";

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
    file << "overlay_ui_scale = " << overlay_ui_scale << "\n\n";

    file << "[Debug]\n";
    file << "show_window = " << (show_window ? "true" : "false") << "\n";
    file << "show_fps = " << (show_fps ? "true" : "false") << "\n";
    file << "window_name = " << window_name << "\n";
    file << "window_size = " << window_size << "\n";
    file << "screenshot_button = " << joinStrings(screenshot_button) << "\n";
    file << "screenshot_delay = " << screenshot_delay << "\n";
    file << "always_on_top = " << (always_on_top ? "true" : "false") << "\n";
    file << "verbose = " << (verbose ? "true" : "false") << "\n\n";

    file << "[Classes]\n";
    file << "HeadClassName = " << head_class_name << "\n\n";

    file << "[ClassSettings]\n";
    file << "Count = " << class_settings.size() << "\n";
    for (size_t i = 0; i < class_settings.size(); ++i) {
        file << "Class_" << i << "_ID = " << class_settings[i].id << "\n";
        file << "Class_" << i << "_Name = " << class_settings[i].name << "\n";
        file << "Class_" << i << "_Ignore = " << (class_settings[i].ignore ? "true" : "false") << "\n";
    }
    file << "\n";


    file << "[HSVFilter]\n";
    file << "enable_hsv_filter = " << (enable_hsv_filter ? "true" : "false") << "\n";
    file << "hsv_lower_h = " << hsv_lower_h << "\n";
    file << "hsv_lower_s = " << hsv_lower_s << "\n";
    file << "hsv_lower_v = " << hsv_lower_v << "\n";
    file << "hsv_upper_h = " << hsv_upper_h << "\n";
    file << "hsv_upper_s = " << hsv_upper_s << "\n";
    file << "hsv_upper_v = " << hsv_upper_v << "\n";
    file << "min_hsv_pixels = " << min_hsv_pixels << "\n";
    file << "remove_hsv_matches = " << (remove_hsv_matches ? "true" : "false") << "\n\n";
    
    file.close();
    return true;
}

std::vector<std::string> Config::listProfiles() {
    std::vector<std::string> profiles;
    std::string current_path_str = "."; 
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(current_path_str)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string extension = entry.path().extension().string();

                if (extension == ".ini" && filename != "config.ini") { 
                    std::string profileName = entry.path().stem().string(); 
                    profiles.push_back(profileName);
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
    }
    
    std::sort(profiles.begin(), profiles.end());
    return profiles;
}

bool Config::saveProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
        return false;
    }
    std::string filename = profileName + ".ini";
    return saveConfig(filename); 
}

bool Config::loadProfile(const std::string& profileName) {
     if (profileName.empty()) {
        return false;
    }
    std::string filename = profileName + ".ini";

    if (!std::filesystem::exists(filename)) {
         return false; 
    }
    return loadConfig(filename); 
}

bool Config::deleteProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
        return false;
    }
    std::string filename = profileName + ".ini";

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
    } catch (const std::filesystem::filesystem_error& e) {
        return false;
    }
}

void Config::resetConfig()
{
    
    
    loadConfig("__dummy_nonexistent_file_for_reset__.ini"); 
    
}


Config::Config()
{
    // Ensure we use the correct path for config files
    std::string exePath = getExecutableDir();
    
    // Use empty string to trigger default path logic
    loadConfig("");
    loadWeaponProfiles(getConfigPath("weapon_profiles.ini"));
    initializeDefaultWeaponProfiles();
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
        profile.base_strength = (float)get_double_ini(section.c_str(), "base_strength", 3.0);
        profile.fire_rate_multiplier = (float)get_double_ini(section.c_str(), "fire_rate_multiplier", 1.0);
        profile.scope_mult_1x = (float)get_double_ini(section.c_str(), "scope_mult_1x", 0.8);
        profile.scope_mult_2x = (float)get_double_ini(section.c_str(), "scope_mult_2x", 1.0);
        profile.scope_mult_3x = (float)get_double_ini(section.c_str(), "scope_mult_3x", 1.2);
        profile.scope_mult_4x = (float)get_double_ini(section.c_str(), "scope_mult_4x", 1.4);
        profile.scope_mult_6x = (float)get_double_ini(section.c_str(), "scope_mult_6x", 1.6);
        profile.scope_mult_8x = (float)get_double_ini(section.c_str(), "scope_mult_8x", 1.8);
        profile.start_delay_ms = get_long_ini(section.c_str(), "start_delay_ms", 0);
        profile.end_delay_ms = get_long_ini(section.c_str(), "end_delay_ms", 0);
        profile.recoil_ms = (float)get_double_ini(section.c_str(), "recoil_ms", 10.0);
        
        weapon_profiles.push_back(profile);
    }

    return true;
}

