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
#include "modules/SimpleIni.h"
#include "keyboard/keyboard_listener.h"

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
    if (!std::filesystem::exists(filename))
    {
        std::cerr << "[Config] Config file does not exist, creating default config: " << filename << std::endl;

        
        detection_resolution = 320;
        capture_fps = 60;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        capture_use_cuda = true;
        use_1ms_capture = false;
        capture_timeout_ms = 5; 
        target_fps = 120.0f; 

        
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        offset_step = 0.01f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;

        
        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        norecoil_step = 5.0f;
        norecoil_ms = 10.0f;
        input_method = "WIN32";
        easynorecoil_start_delay_ms = 0;
        easynorecoil_end_delay_ms = 0;

        
        active_scope_magnification = 0;
        recoil_mult_2x = 1.0f;
        recoil_mult_3x = 1.0f;
        recoil_mult_4x = 1.0f;
        recoil_mult_6x = 1.0f;

        
        

        
        enable_target_locking = false;
        derivative_smoothing_factor = 0.8f; // Default value for smoothing factor
        target_locking_iou_threshold = 0.5f;
        target_locking_max_lost_frames = 10;

        
        kp_x = 0.5; 
        ki_x = 0.0;
        kd_x = 0.1;
        kp_y = 0.4; 
        ki_y = 0.0;
        kd_y = 0.15;

        
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
        max_detections = 100;
        postprocess = "yolo10";
        export_enable_fp8 = false;
        export_enable_fp16 = true;
        onnx_input_resolution = 640;

        
        cuda_device_id = 0;

        
        button_targeting = splitString("RightMouseButton");
        button_shoot = splitString("LeftMouseButton");
        button_zoom = splitString("RightMouseButton");
        button_exit = splitString("F2");
        button_pause = splitString("F3");
        button_reload_config = splitString("F4");
        button_open_overlay = splitString("Home");
        button_disable_upward_aim = splitString("None");
        button_auto_shoot = splitString("None"); 
        button_silent_aim = splitString("None"); 

        
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

        
        enable_optical_flow = false;
        draw_optical_flow = false;
        optical_flow_alpha_cpu = 10.0f;
        draw_optical_flow_steps = 16;
        optical_flow_magnitudeThreshold = 0.5f;
        staticFrameThreshold = 1.0f;
        fovX = 90.0f; 
        fovY = 60.0f; 

        
        enable_hsv_filter = false;
        hsv_lower_h = 0;
        hsv_lower_s = 0;
        hsv_lower_v = 0;
        hsv_upper_h = 179;
        hsv_upper_s = 255;
        hsv_upper_v = 255;
        min_hsv_pixels = 10;
        remove_hsv_matches = false;

        saveConfig(filename); 
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(filename.c_str());
    if (rc < 0) {
        std::cerr << "[Config] Error parsing INI file: " << filename << std::endl;
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
    capture_fps = get_long_ini("Capture", "capture_fps", 60);
    monitor_idx = get_long_ini("Capture", "monitor_idx", 0);
    circle_mask = get_bool_ini("Capture", "circle_mask", true);
    capture_borders = get_bool_ini("Capture", "capture_borders", true);
    capture_cursor = get_bool_ini("Capture", "capture_cursor", true);
    capture_use_cuda = get_bool_ini("Capture", "capture_use_cuda", true);
    use_1ms_capture = get_bool_ini("Capture", "use_1ms_capture", false);
    capture_timeout_ms = get_long_ini("Capture", "capture_timeout_ms", 5);
    target_fps = (float)get_double_ini("Capture", "target_fps", 120.0);

    body_y_offset = (float)get_double_ini("Target", "body_y_offset", 0.15);
    head_y_offset = (float)get_double_ini("Target", "head_y_offset", 0.05);
    offset_step = (float)get_double_ini("Target", "offset_step", 0.01);
    ignore_third_person = get_bool_ini("Target", "ignore_third_person", false);
    shooting_range_targets = get_bool_ini("Target", "shooting_range_targets", false);
    auto_aim = get_bool_ini("Target", "auto_aim", false);

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
    derivative_smoothing_factor = (float)get_double_ini("PID", "derivative_smoothing_factor", 0.8);

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
    max_detections = get_long_ini("AI", "max_detections", 20);
    postprocess = get_string_ini("AI", "postprocess", "yolo10");
    export_enable_fp8 = get_bool_ini("AI", "export_enable_fp8", false);
    export_enable_fp16 = get_bool_ini("AI", "export_enable_fp16", true);
    onnx_input_resolution = get_long_ini("AI", "onnx_input_resolution", 640);

    cuda_device_id = get_long_ini("CUDA", "cuda_device_id", 0);

    button_targeting = splitString(get_string_ini("Buttons", "button_targeting", "RightMouseButton"));
    button_shoot = splitString(get_string_ini("Buttons", "button_shoot", "LeftMouseButton"));
    button_zoom = splitString(get_string_ini("Buttons", "button_zoom", "RightMouseButton"));
    button_exit = splitString(get_string_ini("Buttons", "button_exit", "F2"));
    button_pause = splitString(get_string_ini("Buttons", "button_pause", "F3"));
    button_reload_config = splitString(get_string_ini("Buttons", "button_reload_config", "F4"));
    button_open_overlay = splitString(get_string_ini("Buttons", "button_open_overlay", "Home"));
    button_disable_upward_aim = splitString(get_string_ini("Buttons", "button_disable_upward_aim", "None"));
    button_auto_shoot = splitString(get_string_ini("Buttons", "button_auto_shoot", "None"));
    button_silent_aim = splitString(get_string_ini("Buttons", "button_silent_aim", "None")); 

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

    
    enable_optical_flow = get_bool_ini("OpticalFlow", "enable_optical_flow", false);
    draw_optical_flow = get_bool_ini("OpticalFlow", "draw_optical_flow", false);
    optical_flow_alpha_cpu = (float)get_double_ini("OpticalFlow", "optical_flow_alpha_cpu", 10.0);
    draw_optical_flow_steps = get_long_ini("OpticalFlow", "draw_optical_flow_steps", 16);
    optical_flow_magnitudeThreshold = (float)get_double_ini("OpticalFlow", "optical_flow_magnitudeThreshold", 0.5);
    staticFrameThreshold = (float)get_double_ini("OpticalFlow", "staticFrameThreshold", 1.0);
    fovX = (float)get_double_ini("OpticalFlow", "fovX", 90.0);
    fovY = (float)get_double_ini("OpticalFlow", "fovY", 60.0);

    
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
    
    

    
    enable_target_locking = get_bool_ini("TargetLocking", "enable_target_locking", false);
    target_locking_iou_threshold = (float)get_double_ini("TargetLocking", "target_locking_iou_threshold", 0.5);
    target_locking_max_lost_frames = get_long_ini("TargetLocking", "target_locking_max_lost_frames", 10);

    return true;
}

bool Config::saveConfig(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening config for writing: " << filename << std::endl;
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
    file << "capture_use_cuda = " << (capture_use_cuda ? "true" : "false") << "\n";
    file << "use_1ms_capture = " << (use_1ms_capture ? "true" : "false") << "\n";
    file << "capture_timeout_ms = " << capture_timeout_ms << "\n";
    file << "target_fps = " << target_fps << "\n\n";

    file << "[Target]\n";
    file << std::fixed << std::setprecision(6);
    file << "body_y_offset = " << body_y_offset << "\n";
    file << "head_y_offset = " << head_y_offset << "\n";
    file << "offset_step = " << offset_step << "\n";
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

    
    
    file << "[TargetLocking]\n";
    file << "enable_target_locking = " << (enable_target_locking ? "true" : "false") << "\n";
    file << std::fixed << std::setprecision(6);
    file << "target_locking_iou_threshold = " << target_locking_iou_threshold << "\n";
    file << std::noboolalpha;
    file << "target_locking_max_lost_frames = " << target_locking_max_lost_frames << "\n\n";
    
    file << "[PID]\n";
    file << std::fixed << std::setprecision(6);
    file << "kp_x = " << kp_x << "\n";
    file << "ki_x = " << ki_x << "\n";
    file << "kd_x = " << kd_x << "\n";
    file << "kp_y = " << kp_y << "\n";
    file << "ki_y = " << ki_y << "\n";
    file << "kd_y = " << kd_y << "\n";
    file << "derivative_smoothing_factor = " << derivative_smoothing_factor << "\n\n";

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
    file << std::noboolalpha;
    file << "max_detections = " << max_detections << "\n";
    file << "postprocess = " << postprocess << "\n";
    file << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n";
    file << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n";
    file << "onnx_input_resolution = " << onnx_input_resolution << "\n\n";

    file << "[CUDA]\n";
    file << "cuda_device_id = " << cuda_device_id << "\n\n";

    file << "[Buttons]\n";
    file << "button_targeting = " << joinStrings(button_targeting) << "\n";
    file << "button_shoot = " << joinStrings(button_shoot) << "\n";
    file << "button_zoom = " << joinStrings(button_zoom) << "\n";
    file << "button_exit = " << joinStrings(button_exit) << "\n";
    file << "button_pause = " << joinStrings(button_pause) << "\n";
    file << "button_reload_config = " << joinStrings(button_reload_config) << "\n";
    file << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n";
    file << "button_disable_upward_aim = " << joinStrings(button_disable_upward_aim) << "\n";
    file << "button_auto_shoot = " << joinStrings(button_auto_shoot) << "\n";
    file << "button_silent_aim = " << joinStrings(button_silent_aim) << "\n\n"; 

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
        std::cerr << "[Config] Error listing profiles in directory '" << current_path_str << "': " << e.what() << std::endl;
    }
    
    std::sort(profiles.begin(), profiles.end());
    return profiles;
}

bool Config::saveProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
        std::cerr << "[Config] Invalid profile name for saving: '" << profileName << "'. Cannot save." << std::endl;
        return false;
    }
    std::string filename = profileName + ".ini";
    std::cout << "[Config] Saving current settings to profile: " << filename << std::endl;
    return saveConfig(filename); 
}

bool Config::loadProfile(const std::string& profileName) {
     if (profileName.empty()) {
        std::cerr << "[Config] Invalid profile name for loading: '" << profileName << "'. Cannot load." << std::endl;
        return false;
    }
    std::string filename = profileName + ".ini";
     std::cout << "[Config] Loading settings from profile: " << filename << std::endl;

    if (!std::filesystem::exists(filename)) {
         std::cerr << "[Config] Profile file not found: " << filename << std::endl;
         return false; 
    }
    return loadConfig(filename); 
}

bool Config::deleteProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
         std::cerr << "[Config] Invalid profile name for deletion: '" << profileName << "'. Cannot delete." << std::endl;
        return false;
    }
    std::string filename = profileName + ".ini";
    std::cout << "[Config] Attempting to delete profile: " << filename << std::endl;

    try {
        if (std::filesystem::exists(filename)) {
            if (std::filesystem::remove(filename)) {
                std::cout << "[Config] Profile deleted successfully: " << filename << std::endl;
                return true;
            } else {
                std::cerr << "[Config] Failed to delete profile file: " << filename << std::endl;
                return false;
            }
        } else {
            std::cerr << "[Config] Profile file to delete not found: " << filename << std::endl;
            return false; 
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[Config] Filesystem error while deleting profile '" << filename << "': " << e.what() << std::endl;
        return false;
    }
}

void Config::resetConfig()
{
    
    
    loadConfig("__dummy_nonexistent_file_for_reset__.ini"); 
    
}


Config::Config()
{
    
    
    
    loadConfig("config.ini"); 
}

