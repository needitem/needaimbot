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

        // Capture
        detection_resolution = 320;
        capture_fps = 60;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        virtual_camera_name = "None";
        capture_use_cuda = true;

        // Target
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        offset_step = 0.01f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;

        // Mouse
        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        norecoil_step = 5.0f;
        norecoil_ms = 10.0f;
        input_method = "WIN32";

        // Scope Recoil Control
        active_scope_magnification = 0;
        recoil_mult_2x = 1.0f;
        recoil_mult_3x = 1.0f;
        recoil_mult_4x = 1.0f;
        recoil_mult_6x = 1.0f;

        // Prediction Algorithm Settings (Defaults from your original file)
        prediction_algorithm = "None"; 
        velocity_prediction_ms = 16.0f;
        lr_past_points = 10;
        es_alpha = 0.5f;
        kalman_q = 0.1;
        kalman_r = 0.1;
        kalman_p = 0.1;

        // Separated X/Y PID Controllers
        kp_x = 0.5; 
        ki_x = 0.0;
        kd_x = 0.1;
        kp_y = 0.4; 
        ki_y = 0.0;
        kd_y = 0.15;

        // Arduino
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;

		// KMBOX net params:
        kmbox_ip = "192.168.2.188";
        kmbox_port = "16896";
        kmbox_mac = "46405c53";
        
        // Mouse shooting
        bScope_multiplier = 1.0f;

        // AI
        ai_model = "sunxds_0.5.6.engine"; // Example, use your actual default
        confidence_threshold = 0.15f;
        nms_threshold = 0.50f;
        max_detections = 100;
        postprocess = "yolo10";
        export_enable_fp8 = false;
        export_enable_fp16 = true;
        onnx_input_resolution = 640;

        // CUDA
        use_pinned_memory = true;
        cuda_device_id = 0;

        // Buttons
        button_targeting = splitString("RightMouseButton");
        button_shoot = splitString("LeftMouseButton");
        button_zoom = splitString("RightMouseButton");
        button_exit = splitString("F2");
        button_pause = splitString("F3");
        button_reload_config = splitString("F4");
        button_open_overlay = splitString("Home");
        button_disable_upward_aim = splitString("None");
        button_auto_shoot = splitString("None"); 

        // Overlay
        overlay_opacity = 225;
        overlay_ui_scale = 1.0f;

        // --- Custom Class Settings Defaults ---
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

        // Debug
        show_window = true;
        show_fps = true;
        window_name = "Debug";
        window_size = 80;
        screenshot_button = splitString("None");
        screenshot_delay = 500;
        always_on_top = true;
        verbose = false;

        saveConfig(filename); // Save the newly created default config
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(filename.c_str());
    if (rc < 0) {
        std::cerr << "[Config] Error parsing INI file: " << filename << std::endl;
        return false;
    }

    // Helper lambdas (these were already well-defined in your code)
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

    // Load settings from designated sections
    detection_resolution = get_long_ini("Capture", "detection_resolution", 320);
    capture_fps = get_long_ini("Capture", "capture_fps", 60);
    monitor_idx = get_long_ini("Capture", "monitor_idx", 0);
    circle_mask = get_bool_ini("Capture", "circle_mask", true);
    capture_borders = get_bool_ini("Capture", "capture_borders", true);
    capture_cursor = get_bool_ini("Capture", "capture_cursor", true);
    virtual_camera_name = get_string_ini("Capture", "virtual_camera_name", "None");
    capture_use_cuda = get_bool_ini("Capture", "capture_use_cuda", true);

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
    bScope_multiplier = (float)get_double_ini("Mouse", "bScope_multiplier", 1.2);

    active_scope_magnification = get_long_ini("Recoil", "active_scope_magnification", 0);
    recoil_mult_2x = (float)get_double_ini("Recoil", "recoil_mult_2x", 1.0);
    recoil_mult_3x = (float)get_double_ini("Recoil", "recoil_mult_3x", 1.0);
    recoil_mult_4x = (float)get_double_ini("Recoil", "recoil_mult_4x", 1.0);
    recoil_mult_6x = (float)get_double_ini("Recoil", "recoil_mult_6x", 1.0);

    prediction_algorithm = get_string_ini("Prediction", "prediction_algorithm", "None");
    velocity_prediction_ms = (float)get_double_ini("Prediction", "velocity_prediction_ms", 16.0);
    lr_past_points = get_long_ini("Prediction", "lr_past_points", 10);
    es_alpha = (float)get_double_ini("Prediction", "es_alpha", 0.5);
    kalman_q = get_double_ini("Prediction", "kalman_q", 0.1);
    kalman_r = get_double_ini("Prediction", "kalman_r", 0.1);
    kalman_p = get_double_ini("Prediction", "kalman_p", 0.1);
    
    kp_x = get_double_ini("PID", "kp_x", 0.5);
    ki_x = get_double_ini("PID", "ki_x", 0.0);
    kd_x = get_double_ini("PID", "kd_x", 0.1);
    kp_y = get_double_ini("PID", "kp_y", 0.4);
    ki_y = get_double_ini("PID", "ki_y", 0.0);
    kd_y = get_double_ini("PID", "kd_y", 0.15);

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
    max_detections = get_long_ini("AI", "max_detections", 20);
    postprocess = get_string_ini("AI", "postprocess", "yolo10");
    export_enable_fp8 = get_bool_ini("AI", "export_enable_fp8", false);
    export_enable_fp16 = get_bool_ini("AI", "export_enable_fp16", true);
    onnx_input_resolution = get_long_ini("AI", "onnx_input_resolution", 640);

    use_pinned_memory = get_bool_ini("CUDA", "use_pinned_memory", true);
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

    // --- Load Custom Class Settings --- 
    head_class_name = get_string_ini("Classes", "HeadClassName", "Head");

    int classSettingsCount = ini.GetLongValue("ClassSettings", "Count", -1); 

    class_settings.clear(); // Clear before loading or migrating
    if (classSettingsCount != -1) { // New format exists
        for (int i = 0; i < classSettingsCount; ++i) {
            std::string id_key = "Class_" + std::to_string(i) + "_ID";
            std::string name_key = "Class_" + std::to_string(i) + "_Name";
            std::string ignore_key = "Class_" + std::to_string(i) + "_Ignore";
            
            int id_val = ini.GetLongValue("ClassSettings", id_key.c_str(), i);
            std::string name_val = ini.GetValue("ClassSettings", name_key.c_str(), ""); // Use GetValue for string
            bool ignore_val = ini.GetBoolValue("ClassSettings", ignore_key.c_str(), false);
            
            if (name_val.empty()) { 
                name_val = "Class " + std::to_string(id_val);
            }
            class_settings.emplace_back(id_val, name_val, ignore_val);
        }
    } else { // Old format or no custom settings: Migrate from [Ignore Classes] section values if they exist
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
    // Removed loading of individual class_X and ignore_class_X into Config object members.
    // They are now fully managed by class_settings.

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

    file << "# Config file generated by sunone_aimbot_cpp\n";
    file << "# https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md\n\n";

    file << "[Capture]\n";
    file << "detection_resolution = " << detection_resolution << "\n";
    file << "capture_fps = " << capture_fps << "\n";
    file << "monitor_idx = " << monitor_idx << "\n";
    file << "circle_mask = " << (circle_mask ? "true" : "false") << "\n";
    file << "capture_borders = " << (capture_borders ? "true" : "false") << "\n";
    file << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n";
    file << "virtual_camera_name = " << virtual_camera_name << "\n";
    file << "capture_use_cuda = " << (capture_use_cuda ? "true" : "false") << "\n\n";

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
    file << "bScope_multiplier = " << bScope_multiplier << "\n\n";

    file << "[Recoil]\n";
    file << "active_scope_magnification = " << active_scope_magnification << "\n";
    file << std::fixed << std::setprecision(6);
    file << "recoil_mult_2x = " << recoil_mult_2x << "\n";
    file << "recoil_mult_3x = " << recoil_mult_3x << "\n";
    file << "recoil_mult_4x = " << recoil_mult_4x << "\n";
    file << "recoil_mult_6x = " << recoil_mult_6x << "\n\n";

    file << "[Prediction]\n";
    file << "prediction_algorithm = " << prediction_algorithm << "\n";
    file << std::fixed << std::setprecision(6);
    file << "velocity_prediction_ms = " << velocity_prediction_ms << "\n";
    file << std::noboolalpha;
    file << "lr_past_points = " << lr_past_points << "\n";
    file << std::fixed << std::setprecision(6);
    file << "es_alpha = " << es_alpha << "\n";
    file << "kalman_q = " << kalman_q << "\n";
    file << "kalman_r = " << kalman_r << "\n";
    file << "kalman_p = " << kalman_p << "\n\n";
    
    file << "[PID]\n";
    file << std::fixed << std::setprecision(6);
    file << "kp_x = " << kp_x << "\n";
    file << "ki_x = " << ki_x << "\n";
    file << "kd_x = " << kd_x << "\n";
    file << "kp_y = " << kp_y << "\n";
    file << "ki_y = " << ki_y << "\n";
    file << "kd_y = " << kd_y << "\n\n";

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
    file << std::noboolalpha;
    file << "max_detections = " << max_detections << "\n";
    file << "postprocess = " << postprocess << "\n";
    file << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n";
    file << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n";
    file << "onnx_input_resolution = " << onnx_input_resolution << "\n\n";

    file << "[CUDA]\n";
    file << "use_pinned_memory = " << (use_pinned_memory ? "true" : "false") << "\n";
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
    file << "button_auto_shoot = " << joinStrings(button_auto_shoot) << "\n\n";

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
    // This will call loadConfig with a non-existent filename, triggering the default value assignments
    // and then save it.
    loadConfig("__dummy_nonexistent_file_for_reset__.ini"); 
    // saveConfig("config.ini"); // loadConfig already calls saveConfig if file doesn't exist
}

// The default constructor which calls loadConfig("config.ini")
Config::Config()
{
    // Attempt to load the default config file.
    // If it fails to load (e.g., parse error but file exists), it will print an error.
    // If the file doesn't exist, loadConfig will create it with defaults.
    loadConfig("config.ini"); 
}
