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

#include "config.h"
#include "modules/SimpleIni.h"

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
        // capture_method = "duplication_api"; // Removed
        detection_resolution = 320;
        capture_fps = 60;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        virtual_camera_name = "None";
        capture_use_cuda = true;

        // Target
        // disable_headshot = false; // Removed
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        offset_step = 0.01f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;

        // Target Stickiness
        // sticky_bonus = -50.0f; // Default bonus (negative because lower score is better)
        // sticky_iou_threshold = 0.3f; // Default IoU threshold

        // Mouse
        // dpi = 1000;
        // fovX = 50;
        // fovY = 50;
        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        norecoil_step = 5.0f;
        norecoil_ms = 10.0f;
        input_method = "WIN32";

        // Scope Recoil Control (Defaults)
        active_scope_magnification = 0;
        recoil_mult_2x = 1.0f;
        recoil_mult_3x = 1.0f;
        recoil_mult_4x = 1.0f;
        recoil_mult_6x = 1.0f;

        // Kalman Filter settings
        prediction_time_ms = 16.0f; // Default value
        kalman_process_noise = 1.0f; // Default value
        kalman_measurement_noise = 10.0f; // Default value
        enable_prediction = true; // Default: enable prediction

        // PID Controller
        // kp = 0.5;
        
        // Separated X/Y PID Controllers
        kp_x = 0.5;  // 초기값은 공통 값과 동일하게 설정
        ki_x = 0.0;
        kd_x = 0.1;
        kp_y = 0.4;  // Y축은 약간 낮게 설정 (과도한 하향 조준 방지)
        ki_y = 0.0;
        kd_y = 0.15; // Y축은 미분 게인을 약간 높게 설정 (더 빠른 감속)

        // Arduino
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;

        // Mouse shooting
        auto_shoot = false;
        bScope_multiplier = 1.0f;

        // AI
        ai_model = "sunxds_0.5.6.engine";
        confidence_threshold = 0.15f;
        nms_threshold = 0.50f;
        max_detections = 100;
        postprocess = "yolo10";
        export_enable_fp8 = false;
        export_enable_fp16 = true;

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

        // Overlay
        overlay_opacity = 225;
        overlay_ui_scale = 1.0f;

        // Custom classes
        class_player = 0;
        class_bot = 1;
        class_weapon = 2;
        class_outline = 3;
        class_dead_body = 4;
        class_hideout_target_human = 5;
        class_hideout_target_balls = 6;
        class_head = 7;
        class_smoke = 8;
        class_fire = 9;
        class_third_person = 10;

        // Debug
        show_window = true;
        show_fps = true;
        window_name = "Debug";
        window_size = 80;
        screenshot_button = splitString("None");
        screenshot_delay = 500;
        always_on_top = true;
        verbose = false;

        // Default ignore flags
        ignore_class_0 = false; // player
        ignore_class_1 = false; // bot
        ignore_class_2 = false; // weapon
        ignore_class_3 = false; // outline
        ignore_class_4 = false; // dead_body
        ignore_class_5 = false; // hideout_target_human
        ignore_class_6 = false; // hideout_target_balls
        ignore_class_7 = false; // head
        ignore_class_8 = false; // smoke
        ignore_class_9 = false; // fire
        ignore_class_10 = false; // third_person

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

    auto get_string = [&](const char* key, const char* defval)
    {
        const char* val = ini.GetValue("", key, defval);
        return std::string(val ? val : "");
    };

    auto get_bool = [&](const char* key, bool defval)
    {
        return ini.GetBoolValue("", key, defval);
    };

    auto get_long = [&](const char* key, long defval)
    {
        return (int)ini.GetLongValue("", key, defval);
    };

    auto get_double = [&](const char* key, double defval)
    {
        return ini.GetDoubleValue("", key, defval);
    };

    // Capture
    // capture_method = get_string("capture_method", "duplication_api"); // Removed
    detection_resolution = get_long("detection_resolution", 320);
    capture_fps = get_long("capture_fps", 60);
    monitor_idx = get_long("monitor_idx", 0);
    circle_mask = get_bool("circle_mask", true);
    capture_borders = get_bool("capture_borders", true);
    capture_cursor = get_bool("capture_cursor", true);
    virtual_camera_name = get_string("virtual_camera_name", "None");
    capture_use_cuda = get_bool("capture_use_cuda", true);

    // Target
    // disable_headshot = get_bool("disable_headshot", false); // Removed
    body_y_offset = (float)get_double("body_y_offset", 0.15);
    head_y_offset = (float)get_double("head_y_offset", 0.05);
    offset_step = (float)get_double("offset_step", 0.01);
    ignore_third_person = get_bool("ignore_third_person", false);
    shooting_range_targets = get_bool("shooting_range_targets", false);
    auto_aim = get_bool("auto_aim", false);

    // Removed Target Stickiness loading
    // sticky_bonus = (float)get_double("sticky_bonus", -50.0);
    // sticky_iou_threshold = (float)get_double("sticky_iou_threshold", 0.3);

    // Mouse
    // Remove dpi saving (Assume done)
    // ini.SetLongValue("", "dpi", dpi);
    // Remove fovX and fovY saving
    // ini.SetLongValue("", "fovX", fovX);
    // ini.SetLongValue("", "fovY", fovY);
    easynorecoil = get_bool("easynorecoil", false);
    easynorecoilstrength = (float)get_double("easynorecoilstrength", 0.0);
    norecoil_step = (float)get_double("norecoil_step", 5.0);
    norecoil_ms = (float)get_double("norecoil_ms", 10.0);
    input_method = get_string("input_method", "WIN32");

    // Scope Recoil Control (Load)
    active_scope_magnification = get_long("active_scope_magnification", 0);
    recoil_mult_2x = (float)get_double("recoil_mult_2x", 1.0);
    recoil_mult_3x = (float)get_double("recoil_mult_3x", 1.0);
    recoil_mult_4x = (float)get_double("recoil_mult_4x", 1.0);
    recoil_mult_6x = (float)get_double("recoil_mult_6x", 1.0);

    // Kalman Filter settings
    prediction_time_ms = (float)get_double("prediction_time_ms", 16.0);
    kalman_process_noise = (float)get_double("kalman_process_noise", 1.0);
    kalman_measurement_noise = (float)get_double("kalman_measurement_noise", 10.0);
    enable_prediction = get_bool("enable_prediction", true); // Load prediction enable flag

    // PID Controller
    // kp = (double)get_double("kp", 0.5);
    // ki = (double)get_double("ki", 0.0);
    // kd = (double)get_double("kd", 0.1);

    // Separated X/Y PID Controllers
    kp_x = (double)get_double("kp_x", 0.5);
    ki_x = (double)get_double("ki_x", 0.0);
    kd_x = (double)get_double("kd_x", 0.1);
    kp_y = (double)get_double("kp_y", 0.4);
    ki_y = (double)get_double("ki_y", 0.0);
    kd_y = (double)get_double("kd_y", 0.15);

    // Arduino
    arduino_baudrate = get_long("arduino_baudrate", 115200);
    arduino_port = get_string("arduino_port", "COM0");
    arduino_16_bit_mouse = get_bool("arduino_16_bit_mouse", false);
    arduino_enable_keys = get_bool("arduino_enable_keys", false);

    // Mouse shooting
    auto_shoot = get_bool("auto_shoot", false);
    bScope_multiplier = (float)get_double("bScope_multiplier", 1.2);

    // AI
    ai_model = get_string("ai_model", "sunxds_0.5.6.engine");
    confidence_threshold = (float)get_double("confidence_threshold", 0.15);
    nms_threshold = (float)get_double("nms_threshold", 0.50);
    max_detections = get_long("max_detections", 20);
    postprocess = get_string("postprocess", "yolo11");
    export_enable_fp8 = get_bool("export_enable_fp8", true);
    export_enable_fp16 = get_bool("export_enable_fp16", true);

    // CUDA
    use_pinned_memory = get_bool("use_pinned_memory", true);
    cuda_device_id = get_long("cuda_device_id", 0);

    // Buttons
    button_targeting = splitString(get_string("button_targeting", "RightMouseButton"));
    button_shoot = splitString(get_string("button_shoot", "LeftMouseButton"));
    button_zoom = splitString(get_string("button_zoom", "RightMouseButton"));
    button_exit = splitString(get_string("button_exit", "F2"));
    button_pause = splitString(get_string("button_pause", "F3"));
    button_reload_config = splitString(get_string("button_reload_config", "F4"));
    button_open_overlay = splitString(get_string("button_open_overlay", "Home"));

    // Overlay
    overlay_opacity = get_long("overlay_opacity", 225);
    overlay_ui_scale = (float)get_double("overlay_ui_scale", 1.0);

    // Custom Classes
    class_player = get_long("class_player", 0);
    class_bot = get_long("class_bot", 1);
    class_weapon = get_long("class_weapon", 2);
    class_outline = get_long("class_outline", 3);
    class_dead_body = get_long("class_dead_body", 4);
    class_hideout_target_human = get_long("class_hideout_target_human", 5);
    class_hideout_target_balls = get_long("class_hideout_target_balls", 6);
    class_head = get_long("class_head", 7);
    class_smoke = get_long("class_smoke", 8);
    class_fire = get_long("class_fire", 9);
    class_third_person = get_long("class_third_person", 10);

    // Debug window
    show_window = get_bool("show_window", true);
    show_fps = get_bool("show_fps", true);
    window_name = get_string("window_name", "Debug");
    window_size = get_long("window_size", 80);
    screenshot_button = splitString(get_string("screenshot_button", "None"));
    screenshot_delay = get_long("screenshot_delay", 500);
    always_on_top = get_bool("always_on_top", true);
    verbose = get_bool("verbose", false);

    // Load ignore flags
    ignore_class_0 = get_bool("ignore_class_0", false);
    ignore_class_1 = get_bool("ignore_class_1", false);
    ignore_class_2 = get_bool("ignore_class_2", false);
    ignore_class_3 = get_bool("ignore_class_3", false);
    ignore_class_4 = get_bool("ignore_class_4", false);
    ignore_class_5 = get_bool("ignore_class_5", false);
    ignore_class_6 = get_bool("ignore_class_6", false);
    ignore_class_7 = get_bool("ignore_class_7", false);
    ignore_class_8 = get_bool("ignore_class_8", false);
    ignore_class_9 = get_bool("ignore_class_9", false);
    ignore_class_10 = get_bool("ignore_class_10", false);

    ini.SetValue("", "input_method", input_method.c_str());

    // PID Controller
    // ini.SetDoubleValue("", "kp", kp);
    // ini.SetDoubleValue("", "ki", ki);
    // ini.SetDoubleValue("", "kd", kd);

    // Separated X/Y PID Controllers
    ini.SetDoubleValue("", "kp_x", kp_x);
    ini.SetDoubleValue("", "ki_x", ki_x);
    ini.SetDoubleValue("", "kd_x", kd_x);
    ini.SetDoubleValue("", "kp_y", kp_y);
    ini.SetDoubleValue("", "ki_y", ki_y);
    ini.SetDoubleValue("", "kd_y", kd_y);

    // Kalman Filter settings
    ini.SetDoubleValue("", "prediction_time_ms", prediction_time_ms);
    ini.SetDoubleValue("", "kalman_process_noise", kalman_process_noise);
    ini.SetDoubleValue("", "kalman_measurement_noise", kalman_measurement_noise);
    ini.SetBoolValue("", "enable_prediction", enable_prediction); // Save prediction enable flag

    // CUDA
    ini.SetBoolValue("", "use_pinned_memory", use_pinned_memory);
    ini.SetLongValue("", "cuda_device_id", cuda_device_id);

    // Buttons
    ini.SetValue("", "button_targeting", joinStrings(button_targeting, " ").c_str());

    ini.SetDoubleValue("", "norecoil_step", norecoil_step, nullptr, true);
    ini.SetDoubleValue("", "norecoil_ms", norecoil_ms, nullptr, true);

    ini.SetDoubleValue("", "bScope_multiplier", bScope_multiplier);

    // Scope Recoil Control (Save)
    ini.SetLongValue("", "active_scope_magnification", active_scope_magnification);
    ini.SetDoubleValue("", "recoil_mult_2x", recoil_mult_2x, nullptr, true);
    ini.SetDoubleValue("", "recoil_mult_3x", recoil_mult_3x, nullptr, true);
    ini.SetDoubleValue("", "recoil_mult_4x", recoil_mult_4x, nullptr, true);
    ini.SetDoubleValue("", "recoil_mult_6x", recoil_mult_6x, nullptr, true);

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

    file << "# An explanation of the options can be found at:\n";
    file << "# https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md\n\n";

    // Capture
    // ini.SetValue("", "capture_method", capture_method.c_str()); // Removed
    file << "# Capture\n"
        << "detection_resolution = " << detection_resolution << "\n"
        << "capture_fps = " << capture_fps << "\n"
        << "monitor_idx = " << monitor_idx << "\n"
        << "circle_mask = " << (circle_mask ? "true" : "false") << "\n"
        << "capture_borders = " << (capture_borders ? "true" : "false") << "\n"
        << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n"
        << "virtual_camera_name = " << virtual_camera_name << "\n"
        << "capture_use_cuda = " << (capture_use_cuda ? "true" : "false") << "\n\n";

    // Target
    file << "# Target\n"
        // << "disable_headshot = " << (disable_headshot ? "true" : "false") << "\n" // Removed
        << std::fixed << std::setprecision(2)
        << "body_y_offset = " << body_y_offset << "\n"
        << "head_y_offset = " << head_y_offset << "\n"
        << "offset_step = " << offset_step << "\n"
        << "ignore_third_person = " << (ignore_third_person ? "true" : "false") << "\n"
        << "shooting_range_targets = " << (shooting_range_targets ? "true" : "false") << "\n"
        << "auto_aim = " << (auto_aim ? "true" : "false") << "\n\n";

    // Removed Target Stickiness saving
    // writeLine(file, "", "Target Stickiness", "", "");
    // writeLine(file, "sticky_bonus", sticky_bonus);
    // writeLine(file, "sticky_iou_threshold", sticky_iou_threshold);

    // Mouse
    // Remove dpi saving (Assume done)
    // ini.SetLongValue("", "dpi", dpi);
    // Remove fovX and fovY saving
    // ini.SetLongValue("", "fovX", fovX);
    // ini.SetLongValue("", "fovY", fovY);
    file << "# Mouse move\n"
        << "easynorecoil = " << (easynorecoil ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "easynorecoilstrength = " << easynorecoilstrength << "\n"
        << "norecoil_step = " << norecoil_step << "\n"
        << "norecoil_ms = " << norecoil_ms << "\n"
        << "# WIN32, GHUB, ARDUINO\n"
        << "input_method = " << input_method << "\n\n"

        << "# Separated X/Y PID Controllers\n"
        << std::fixed << std::setprecision(3)
        << "kp_x = " << kp_x << "\n"
        << "ki_x = " << ki_x << "\n"
        << "kd_x = " << kd_x << "\n"
        << "kp_y = " << kp_y << "\n"
        << "ki_y = " << ki_y << "\n"
        << "kd_y = " << kd_y << "\n\n";

    // Kalman Filter settings
    file << "# Kalman Filter Prediction\n"
        << std::fixed << std::setprecision(1)
        << "prediction_time_ms = " << prediction_time_ms << "\n"
        << std::fixed << std::setprecision(3)
        << "kalman_process_noise = " << kalman_process_noise << "\n"
        << std::fixed << std::setprecision(2)
        << "kalman_measurement_noise = " << kalman_measurement_noise << "\n"
        << "enable_prediction = " << (enable_prediction ? "true" : "false") << "\n\n";

    // Arduino
    file << "# Arduino\n"
        << "arduino_baudrate = " << arduino_baudrate << "\n"
        << "arduino_port = " << arduino_port << "\n"
        << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n"
        << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    // Mouse shooting
    file << "# Mouse shooting\n"
        << "auto_shoot = " << (auto_shoot ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "bScope_multiplier = " << bScope_multiplier << "\n\n";

    // Scope Recoil Control (Save)
    file << "# Scope Recoil Control\n"
        << "active_scope_magnification = " << active_scope_magnification << "\n"
        << std::fixed << std::setprecision(2)
        << "recoil_mult_2x = " << recoil_mult_2x << "\n"
        << "recoil_mult_3x = " << recoil_mult_3x << "\n"
        << "recoil_mult_4x = " << recoil_mult_4x << "\n"
        << "recoil_mult_6x = " << recoil_mult_6x << "\n\n";

    // AI
    file << "# AI\n"
        << "ai_model = " << ai_model << "\n"
        << std::fixed << std::setprecision(2)
        << "confidence_threshold = " << confidence_threshold << "\n"
        << "nms_threshold = " << nms_threshold << "\n"
        << std::setprecision(0)
        << "max_detections = " << max_detections << "\n"
        << "postprocess = " << postprocess << "\n"
        << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n"
        << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n\n";

    // CUDA
    file << "# CUDA\n"
        << "use_pinned_memory = " << (use_pinned_memory ? "true" : "false") << "\n"
        << "cuda_device_id = " << cuda_device_id << "\n\n";

    // Buttons
    file << "# Buttons\n"
        << "button_targeting = " << joinStrings(button_targeting) << "\n"
        << "button_shoot = " << joinStrings(button_shoot) << "\n"
        << "button_zoom = " << joinStrings(button_zoom) << "\n"
        << "button_exit = " << joinStrings(button_exit) << "\n"
        << "button_pause = " << joinStrings(button_pause) << "\n"
        << "button_reload_config = " << joinStrings(button_reload_config) << "\n"
        << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n\n";

    // Overlay
    file << "# Overlay\n"
        << "overlay_opacity = " << overlay_opacity << "\n"
        << std::fixed << std::setprecision(2)
        << "overlay_ui_scale = " << overlay_ui_scale << "\n\n";

    // Custom Classes
    file << "# Custom Classes\n"
        << "class_player = " << class_player << "\n"
        << "class_bot = " << class_bot << "\n"
        << "class_weapon = " << class_weapon << "\n"
        << "class_outline = " << class_outline << "\n"
        << "class_dead_body = " << class_dead_body << "\n"
        << "class_hideout_target_human = " << class_hideout_target_human << "\n"
        << "class_hideout_target_balls = " << class_hideout_target_balls << "\n"
        << "class_head = " << class_head << "\n"
        << "class_smoke = " << class_smoke << "\n"
        << "class_fire = " << class_fire << "\n"
        << "class_third_person = " << class_third_person << "\n\n";

    // Debug
    file << "# Debug window\n"
        << "show_window = " << (show_window ? "true" : "false") << "\n"
        << "show_fps = " << (show_fps ? "true" : "false") << "\n"
        << "window_name = " << window_name << "\n"
        << "window_size = " << window_size << "\n"
        << "screenshot_button = " << joinStrings(screenshot_button) << "\n"
        << "screenshot_delay = " << screenshot_delay << "\n"
        << "always_on_top = " << (always_on_top ? "true" : "false") << "\n"
        << "verbose = " << (verbose ? "true" : "false") << "\n"
        << "\n"
        << "[Ignore Classes]\n"
        << "ignore_class_0 = " << (ignore_class_0 ? "true" : "false") << " ; player\n"
        << "ignore_class_1 = " << (ignore_class_1 ? "true" : "false") << " ; bot\n"
        << "ignore_class_2 = " << (ignore_class_2 ? "true" : "false") << " ; weapon\n"
        << "ignore_class_3 = " << (ignore_class_3 ? "true" : "false") << " ; outline\n"
        << "ignore_class_4 = " << (ignore_class_4 ? "true" : "false") << " ; dead_body\n"
        << "ignore_class_5 = " << (ignore_class_5 ? "true" : "false") << " ; hideout_target_human\n"
        << "ignore_class_6 = " << (ignore_class_6 ? "true" : "false") << " ; hideout_target_balls\n"
        << "ignore_class_7 = " << (ignore_class_7 ? "true" : "false") << " ; head\n"
        << "ignore_class_8 = " << (ignore_class_8 ? "true" : "false") << " ; smoke\n"
        << "ignore_class_9 = " << (ignore_class_9 ? "true" : "false") << " ; fire\n"
        << "ignore_class_10 = " << (ignore_class_10 ? "true" : "false") << " ; third_person\n";

    file.close();
    return true;
}

std::vector<std::string> Config::listProfiles() {
    std::vector<std::string> profiles;
    std::string current_path_str = "."; // Assuming profiles are in the same directory as the executable
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(current_path_str)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string extension = entry.path().extension().string();

                if (extension == ".ini" && filename != "config.ini") { // Find .ini files, excluding the main config.ini
                    std::string profileName = entry.path().stem().string(); // Get filename without extension
                    profiles.push_back(profileName);
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[Config] Error listing profiles in directory '" << current_path_str << "': " << e.what() << std::endl;
    }
    
    // Sort profiles alphabetically for consistency
    std::sort(profiles.begin(), profiles.end());

    // Optionally add a default entry if config.ini exists and you want to represent it
    // profiles.insert(profiles.begin(), "Default"); 

    return profiles;
}

bool Config::saveProfile(const std::string& profileName) {
    if (profileName.empty() || profileName == "config") {
        std::cerr << "[Config] Invalid profile name for saving: '" << profileName << "'. Cannot save." << std::endl;
        return false;
    }
    std::string filename = profileName + ".ini";
    std::cout << "[Config] Saving current settings to profile: " << filename << std::endl;
    return saveConfig(filename); // Reuse existing save logic
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
         return false; // Indicate failure if profile doesn't exist
    }
    return loadConfig(filename); // Reuse existing load logic
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
            return false; // Indicate failure as file wasn't found
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[Config] Filesystem error while deleting profile '" << filename << "': " << e.what() << std::endl;
        return false;
    }
}