#pragma once

#define NOMINMAX  // Prevent Windows.h from defining min/max macros

#include <algorithm>
#include <string>
#include <cmath>
#include "config.h"
#include "../utils/constants.h"
#include "../core/constants.h"

class ConfigValidator {
public:
    static void validateAndCorrect(Config& config) {
        // Detection resolution
        config.detection_resolution = std::clamp(
            config.detection_resolution,
            Constants::MIN_DETECTION_RESOLUTION,
            Constants::MAX_DETECTION_RESOLUTION
        );

        // Capture FPS
        config.capture_fps = std::clamp(
            config.capture_fps,
            0,
            Constants::MAX_CAPTURE_FPS
        );

        // Target FPS
        config.target_fps = (std::max)(1.0f, (std::min)(config.target_fps, static_cast<float>(Constants::MAX_CAPTURE_FPS)));

        // Monitor index
        config.monitor_idx = (std::max)(0, config.monitor_idx);

        // Offsets (0.0 to 1.0)
        config.body_y_offset = std::clamp(config.body_y_offset, 0.0f, 1.0f);
        config.head_y_offset = std::clamp(config.head_y_offset, 0.0f, 1.0f);
        config.offset_step = std::clamp(config.offset_step, 0.001f, 0.1f);

        // Crosshair offsets
        config.crosshair_offset_x = std::clamp(config.crosshair_offset_x, -100.0f, 100.0f);
        config.crosshair_offset_y = std::clamp(config.crosshair_offset_y, -100.0f, 100.0f);

        // Recoil settings
        config.easynorecoilstrength = std::clamp(config.easynorecoilstrength, 0.0f, 10.0f);
        config.norecoil_step = std::clamp(config.norecoil_step, 0.1f, 5.0f);
        config.norecoil_ms = static_cast<float>((std::max)(1, (std::min)(static_cast<int>(config.norecoil_ms), 100)));
        config.easynorecoil_start_delay_ms = (std::max)(0, config.easynorecoil_start_delay_ms);
        config.easynorecoil_end_delay_ms = (std::max)(0, config.easynorecoil_end_delay_ms);

        // Scope multipliers
        config.bScope_multiplier = std::clamp(config.bScope_multiplier, 0.1f, 10.0f);
        config.recoil_mult_2x = std::clamp(config.recoil_mult_2x, 0.1f, 10.0f);
        config.recoil_mult_3x = std::clamp(config.recoil_mult_3x, 0.1f, 10.0f);
        config.recoil_mult_4x = std::clamp(config.recoil_mult_4x, 0.1f, 10.0f);
        config.recoil_mult_6x = std::clamp(config.recoil_mult_6x, 0.1f, 10.0f);

        // PID parameters
        config.kp_x = (std::max)(0.0f, (std::min)(static_cast<float>(config.kp_x), 10.0f));
        config.ki_x = (std::max)(0.0f, (std::min)(static_cast<float>(config.ki_x), 10.0f));
        config.kd_x = (std::max)(0.0f, (std::min)(static_cast<float>(config.kd_x), 10.0f));
        config.kp_y = (std::max)(0.0f, (std::min)(static_cast<float>(config.kp_y), 10.0f));
        config.ki_y = (std::max)(0.0f, (std::min)(static_cast<float>(config.ki_y), 10.0f));
        config.kd_y = (std::max)(0.0f, (std::min)(static_cast<float>(config.kd_y), 10.0f));

        // Detection parameters
        config.confidence_threshold = std::clamp(config.confidence_threshold, 0.01f, 1.0f);
        config.nms_threshold = std::clamp(config.nms_threshold, 0.01f, 1.0f);
        config.max_detections = std::clamp(config.max_detections, 1, Constants::MAX_DETECTIONS);
        config.distance_weight = std::clamp(config.distance_weight, 0.0f, 10.0f);
        config.confidence_weight = std::clamp(config.confidence_weight, 0.0f, 10.0f);
        config.sticky_target_threshold = std::clamp(config.sticky_target_threshold, 0.0f, 1.0f);

        // Overlay settings
        config.overlay_opacity = std::clamp(
            config.overlay_opacity,
            UtilConstants::MIN_OVERLAY_OPACITY,
            UtilConstants::MAX_OVERLAY_OPACITY
        );
        config.overlay_ui_scale = std::clamp(config.overlay_ui_scale, 0.5f, 3.0f);
        config.window_size = static_cast<int>((std::max)(0.1f, (std::min)(static_cast<float>(config.window_size), 3.0f)));

        // Screenshot delay
        config.screenshot_delay = (std::max)(0, config.screenshot_delay);

        // HSV filter parameters
        config.hsv_lower_h = std::clamp(config.hsv_lower_h, 0, 179);
        config.hsv_lower_s = std::clamp(config.hsv_lower_s, 0, 255);
        config.hsv_lower_v = std::clamp(config.hsv_lower_v, 0, 255);
        config.hsv_upper_h = std::clamp(config.hsv_upper_h, 0, 179);
        config.hsv_upper_s = std::clamp(config.hsv_upper_s, 0, 255);
        config.hsv_upper_v = std::clamp(config.hsv_upper_v, 0, 255);
        config.min_hsv_pixels = (std::max)(0, config.min_hsv_pixels);

        // Validate string values
        if (config.capture_method != "simple" && 
            config.capture_method != "duplication" && 
            config.capture_method != "bitblt" &&
            config.capture_method != "virtual_camera" &&
            config.capture_method != "ndi") {
            config.capture_method = "duplication";
        }

        if (config.input_method != "WIN32" &&
            config.input_method != "GHUB" &&
            config.input_method != "ARDUINO" &&
            config.input_method != "RAZER" &&
            config.input_method != "KMBOX" &&
            config.input_method != "MAKCU") {
            config.input_method = "WIN32";
        }

        // Validate head class name
        if (config.head_class_name.empty()) {
            config.head_class_name = "head";
        }
    }
};