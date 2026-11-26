#pragma once

// Nonlinear P(D) aim controller: owns the tuned gains (kp/kd/softness per aim
// profile) and the user-facing config knobs, with JSON load/save/print.
// Internal tuned constants live in pd_controller.cuh as __device__ constexpr.

#include <algorithm>
#include <iostream>

#include "simple_postprocess.h"
#include "json.hpp"

namespace pd_controller {

struct Gains {
    float kp_x = 0.55f;
    float kp_y = 0.6f;
    float softness_x = 11.0f;
    float softness_y = 10.0f;
    float kd_x = 0.18f;
    float kd_y = 0.22f;
};

class Settings {
public:
    Gains right{0.445f, 0.406f, 8.6f, 6.57f, 0.052f, 0.037f};
    Gains thumb{0.709f, 0.408f, 4.66f, 9.77f, 0.02f, 0.082f};

    float inflight_deadtime_frames = 2.85f;
    float ff_gain = 2.924f;
    float predict_frames = 5.47f;
    float aim_y_scale = 1.0f;
    float max_step = 11.37f;
    float oneeuro_min_cutoff = 0.049f;
    float oneeuro_beta = 0.02f;

    // IoU stickiness threshold is passed separately to the kernel.
    static constexpr float kIouStickinessThreshold = 0.3f;

    gpa::AimConfig rightGpuConfig() const { return toGpuConfig(right); }
    gpa::AimConfig thumbGpuConfig() const { return toGpuConfig(thumb); }

    void load(const nlohmann::json& j) {
        if (j.contains("aim_kp_x")) right.kp_x = j["aim_kp_x"];
        if (j.contains("aim_kp_y")) right.kp_y = j["aim_kp_y"];
        if (j.contains("aim_softness_x")) right.softness_x = j["aim_softness_x"];
        if (j.contains("aim_softness_y")) right.softness_y = j["aim_softness_y"];
        if (j.contains("aim_kd_x")) right.kd_x = j["aim_kd_x"];
        if (j.contains("aim_kd_y")) right.kd_y = j["aim_kd_y"];

        if (j.contains("thumb_aim_kp_x")) thumb.kp_x = j["thumb_aim_kp_x"];
        if (j.contains("thumb_aim_kp_y")) thumb.kp_y = j["thumb_aim_kp_y"];
        thumb.softness_x = j.contains("thumb_aim_softness_x")
                                ? j["thumb_aim_softness_x"].get<float>()
                                : right.softness_x;
        thumb.softness_y = j.contains("thumb_aim_softness_y")
                                ? j["thumb_aim_softness_y"].get<float>()
                                : right.softness_y;
        thumb.kd_x = j.contains("thumb_aim_kd_x")
                         ? j["thumb_aim_kd_x"].get<float>()
                         : right.kd_x;
        thumb.kd_y = j.contains("thumb_aim_kd_y")
                         ? j["thumb_aim_kd_y"].get<float>()
                         : right.kd_y;

        if (j.contains("inflight_deadtime_frames")) {
            const float v = j["inflight_deadtime_frames"];
            inflight_deadtime_frames = std::clamp(v, 0.0f, 4.0f);
        }
        if (j.contains("ff_gain")) ff_gain = j["ff_gain"];
        if (j.contains("predict_frames")) {
            const float v = j["predict_frames"];
            predict_frames = std::clamp(v, 0.0f, 6.0f);
        }
        if (j.contains("aim_y_scale")) {
            const float v = j["aim_y_scale"];
            aim_y_scale = std::clamp(v, 0.0f, 4.0f);
        }
        if (j.contains("aim_max_step")) max_step = j["aim_max_step"];
        if (j.contains("oneeuro_min_cutoff")) oneeuro_min_cutoff = j["oneeuro_min_cutoff"];
        if (j.contains("oneeuro_beta")) oneeuro_beta = j["oneeuro_beta"];
    }

    void save(nlohmann::json& j) const {
        j["aim_kp_x"] = right.kp_x;
        j["aim_kp_y"] = right.kp_y;
        j["aim_softness_x"] = right.softness_x;
        j["aim_softness_y"] = right.softness_y;
        j["aim_kd_x"] = right.kd_x;
        j["aim_kd_y"] = right.kd_y;
        j["thumb_aim_kp_x"] = thumb.kp_x;
        j["thumb_aim_kp_y"] = thumb.kp_y;
        j["thumb_aim_softness_x"] = thumb.softness_x;
        j["thumb_aim_softness_y"] = thumb.softness_y;
        j["thumb_aim_kd_x"] = thumb.kd_x;
        j["thumb_aim_kd_y"] = thumb.kd_y;

        j["inflight_deadtime_frames"] = inflight_deadtime_frames;
        j["ff_gain"] = ff_gain;
        j["predict_frames"] = predict_frames;
        j["aim_y_scale"] = aim_y_scale;
        j["aim_max_step"] = max_step;
        j["oneeuro_min_cutoff"] = oneeuro_min_cutoff;
        j["oneeuro_beta"] = oneeuro_beta;
    }

    void print() const {
        std::cout << "[Config] Right-click P: Kp(" << right.kp_x << "," << right.kp_y
                  << ") Softness(" << right.softness_x << "," << right.softness_y
                  << ") Kd(" << right.kd_x << "," << right.kd_y << ")" << std::endl;
        std::cout << "[Config] Thumb P: Kp(" << thumb.kp_x << "," << thumb.kp_y
                  << ") Softness(" << thumb.softness_x << "," << thumb.softness_y
                  << ") Kd(" << thumb.kd_x << "," << thumb.kd_y << ")" << std::endl;
        std::cout << "[Config] Dead-time: lag=" << inflight_deadtime_frames
                  << " frames" << std::endl;
        std::cout << "[Config] Lead ff_gain: "
                  << (ff_gain > 0.0f ? std::to_string(ff_gain) : std::string("OFF"))
                  << std::endl;
        std::cout << "[Config] Predict frames: "
                  << (predict_frames > 0.0f ? std::to_string(predict_frames) : std::string("OFF"))
                  << std::endl;
        std::cout << "[Config] Vertical assist scale: " << aim_y_scale << std::endl;
        std::cout << "[Config] Aim max step: " << max_step
                  << (max_step > 0.0f ? " px/frame" : " (disabled)") << std::endl;
        std::cout << "[Config] One Euro: min_cutoff=" << oneeuro_min_cutoff
                  << ", beta=" << oneeuro_beta << std::endl;
    }

private:
    gpa::AimConfig toGpuConfig(const Gains& gains) const {
        gpa::AimConfig aim;
        aim.kp_x = gains.kp_x;
        aim.kp_y = gains.kp_y;
        aim.p_softness_x = gains.softness_x;
        aim.p_softness_y = gains.softness_y;
        aim.kd_x = gains.kd_x;
        aim.kd_y = gains.kd_y;
        aim.deadtime_frames = inflight_deadtime_frames;
        aim.ff_gain = ff_gain;
        aim.predict_frames = predict_frames;
        aim.aim_y_scale = aim_y_scale;
        aim.max_step = max_step;
        aim.oneeuro_min_cutoff = oneeuro_min_cutoff;
        aim.oneeuro_beta = oneeuro_beta;
        return aim;
    }
};

}  // namespace pd_controller
