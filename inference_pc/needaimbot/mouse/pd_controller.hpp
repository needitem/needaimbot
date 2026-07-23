#pragma once

// Nonlinear P(D) aim controller: owns the tuned gains (kp/kd/softness per aim
// profile), the shared shaping around them (coast gap-glide, One Euro
// pre-filter, same-target stickiness, per-frame max step), and their JSON
// load/save/print - decoupled from simple_main.cpp's
// app-wide Config. The actual PD math runs on the GPU (see
// needaimbot/cuda/simple_postprocess.cu); this header only builds the
// gpa::AimConfig the GPU kernel consumes.

#include <algorithm>
#include <iostream>

#include "simple_postprocess.h"
#include "json.hpp"

namespace pd_controller {

// One aim profile's tuned gains (there are two: right-click and thumb/Side2).
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
    Gains right{0.55f, 0.6f, 11.0f, 10.0f, 0.18f, 0.22f};
    Gains thumb{0.6f, 0.62f, 11.0f, 10.0f, 0.25f, 0.35f};

    // Same-target stickiness for tracking (shared by both profiles).
    float iou_stickiness_threshold = 0.3f;
    float distance_stickiness_factor = 0.5f;
    int track_persistence_frames = 5;

    // Coast: bridge brief detection gaps by gliding from the last movement
    // (decayed) instead of freezing or shaking.
    bool coast_enabled = true;
    float coast_decay = 0.85f;

    // Dead-time compensation: subtract the moves still in flight (emitted within
    // the last deadtime_frames, not yet visible in the detection) from the
    // error. Removes the double-correction that causes overshoot/ringing.
    // 0 = off (legacy). deadtime_frames must match the rig's measured
    // emit->visible lag (bench/calibrate.py STEP-RESPONSE).
    float inflight_comp = 0.0f;
    int inflight_deadtime_frames = 1;

    // Per-frame max move (output px). 0 = disabled (unbounded).
    float max_step = 30.0f;

    // One Euro adaptive low-pass on the target center (jitter suppression).
    bool oneeuro_enabled = true;
    float oneeuro_min_cutoff = 0.1f;
    float oneeuro_beta = 0.02f;
    float oneeuro_dcutoff = 0.5f;

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
        // Thumb softness/kd default to the right-click profile's values when
        // not explicitly configured.
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

        if (j.contains("iou_stickiness_threshold")) iou_stickiness_threshold = j["iou_stickiness_threshold"];
        if (j.contains("distance_stickiness_factor")) distance_stickiness_factor = j["distance_stickiness_factor"];
        if (j.contains("track_persistence_frames")) {
            const int v = j["track_persistence_frames"];
            track_persistence_frames = std::clamp(v, 0, 60);
        }

        if (j.contains("coast_enabled")) coast_enabled = j["coast_enabled"];
        if (j.contains("coast_decay")) coast_decay = j["coast_decay"];
        if (j.contains("inflight_comp")) inflight_comp = j["inflight_comp"];
        if (j.contains("inflight_deadtime_frames")) {
            const int v = j["inflight_deadtime_frames"];
            inflight_deadtime_frames = std::clamp(v, 0, 4);
        }
        if (j.contains("aim_max_step")) max_step = j["aim_max_step"];

        if (j.contains("oneeuro_enabled")) oneeuro_enabled = j["oneeuro_enabled"];
        if (j.contains("oneeuro_min_cutoff")) oneeuro_min_cutoff = j["oneeuro_min_cutoff"];
        if (j.contains("oneeuro_beta")) oneeuro_beta = j["oneeuro_beta"];
        if (j.contains("oneeuro_dcutoff")) oneeuro_dcutoff = j["oneeuro_dcutoff"];
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

        j["_section_tracking"] = "===== Target tracking / stickiness =====";
        j["iou_stickiness_threshold"] = iou_stickiness_threshold;
        j["distance_stickiness_factor"] = distance_stickiness_factor;
        j["track_persistence_frames"] = track_persistence_frames;
        j["coast_enabled"] = coast_enabled;
        j["coast_decay"] = coast_decay;
        j["inflight_comp"] = inflight_comp;
        j["inflight_deadtime_frames"] = inflight_deadtime_frames;
        j["aim_max_step"] = max_step;

        j["_section_oneeuro"] = "===== One Euro center filter (jitter suppression) =====";
        j["oneeuro_enabled"] = oneeuro_enabled;
        j["oneeuro_min_cutoff"] = oneeuro_min_cutoff;
        j["oneeuro_beta"] = oneeuro_beta;
        j["oneeuro_dcutoff"] = oneeuro_dcutoff;
    }

    void print() const {
        std::cout << "[Config] Right-click P: Kp(" << right.kp_x << "," << right.kp_y
                  << ") Softness(" << right.softness_x << "," << right.softness_y
                  << ") Kd(" << right.kd_x << "," << right.kd_y << ")" << std::endl;
        std::cout << "[Config] Thumb P: Kp(" << thumb.kp_x << "," << thumb.kp_y
                  << ") Softness(" << thumb.softness_x << "," << thumb.softness_y
                  << ") Kd(" << thumb.kd_x << "," << thumb.kd_y << ")" << std::endl;
        std::cout << "[Config] IoU stickiness: " << iou_stickiness_threshold << std::endl;
        std::cout << "[Config] Distance stickiness factor: " << distance_stickiness_factor
                  << (distance_stickiness_factor > 0.0f ? " (ON)" : " (OFF)") << std::endl;
        std::cout << "[Config] Coast (gap glide): " << (coast_enabled ? "ON" : "OFF")
                  << " (decay=" << coast_decay << ", window=" << track_persistence_frames
                  << " frames)" << std::endl;
        std::cout << "[Config] Dead-time compensation: "
                  << (inflight_comp > 0.0f
                          ? "ON (comp=" + std::to_string(inflight_comp) + ", lag=" +
                                std::to_string(inflight_deadtime_frames) + " frames)"
                          : std::string("OFF"))
                  << std::endl;
        std::cout << "[Config] Aim max step: " << max_step
                  << (max_step > 0.0f ? " px/frame" : " (disabled)") << std::endl;
        std::cout << "[Config] One Euro center filter: " << (oneeuro_enabled ? "ON" : "OFF")
                  << " (min_cutoff=" << oneeuro_min_cutoff << ", beta=" << oneeuro_beta
                  << ", dcutoff=" << oneeuro_dcutoff << ")" << std::endl;
        std::cout << "[Config] Track persistence: " << track_persistence_frames
                  << " frame(s)"
                  << (track_persistence_frames > 0 ? " (ON)" : " (OFF)") << std::endl;
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
        aim.distance_stickiness_factor = distance_stickiness_factor;
        aim.track_persistence_frames = track_persistence_frames;
        aim.coast_enabled = coast_enabled ? 1.0f : 0.0f;
        aim.coast_decay = coast_decay;
        aim.inflight_comp = inflight_comp;
        aim.deadtime_frames = inflight_deadtime_frames;
        aim.oneeuro_enabled = oneeuro_enabled ? 1.0f : 0.0f;
        aim.oneeuro_min_cutoff = oneeuro_min_cutoff;
        aim.oneeuro_beta = oneeuro_beta;
        aim.oneeuro_dcutoff = oneeuro_dcutoff;
        aim.max_step = max_step;
        return aim;
    }
};

}  // namespace pd_controller
