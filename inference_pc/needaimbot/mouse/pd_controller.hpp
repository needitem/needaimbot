#pragma once

// Nonlinear P(D) aim controller: owns the tuned gains (kp/kd/softness per aim
// profile), the shared shaping around them (coast gap-glide, velocity
// feedforward, One Euro pre-filter, same-target stickiness, per-frame max
// step), and their JSON load/save/print - decoupled from simple_main.cpp's
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
    float feedforward_gain = 0.9f;

    // Per-frame max move (output px). 0 = disabled (unbounded).
    float max_step = 30.0f;

    // One Euro adaptive low-pass on the target center (jitter suppression).
    bool oneeuro_enabled = true;
    float oneeuro_min_cutoff = 0.1f;
    float oneeuro_beta = 0.02f;
    float oneeuro_dcutoff = 0.5f;

    // Steady-state motor noise on the real per-frame movement (Harris-Wolpert
    // SDN + velocity-modulated tremor), shared by both profiles - see
    // gpa::AimConfig/AimState. Off (0) by default; opt in once verified,
    // since this adds jitter to live tracking precision.
    float aim_sdn_k = 0.0f;
    float aim_tremor_amp = 0.0f;
    float aim_tremor_freq_min = 8.0f;
    float aim_tremor_freq_max = 12.0f;
    float aim_tremor_dt_ms = 7.8f;

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
        if (j.contains("feedforward_gain")) feedforward_gain = j["feedforward_gain"];
        if (j.contains("aim_max_step")) max_step = j["aim_max_step"];

        if (j.contains("oneeuro_enabled")) oneeuro_enabled = j["oneeuro_enabled"];
        if (j.contains("oneeuro_min_cutoff")) oneeuro_min_cutoff = j["oneeuro_min_cutoff"];
        if (j.contains("oneeuro_beta")) oneeuro_beta = j["oneeuro_beta"];
        if (j.contains("oneeuro_dcutoff")) oneeuro_dcutoff = j["oneeuro_dcutoff"];

        if (j.contains("aim_sdn_k")) aim_sdn_k = j["aim_sdn_k"];
        if (j.contains("aim_tremor_amp")) aim_tremor_amp = j["aim_tremor_amp"];
        if (j.contains("aim_tremor_freq_min")) aim_tremor_freq_min = j["aim_tremor_freq_min"];
        if (j.contains("aim_tremor_freq_max")) aim_tremor_freq_max = j["aim_tremor_freq_max"];
        if (j.contains("aim_tremor_dt_ms")) aim_tremor_dt_ms = j["aim_tremor_dt_ms"];
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

        j["iou_stickiness_threshold"] = iou_stickiness_threshold;
        j["distance_stickiness_factor"] = distance_stickiness_factor;
        j["track_persistence_frames"] = track_persistence_frames;
        j["coast_enabled"] = coast_enabled;
        j["coast_decay"] = coast_decay;
        j["feedforward_gain"] = feedforward_gain;
        j["aim_max_step"] = max_step;

        j["_section_oneeuro"] = "===== One Euro center filter (jitter suppression) =====";
        j["oneeuro_enabled"] = oneeuro_enabled;
        j["oneeuro_min_cutoff"] = oneeuro_min_cutoff;
        j["oneeuro_beta"] = oneeuro_beta;
        j["oneeuro_dcutoff"] = oneeuro_dcutoff;

        j["_section_motor_noise"] = "===== Steady-state motor noise (SDN + tremor) =====";
        j["aim_sdn_k"] = aim_sdn_k;
        j["aim_tremor_amp"] = aim_tremor_amp;
        j["aim_tremor_freq_min"] = aim_tremor_freq_min;
        j["aim_tremor_freq_max"] = aim_tremor_freq_max;
        j["aim_tremor_dt_ms"] = aim_tremor_dt_ms;
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
        std::cout << "[Config] Velocity feedforward: " << feedforward_gain << std::endl;
        std::cout << "[Config] Aim max step: " << max_step
                  << (max_step > 0.0f ? " px/frame" : " (disabled)") << std::endl;
        std::cout << "[Config] One Euro center filter: " << (oneeuro_enabled ? "ON" : "OFF")
                  << " (min_cutoff=" << oneeuro_min_cutoff << ", beta=" << oneeuro_beta
                  << ", dcutoff=" << oneeuro_dcutoff << ")" << std::endl;
        std::cout << "[Config] Track persistence: " << track_persistence_frames
                  << " frame(s)"
                  << (track_persistence_frames > 0 ? " (ON)" : " (OFF)") << std::endl;
        std::cout << "[Config] Steady-state motor noise: SDN k=" << aim_sdn_k
                  << (aim_sdn_k > 0.0f ? " (ON)" : " (OFF)")
                  << ", tremor amp=" << aim_tremor_amp
                  << (aim_tremor_amp > 0.0f ? " (ON)" : " (OFF)") << std::endl;
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
        aim.feedforward_gain = feedforward_gain;
        aim.oneeuro_enabled = oneeuro_enabled ? 1.0f : 0.0f;
        aim.oneeuro_min_cutoff = oneeuro_min_cutoff;
        aim.oneeuro_beta = oneeuro_beta;
        aim.oneeuro_dcutoff = oneeuro_dcutoff;
        aim.max_step = max_step;
        aim.sdn_k = aim_sdn_k;
        aim.tremor_amp = aim_tremor_amp;
        aim.tremor_freq_min = aim_tremor_freq_min;
        aim.tremor_freq_max = aim_tremor_freq_max;
        aim.tremor_dt_ms = aim_tremor_dt_ms;
        return aim;
    }
};

}  // namespace pd_controller
