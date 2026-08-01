#pragma once

// Nonlinear P(D) aim controller: owns the tuned gains (kp/kd/softness per aim
// profile), the shared shaping around them (One Euro pre-filter, same-target
// stickiness, per-frame max step), and their JSON
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
    Gains right{0.765f, 0.698f, 8.6f, 6.57f, 0.052f, 0.037f};
    Gains thumb{0.6f, 0.62f, 11.0f, 10.0f, 0.25f, 0.35f};

    // Same-target stickiness for tracking (shared by both profiles).
    float iou_stickiness_threshold = 0.3f;
    float distance_stickiness_factor = 0.4f;
    int track_persistence_frames = 3;

    // Dead-time compensation: subtract the moves still in flight (emitted within
    // the last deadtime_frames, not yet visible in the detection) from the
    // error. Removes the double-correction that causes overshoot/ringing.
    // 0 = off (legacy). deadtime_frames must match the rig's measured
    // emit->visible lag (bench/calibrate.py STEP-RESPONSE).
    // FRACTIONAL: the real emit->visible lag is ~1.13 frames here, not an
    // integer, so rounding it either under- or over-compensates. 1.0 reproduces
    // the old integer behaviour exactly.
    float inflight_comp = 1.0f;
    float inflight_deadtime_frames = 1.25f;

    // Lead / feedforward on the ego-corrected target velocity (see
    // needaimbot/cuda/simple_postprocess.h for why this is sound now and was not
    // before the in-flight ring existed). ff_gain 0 = off (pure P+D).
    float ff_gain = 1.7f;
    float ff_ego_lag = 2.25f;
    float ff_v_ema = 0.235f;
    // ONE shared gate pair for BOTH lead terms. Per-term gates (pred_vgate /
    // pred_err_gate) were removed: they gated the same velocity against the same
    // error and bought <0.5% for twice the tuning surface; a properly tuned shared
    // pair beat the four-knob version (-0.8% error at equal ringing).
    float lead_vgate = 14.79f;
    float lead_err_gate = 22.25f;
    // Symmetric dead-time comp: extrapolate the TARGET forward over the dead time
    // (inflight_comp only removed OUR motion). 0 = off.
    float predict_frames = 3.18f;
    // Reject the head<->body anchor-flip artifact (the selected classId already
    // identifies those frames exactly). 1 = on. See simple_postprocess.h.
    float class_switch_reject = 1.0f;
    // body 가 잡히면 항상 body 를 쓰고, head 는 body 가 없을 때만 쓴다.
    float head_deprioritized = 1.0f;
    // body 조준점을 y1+k*h 대신 cy+(k-0.5)*h_ema 로 관측 (같은 점, 더 조용함).
    // 0 = 끔. 근거·실측치는 simple_postprocess.h 의 aim_h_ema 주석.
    float aim_h_ema = 0.2f;

    // Per-frame max move (output px). 0 = disabled (unbounded).
    float max_step = 19.56f;

    // One Euro adaptive low-pass on the target center (jitter suppression).
    bool oneeuro_enabled = true;
    float oneeuro_min_cutoff = 0.084f;
    float oneeuro_beta = 0.02f;

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

        if (j.contains("inflight_comp")) inflight_comp = j["inflight_comp"];
        if (j.contains("inflight_deadtime_frames")) {
            const float v = j["inflight_deadtime_frames"];
            inflight_deadtime_frames = std::clamp(v, 0.0f, 4.0f);
        }
        if (j.contains("ff_gain")) ff_gain = j["ff_gain"];
        if (j.contains("ff_ego_lag")) {
            const float v = j["ff_ego_lag"];
            ff_ego_lag = std::clamp(v, 1.0f, 4.0f);
        }
        if (j.contains("ff_v_ema")) {
            const float v = j["ff_v_ema"];
            ff_v_ema = std::clamp(v, 0.0f, 1.0f);
        }
        if (j.contains("lead_vgate")) lead_vgate = j["lead_vgate"];
        if (j.contains("lead_err_gate")) lead_err_gate = j["lead_err_gate"];
        if (j.contains("predict_frames")) {
            const float v = j["predict_frames"];
            predict_frames = std::clamp(v, 0.0f, 6.0f);
        }
        if (j.contains("head_deprioritized")) {
            const bool v = j["head_deprioritized"];
            head_deprioritized = v ? 1.0f : 0.0f;
        }
        if (j.contains("aim_h_ema")) {
            const float v = j["aim_h_ema"];
            aim_h_ema = std::clamp(v, 0.0f, 1.0f);
        }
        if (j.contains("class_switch_reject")) {
            const bool v = j["class_switch_reject"];
            class_switch_reject = v ? 1.0f : 0.0f;
        }
        if (j.contains("aim_max_step")) max_step = j["aim_max_step"];

        if (j.contains("oneeuro_enabled")) oneeuro_enabled = j["oneeuro_enabled"];
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

        j["_section_tracking"] = "===== Target tracking / stickiness =====";
        j["iou_stickiness_threshold"] = iou_stickiness_threshold;
        j["distance_stickiness_factor"] = distance_stickiness_factor;
        j["track_persistence_frames"] = track_persistence_frames;
        j["inflight_comp"] = inflight_comp;
        j["inflight_deadtime_frames"] = inflight_deadtime_frames;
        j["aim_max_step"] = max_step;

        j["_section_lead"] = "===== Lead term (ego-corrected velocity feedforward) =====";
        j["ff_gain"] = ff_gain;
        j["ff_ego_lag"] = ff_ego_lag;
        j["ff_v_ema"] = ff_v_ema;
        j["lead_vgate"] = lead_vgate;
        j["lead_err_gate"] = lead_err_gate;
        j["predict_frames"] = predict_frames;
        j["class_switch_reject"] = (class_switch_reject != 0.0f);
        j["head_deprioritized"] = (head_deprioritized != 0.0f);
        j["aim_h_ema"] = aim_h_ema;

        j["_section_oneeuro"] = "===== One Euro center filter (jitter suppression) =====";
        j["oneeuro_enabled"] = oneeuro_enabled;
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
        std::cout << "[Config] IoU stickiness: " << iou_stickiness_threshold << std::endl;
        std::cout << "[Config] Distance stickiness factor: " << distance_stickiness_factor
                  << (distance_stickiness_factor > 0.0f ? " (ON)" : " (OFF)") << std::endl;
        std::cout << "[Config] Dead-time compensation: "
                  << (inflight_comp > 0.0f
                          ? "ON (comp=" + std::to_string(inflight_comp) + ", lag=" +
                                std::to_string(inflight_deadtime_frames) + " frames)"
                          : std::string("OFF"))
                  << std::endl;
        std::cout << "[Config] Lead (ego-corrected ff): "
                  << (ff_gain > 0.0f
                          ? "ON (gain=" + std::to_string(ff_gain) + ", ego_lag=" +
                                std::to_string(ff_ego_lag) + ", vgate=" +
                                std::to_string(lead_vgate) + ", err_gate=" +
                                std::to_string(lead_err_gate) + ")"
                          : std::string("OFF"))
                  << std::endl;
        std::cout << "[Config] Target extrapolation: "
                  << (predict_frames > 0.0f
                          ? "ON (" + std::to_string(predict_frames) + " frames)"
                          : std::string("OFF"))
                  << std::endl;
        std::cout << "[Config] Body-priority selection: "
                  << (head_deprioritized != 0.0f ? "ON (head = fallback only)" : "OFF")
                  << std::endl;
        std::cout << "[Config] Body aim-point observation: "
                  << (aim_h_ema > 0.0f
                          ? "centre + smoothed height (a=" + std::to_string(aim_h_ema) + ")"
                          : std::string("raw y1 + k*h"))
                  << std::endl;
        std::cout << "[Config] Class-switch rejection: "
                  << (class_switch_reject != 0.0f ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] Aim max step: " << max_step
                  << (max_step > 0.0f ? " px/frame" : " (disabled)") << std::endl;
        std::cout << "[Config] One Euro center filter: " << (oneeuro_enabled ? "ON" : "OFF")
                  << " (min_cutoff=" << oneeuro_min_cutoff << ", beta=" << oneeuro_beta << ")" << std::endl;
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
        aim.inflight_comp = inflight_comp;
        aim.deadtime_frames = inflight_deadtime_frames;
        aim.ff_gain = ff_gain;
        aim.ff_ego_lag = ff_ego_lag;
        aim.ff_v_ema = ff_v_ema;
        aim.lead_vgate = lead_vgate;
        aim.lead_err_gate = lead_err_gate;
        aim.predict_frames = predict_frames;
        aim.class_switch_reject = class_switch_reject;
        aim.head_deprioritized = head_deprioritized;
        aim.aim_h_ema = aim_h_ema;
        aim.oneeuro_enabled = oneeuro_enabled ? 1.0f : 0.0f;
        aim.oneeuro_min_cutoff = oneeuro_min_cutoff;
        aim.oneeuro_beta = oneeuro_beta;
        aim.max_step = max_step;
        return aim;
    }
};

}  // namespace pd_controller
