#pragma once

// Application configuration: Config struct (fields, load, save, print,
// getAllowedClassMask) and rate-rescale helpers.  Header-only, following the
// existing needaimbot/app/ convention.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "simple_postprocess.h"
#include "pd_controller.hpp"
#include "json.hpp"

// Current shipped-config generation. See Config::configVersion.
//   1 = 2026-08-01: adaptive dead time + the gains re-optimised for it, the
//       thumb profile's first tuning, aim_h_ema, ego_frame_filter removed.
//   2 = 2026-08-03: first direct measurement of the emit->visible dead time
//       (11.6ms = 2.85 frames, vs the 1.10 the sim had assumed) plus the 240Hz
//       monitor swap. Every frame-denominated value moves; a v1 file left in
//       place would keep the old ones and ring hard - overshoot 37 measured
//       under the real plant. See CONFIG_REFERENCE.
//   6 = 2026-09-11: the x2 capture-upscale conversion is REMOVED from
//       aim_softness_x and the two lead gates (they go back to the 320 values:
//       8.6 / 14.79 / 22.25) - simulation at both noise levels showed the doubled
//       values were leaving tracking lag on the horizontal axis (strafe X error
//       -20..-26%, dwell +7pt, lost -3pt at the production y_scale). soft_y keeps
//       the doubled value; aim_y_scale 0.1 becomes the shipped default (vertical
//       assist deliberately kept human-like). A v5 file keeps the old x values.
inline constexpr int kConfigVersion = 6;

// Max per-frame swing of the adaptive dead-time correction, in frames. Bounds
// the damage a bad clock-offset sample can do, and the in-flight ring is 4 deep.
inline constexpr double kDeadtimeAdaptMax = 0.75;
// Bounds on the frame-rate rescale. A wild period estimate must not be able to
// swing the loop gain; outside this the capture is broken anyway.
inline constexpr long kFramePeriodWarmupSkip = 30;   // graph capture / first frames
inline constexpr long kFramePeriodLogAfter   = 600;  // ~2.6s at 230fps, mean settled
inline constexpr double kRateScaleMin = 0.35;   // ~410fps against a 143 baseline
inline constexpr double kRateScaleMax = 2.00;   // ~72fps

// Convert an AimConfig authored at one capture rate to the rate actually running.
// r = T_actual / T_tuned.
//
//   x r   px/frame or per-frame coefficient : kp, max_step, lead_vgate, ff_v_ema,
//                                             oneeuro_min_cutoff
//   / r   frame counts, velocity multipliers : predict_frames, ff_gain
//   fixed px, and things already carrying T  : softness, kd, lead_err_gate, beta
//
// kd looks like it should scale and must not: it multiplies derr, which is px per
// FRAME and so already carries a factor of T, and the wanted per-frame output
// carries the same one. They cancel. oneeuro_min_cutoff does scale, because
// oneEuroAlpha() assumes Te = 1 frame, making its cutoff cycles-per-frame.
// beta stays fixed for the same reason kd does - it multiplies a px/frame speed.
inline void rescaleAimConfigForRate(gpa::AimConfig& c, double r) {
    const float fr = static_cast<float>(r);
    c.kp_x *= fr;
    c.kp_y *= fr;
    c.max_step *= fr;
    c.oneeuro_min_cutoff *= fr;
    c.predict_frames /= fr;
    c.ff_gain /= fr;
}

// Configuration
struct Config {
    // Engine
    // NOTE: We intentionally run the plain fp16 engine, not fp16io. Switching the
    // I/O tensors to fp16 (fp16io) degrades detection accuracy too much to be
    // worth the small bandwidth/latency gain, so do not "optimize" this to fp16io.
    std::string enginePath = "engines/sunxds_0.8.2_256_fp16.engine";
    std::string makcuPort = "/dev/ttyACM0";
    int udpPort = 5007;

    // Detection
    float confThreshold = 0.35f;
    int headClassId = 1;      // Head class for headshot priority
    int maxDetections = 100;  // Maximum detections per frame

    // Class filtering (max 32 classes)
    std::vector<bool> classAllowed = std::vector<bool>(32, true);  // Which classes to target
    int maxClasses = 32;

    // Aiming (0 = top, 1 = bottom of bbox)
    float headAimPoint = 1.0f;   // Head: aim at bottom (neck area)
    float bodyAimPoint = 0.15f;  // Body: aim near top (chest area)

    // Nonlinear P(D) aim controller (gains, One Euro, stickiness) -
    // see needaimbot/mouse/pd_controller.hpp.
    pd_controller::Settings pd;

    // No-recoil
    bool noRecoilEnabled = true;
    float recoilCompX = 0.0f;
    float recoilCompY = 0.8f;
    int recoilTickMs = 10;

    // Static shoot-offset aim-shift (applied while aiming + shooting).
    float shootOffsetX = 0.0f;
    float shootOffsetY = -13.0f;

    // Mouse rate limiting. 0 = no rate limit AND enables the zero-latency
    // direct-callback aim path (submitAimMovement sends straight from the GPU
    // completion callback instead of hopping through the sender thread/queue).
    // See directAimMoveInCallback below - it is gated on mouseMinIntervalMs <= 0.
    int mouseMinIntervalMs = 0;
    int maxInFlightFrames = 1;    // Keep latency low by avoiding stale queued frames
    int frameCreditDepth = 2;     // Allow the Game PC to pre-send the newest next frame
    bool directAimMoveInCallback = true;

    // Makcu settings
    int makcuBaudrate = 4000000;

    // CPU core affinity (latency stability). When disabled, the receive and
    // callback threads keep their built-in default pinning (last / last-1 core)
    // and the main/sender threads float. When enabled, each thread is pinned to
    // the configured core; -1 leaves that thread unpinned.
    bool cpuAffinityEnabled = false;
    int affinityCoreMain = 5;       // frame-acquire + submit loop
    int affinityCoreReceive = 7;    // UDP receive thread
    int affinityCoreCallback = 6;   // GPU completion / mouse-move worker
    int affinityCoreSender = 4;     // move-sender thread (used when not direct)

    // Runtime diagnostics. The screen shows only user-facing health (FPS, aim/
    // shoot, packet drops); the full per-window metrics (latency avg/max +
    // p50/p95/p99, stage breakdown) are appended to perfLogPath instead. A
    // relative path is resolved next to the config file; empty disables the log.
    // Off by default so a normal install writes nothing to disk - turn on only
    // when measuring. When on, the log is size-capped (perfLogMaxBytes) with one
    // rotated backup, so total disk use is bounded even over long runs.
    bool perfStatsEnabled = false;
    // Separate from perf stats since v4. They answer different questions - perf
    // stats is "where does the time go in our pipeline", calibration is "what did
    // the detector see and what did we emit" - and a play session wants the second
    // without the first. They were one switch only because measurement always
    // wanted both.
    bool calibrationLogEnabled = false;
    int perfStatsIntervalMs = 1000;
    std::string perfLogPath = "perf_stats.log";
    int perfLogMaxBytes = 33554432;       // rotate at 32 MiB (0 = no rotation, grows unbounded)
    bool perfLogTruncateOnStart = false;  // true = overwrite log each run instead of appending
    bool realtimeThreadsEnabled = true;
    bool forceAimOn = false;  // Benchmark/testing override (keeps inference loop active)
    bool idleGraphPrecaptureEnabled = true;
    int idleGraphPrecaptureIntervalMs = 100;
    // Shapes whose CUDA graph should be captured at startup (before the first
    // frame arrives). Each entry is {sourceWidth, sourceHeight} in pixels.
    // Empty by default - capture happens lazily on first incoming frame.
    std::vector<std::pair<int, int>> preCaptureShapes;
    bool stageTimingEnabled = false;  // Per-stage CUDA event timings (opt-in)
    // Use each frame's measured staleness for the in-flight dead-time subtraction
    // instead of a fixed deadtime_frames. See trySubmitLatestFrame.
    bool deadtimeAdaptive = true;
    // Frame-denominated parameters mean different physical things at different
    // capture rates, and the rate is NOT fixed - one 400s session ran 226-246fps,
    // and three captures on 2026-08-07 read 199, 206 and 237 depending on what was
    // being measured. Rescaling by hand got it wrong three times in one day, so the
    // conversion happens per frame from the measured period instead. The numbers in
    // the config mean what they say at aim_tuned_fps; the runtime does the rest.
    bool aimRateAdaptive = true;
    float aimTunedFps = 143.0f;
    // The rate-independent part of the emit->visible dead time: USB plus the game's
    // own render pipeline. The rest is capture sampling (~T/2), which follows the
    // rate. Injection measured 11.60ms total at 143fps (T=6.99), so the fixed part
    // is 8.10 - and expressed this way it reproduces that measurement at the rate it
    // was taken instead of only at one rate.
    //
    // The shipped 10.2 is deliberately ~2ms ABOVE that. It is a damping choice, not
    // a measurement, for two reasons:
    //  - the real dead time is not knowable per session. It varies with load and
    //    with the game's own frame time, and it cannot be measured while playing:
    //    estimating it from our own emits (bench/deadtime_closedloop.py) returns
    //    1.45-4.85 frames on captures whose injected truth is 1.66.
    //  - the penalty is sharply asymmetric. Sweeping the plant 8-15ms at 231fps,
    //    a w BELOW the plant blows up (overshoot 16.97 at w=2.10 vs plant 3.40)
    //    while a w above it just slows down (reach 22 -> 38ms). Total-error spread
    //    across the same plant range narrows monotonically with w: 30.8% at w=2.10,
    //    17.3% at 2.85, 14.3% at 3.10.
    // So this errs high on purpose. Lower it for speed, raise it for calm.
    float inflightDeadtimeMs = 10.2f;

    // Calibration capture path (for bench/calibrate.py). Only the WHERE - the
    // logger is enabled by perf_stats_enabled, not by this path. A relative path
    // is written next to the binary's working dir.
    std::string calibrationLogPath = "calib.csv";
    // Step-response dead-time measurement. >0 = inject +-this many mouse counts
    // (X) every calibrationStepPeriodMs while the log is on. Aim OFF at a
    // stationary target; calibrate.py cross-correlates cx vs the injection to
    // find the full emit->visible dead-time. 0 = off.
    // Bumped whenever a shipped default changes in a way an existing config
    // would silently override. The file's values still win - this only makes the
    // staleness visible instead of leaving it to be discovered by feel.
    int configVersion = 0;

    int calibrationStepPx = 0;
    int calibrationStepPeriodMs = 250;

    // Inference keep-warm tail. Inference normally runs ONLY while the aim key
    // is held (saves GPU/power), so the first frame of a fresh aim pays one cold
    // detection cycle before the crosshair reacts. This keeps inference running
    // for N ms AFTER aiming stops, so a quick re-aim within the tail reacquires
    // instantly (already-warm detection) at the cost of some idle GPU work.
    //   0  = strict gate (default, max power saving, cold reacquire every aim)
    //  >0  = keep inferring this many ms after the aim key releases
    //  <0  = always on (never gate; no cold reacquire, full idle GPU cost)
    int inferenceKeepwarmMs = 0;

    // Busy-poll the UDP receive socket instead of sleeping in the kernel.
    // Skips the IRQ->wakeup latency (~5-15us) on each frame's first packet at
    // the cost of pinning the receive core at 100% while running.
    bool udpBusySpin = false;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;

        try {
            nlohmann::json j;
            f >> j;

            if (j.contains("engine_path")) enginePath = j["engine_path"];
            if (j.contains("makcu_port")) makcuPort = j["makcu_port"];
            if (j.contains("udp_port")) udpPort = j["udp_port"];

            if (j.contains("conf_threshold")) confThreshold = j["conf_threshold"];
            if (j.contains("head_class_id")) headClassId = j["head_class_id"];
            if (j.contains("max_detections")) maxDetections = j["max_detections"];

            if (j.contains("head_aim_point")) headAimPoint = j["head_aim_point"];
            if (j.contains("body_aim_point")) bodyAimPoint = j["body_aim_point"];

            pd.load(j);

            if (j.contains("no_recoil_enabled")) noRecoilEnabled = j["no_recoil_enabled"];
            if (j.contains("recoil_comp_x")) recoilCompX = j["recoil_comp_x"];
            if (j.contains("recoil_comp_y")) recoilCompY = j["recoil_comp_y"];
            if (j.contains("recoil_tick_ms")) recoilTickMs = j["recoil_tick_ms"];

            if (j.contains("shoot_offset_x")) shootOffsetX = j["shoot_offset_x"];
            if (j.contains("shoot_offset_y")) shootOffsetY = j["shoot_offset_y"];

            if (j.contains("mouse_min_interval_ms")) mouseMinIntervalMs = j["mouse_min_interval_ms"];
            if (j.contains("max_inflight_frames")) maxInFlightFrames = j["max_inflight_frames"];
            if (j.contains("frame_credit_depth")) frameCreditDepth = j["frame_credit_depth"];
            if (j.contains("direct_aim_move_in_callback")) directAimMoveInCallback = j["direct_aim_move_in_callback"];
            if (j.contains("makcu_baudrate")) makcuBaudrate = j["makcu_baudrate"];

            if (j.contains("perf_stats_enabled")) perfStatsEnabled = j["perf_stats_enabled"];
            if (j.contains("perf_stats_interval_ms")) perfStatsIntervalMs = j["perf_stats_interval_ms"];
            if (j.contains("perf_log_path")) perfLogPath = j["perf_log_path"];
            if (j.contains("perf_log_max_bytes")) perfLogMaxBytes = j["perf_log_max_bytes"];
            if (j.contains("perf_log_truncate_on_start")) perfLogTruncateOnStart = j["perf_log_truncate_on_start"];
            if (j.contains("realtime_threads_enabled")) realtimeThreadsEnabled = j["realtime_threads_enabled"];
            if (j.contains("cpu_affinity_enabled")) cpuAffinityEnabled = j["cpu_affinity_enabled"];
            if (j.contains("affinity_core_main")) affinityCoreMain = j["affinity_core_main"];
            if (j.contains("affinity_core_receive")) affinityCoreReceive = j["affinity_core_receive"];
            if (j.contains("affinity_core_callback")) affinityCoreCallback = j["affinity_core_callback"];
            if (j.contains("affinity_core_sender")) affinityCoreSender = j["affinity_core_sender"];
            if (j.contains("force_aim_on")) forceAimOn = j["force_aim_on"];
            if (j.contains("idle_graph_precapture_enabled")) idleGraphPrecaptureEnabled = j["idle_graph_precapture_enabled"];
            if (j.contains("idle_graph_precapture_interval_ms")) idleGraphPrecaptureIntervalMs = j["idle_graph_precapture_interval_ms"];
            if (j.contains("config_version")) configVersion = j["config_version"];
            if (j.contains("stage_timing_enabled")) stageTimingEnabled = j["stage_timing_enabled"];
            if (j.contains("deadtime_adaptive")) deadtimeAdaptive = j["deadtime_adaptive"];
            // A pre-v5 file holds gains that were ALREADY rescaled by hand for the
            // rate it was authored at. Letting the new default turn rate adaptation
            // on would rescale them a second time - about 0.6x at play rates - and
            // the version banner would not stop it, because the banner only prints;
            // the file always wins (see the config_version note in the reference).
            // So the switch defaults off for old files and on for v5+. An explicit
            // key still wins either way.
            aimRateAdaptive = (configVersion >= 5);
            if (j.contains("aim_rate_adaptive")) aimRateAdaptive = j["aim_rate_adaptive"];
            if (j.contains("aim_tuned_fps")) aimTunedFps = j["aim_tuned_fps"];
            if (j.contains("inflight_deadtime_ms")) inflightDeadtimeMs = j["inflight_deadtime_ms"];
            // Split out of perf_stats_enabled in v4. A file written before that has
            // no such key and was authored expecting the old master-switch, so
            // inherit instead of defaulting to off - silently not recording a
            // session someone asked for is worse than writing one extra file.
            calibrationLogEnabled = perfStatsEnabled;
            if (j.contains("calibration_log_enabled")) calibrationLogEnabled = j["calibration_log_enabled"];
            if (j.contains("calibration_log_path")) calibrationLogPath = j["calibration_log_path"];
            if (j.contains("calibration_step_px")) calibrationStepPx = j["calibration_step_px"];
            if (j.contains("calibration_step_period_ms")) calibrationStepPeriodMs = j["calibration_step_period_ms"];
            if (j.contains("inference_keepwarm_ms")) inferenceKeepwarmMs = j["inference_keepwarm_ms"];
            if (j.contains("udp_busy_spin")) udpBusySpin = j["udp_busy_spin"];
            preCaptureShapes.clear();
            if (j.contains("pre_capture_shapes")) {
                for (const auto& entry : j["pre_capture_shapes"]) {
                    if (!entry.is_array() || entry.size() != 2) continue;
                    const int w = entry[0].get<int>();
                    const int h = entry[1].get<int>();
                    if (w > 0 && h > 0 && w <= 8192 && h <= 8192) {
                        preCaptureShapes.emplace_back(w, h);
                    }
                }
            }

            // Class filtering - either "allowed_classes": [0, 1, 7] or detailed class_settings
            classAllowed.resize(maxClasses, true);  // Default: all classes allowed

            if (j.contains("allowed_classes")) {
                // Simple format: list of allowed class IDs
                std::fill(classAllowed.begin(), classAllowed.end(), false);
                for (const auto& cls : j["allowed_classes"]) {
                    int id = cls.get<int>();
                    if (id >= 0 && id < maxClasses) {
                        classAllowed[id] = true;
                    }
                }
            } else if (j.contains("class_settings")) {
                // Detailed format: array of {id, name, allow}
                std::fill(classAllowed.begin(), classAllowed.end(), false);
                for (const auto& cs : j["class_settings"]) {
                    if (cs.contains("id") && cs.contains("allow")) {
                        int id = cs["id"].get<int>();
                        bool allow = cs["allow"].get<bool>();
                        if (id >= 0 && id < maxClasses) {
                            classAllowed[id] = allow;
                        }
                    }
                }
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Config] Error parsing: " << e.what() << std::endl;
            return false;
        }
    }

    bool save(const std::string& path) const {
        try {
            nlohmann::json j;
            // Config is written in labelled sections. The "_section_*" keys are
            // pure visual dividers (JSON has no comments) - the loader ignores
            // any key it does not recognise, so they are harmless. Keys are
            // grouped by what you tune together; see CONFIG_REFERENCE.md for a
            // per-key description, default and typical range.
            auto section = [&j](const char* title) {
                j[std::string("_section_") + title] =
                    std::string("========== ") + title + " ==========";
            };

            section("ENGINE / NETWORK");
            j["engine_path"] = enginePath;
            j["udp_port"] = udpPort;
            j["makcu_port"] = makcuPort;
            j["makcu_baudrate"] = makcuBaudrate;

            section("DETECTION");
            j["conf_threshold"] = confThreshold;
            j["head_class_id"] = headClassId;
            j["max_detections"] = maxDetections;
            {
                nlohmann::json allowedList = nlohmann::json::array();
                if (classAllowed.empty()) {
                    for (int i = 0; i < maxClasses; i++) allowedList.push_back(i);
                } else {
                    for (size_t i = 0; i < classAllowed.size(); i++) {
                        if (classAllowed[i]) allowedList.push_back(static_cast<int>(i));
                    }
                }
                j["allowed_classes"] = allowedList;
            }
            {
                nlohmann::json shapes = nlohmann::json::array();
                for (const auto& wh : preCaptureShapes) {
                    shapes.push_back({wh.first, wh.second});
                }
                j["pre_capture_shapes"] = shapes;
            }

            section("AIM POINT");
            j["head_aim_point"] = headAimPoint;
            j["body_aim_point"] = bodyAimPoint;
            j["shoot_offset_x"] = shootOffsetX;
            j["shoot_offset_y"] = shootOffsetY;

            section("AIM CONTROLLER + CENTER FILTER");
            pd.save(j);  // PD gains, thumb gains, stickiness, One Euro

            section("RECOIL COMPENSATION");
            j["no_recoil_enabled"] = noRecoilEnabled;
            j["recoil_comp_x"] = recoilCompX;
            j["recoil_comp_y"] = recoilCompY;
            j["recoil_tick_ms"] = recoilTickMs;

            section("PIPELINE / LATENCY");
            j["max_inflight_frames"] = maxInFlightFrames;
            j["frame_credit_depth"] = frameCreditDepth;
            j["mouse_min_interval_ms"] = mouseMinIntervalMs;
            j["direct_aim_move_in_callback"] = directAimMoveInCallback;
            j["inference_keepwarm_ms"] = inferenceKeepwarmMs;
            j["udp_busy_spin"] = udpBusySpin;
            j["idle_graph_precapture_enabled"] = idleGraphPrecaptureEnabled;
            j["idle_graph_precapture_interval_ms"] = idleGraphPrecaptureIntervalMs;

            section("SYSTEM (threads / CPU affinity)");
            j["realtime_threads_enabled"] = realtimeThreadsEnabled;
            j["cpu_affinity_enabled"] = cpuAffinityEnabled;
            j["affinity_core_main"] = affinityCoreMain;
            j["affinity_core_receive"] = affinityCoreReceive;
            j["affinity_core_callback"] = affinityCoreCallback;
            j["affinity_core_sender"] = affinityCoreSender;

            section("DIAGNOSTICS (perf log / calibration / bench)");
            j["perf_stats_enabled"] = perfStatsEnabled;
            j["perf_stats_interval_ms"] = perfStatsIntervalMs;
            j["perf_log_path"] = perfLogPath;
            j["perf_log_max_bytes"] = perfLogMaxBytes;
            j["perf_log_truncate_on_start"] = perfLogTruncateOnStart;
            j["config_version"] = kConfigVersion;
            j["stage_timing_enabled"] = stageTimingEnabled;
            j["deadtime_adaptive"] = deadtimeAdaptive;
            j["aim_rate_adaptive"] = aimRateAdaptive;
            j["aim_tuned_fps"] = aimTunedFps;
            j["inflight_deadtime_ms"] = inflightDeadtimeMs;
            j["force_aim_on"] = forceAimOn;
            j["calibration_log_enabled"] = calibrationLogEnabled;
            j["calibration_log_path"] = calibrationLogPath;
            j["calibration_step_px"] = calibrationStepPx;
            j["calibration_step_period_ms"] = calibrationStepPeriodMs;

            const auto configPath = std::filesystem::path(path);
            const auto parent = configPath.parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
            }

            std::ofstream f(path);
            if (!f) return false;
            f << j.dump(4) << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Config] Error saving: " << e.what() << std::endl;
            return false;
        }
    }

    void print() const {
        std::cout << "[Config] Engine: " << enginePath << std::endl;
        std::cout << "[Config] Makcu: " << makcuPort << std::endl;
        std::cout << "[Config] UDP port: " << udpPort << std::endl;
        std::cout << "[Config] Confidence: " << confThreshold << std::endl;
        pd.print();
        std::cout << "[Config] Max detections: " << maxDetections << std::endl;
        std::cout << "[Config] No-recoil: " << (noRecoilEnabled ? "ON" : "OFF")
                  << " (Y=" << recoilCompY << ", tick=" << recoilTickMs << "ms)" << std::endl;
        std::cout << "[Config] Shoot offset: (" << shootOffsetX << ", " << shootOffsetY << ")"
                  << ((shootOffsetX != 0.0f || shootOffsetY != 0.0f) ? "" : " (off)") << std::endl;
        std::cout << "[Config] Max in-flight frames: " << maxInFlightFrames << std::endl;
        std::cout << "[Config] Frame credit depth: " << frameCreditDepth << std::endl;
        std::cout << "[Config] Direct aim move in callback: "
                  << (directAimMoveInCallback ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] Perf stats: " << (perfStatsEnabled ? "ON" : "OFF")
                  << " (interval=" << perfStatsIntervalMs << "ms)" << std::endl;
        std::cout << "[Config] Inference gate: "
                  << (inferenceKeepwarmMs < 0 ? std::string("always-on (no gate)")
                      : inferenceKeepwarmMs == 0 ? std::string("aim-only (strict, cold reacquire)")
                      : std::string("aim + ") + std::to_string(inferenceKeepwarmMs) + "ms keep-warm tail")
                  << std::endl;
        std::cout << "[Config] Perf log: "
                  << (perfLogPath.empty() ? std::string("(disabled)") : perfLogPath);
        if (!perfLogPath.empty()) {
            std::cout << " (rotate=" << (perfLogMaxBytes > 0 ? std::to_string(perfLogMaxBytes) + "B" : "off")
                      << ", truncate_on_start=" << (perfLogTruncateOnStart ? "yes" : "no") << ")";
        }
        std::cout << std::endl;
        std::cout << "[Config] Realtime thread hints: " << (realtimeThreadsEnabled ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] CPU affinity: " << (cpuAffinityEnabled ? "ON" : "OFF");
        if (cpuAffinityEnabled) {
            std::cout << " (main=" << affinityCoreMain << ", recv=" << affinityCoreReceive
                      << ", cb=" << affinityCoreCallback << ", sender=" << affinityCoreSender << ")";
        }
        std::cout << std::endl;
        std::cout << "[Config] Force aim override: " << (forceAimOn ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] Idle graph pre-capture: "
                  << (idleGraphPrecaptureEnabled ? "ON" : "OFF")
                  << " (interval=" << idleGraphPrecaptureIntervalMs << "ms)" << std::endl;
        if (configVersion < kConfigVersion) {
            std::cout << "[Config] *** This config predates the current tuning (file v"
                      << configVersion << " < v" << kConfigVersion << ") ***" << std::endl;
            std::cout << "[Config]     Its values OVERRIDE the shipped defaults, so any"
                         " key it already has stays at the old value." << std::endl;
            std::cout << "[Config]     Most likely stale: thumb_aim_kd_*, aim_kp_*,"
                         " aim_softness_*, aim_max_step, oneeuro_min_cutoff." << std::endl;
            if (configVersion < 5 && !aimRateAdaptive) {
                std::cout << "[Config]     Rate normalisation stays OFF: a pre-v5 file's gains"
                             " were already rescaled by hand, and doing it again"
                             " would land at ~0.6x." << std::endl;
            } else if (configVersion < 5) {
                // The explicit key wins, as it should - but say what it implies, since
                // copying the flag into an old file without also reverting the gains to
                // the aim_tuned_fps baseline is the one way back into the 0.6x case.
                std::cout << "[Config]     *** aim_rate_adaptive is ON in a pre-v5 file. Its gains"
                             " must be the aim_tuned_fps baseline, not already-rescaled"
                             " values, or they get rescaled twice. ***" << std::endl;
            }
            std::cout << "[Config]     Refresh with tools/preset.sh {320|160}, or delete"
                         " the file to be reseeded from simple_config.default.json."
                      << std::endl;
        }
        std::cout << "[Config] Adaptive dead-time: "
                  << (deadtimeAdaptive ? "ON (per-frame measured staleness)"
                                       : "OFF (fixed deadtime_frames)") << std::endl;
        std::cout << "[Config] Stage timing: "
                  << (stageTimingEnabled ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] Rate normalisation: ";
        if (aimRateAdaptive)
            std::cout << "ON (values authored at " << aimTunedFps << " fps, rescaled per frame;"
                      << " deadtime " << inflightDeadtimeMs << "ms + T/2)";
        else
            std::cout << "OFF (values used as-is)";
        std::cout << std::endl;
        std::cout << "[Config] Calibration log: ";
        if (calibrationLogEnabled)
            std::cout << (calibrationLogPath.empty() ? std::string("calib.csv") : calibrationLogPath);
        else
            std::cout << "(off; calibration_log_enabled=true to log to "
                      << (calibrationLogPath.empty() ? std::string("calib.csv") : calibrationLogPath)
                      << ")";
        if (calibrationStepPx > 0)
            std::cout << " [step-response +-" << calibrationStepPx << " cnt/" << calibrationStepPeriodMs << "ms]";
        std::cout << std::endl;
        std::cout << "[Config] Pre-capture shapes: ";
        if (preCaptureShapes.empty()) {
            std::cout << "(none, lazy)";
        } else {
            for (size_t i = 0; i < preCaptureShapes.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << preCaptureShapes[i].first << "x" << preCaptureShapes[i].second;
            }
        }
        std::cout << std::endl;
        // Print allowed classes
        std::cout << "[Config] Allowed classes: ";
        bool first = true;
        for (size_t i = 0; i < classAllowed.size(); i++) {
            if (classAllowed[i]) {
                if (!first) std::cout << ", ";
                std::cout << i;
                first = false;
            }
        }
        if (first) std::cout << "(none)";
        std::cout << std::endl;
    }

    // Get bitmask of allowed classes for GPU
    uint32_t getAllowedClassMask() const {
        uint32_t mask = 0;
        for (size_t i = 0; i < classAllowed.size() && i < 32; i++) {
            if (classAllowed[i]) mask |= (1u << i);
        }
        return mask;
    }
};
