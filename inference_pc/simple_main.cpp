// Simple aimbot - clean implementation
// UDP capture + TensorRT inference + Mouse control via Makcu
// Features: Full GPU pipeline (inference + postprocess + nonlinear P), No-recoil
// GPU Callback API for lowest latency (no cudaStreamSync wait)
// Minimal CPU usage - frame receive on CPU, inference/postprocess/movement on GPU

#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <condition_variable>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include "needaimbot/cuda/simple_inference.h"
#include "needaimbot/cuda/simple_postprocess.h"
#include "needaimbot/capture/udp_capture.h"
#include "needaimbot/mouse/input_drivers/MakcuConnection.h"
#include "needaimbot/mouse/controller.h"
#include "needaimbot/mouse/pd_controller.hpp"
#include "needaimbot/mouse/warped_replay.hpp"
#include "needaimbot/app/engine_locator.hpp"
#include "needaimbot/app/runtime_options.hpp"
#include "needaimbot/app/runtime_diagnostics.hpp"
#include "needaimbot/capture/debug_frame_dump.hpp"

// Third-party JSON parser (header-only)
#include "needaimbot/modules/json.hpp"
using json = nlohmann::json;

using Clock = std::chrono::steady_clock;

std::atomic<bool> g_running{true};
std::atomic<int> g_frameCount{0};  // Completed inference callbacks per stat window
std::atomic<uint64_t> g_callbackLatencySamples{0};
std::atomic<int64_t> g_callbackLatencyTotalUs{0};
std::atomic<int64_t> g_callbackLatencyMaxUs{0};
AtomicLatencyHistogram g_callbackLatencyHist;  // submit->completion latency percentiles

// Capture->inference-complete end-to-end latency (game-PC capture timestamp in
// the UDP header vs. this PC's completion time). Assumes NTP-synced wall clocks;
// clock-skew outliers are dropped at record time. Mouse actuation is NOT included.
std::atomic<uint64_t> g_e2eLatencySamples{0};
std::atomic<int64_t> g_e2eLatencyTotalUs{0};
std::atomic<int64_t> g_e2eLatencyMaxUs{0};
AtomicLatencyHistogram g_e2eLatencyHist;

// This PC's wall clock (system_clock) epoch microseconds, for subtracting the
// game-PC capture timestamp carried in the UDP header.
static int64_t nowUnixMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Wall-clock stamp for perf-log lines (local time, millisecond resolution) so a
// window's metrics can be correlated with events on the game PC.
static std::string formatWallClock() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) % 1000;
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    std::ostringstream os;
    os << std::put_time(&tmv, "%Y-%m-%d %H:%M:%S") << '.'
       << std::setfill('0') << std::setw(3) << ms.count();
    return os.str();
}

void signalHandler(int sig) {
    std::cout << "\n[Simple] Received signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

// Forward declaration
struct Config;
struct CallbackContext;

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

    // Nonlinear P(D) aim controller (gains, coast, One Euro, stickiness) -
    // see needaimbot/mouse/pd_controller.hpp.
    pd_controller::Settings pd;

    // Humanized acquisition-flick trajectory, played back in place of
    // pd_controller's own output for the first stretch of a freshly locked
    // target - see needaimbot/mouse/warped_replay.hpp.
    bool flickEnabled = true;
    warped_replay::config flick;

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
    // Wire encoding for mouse moves: false = ASCII km.move (proven default),
    // true = MAKCU binary frame (8B vs ~14B). Opt-in; validate on real hardware
    // before trusting it - see MakcuConnection::setBinaryMove.
    bool makcuBinaryMove = false;

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

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;

        try {
            json j;
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

            if (j.contains("flick_enabled")) flickEnabled = j["flick_enabled"];
            flick.load(j);

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
            if (j.contains("makcu_binary_move")) makcuBinaryMove = j["makcu_binary_move"];

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
            if (j.contains("stage_timing_enabled")) stageTimingEnabled = j["stage_timing_enabled"];
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
            json j;
            j["engine_path"] = enginePath;
            j["makcu_port"] = makcuPort;
            j["udp_port"] = udpPort;

            j["conf_threshold"] = confThreshold;
            j["head_class_id"] = headClassId;
            j["max_detections"] = maxDetections;

            j["head_aim_point"] = headAimPoint;
            j["body_aim_point"] = bodyAimPoint;

            pd.save(j);

            j["flick_enabled"] = flickEnabled;
            flick.save(j);

            j["no_recoil_enabled"] = noRecoilEnabled;
            j["recoil_comp_x"] = recoilCompX;
            j["recoil_comp_y"] = recoilCompY;
            j["recoil_tick_ms"] = recoilTickMs;

            j["shoot_offset_x"] = shootOffsetX;
            j["shoot_offset_y"] = shootOffsetY;

            j["mouse_min_interval_ms"] = mouseMinIntervalMs;
            j["max_inflight_frames"] = maxInFlightFrames;
            j["frame_credit_depth"] = frameCreditDepth;
            j["direct_aim_move_in_callback"] = directAimMoveInCallback;
            j["makcu_baudrate"] = makcuBaudrate;
            j["makcu_binary_move"] = makcuBinaryMove;

            j["perf_stats_enabled"] = perfStatsEnabled;
            j["perf_stats_interval_ms"] = perfStatsIntervalMs;
            j["perf_log_path"] = perfLogPath;
            j["perf_log_max_bytes"] = perfLogMaxBytes;
            j["perf_log_truncate_on_start"] = perfLogTruncateOnStart;
            j["realtime_threads_enabled"] = realtimeThreadsEnabled;
            j["cpu_affinity_enabled"] = cpuAffinityEnabled;
            j["affinity_core_main"] = affinityCoreMain;
            j["affinity_core_receive"] = affinityCoreReceive;
            j["affinity_core_callback"] = affinityCoreCallback;
            j["affinity_core_sender"] = affinityCoreSender;
            j["force_aim_on"] = forceAimOn;
            j["idle_graph_precapture_enabled"] = idleGraphPrecaptureEnabled;
            j["idle_graph_precapture_interval_ms"] = idleGraphPrecaptureIntervalMs;
            j["stage_timing_enabled"] = stageTimingEnabled;
            {
                json shapes = json::array();
                for (const auto& wh : preCaptureShapes) {
                    shapes.push_back({wh.first, wh.second});
                }
                j["pre_capture_shapes"] = shapes;
            }

            // Save allowed classes as simple list
            json allowedList = json::array();
            if (classAllowed.empty()) {
                // Default: all classes allowed
                for (int i = 0; i < maxClasses; i++) allowedList.push_back(i);
            } else {
                for (size_t i = 0; i < classAllowed.size(); i++) {
                    if (classAllowed[i]) allowedList.push_back(static_cast<int>(i));
                }
            }
            j["allowed_classes"] = allowedList;

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
        std::cout << "[Config] Acquisition flick: " << (flickEnabled ? "ON" : "OFF") << std::endl;
        if (flickEnabled) flick.print();
        std::cout << "[Config] Max detections: " << maxDetections << std::endl;
        std::cout << "[Config] No-recoil: " << (noRecoilEnabled ? "ON" : "OFF")
                  << " (Y=" << recoilCompY << ", tick=" << recoilTickMs << "ms)" << std::endl;
        std::cout << "[Config] Shoot offset: (" << shootOffsetX << ", " << shootOffsetY << ")"
                  << ((shootOffsetX != 0.0f || shootOffsetY != 0.0f) ? "" : " (off)") << std::endl;
        std::cout << "[Config] Max in-flight frames: " << maxInFlightFrames << std::endl;
        std::cout << "[Config] Frame credit depth: " << frameCreditDepth << std::endl;
        std::cout << "[Config] Direct aim move in callback: "
                  << (directAimMoveInCallback ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] Makcu move encoding: "
                  << (makcuBinaryMove ? "BINARY (8B frame)" : "ASCII (km.move)") << std::endl;
        std::cout << "[Config] Perf stats: " << (perfStatsEnabled ? "ON" : "OFF")
                  << " (interval=" << perfStatsIntervalMs << "ms)" << std::endl;
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
        std::cout << "[Config] Stage timing: "
                  << (stageTimingEnabled ? "ON" : "OFF") << std::endl;
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

// =============================================================================
// GPU Callback Handler
// =============================================================================
// This callback runs from the completion worker when GPU inference finishes.
// It only decides *whether* to move (aiming active + target present) and then
// hands the raw (dx, dy) off to the controller, which owns everything about
// *how* the move actually reaches the mouse (see needaimbot/mouse/controller.h).
//
// OPTIMIZATION: Cached config values eliminate pointer indirection in hot path.
// All frequently accessed values are copied to the context struct at init time.

struct CallbackContext {
    // Hardware/collaborator references (only things we can't cache)
    MakcuConnection* makcu;
    UDPCapture* udpCapture;
    controller::MouseController* controller = nullptr;
    std::condition_variable* pipelineCv = nullptr;
    std::mutex* pipelineCvMutex = nullptr;

    // Cached config values (lock-free, no pointer chasing)
    bool forceAimOn = false;
    bool perfStatsEnabled = false;

    // Acquisition-flick playback (warped_replay) - started on freshAcquire,
    // sampled instead of the PD movement until it finishes. Callback-thread
    // only (see needaimbot/mouse/warped_replay.hpp), so a single instance per
    // context is fine.
    bool flickEnabled = true;
    warped_replay::config flickConfig;
    warped_replay::FlickPlayback flickPlayback;

    // Initialize cached values from config
    void initFromConfig(const Config& cfg) {
        forceAimOn = cfg.forceAimOn;
        perfStatsEnabled = cfg.perfStatsEnabled;
        flickEnabled = cfg.flickEnabled;
        flickConfig = cfg.flick;
        // Preload the replay DB now (startup), not on the first flick's
        // callback - the ~130ms JSON parse must not land on the hot path.
        if (flickEnabled) warped_replay::warmup(flickConfig);
    }
};

struct CallbackTicket {
    CallbackContext* ctx = nullptr;
    int bufferIndex = -1;
    Clock::time_point submitTime{};
    uint64_t captureUnixMicros = 0;  // game-PC capture wall time, for E2E latency
    std::atomic<bool> busy{false};
};

// GPU callback handler - called immediately when inference completes
// OPTIMIZED: Uses cached config values, no pointer indirection
void inferenceCallback(const gpa::InferenceResult& result, void* userData) {
    auto* ticket = static_cast<CallbackTicket*>(userData);
    if (!ticket || !ticket->ctx) {
        return;
    }

    CallbackContext* ctx = ticket->ctx;
    if (ctx->perfStatsEnabled && ticket->submitTime.time_since_epoch().count() != 0) {
        const int64_t latencyUs = elapsedUs(ticket->submitTime, Clock::now());
        g_callbackLatencySamples.fetch_add(1, std::memory_order_relaxed);
        g_callbackLatencyTotalUs.fetch_add(latencyUs, std::memory_order_relaxed);
        atomicMax(g_callbackLatencyMaxUs, latencyUs);
        g_callbackLatencyHist.record(latencyUs);
    }
    // Capture->inference-complete end-to-end latency (excludes mouse actuation).
    // The game-PC capture time is on the game clock, so add the measured clock
    // offset (game - inference) to map our completion time onto the same clock.
    // Skipped until the first ping/pong round trip has produced an offset.
    if (ctx->perfStatsEnabled && ticket->captureUnixMicros != 0 && ctx->udpCapture) {
        bool offsetValid = false;
        const int64_t offset = ctx->udpCapture->GetClockOffsetMicros(&offsetValid);
        if (offsetValid) {
            const int64_t e2eUs =
                (nowUnixMicros() + offset) - static_cast<int64_t>(ticket->captureUnixMicros);
            // Residual out-of-range means the offset estimate is still settling
            // or a transient; drop so it doesn't poison the stats.
            if (e2eUs >= 0 && e2eUs < 1000000) {
                g_e2eLatencySamples.fetch_add(1, std::memory_order_relaxed);
                g_e2eLatencyTotalUs.fetch_add(e2eUs, std::memory_order_relaxed);
                atomicMax(g_e2eLatencyMaxUs, e2eUs);
                g_e2eLatencyHist.record(e2eUs);
            }
        }
    }

    auto releaseTicket = [ticket, ctx]() {
        const int completedBuffer = ticket->bufferIndex;
        if (completedBuffer >= 0 && ctx->udpCapture) {
            ctx->udpCapture->ReleaseFrame(completedBuffer);
        }
        auto markReleased = [&]() {
            ticket->bufferIndex = -1;
            ticket->submitTime = Clock::time_point{};
            ticket->captureUnixMicros = 0;
            ticket->busy.store(false, std::memory_order_release);
        };
        if (ctx->pipelineCv && ctx->pipelineCvMutex) {
            {
                std::lock_guard<std::mutex> lock(*ctx->pipelineCvMutex);
                markReleased();
            }
            ctx->pipelineCv->notify_one();
        } else {
            markReleased();
        }
    };

    // Count every completed inference callback (target/no-target)
    g_frameCount.fetch_add(1, std::memory_order_relaxed);

    const uint8_t callbackButtonMask = ctx->makcu->buttonMask();
    const bool aimingActive = ctx->forceAimOn || controller::maskAiming(callbackButtonMask);
    if (!aimingActive) {
        // Aim released mid-flick: cancel rather than leave it active. GPU-side
        // tracking (has_track) keeps running while aiming is off, so the same
        // track can still be live with no fresh acquire when aiming resumes -
        // left active, sample() would see a huge stale elapsed time and fire
        // one giant catch-up jump to the flick's endpoint.
        ctx->flickPlayback.cancel();
        releaseTicket();
        return;
    }

    if (!result.hasTarget) {
        // Same reasoning as above: target lost (not just coasted) mid-flick.
        ctx->flickPlayback.cancel();
        releaseTicket();
        return;
    }

    // A fresh lock starts (and immediately takes over from) a humanized
    // acquisition flick; a continued/coasted track never (re)starts one.
    if (ctx->flickEnabled && result.freshAcquire) {
        // Use the actually-acquired target's own size (Fitts' law "W") rather
        // than a fixed config guess, so a small target gets a longer, more-
        // corrected flick and a large one a quicker, more direct one - same
        // as a human. Averaged width/height since the flick doesn't reason
        // about movement-axis-relative target extent.
        const double detectedTargetWidth =
            0.5 * ((result.targetX2 - result.targetX1) + (result.targetY2 - result.targetY1));
        ctx->flickPlayback.start(result.errorX, result.errorY,
                                  result.movementScaleX, result.movementScaleY,
                                  detectedTargetWidth,
                                  ctx->flickConfig);
    }

    int moveDx = result.movement.dx;
    int moveDy = result.movement.dy;
    if (ctx->flickPlayback.active()) {
        if (auto delta = ctx->flickPlayback.sample(Clock::now())) {
            moveDx = delta->dx;
            moveDy = delta->dy;
        }
    }

    // Inference is done - hand the result off to the controller, which
    // decides how to turn it into physical mouse motion. Pass the shooting
    // state so it can apply the static shoot-offset aim-shift (aiming is
    // already gated above).
    ctx->controller->submitAimMovement(
        moveDx, moveDy, controller::maskShooting(callbackButtonMask));

    releaseTicket();
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "=== Simple Aimbot (Full GPU Pipeline) ===" << std::endl;

    RuntimeOptions runtimeOptions;
    if (!parseRuntimeOptions(argc, argv, runtimeOptions)) {
        return 1;
    }
    if (runtimeOptions.helpRequested) {
        return 0;
    }

    // Load config
    Config cfg;
    const std::filesystem::path exePath = engine_locator::currentExecutablePath(argv[0]);
    const std::filesystem::path exeDir =
        exePath.has_parent_path() ? exePath.parent_path() : std::filesystem::current_path();
    const auto repoRoot = engine_locator::findRepoRoot(exeDir);
    std::filesystem::path configPath;
    if (runtimeOptions.hasConfigPath) {
        configPath = runtimeOptions.configPath;
    } else {
        configPath = engine_locator::chooseConfigPath(exeDir);
    }

    configPath = engine_locator::safeAbsolute(configPath);
    const std::string configPathStr = configPath.lexically_normal().string();
    bool configDirty = false;

    if (cfg.load(configPathStr)) {
        std::cout << "[Config] Loaded from " << configPathStr << std::endl;
    } else {
        std::cout << "[Config] Using defaults; will save to " << configPathStr << std::endl;
        configDirty = true;
    }

    if (runtimeOptions.buttonTest) {
        std::cout << "[ButtonTest] Opening Makcu at " << cfg.makcuPort
                  << " (baudrate: " << cfg.makcuBaudrate << ")" << std::endl;
        MakcuConnection makcu(cfg.makcuPort, cfg.makcuBaudrate);
        if (!makcu.isOpen()) {
            std::cerr << "[ButtonTest] Failed to open Makcu" << std::endl;
            return 1;
        }

        auto printButtonMask = [](uint8_t mask) {
            std::cout << "\r[ButtonTest] mask=0x"
                      << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
                      << static_cast<int>(mask)
                      << std::dec << std::nouppercase << std::setfill(' ')
                      << " L:" << ((mask & controller::kMakcuLeftMask) ? "ON " : "OFF")
                      << " R:" << ((mask & controller::kMakcuRightMask) ? "ON " : "OFF")
                      << " M:" << ((mask & controller::kMakcuMiddleMask) ? "ON " : "OFF")
                      << " S1:" << ((mask & controller::kMakcuSide1Mask) ? "ON " : "OFF")
                      << " S2:" << ((mask & controller::kMakcuSide2Mask) ? "ON " : "OFF")
                      << "    " << std::flush;
        };

        std::cout << "[ButtonTest] Press mouse buttons. Ctrl+C to exit." << std::endl;
        uint64_t lastSeq = makcu.buttonSequence();
        uint8_t lastMask = makcu.buttonMask();
        printButtonMask(lastMask);
        while (g_running.load(std::memory_order_relaxed)) {
            makcu.waitForButtonEvent(lastSeq, 1000);
            const uint64_t seq = makcu.buttonSequence();
            const uint8_t mask = makcu.buttonMask();
            if (seq != lastSeq || mask != lastMask) {
                lastSeq = seq;
                lastMask = mask;
                printButtonMask(mask);
            }
        }
        std::cout << std::endl;
        return 0;
    }

    if (runtimeOptions.collectCalib) {
        const int target = runtimeOptions.collectCalibCount;
        std::error_code ec;
        std::filesystem::create_directories(runtimeOptions.collectCalibDir, ec);
        UDPCapture cap;
        if (!cap.Initialize(cfg.udpPort) || !cap.StartCapture()) {
            std::cerr << "[Collect] Failed to start UDP capture on port " << cfg.udpPort << std::endl;
            return 1;
        }
        std::cout << "[Collect] Saving up to " << target << " unique RGB frames to "
                  << runtimeOptions.collectCalibDir.string() << std::endl;
        std::cout << "[Collect] Play the game normally (move around, aim at enemies). Ctrl+C to stop early." << std::endl;
        int saved = 0;
        uint64_t lastFid = UINT64_MAX;
        while (g_running.load(std::memory_order_relaxed) && saved < target) {
            cap.SendFrameCredit(0, 2);  // keep the Game PC streaming
            void* px = nullptr;
            unsigned int w = 0, h = 0;
            uint64_t fid = 0;
            int bi = -1;
            uint8_t bpp = 3, fmt = UDP_PIXEL_FORMAT_RGB;
            const bool got = cap.AcquireFramePinned(&px, &w, &h, &fid, &bi, 200, &bpp, &fmt);
            if (got) {
                const bool valid = px && w != 0 && h != 0 && fmt == UDP_PIXEL_FORMAT_RGB && bpp == 3;
                if (valid && fid != lastFid) {  // skip duplicate (latest-frame) re-reads
                    lastFid = fid;
                    std::ostringstream name;
                    name << "f" << std::setw(6) << std::setfill('0') << saved
                         << "_" << w << "x" << h << "_rgb.bin";
                    std::ofstream of((runtimeOptions.collectCalibDir / name.str()).string(),
                                     std::ios::binary);
                    if (of) {
                        of.write(reinterpret_cast<const char*>(px),
                                 static_cast<std::streamsize>(static_cast<size_t>(w) * h * 3));
                        ++saved;
                        if (saved % 25 == 0 || saved == target) {
                            std::cout << "\r[Collect] " << saved << "/" << target << std::flush;
                        }
                    }
                }
                if (bi >= 0) cap.ReleaseFrame(bi);
            }
        }
        std::cout << "\n[Collect] Saved " << saved << " frames to "
                  << runtimeOptions.collectCalibDir.string() << std::endl;
        cap.StopCapture();
        return 0;
    }

    const std::string requestedEnginePath = cfg.enginePath;
    auto resolvedEnginePath = engine_locator::resolveEnginePath(cfg.enginePath, configPath, exeDir, repoRoot);
    if (!resolvedEnginePath) {
        std::cerr << "[Config] Engine not found: " << requestedEnginePath << std::endl;
        std::cerr << "[Config] Set engine_path to an existing .engine file in "
                  << configPathStr << std::endl;
        return 1;
    }
    cfg.enginePath = resolvedEnginePath->lexically_normal().string();
    if (cfg.enginePath != requestedEnginePath) {
        std::cout << "[Config] Engine resolved: " << requestedEnginePath
                  << " -> " << cfg.enginePath << std::endl;
        configDirty = true;
    }

    if (configDirty) {
        if (cfg.save(configPathStr)) {
            std::cout << "[Config] Saved normalized config to " << configPathStr << std::endl;
        } else {
            std::cerr << "[Config] Warning: failed to save " << configPathStr << std::endl;
        }
    }

    DebugFrameDumper debugFrameDumper;
    if (runtimeOptions.debugFrameDump) {
        std::filesystem::path debugDir;
        if (runtimeOptions.hasDebugDir) {
            debugDir = runtimeOptions.debugDir;
        } else if (repoRoot) {
            debugDir = *repoRoot / "inference_pc" / "debug";
        } else {
            debugDir = configPath.parent_path() / "debug";
        }
        debugFrameDumper.enabled = true;
        debugFrameDumper.outputPath = engine_locator::safeAbsolute(debugDir / "received_frame.bmp").lexically_normal();
        debugFrameDumper.nextCaptureTime = Clock::now();
        debugFrameDumper.shootOffsetX = cfg.shootOffsetX;
        debugFrameDumper.shootOffsetY = cfg.shootOffsetY;
        std::cout << "[Debug] Frame dump: ON -> "
                  << debugFrameDumper.outputPath.string()
                  << " (1 Hz, overwrite)" << std::endl;
    }

    cfg.print();
    if (cfg.realtimeThreadsEnabled) {
        applyRealtimeHint("simple-main", 4);
    }
    if (cfg.cpuAffinityEnabled) {
        pinThreadToCore(cfg.affinityCoreMain);
    }

    // 1. Load TensorRT engine
    gpa::SimpleInference inference;
    inference.setMaxDetections(cfg.maxDetections);
    inference.setStageTimingEnabled(cfg.stageTimingEnabled);
    if (cfg.cpuAffinityEnabled) {
        inference.setCallbackAffinity(cfg.affinityCoreCallback);
    }
    if (!inference.loadEngine(cfg.enginePath)) {
        std::cerr << "[Simple] Failed to load engine" << std::endl;
        return 1;
    }

    // 2. Initialize Makcu connection
    MakcuConnection makcu(cfg.makcuPort, cfg.makcuBaudrate);
    if (!makcu.isOpen()) {
        std::cerr << "[Simple] Failed to open Makcu at " << cfg.makcuPort
                  << " (baudrate: " << cfg.makcuBaudrate << ")" << std::endl;
        return 1;
    }
    makcu.setBinaryMove(cfg.makcuBinaryMove);
    std::cout << "[Simple] Makcu connected at " << cfg.makcuBaudrate << " baud"
              << " (move encoding: " << (cfg.makcuBinaryMove ? "binary" : "ASCII") << ")"
              << std::endl;

    // 3. Initialize UDP capture
    UDPCapture udpCapture;
    if (cfg.cpuAffinityEnabled) {
        udpCapture.SetReceiveAffinity(cfg.affinityCoreReceive);
    }
    if (!udpCapture.Initialize(cfg.udpPort)) {
        std::cerr << "[Simple] Failed to initialize UDP capture" << std::endl;
        return 1;
    }
    // StartCapture() (spins up the receive thread) is deferred until the frame-
    // ready callback below is registered - the callback must be set before the
    // receive thread can observe a completed frame.

    // 4. State - all GPU now, minimal CPU state
    const gpa::AimConfig rightGpuAimConfig = cfg.pd.rightGpuConfig();
    const gpa::AimConfig thumbGpuAimConfig = cfg.pd.thumbGpuConfig();
    auto selectGpuAimConfig = [&](uint8_t buttonMask) -> const gpa::AimConfig& {
        return controller::maskThumbAiming(buttonMask) ? thumbGpuAimConfig : rightGpuAimConfig;
    };
    const uint32_t allowedClassMask = cfg.getAllowedClassMask();

    // Movement controller: owns the move queue, sender thread, and no-recoil
    // ticking. main only ever hands it inference results / button state.
    controller::MouseController movementController;
    controller::Settings controllerSettings;
    controllerSettings.forceAimOn = cfg.forceAimOn;
    controllerSettings.directAimMoveInCallback =
        cfg.directAimMoveInCallback && cfg.mouseMinIntervalMs <= 0;
    controllerSettings.mouseMinIntervalMs = cfg.mouseMinIntervalMs;
    controllerSettings.noRecoilEnabled = cfg.noRecoilEnabled;
    controllerSettings.recoilCompX = cfg.recoilCompX;
    controllerSettings.recoilCompY = cfg.recoilCompY;
    controllerSettings.recoilTickMs = cfg.recoilTickMs;
    // shoot-offset is applied as a GPU-side aim-reference shift (fed per-frame
    // via frameAimConfig), not through the controller - see the callback above.
    movementController.configure(makcu, controllerSettings);
    movementController.start([&]() {
        if (cfg.realtimeThreadsEnabled) {
            applyRealtimeHint("move-sender", 2);
        }
        if (cfg.cpuAffinityEnabled) {
            pinThreadToCore(cfg.affinityCoreSender);
        }
    });

    // Setup callback context with cached config values
    CallbackContext callbackCtx;
    callbackCtx.makcu = &makcu;
    callbackCtx.udpCapture = &udpCapture;
    callbackCtx.controller = &movementController;
    std::condition_variable pipelineCv;
    std::mutex pipelineCvMutex;
    callbackCtx.pipelineCv = &pipelineCv;
    callbackCtx.pipelineCvMutex = &pipelineCvMutex;
    callbackCtx.initFromConfig(cfg);  // Cache config values (lock-free)
    constexpr size_t kCallbackTicketCount = 4;
    std::array<CallbackTicket, kCallbackTicketCount> callbackTickets{};
    for (auto& ticket : callbackTickets) {
        ticket.ctx = &callbackCtx;
        ticket.bufferIndex = -1;
        ticket.busy.store(false, std::memory_order_relaxed);
    }
    size_t callbackTicketCursor = 0;
    auto acquireCallbackTicket = [&]() -> CallbackTicket* {
        for (size_t attempt = 0; attempt < callbackTickets.size(); ++attempt) {
            const size_t idx = (callbackTicketCursor + attempt) % callbackTickets.size();
            bool expected = false;
            if (callbackTickets[idx].busy.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                callbackTickets[idx].bufferIndex = -1;
                callbackTicketCursor = (idx + 1) % callbackTickets.size();
                return &callbackTickets[idx];
            }
        }
        return nullptr;
    };
    auto busyCallbackTicketCount = [&]() {
        int busyCount = 0;
        for (const auto& ticket : callbackTickets) {
            if (ticket.busy.load(std::memory_order_acquire)) {
                ++busyCount;
            }
        }
        return busyCount;
    };
    const int maxPipelineInFlight = std::clamp(cfg.maxInFlightFrames, 1, 4);
    const uint32_t frameCreditDepth = static_cast<uint32_t>(std::clamp(cfg.frameCreditDepth, 1, 4));
    const int perfStatsIntervalMs = std::clamp(cfg.perfStatsIntervalMs, 250, 10000);
    const int idleGraphPrecaptureIntervalMs = std::clamp(cfg.idleGraphPrecaptureIntervalMs, 20, 1000);

    auto lastStatTime = std::chrono::steady_clock::now();

    std::cout << "[Simple] Full GPU pipeline: ENABLED (inference + decode + target + nonlinear P)" << std::endl;
    std::cout << "[Simple] GPU Callback API: ENABLED (lowest latency, no sync wait)" << std::endl;
    std::cout << "[Simple] Lock-free config cache: ENABLED" << std::endl;
    std::cout << "[Simple] Movement post-processing: shoot-offset" << std::endl;
    std::cout << "[Simple] IoU-based target stickiness: ENABLED" << std::endl;
    std::cout << "[Simple] Pinned receive buffers: " << (udpCapture.IsPinnedMemoryEnabled() ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "[Simple] Latest-frame in-flight limit: " << maxPipelineInFlight << std::endl;
    std::cout << "[Simple] Frame credit depth: " << frameCreditDepth << std::endl;

    // 5. Pre-capture CUDA graphs for the shapes listed in config (if any).
    // The multi-shape graph cache holds up to kMaxGraphShapes buckets, so any
    // configured shape that matches an incoming frame avoids the first-frame
    // capture cost entirely.
    int preCaptureSuccess = 0;
    for (const auto& shape : cfg.preCaptureShapes) {
        const int w = shape.first;
        const int h = shape.second;
        std::cout << "[Simple] Pre-capturing CUDA graph for " << w << "x" << h
                  << "..." << std::endl;
        if (inference.captureFullGraphForShape(
                w, h, maxPipelineInFlight,
                cfg.confThreshold, cfg.headClassId,
                allowedClassMask, rightGpuAimConfig,
                cfg.pd.iou_stickiness_threshold, cfg.headAimPoint, cfg.bodyAimPoint)) {
            ++preCaptureSuccess;
        } else {
            std::cerr << "[Simple] Pre-capture FAILED for " << w << "x" << h << std::endl;
        }
    }
    std::cout << "[Simple] Full CUDA graph: "
              << (preCaptureSuccess > 0
                      ? "PRE-CAPTURED (" + std::to_string(preCaptureSuccess) + " shape" +
                            (preCaptureSuccess == 1 ? "" : "s") + ")"
                      : std::string("DEFERRED until first frame shape"))
              << std::endl;
    std::cout << "[Simple] Idle CUDA graph pre-capture: "
              << (cfg.idleGraphPrecaptureEnabled ? "ENABLED" : "DISABLED")
              << " (interval=" << idleGraphPrecaptureIntervalMs << "ms)" << std::endl;
    std::cout << "[Simple] Stage timing: "
              << (cfg.stageTimingEnabled ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "[Simple] Direct callback aim moves: "
              << (movementController.directAimMoveInCallback() ? "ENABLED" : "DISABLED")
              << std::endl;

    // Perf log: full per-window metrics go here (the screen stays user-facing).
    // Relative paths resolve next to the config file; empty disables logging.
    // The file is size-capped: at perfLogMaxBytes it is rotated to "<path>.1"
    // (one backup) and reopened, so total disk use is bounded to ~2x the cap.
    // Writes happen at most once per stat window (default 1s), not per frame,
    // and flush is batched (every ~5s) to keep file I/O off the hot cadence.
    std::ofstream perfLog;
    std::filesystem::path perfLogPath;
    size_t perfLogBytes = 0;
    const size_t perfLogMaxBytes = cfg.perfLogMaxBytes > 0 ? static_cast<size_t>(cfg.perfLogMaxBytes) : 0;
    auto perfLogHeader = [&]() {
        std::ostringstream h;
        h << "# session " << formatWallClock()
          << " engine=" << cfg.enginePath
          << " interval=" << perfStatsIntervalMs << "ms"
          << " | fields: R/S/D=recv/submit/done per s, B=busyDrop F=submitFail"
          << " C/U=udp recv/drop, Cr=credit I=inflight, Aw=acquire ms(avg/max)"
          << " Awp=acquire p50/95/99 ms, Su=submit us(avg/max), Cb=callback ms(avg/max)"
          << " Cbp=callback p50/95/99 ms,"
          << " E2E=capture->complete ms(avg/max) E2Ep=p50/95/99 ms E2En=samples/window"
          << " (needs NTP-synced clocks; excludes mouse), NF/IF/MD=timeouts/invalid/movedrop,"
          << " G=graph/std/fallback, St=[h2d pre inf post d2h] us(avg/max)\n";
        return h.str();
    };
    if (cfg.perfStatsEnabled && !cfg.perfLogPath.empty()) {
        std::filesystem::path logPath(cfg.perfLogPath);
        if (logPath.is_relative()) logPath = configPath.parent_path() / logPath;
        perfLogPath = logPath.lexically_normal();
        const auto openMode = std::ios::out |
            (cfg.perfLogTruncateOnStart ? std::ios::trunc : std::ios::app);
        perfLog.open(perfLogPath.string(), openMode);
        if (perfLog.is_open()) {
            std::error_code ec;
            if (!cfg.perfLogTruncateOnStart) {
                const auto existing = std::filesystem::file_size(perfLogPath, ec);
                if (!ec) perfLogBytes = static_cast<size_t>(existing);
            }
            const std::string hdr = perfLogHeader();
            perfLog << hdr;
            perfLogBytes += hdr.size();
            perfLog.flush();
            std::cout << "[Simple] Perf log -> " << perfLogPath.string()
                      << (perfLogMaxBytes ? " (rotate at " +
                            std::to_string(perfLogMaxBytes / (1024 * 1024)) + "MiB, 1 backup)"
                                          : std::string(" (no rotation)"))
                      << std::endl;
        } else {
            std::cerr << "[Simple] WARN: could not open perf log at " << perfLogPath.string()
                      << " (metrics will not be recorded)" << std::endl;
        }
    }
    auto lastPerfFlush = std::chrono::steady_clock::now();

    std::cout << "\n[Simple] Running... Press Ctrl+C to exit" << std::endl;
    std::cout << "[Simple] Right-click (or Side2) = AIM" << std::endl;
    std::cout << "[Simple] Left+Right = AIM + NO-RECOIL" << std::endl;

    // 6. Main loop with GPU callback API
    // - Frame submission is event-driven: UDPCapture's receive thread invokes
    //   trySubmitLatestFrame() (below) the instant a frame finishes assembling,
    //   so the newest frame goes straight from network to GPU submit with no
    //   extra hop through the main thread. The main loop itself only drains
    //   (retries a frame that arrived while the pipeline was full) and handles
    //   slow/idle housekeeping (CUDA graph capture, stats, debug dump).
    // - Inference is queued to GPU
    // - Completion callback queues mouse movement without cudaStreamSync wait.
    int recvFramesWindow = 0;
    int submittedFramesWindow = 0;
    int busyDropWindow = 0;
    int submitFailWindow = 0;
    uint64_t creditSentWindow = 0;
    uint32_t nextCreditMinFrameId = 0;
    PerfWindowStats perfWindow;
    uint64_t lastUdpReceived = udpCapture.GetReceivedFrameCount();
    uint64_t lastUdpDropped = udpCapture.GetDroppedFrameCount();
    size_t lastStatusLineLen = 0;
    bool graphCaptureFailedForShape = false;
    unsigned int failedGraphW = 0;
    unsigned int failedGraphH = 0;
    bool idleGraphPrecaptureDone = false;
    auto nextIdleGraphPrecaptureTime = std::chrono::steady_clock::now();

    // Serializes every touch of udpCapture's single-consumer AcquireFramePinned
    // (m_consumedSeq is not atomic) and every call into `inference` (its
    // CUDA-graph-cache state is not internally synchronized), now that both the
    // receive thread (via trySubmitLatestFrame) and the main thread (drain,
    // idle graph pre-capture, debug dump) can reach them. Always try_lock'd from
    // the receive thread's path so a slow main-thread section (graph capture
    // can take hundreds of ms) never stalls the socket; main-thread call sites
    // may block briefly since only they, not the receive thread, wait on it.
    std::mutex submitMutex;
    // Set by trySubmitLatestFrame (recv thread or drain) when a frame's shape
    // has no captured CUDA graph and the GPU is idle. Packed (width<<32|height).
    // The main loop performs the actual (slow) capture, never the recv thread.
    std::atomic<uint64_t> pendingCaptureShape{0};

    auto isRgbFrameInput = [](uint8_t framePixelFormat, uint8_t frameBytesPerPixel) {
        return framePixelFormat == UDP_PIXEL_FORMAT_RGB && frameBytesPerPixel == 3;
    };
    // A small credit depth lets the Game PC pre-send the newest next frame while
    // the GPU is busy, reducing inter-frame gaps without allowing a deep queue.
    // Callers must hold submitMutex (creditSentWindow is not atomic).
    auto requestFrameCredits = [&]() {
        if (udpCapture.SendFrameCredit(nextCreditMinFrameId, frameCreditDepth)) {
            creditSentWindow += frameCreditDepth;
        }
    };

    // Callers must hold submitMutex - this can call captureFullGraphForShape,
    // which takes hundreds of ms, and touches `inference`'s graph-cache state
    // with no internal locking of its own.
    auto ensureFullGraphReady = [&](unsigned int sourceW, unsigned int sourceH,
                                    const char* logPrefix) {
        const bool graphReady = inference.isFullGraphReadyForShape(
            static_cast<int>(sourceW), static_cast<int>(sourceH),
            maxPipelineInFlight,
            cfg.confThreshold, cfg.headClassId,
            allowedClassMask, rightGpuAimConfig,
            cfg.pd.iou_stickiness_threshold, cfg.headAimPoint, cfg.bodyAimPoint);
        const bool graphFailedForThisShape =
            graphCaptureFailedForShape && failedGraphW == sourceW && failedGraphH == sourceH;
        if (graphReady || graphFailedForThisShape) {
            idleGraphPrecaptureDone = true;
            return true;
        }
        if (inference.getCallbacksInFlight() != 0) {
            return true;
        }

        std::cout << "\n[Simple] " << logPrefix << "Capturing full CUDA graph for source "
                  << sourceW << "x" << sourceH << "..." << std::endl;
        if (inference.captureFullGraphForShape(
                static_cast<int>(sourceW), static_cast<int>(sourceH),
                maxPipelineInFlight,
                cfg.confThreshold, cfg.headClassId,
                allowedClassMask, rightGpuAimConfig,
                cfg.pd.iou_stickiness_threshold, cfg.headAimPoint, cfg.bodyAimPoint)) {
            graphCaptureFailedForShape = false;
            failedGraphW = 0;
            failedGraphH = 0;
            idleGraphPrecaptureDone = true;
            std::cout << "[Simple] Full CUDA graph: ENABLED for source "
                      << sourceW << "x" << sourceH << std::endl;
        } else {
            graphCaptureFailedForShape = true;
            failedGraphW = sourceW;
            failedGraphH = sourceH;
            idleGraphPrecaptureDone = true;
            std::cout << "[Simple] Full CUDA graph: DISABLED for source "
                      << sourceW << "x" << sourceH
                      << " (using standard execution)" << std::endl;
        }
        return false;
    };

    // Attempts to submit exactly one frame (the newest available) to inference.
    // Called from two places:
    //  (a) UDPCapture's receive thread, via the frame-ready callback, the
    //      instant a new frame is published - the hot path (network -> GPU
    //      submit with no extra hop).
    //  (b) the main loop's periodic 1ms drain, to pick up a frame that arrived
    //      while the pipeline was full or while this function lost the
    //      submitMutex race (see below).
    // Never blocks: submitMutex is only try-locked and AcquireFramePinned is
    // called with timeoutMs=0. This is required for (a): the receive thread
    // must keep draining the socket and must never wait on GPU work or on a
    // lock a slow main-thread operation might be holding (CUDA graph capture
    // can take hundreds of ms). A frame this function can't claim right now is
    // never lost - UDPCapture's buffers are newest-wins, so the next call (from
    // either caller) simply picks up whatever is newest at that time.
    auto trySubmitLatestFrame = [&]() {
        // Refuse new GPU submissions once shutdown has begun. This is called
        // from the receive thread (via the frame-ready callback), which keeps
        // running until udpCapture.StopCapture() joins it - strictly after the
        // main loop's own `while (g_running)` has already exited. Without this
        // check, a frame arriving in that window could launch new GPU work
        // referencing stack-local state (callbackTickets, callbackCtx) that is
        // about to be destroyed, after the shutdown path's one-time
        // cudaStreamSynchronize has already run (see the code near main's
        // `return`, which stops capture - and thus this callback - before that
        // sync for exactly this reason).
        if (!g_running.load(std::memory_order_relaxed)) return;

        std::unique_lock<std::mutex> lock(submitMutex, std::try_to_lock);
        if (!lock.owns_lock()) return;  // Contended; the next frame/tick retries.

        if (busyCallbackTicketCount() >= maxPipelineInFlight) return;  // Full; drained on the next slot-free notify.

        // Only pull a frame into the inference pipeline while actively aiming -
        // otherwise leave it in UDPCapture's ring (saves GPU/CPU while idle).
        const uint8_t idleButtonMask = makcu.buttonMask();
        if (!(cfg.forceAimOn || controller::maskAiming(idleButtonMask))) return;

        requestFrameCredits();

        void* pinnedRgbData = nullptr;
        unsigned int width = 0, height = 0;
        uint64_t acquiredFrameId = 0;
        int bufferIndex = -1;
        uint8_t bytesPerPixel = 3;
        uint8_t pixelFormat = UDP_PIXEL_FORMAT_RGB;
        uint64_t acquiredCaptureUnixMicros = 0;

        Clock::time_point acquireStart{};
        if (cfg.perfStatsEnabled) {
            acquireStart = Clock::now();
        }
        // Non-blocking - see the function comment above.
        const bool gotFrame = udpCapture.AcquireFramePinned(
            &pinnedRgbData, &width, &height, &acquiredFrameId, &bufferIndex, /*timeoutMs=*/0,
            &bytesPerPixel, &pixelFormat, &acquiredCaptureUnixMicros);
        if (cfg.perfStatsEnabled) {
            perfWindow.recordAcquire(elapsedUs(acquireStart, Clock::now()), gotFrame);
        }
        if (!gotFrame) return;

        if (!pinnedRgbData || width == 0 || height == 0) {
            if (cfg.perfStatsEnabled) {
                ++perfWindow.invalidFrames;
            }
            if (bufferIndex >= 0) udpCapture.ReleaseFrame(bufferIndex);
            return;
        }
        nextCreditMinFrameId = static_cast<uint32_t>(acquiredFrameId + 1);

        if (!isRgbFrameInput(pixelFormat, bytesPerPixel)) {
            if (cfg.perfStatsEnabled) {
                ++perfWindow.invalidFrames;
            }
            udpCapture.ReleaseFrame(bufferIndex);
            return;
        }

        const auto nowLocal = Clock::now();
        if (debugFrameDumper.due(nowLocal)) {
            debugFrameDumper.scheduleNext(nowLocal);
            debugFrameDumper.save(pinnedRgbData, width, height, acquiredFrameId);
        }

        // Re-check button state fresh (it may have changed since the top-of-
        // function check above) and select the matching aim config.
        const uint8_t frameButtonMask = makcu.buttonMask();
        const bool aiming = cfg.forceAimOn || controller::maskAiming(frameButtonMask);
        if (!aiming) {
            // Skip inference when not aiming (save power)
            udpCapture.ReleaseFrame(bufferIndex);
            return;
        }
        // Copy (not reference) so the per-frame static shoot-offset can be
        // folded in: while shooting, shift the aim reference point by the
        // configured offset; otherwise leave it at screen center. The GPU
        // controller treats this as a setpoint shift (see AimConfig), so the
        // aim settles at the offset instead of drifting up like an additive
        // per-frame nudge would.
        gpa::AimConfig frameAimConfig = selectGpuAimConfig(frameButtonMask);
        if (controller::maskShooting(frameButtonMask)) {
            frameAimConfig.shoot_offset_x = cfg.shootOffsetX;
            frameAimConfig.shoot_offset_y = cfg.shootOffsetY;
        }

        const bool graphReady = inference.isFullGraphReadyForShape(
            static_cast<int>(width), static_cast<int>(height),
            maxPipelineInFlight,
            cfg.confThreshold, cfg.headClassId,
            allowedClassMask, rightGpuAimConfig,
            cfg.pd.iou_stickiness_threshold, cfg.headAimPoint, cfg.bodyAimPoint);
        const bool graphFailedForThisShape =
            graphCaptureFailedForShape && failedGraphW == width && failedGraphH == height;
        if (graphReady || graphFailedForThisShape) {
            idleGraphPrecaptureDone = true;
        }
        if (!graphReady && !graphFailedForThisShape && inference.getCallbacksInFlight() == 0) {
            // No graph for this shape yet and the GPU is idle. Don't capture it
            // here - that can take hundreds of ms (see this function's comment).
            // Ask the main loop to do it and skip this frame; the next
            // completed frame will retry once the graph exists.
            udpCapture.ReleaseFrame(bufferIndex);
            pendingCaptureShape.store((static_cast<uint64_t>(width) << 32) | height,
                                      std::memory_order_relaxed);
            return;
        }

        recvFramesWindow++;

        CallbackTicket* ticket = acquireCallbackTicket();
        if (!ticket) {
            udpCapture.ReleaseFrame(bufferIndex);
            ++busyDropWindow;
            return;
        }
        ticket->bufferIndex = bufferIndex;

        // GPU CALLBACK API: Queue inference, callback fires when GPU completes.
        // No cudaStreamSynchronize - mouse movement happens in callback thread.
        Clock::time_point submitStart{};
        if (cfg.perfStatsEnabled) {
            submitStart = Clock::now();
        }
        ticket->submitTime = submitStart;
        ticket->captureUnixMicros = acquiredCaptureUnixMicros;
        bool submitted = inference.runInferenceWithCallback(
            pinnedRgbData, width, height,
            cfg.confThreshold, cfg.headClassId,
            allowedClassMask,
            frameAimConfig,
            cfg.pd.iou_stickiness_threshold,
            cfg.headAimPoint, cfg.bodyAimPoint,
            inferenceCallback, ticket);
        if (cfg.perfStatsEnabled) {
            perfWindow.recordSubmit(elapsedUs(submitStart, Clock::now()));
        }

        if (submitted) {
            submittedFramesWindow++;
        } else {
            int failedBuffer = ticket->bufferIndex;
            if (failedBuffer >= 0) {
                udpCapture.ReleaseFrame(failedBuffer);
            }
            ticket->bufferIndex = -1;
            ticket->submitTime = Clock::time_point{};
            ticket->captureUnixMicros = 0;
            ticket->busy.store(false, std::memory_order_release);
            if (inference.getCallbacksInFlight() >= maxPipelineInFlight) {
                busyDropWindow++;
            } else {
                submitFailWindow++;
            }
        }
        // On success, buffer is released by callback after GPU work completes.
    };

    // Must be registered before StartCapture() spins up the receive thread.
    udpCapture.SetFrameReadyCallback(trySubmitLatestFrame);
    if (!udpCapture.StartCapture()) {
        std::cerr << "[Simple] Failed to start UDP capture" << std::endl;
        return 1;
    }
    std::cout << "[Simple] UDP capture started on port " << cfg.udpPort << std::endl;

    while (g_running) {
        // Tick: wait briefly for a pipeline slot to free (the completion
        // callback notifies pipelineCv when a ticket is released) or simply
        // time out. Either way, fall through to drain below.
        {
            std::unique_lock<std::mutex> lock(pipelineCvMutex);
            pipelineCv.wait_for(lock, std::chrono::milliseconds(1));
        }

        // Keep the game<->inference clock offset fresh for E2E latency. Self-
        // throttled to ~10/s, so calling it every ~1ms tick is cheap.
        if (cfg.perfStatsEnabled) {
            udpCapture.SendClockSyncPing();
        }

        // Drain path: the hot path is the receive thread's frame-ready callback
        // (see trySubmitLatestFrame's comment); this call only matters when
        // that callback found the pipeline full, lost the submitMutex race, or
        // no frames are arriving other than this 1ms heartbeat.
        trySubmitLatestFrame();

        // A frame in trySubmitLatestFrame found no CUDA graph captured for its
        // shape with the GPU idle - do the (possibly slow) capture here rather
        // than on the receive thread.
        if (const uint64_t packedShape = pendingCaptureShape.exchange(0, std::memory_order_relaxed)) {
            const unsigned int shapeW = static_cast<unsigned int>(packedShape >> 32);
            const unsigned int shapeH = static_cast<unsigned int>(packedShape & 0xffffffffu);
            std::lock_guard<std::mutex> lock(submitMutex);
            (void)ensureFullGraphReady(shapeW, shapeH, "");
        }

        const auto now = std::chrono::steady_clock::now();

        // No-recoil compensation: purely time-based (fires every recoilTickMs
        // while left+right click is held), driven by this loop's own 1ms
        // cadence rather than by frame arrival, since frame submission is now
        // event-driven and may not touch the main loop at all while aiming.
        movementController.tickNoRecoil(makcu.buttonMask());

        // Idle-only housekeeping: graph pre-capture and debug frame dump each
        // pull their own frame directly (the hot path only runs while aiming),
        // so both must serialize against trySubmitLatestFrame via submitMutex.
        const uint8_t idleButtonMask = makcu.buttonMask();
        const bool aimingActiveMain = cfg.forceAimOn || controller::maskAiming(idleButtonMask);
        if (!aimingActiveMain) {
            // trySubmitLatestFrame stops pulling frames the instant aiming
            // drops (see its own "Only pull a frame while actively aiming"
            // check), so the inference callback can go silent for as long as
            // aiming stays off - it never gets a chance to observe the
            // release and cancel an in-progress flick itself. This loop is
            // the only thing still polling button state at that point, so it
            // has to be the one to cancel a stale flick before it can resume
            // as a huge catch-up jump when aiming comes back.
            callbackCtx.flickPlayback.cancel();

            std::lock_guard<std::mutex> lock(submitMutex);
            if (cfg.idleGraphPrecaptureEnabled && !idleGraphPrecaptureDone &&
                now >= nextIdleGraphPrecaptureTime && busyCallbackTicketCount() == 0) {
                nextIdleGraphPrecaptureTime =
                    now + std::chrono::milliseconds(idleGraphPrecaptureIntervalMs);
                requestFrameCredits();
                void* pinnedRgbData = nullptr;
                unsigned int width = 0, height = 0;
                uint64_t idleFrameId = 0;
                int bufferIndex = -1;
                uint8_t bytesPerPixel = 3;
                uint8_t pixelFormat = UDP_PIXEL_FORMAT_RGB;
                const bool gotIdleFrame = udpCapture.AcquireFramePinned(
                    &pinnedRgbData, &width, &height, &idleFrameId, &bufferIndex, 1,
                    &bytesPerPixel, &pixelFormat);
                if (gotIdleFrame) {
                    nextCreditMinFrameId = static_cast<uint32_t>(idleFrameId + 1);
                    const bool validIdleFrame =
                        pinnedRgbData && width != 0 && height != 0 &&
                        isRgbFrameInput(pixelFormat, bytesPerPixel);
                    if (bufferIndex >= 0) {
                        udpCapture.ReleaseFrame(bufferIndex);
                        bufferIndex = -1;
                    }
                    if (validIdleFrame) {
                        (void)ensureFullGraphReady(width, height, "Idle pre-capture: ");
                    } else if (cfg.perfStatsEnabled) {
                        ++perfWindow.invalidFrames;
                    }
                }
            } else if (debugFrameDumper.due(now)) {
                requestFrameCredits();
                void* pinnedRgbData = nullptr;
                unsigned int width = 0, height = 0;
                uint64_t debugFrameId = 0;
                int bufferIndex = -1;
                uint8_t bytesPerPixel = 3;
                uint8_t pixelFormat = UDP_PIXEL_FORMAT_RGB;
                const bool gotDebugFrame = udpCapture.AcquireFramePinned(
                    &pinnedRgbData, &width, &height, &debugFrameId, &bufferIndex, /*timeoutMs=*/16,
                    &bytesPerPixel, &pixelFormat);
                if (gotDebugFrame) {
                    nextCreditMinFrameId = static_cast<uint32_t>(debugFrameId + 1);
                    const bool validDebugFrame =
                        pinnedRgbData && width != 0 && height != 0 &&
                        isRgbFrameInput(pixelFormat, bytesPerPixel);
                    if (validDebugFrame) {
                        debugFrameDumper.save(pinnedRgbData, width, height, debugFrameId);
                        debugFrameDumper.scheduleNext(now);
                    } else {
                        debugFrameDumper.scheduleRetry(now);
                    }
                    if (bufferIndex >= 0) {
                        udpCapture.ReleaseFrame(bufferIndex);
                        bufferIndex = -1;
                    }
                } else {
                    debugFrameDumper.scheduleRetry(now);
                }
            }
        }

        // Stats every interval (using atomic g_frameCount from callbacks, plus
        // the window counters/perfWindow above - those are only ever touched
        // while holding submitMutex, so read-and-reset them under the same
        // lock for a mutually consistent snapshot).
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatTime).count();
        if (elapsed >= perfStatsIntervalMs) {
            int recvSnapshot, submittedSnapshot, busyDropSnapshot, submitFailSnapshot;
            uint64_t creditSnapshot;
            PerfWindowStats perfSnapshot;
            {
                std::lock_guard<std::mutex> lock(submitMutex);
                recvSnapshot = recvFramesWindow;
                submittedSnapshot = submittedFramesWindow;
                busyDropSnapshot = busyDropWindow;
                submitFailSnapshot = submitFailWindow;
                creditSnapshot = creditSentWindow;
                perfSnapshot = perfWindow;
                recvFramesWindow = 0;
                submittedFramesWindow = 0;
                busyDropWindow = 0;
                submitFailWindow = 0;
                creditSentWindow = 0;
                perfWindow.reset();
            }

            int completedFrames = g_frameCount.exchange(0, std::memory_order_relaxed);  // Atomic read and reset
            const auto launchStats = inference.takeLaunchStats();
            const uint64_t callbackLatencySamples =
                g_callbackLatencySamples.exchange(0, std::memory_order_relaxed);
            const int64_t callbackLatencyTotalUs =
                g_callbackLatencyTotalUs.exchange(0, std::memory_order_relaxed);
            const int64_t callbackLatencyMaxUs =
                g_callbackLatencyMaxUs.exchange(0, std::memory_order_relaxed);
            const uint64_t moveQueueDropped = movementController.takeDroppedCount();
            const double callbackAvgMs = callbackLatencySamples == 0
                ? 0.0
                : static_cast<double>(callbackLatencyTotalUs) / callbackLatencySamples / 1000.0;
            const double callbackMaxMs = static_cast<double>(callbackLatencyMaxUs) / 1000.0;
            // Only drain the 512-bucket histograms when stats are on; with stats
            // off nothing is recorded, so skip the per-window work entirely.
            const LatencyHistogram callbackHist =
                cfg.perfStatsEnabled ? g_callbackLatencyHist.drain() : LatencyHistogram{};
            const uint64_t e2eSamples =
                g_e2eLatencySamples.exchange(0, std::memory_order_relaxed);
            const int64_t e2eTotalUs = g_e2eLatencyTotalUs.exchange(0, std::memory_order_relaxed);
            const int64_t e2eMaxUs = g_e2eLatencyMaxUs.exchange(0, std::memory_order_relaxed);
            const double e2eAvgMs = e2eSamples == 0
                ? 0.0
                : static_cast<double>(e2eTotalUs) / e2eSamples / 1000.0;
            const double e2eMaxMs = static_cast<double>(e2eMaxUs) / 1000.0;
            const LatencyHistogram e2eHist =
                cfg.perfStatsEnabled ? g_e2eLatencyHist.drain() : LatencyHistogram{};
            const auto usToMs = [](int64_t us) { return static_cast<double>(us) / 1000.0; };
            uint64_t udpReceivedNow = udpCapture.GetReceivedFrameCount();
            uint64_t udpReceivedDelta = udpReceivedNow - lastUdpReceived;
            lastUdpReceived = udpReceivedNow;
            uint64_t udpDroppedNow = udpCapture.GetDroppedFrameCount();
            uint64_t udpDroppedDelta = udpDroppedNow - lastUdpDropped;
            lastUdpDropped = udpDroppedNow;
            const uint8_t statusButtonMask = makcu.buttonMask();
            const bool aimActive = cfg.forceAimOn || controller::maskAiming(statusButtonMask);
            const bool shootActive = controller::maskShooting(statusButtonMask);

            // Screen: user-facing health only. FPS = completed inferences/s;
            // drop = UDP frames lost this window. Everything diagnostic goes to
            // the perf log below.
            std::ostringstream status;
            status << std::fixed << std::setprecision(1)
                   << "[Simple] FPS:" << (completedFrames * 1000.0f / elapsed)
                   << " aim:" << (aimActive ? "ON" : "OFF")
                   << " shoot:" << (shootActive ? "ON" : "OFF")
                   << " drop:" << udpDroppedDelta;

            // Perf log: the full per-window metric set (avg/max + percentiles +
            // optional stage breakdown), one timestamped line per window.
            if (cfg.perfStatsEnabled && perfLog.is_open()) {
                std::ostringstream logline;
                logline << std::fixed << std::setprecision(1)
                        << formatWallClock()
                        << " R:" << (recvSnapshot * 1000.0f / elapsed)
                        << " S:" << (submittedSnapshot * 1000.0f / elapsed)
                        << " D:" << (completedFrames * 1000.0f / elapsed)
                        << " B:" << busyDropSnapshot
                        << " F:" << submitFailSnapshot
                        << " C:" << udpReceivedDelta
                        << " U:" << udpDroppedDelta
                        << " Cr:" << creditSnapshot
                        << " I:" << busyCallbackTicketCount()
                        << " A:" << (aimActive ? "ON" : "OFF")
                        << " Sh:" << (shootActive ? "ON" : "OFF")
                        << " Aw:" << perfSnapshot.averageAcquireMs() << "/" << perfSnapshot.maxAcquireMs() << "ms"
                        << " Awp[" << usToMs(perfSnapshot.acquireHist.percentileUs(0.50)) << "/"
                        << usToMs(perfSnapshot.acquireHist.percentileUs(0.95)) << "/"
                        << usToMs(perfSnapshot.acquireHist.percentileUs(0.99)) << "]ms"
                        << " Su:" << perfSnapshot.averageSubmitUs() << "/" << perfSnapshot.submitMaxUs << "us"
                        << " Cb:" << callbackAvgMs << "/" << callbackMaxMs << "ms"
                        << " Cbp[" << usToMs(callbackHist.percentileUs(0.50)) << "/"
                        << usToMs(callbackHist.percentileUs(0.95)) << "/"
                        << usToMs(callbackHist.percentileUs(0.99)) << "]ms"
                        << " E2E:" << e2eAvgMs << "/" << e2eMaxMs << "ms"
                        << " E2Ep[" << usToMs(e2eHist.percentileUs(0.50)) << "/"
                        << usToMs(e2eHist.percentileUs(0.95)) << "/"
                        << usToMs(e2eHist.percentileUs(0.99)) << "]ms"
                        << " E2En:" << e2eSamples
                        << " NF:" << perfSnapshot.acquireTimeouts
                        << " IF:" << perfSnapshot.invalidFrames
                        << " MD:" << moveQueueDropped
                        << " G:" << launchStats.graph << "/" << launchStats.standard
                        << "/" << launchStats.graphFallback;
                if (cfg.stageTimingEnabled) {
                    const auto st = inference.takeStageTimingStats();
                    auto avgUs = [&](uint64_t total) {
                        return st.samples == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(st.samples);
                    };
                    logline << " St[h2d:" << avgUs(st.h2dUsTotal) << "/" << st.h2dUsMax
                            << " pre:" << avgUs(st.preprocessUsTotal) << "/" << st.preprocessUsMax
                            << " inf:" << avgUs(st.inferenceUsTotal) << "/" << st.inferenceUsMax
                            << " post:" << avgUs(st.postprocessUsTotal) << "/" << st.postprocessUsMax
                            << " d2h:" << avgUs(st.d2hUsTotal) << "/" << st.d2hUsMax << "us]";
                }
                const std::string line = logline.str();
                perfLog << line << '\n';
                perfLogBytes += line.size() + 1;

                // Size-based rotation: at the cap, move the file to "<path>.1"
                // (replacing any prior backup) and reopen fresh. Reopen happens
                // at most once per cap worth of data (~hours at 1 line/s), so it
                // never lands on the frame path.
                if (perfLogMaxBytes && perfLogBytes >= perfLogMaxBytes) {
                    perfLog.close();
                    std::error_code ec;
                    std::filesystem::rename(perfLogPath,
                        std::filesystem::path(perfLogPath.string() + ".1"), ec);
                    perfLog.open(perfLogPath.string(), std::ios::out | std::ios::trunc);
                    perfLogBytes = 0;
                    if (perfLog.is_open()) {
                        const std::string hdr = perfLogHeader();
                        perfLog << hdr;
                        perfLogBytes += hdr.size();
                    }
                    perfLog.flush();
                    lastPerfFlush = now;
                } else if (now - lastPerfFlush >= std::chrono::seconds(5)) {
                    // Batch flushes: 1 line/s doesn't need a syscall per write.
                    perfLog.flush();
                    lastPerfFlush = now;
                }
            }

            const std::string statusLine = status.str();
            std::cout << '\r' << statusLine;
            if (lastStatusLineLen > statusLine.size()) {
                std::cout << std::string(lastStatusLineLen - statusLine.size(), ' ');
            }
            std::cout << std::flush;
            lastStatusLineLen = statusLine.size();

            lastStatTime = now;
        }
    }

    // Stop the receive thread FIRST: trySubmitLatestFrame is reachable from its
    // frame-ready callback and does not stop just because the main loop above
    // exited, so it could otherwise still launch new GPU work referencing
    // stack-local state (callbackTickets, callbackCtx) after this function
    // returns. Once StopCapture() has joined that thread, no more submissions
    // can happen, and the single cudaStreamSynchronize below is guaranteed to
    // drain everything that was ever launched.
    std::cout << "\n[Simple] Shutting down..." << std::endl;
    udpCapture.StopCapture();

    // Wait for any pending GPU work before the stack-local callback state
    // (callbackTickets, callbackCtx, etc.) is destroyed.
    cudaStreamSynchronize(inference.getStream());
    movementController.stop();

    return 0;
}
