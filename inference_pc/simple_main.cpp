// Simple aimbot - clean implementation
// UDP capture + TensorRT inference + Mouse control via Makcu
// Features: Full GPU pipeline (inference + postprocess + nonlinear P), No-recoil
// GPU Callback API for lowest latency (no cudaStreamSync wait)
// Minimal CPU usage - frame receive on CPU, inference/postprocess/movement on GPU

#include <iostream>
#include <fstream>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <functional>
#include <condition_variable>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include "needaimbot/cuda/simple_inference.h"
#include "needaimbot/cuda/simple_postprocess.h"
#include "needaimbot/capture/udp_capture.h"
#include "needaimbot/mouse/MakcuConnection.h"
#include "needaimbot/mouse/controller.h"
#include "needaimbot/mouse/pd_controller.hpp"
#include "needaimbot/app/engine_locator.hpp"
#include "needaimbot/app/runtime_options.hpp"
#include "needaimbot/app/config.hpp"
#include "needaimbot/app/pipeline_metrics.hpp"
#include "needaimbot/app/run_modes.hpp"
#include "needaimbot/app/runtime_diagnostics.hpp"
#include "needaimbot/capture/debug_frame_dump.hpp"

// Third-party JSON parser (header-only)
#include "third_party/nlohmann/json.hpp"
using json = nlohmann::json;

using Clock = std::chrono::steady_clock;

// Forward declaration
struct CallbackContext;


// GPU Callback Handler
// This callback runs from the completion worker when GPU inference finishes.
// It only decides *whether* to move (aiming active + target present) and then
// hands the raw (dx, dy) off to the controller, which owns everything about
// *how* the move actually reaches the mouse (see needaimbot/mouse/controller.h).
//
// OPTIMIZATION: Cached config values eliminate pointer indirection in hot path.
// All frequently accessed values are copied to the context struct at init time.

// --- Calibration logging (opt-in, off by default) -------------------------
// Records the raw per-frame detection so bench/calibrate.py can measure this
// rig's real detector noise, head-selection rate, dropout and frame timing, and
// feed them back into the sim. Buffered in memory (no hot-path file I/O),
// written once on shutdown. Aim OFF at a stationary target -> the center
// variance is pure detector noise.
struct CalibRec {
    int64_t t_us;      // since logger start
    int64_t lat_us;    // submit->completion (inference pipeline), -1 if unknown
    uint8_t aiming;
    uint8_t hasTarget;
    int   classId;
    float conf;
    float cx, cy, w, h;   // selected target box: centre + size (model-input px)
    int   emitDx, emitDy; // this frame's proposed mouse move (output counts)
    float mScaleX, mScaleY; // model-px error -> output-count scale (to undo ego)
    long long injX;       // cumulative step-response injection at this frame
};
class CalibLogger {
public:
    explicit CalibLogger(std::string path) : path_(std::move(path)), start_(Clock::now()) {
        recs_.reserve(200000);   // ~11 min @300fps; bounded, no realloc on the hot path
    }
    ~CalibLogger() { dump(); }
    Clock::time_point start() const { return start_; }
    void record(const CalibRec& r) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (recs_.size() < recs_.capacity()) {
            recs_.push_back(r);
        } else if (!full_) {
            // Dropping rows silently would look like a short session rather than a
            // truncated one, and the analysis would happily average the first half.
            full_ = true;
            std::cerr << "*** [Calib] buffer full at " << recs_.size()
                      << " rows - the rest of this session is NOT recorded ***" << std::endl;
        }
    }
    void dump() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (dumped_ || path_.empty()) return;
        dumped_ = true;
        std::ofstream f(path_);
        if (!f) { std::cerr << "[Calib] cannot open " << path_ << std::endl; return; }
        f << "t_us,lat_us,aiming,hasTarget,classId,conf,cx,cy,w,h,emit_dx,emit_dy,mscale_x,mscale_y,inject_cum_x\n";
        for (const auto& r : recs_) {
            f << r.t_us << ',' << r.lat_us << ',' << int(r.aiming) << ',' << int(r.hasTarget)
              << ',' << r.classId << ',' << r.conf << ',' << r.cx << ',' << r.cy << ',' << r.w
              << ',' << r.h << ',' << r.emitDx << ',' << r.emitDy
              << ',' << r.mScaleX << ',' << r.mScaleY << ',' << r.injX << '\n';
        }
        std::cout << "[Calib] wrote " << recs_.size() << " rows to " << path_ << std::endl;
    }
private:
    std::string path_;
    std::vector<CalibRec> recs_;
    std::mutex mtx_;
    Clock::time_point start_;
    bool dumped_ = false;
    bool full_ = false;
};

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

    // Calibration logger (nullptr = off). Not owned; lives in main().
    CalibLogger* calib = nullptr;

    void initFromConfig(const Config& cfg) {
        forceAimOn = cfg.forceAimOn;
        perfStatsEnabled = cfg.perfStatsEnabled;
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

    // Releasing the frame buffer and the ticket is the callback's only
    // non-optional job: the pipeline runs one frame in flight, so a path that
    // returns without doing it stalls submission permanently. A scope guard makes
    // that structural instead of relying on every branch remembering to call it -
    // including the branches that throw.
    struct TicketReleaser {
        CallbackTicket* ticket;
        CallbackContext* ctx;
        ~TicketReleaser() {
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
            // The next-frame kick is NOT done here: at this point the worker has
            // not yet released this frame's inference slot / in-flight counter
            // (that happens after this callback returns), so a resubmit from
            // inside the callback would be rejected at max in-flight and drop the
            // newest frame. The kick is instead driven by SimpleInference's
            // post-completion hook, which runs after that teardown. The
            // pipelineCv notify above remains the fallback that wakes the main
            // loop.
        }
    } ticketReleaser{ticket, ctx};

    // Count every completed inference callback (target/no-target)
    g_frameCount.fetch_add(1, std::memory_order_relaxed);

    // Whether this frame's move is emitted is decided ONCE, at submit time, and
    // carried through the GPU controller - which recorded it in the in-flight
    // ring on exactly that basis. Re-deciding here from the live aim key would
    // let the two disagree whenever the key moved while the frame was in flight:
    // the ring would hold a move that was never sent, and the dead-time
    // compensation would then subtract motion that never happened.
    const bool emitMove = (result.outputEnabled != 0);

    // Calibration: log the raw detection BEFORE the aim/target early-returns, so
    // a stationary-target capture with aim OFF still records every frame.
    if (ctx->calib) {
        const uint8_t callbackButtonMask = ctx->makcu->buttonMask();
        const bool aimingActive =
            ctx->forceAimOn || controller::maskAiming(callbackButtonMask);
        const int64_t lat = (ticket->submitTime.time_since_epoch().count() != 0)
            ? elapsedUs(ticket->submitTime, Clock::now()) : -1;
        ctx->calib->record({elapsedUs(ctx->calib->start(), Clock::now()), lat,
            static_cast<uint8_t>(aimingActive), static_cast<uint8_t>(result.hasTarget),
            result.targetClassId, result.targetConf,
            0.5f * (result.targetX1 + result.targetX2), 0.5f * (result.targetY1 + result.targetY2),
            result.targetX2 - result.targetX1, result.targetY2 - result.targetY1,
            result.movement.dx, result.movement.dy,
            result.movementScaleX, result.movementScaleY,
            g_injectCumX.load(std::memory_order_relaxed)});
    }

    if (!emitMove || !result.hasTarget) {
        return;  // ticketReleaser hands the frame buffer back
    }

    // Inference is done - hand the result off to the controller, which
    // decides how to turn it into physical mouse motion. (The static shoot-
    // offset aim-shift lives in the GPU controller's error term, not here.)
    ctx->controller->submitAimMovement(result.movement.dx, result.movement.dy);
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
        return runButtonTest(cfg);
    }

    if (runtimeOptions.collectCalib) {
        return runCollectCalib(cfg, runtimeOptions);
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
    inference.setRealtimeEnabled(cfg.realtimeThreadsEnabled);
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
    std::cout << "[Simple] Makcu connected at " << cfg.makcuBaudrate << " baud"
              << " (move encoding: ASCII km.move)"
              << std::endl;

    // 3. Initialize UDP capture
    UDPCapture udpCapture;
    if (cfg.cpuAffinityEnabled) {
        udpCapture.SetReceiveAffinity(cfg.affinityCoreReceive);
    }
    udpCapture.SetBusySpin(cfg.udpBusySpin);
    udpCapture.SetRealtimeEnabled(cfg.realtimeThreadsEnabled);
    if (cfg.udpBusySpin) {
        std::cout << "[Simple] UDP busy-spin receive: ON (recv core pinned at 100%)"
                  << std::endl;
    }
    if (!udpCapture.Initialize(cfg.udpPort)) {
        std::cerr << "[Simple] Failed to initialize UDP capture" << std::endl;
        return 1;
    }
    // StartCapture() (spins up the receive thread) is deferred until the frame-
    // ready callback below is registered - the callback must be set before the
    // receive thread can observe a completed frame.

    // 4. State - all GPU now, minimal CPU state
    gpa::AimConfig rightGpuAimConfig = cfg.pd.rightGpuConfig();
    gpa::AimConfig thumbGpuAimConfig = cfg.pd.thumbGpuConfig();
    // Normalize the nonlinear-P softness to a fixed input-resolution reference.
    // p_softness is a MODEL-PIXEL knee, so a 640 engine (2x the pixel scale of
    // the 320 the gains are tuned at) would otherwise respond ~2x sharper to the
    // same detection and amplify felt noise. Scaling softness by inputW/ref keeps
    // the aim feel identical across engine resolutions with the config unchanged.
    // kp/kd are dimensionless (gain multipliers) and stay as configured.
    {
        constexpr float kSoftnessReferenceInput = 320.0f;
        const float softScaleX = static_cast<float>(inference.inputWidth()) / kSoftnessReferenceInput;
        const float softScaleY = static_cast<float>(inference.inputHeight()) / kSoftnessReferenceInput;
        for (gpa::AimConfig* c : {&rightGpuAimConfig, &thumbGpuAimConfig}) {
            c->p_softness_x *= softScaleX;
            c->p_softness_y *= softScaleY;
        }
        if (softScaleX != 1.0f || softScaleY != 1.0f) {
            std::cout << "[Aim] softness normalized to 320-ref for "
                      << inference.inputWidth() << "x" << inference.inputHeight()
                      << " engine: x*" << softScaleX << " y*" << softScaleY << std::endl;
        }
    }
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

    // Calibration logger (opt-in). Owns the record buffer; dumps CSV on exit.
    // Its own switch (calibration_log_enabled) - a play session wants the detection
    // trace without the pipeline timing, and measurement wants both.
    // Writes to calibration_log_path, or "calib.csv" if unset. A relative path
    // resolves against the working directory, which needaimbot.sh sets to inference_pc/
    // - NOT the binary's directory. Prefer absolute (tools/measure.sh writes one) so
    // the file does not land somewhere the analysis step is not looking.
    std::unique_ptr<CalibLogger> calibLogger;
    if (cfg.calibrationLogEnabled) {
        const std::string calibPath =
            cfg.calibrationLogPath.empty() ? std::string("calib.csv") : cfg.calibrationLogPath;
        calibLogger = std::make_unique<CalibLogger>(calibPath);
        callbackCtx.calib = calibLogger.get();
        std::cout << "[Calib] logging detections to " << calibPath << std::endl;
    }
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
          << " G=graph/std/fallback, Stn=timed samples StDrop=untimeable (excluded),"
          << " St=[h2d pre inf post d2h] us(avg/max)\n";
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
    // Keep-warm state for inference gating. Written/read only inside
    // trySubmitLatestFrame (and its two gate sites), which is serialized by
    // submitMutex, so no atomics needed. shouldInfer() returns whether to run
    // inference for this frame given the live aim state and the keep-warm tail;
    // it refreshes lastAimingActive whenever aiming is truly active.
    // Per-frame dead-time adaptation state (see the deadtimeAdaptive block in
    // trySubmitLatestFrame). Same serialization as lastAimingActive below:
    // written/read only inside trySubmitLatestFrame, which submitMutex serializes.
    int64_t prevCaptureUnixMicros = 0;
    double framePeriodUsEma = 0.0;
    // How many plausible gaps the EMA has seen. The rescale itself uses the estimate
    // from the first frame - a warming EMA still beats assuming aim_tuned_fps - but
    // the log waits, or it announces the warm-up transient as if it were a rate.
    long framePeriodSamples = 0;
    double ageFramesEma = 0.0;

    Clock::time_point lastAimingActive{};
    auto shouldInfer = [&](bool aimingNow) -> bool {
        if (aimingNow) { lastAimingActive = Clock::now(); return true; }
        if (cfg.inferenceKeepwarmMs < 0) return true;   // always-on
        if (cfg.inferenceKeepwarmMs == 0) return false; // strict aim-only gate
        if (lastAimingActive.time_since_epoch().count() == 0) return false;
        return elapsedUs(lastAimingActive, Clock::now())
               < static_cast<int64_t>(cfg.inferenceKeepwarmMs) * 1000;
    };

    // allowSlowWork = false on the two hot-path callers (the receive thread's
    // frame-ready callback and the GPU completion hook). Anything that reads the
    // frame back on the CPU or touches the filesystem is confined to the main
    // loop's drain call: the debug dump used to run inline on the RECEIVE thread,
    // where a full-image read out of write-combined memory plus a BMP write stops
    // the socket being drained for as long as it takes.
    auto trySubmitLatestFrame = [&](bool allowSlowWork) {
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

        // Only pull a frame into the inference pipeline while actively aiming
        // (or within the keep-warm tail) - otherwise leave it in UDPCapture's
        // ring (saves GPU/CPU while idle).
        const uint8_t idleButtonMask = makcu.buttonMask();
        if (!shouldInfer(cfg.forceAimOn || controller::maskAiming(idleButtonMask))) return;

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

        if (allowSlowWork) {
            const auto nowLocal = Clock::now();
            if (debugFrameDumper.due(nowLocal)) {
                debugFrameDumper.scheduleNext(nowLocal);
                debugFrameDumper.save(pinnedRgbData, width, height, acquiredFrameId);
            }
        }

        // Re-check button state fresh (it may have changed since the top-of-
        // function check above) and select the matching aim config.
        const uint8_t frameButtonMask = makcu.buttonMask();
        const bool aiming = cfg.forceAimOn || controller::maskAiming(frameButtonMask);
        if (!shouldInfer(aiming)) {
            // Not aiming and past the keep-warm tail: skip inference (save power).
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
        // Inference also runs while the aim key is up (keep-warm), and nothing is
        // sent on those frames. The GPU has to know, because it records every
        // computed move in the in-flight ring and the dead-time compensation then
        // subtracts it from the next measurement - so a warm-up frame used to
        // leave a move in the ring that the mouse never made. Decided once, here,
        // and echoed back in the result so the callback emits on exactly these
        // frames.
        frameAimConfig.aim_output_enabled = aiming ? 1.0f : 0.0f;
        // --- Per-frame dead-time compensation ---
        // inflight_comp subtracts the moves emitted but not yet visible in this
        // measurement, and deadtime_frames says HOW MANY frames' worth. That was a
        // tuned constant, but the real transport delay jitters (measured E2E p50
        // 2.7ms / p95 4.8ms), so a constant is wrong on most frames - and that
        // error shows up as overshoot that no gain setting can remove.
        // This frame's staleness is already known here: capture timestamp vs now.
        // Feeding it in costs nothing on the GPU (deadtime_frames is already
        // fractional and uploaded per frame) and is what lets the ringing/speed
        // trade-off move instead of just sliding along.
        //   sim (3 blocks x60): step overshoot -34%, reach40 -1.7%, error -0.4%,
        //   reversal peak -0.1% - no metric regresses.
        // Only the DEVIATION from this session's own typical staleness is applied,
        // so a constant clock-offset bias cancels out and only jitter is corrected.
        // The frame period is just the gap between consecutive capture stamps - it
        // needs no clock offset, so it is estimated outside the adaptive block that
        // does. (It used to live inside, which meant the rate estimate died with the
        // offset.) One stall must not poison it, hence the plausibility window.
        if (acquiredCaptureUnixMicros != 0 && prevCaptureUnixMicros != 0) {
            const int64_t periodUs =
                static_cast<int64_t>(acquiredCaptureUnixMicros) - prevCaptureUnixMicros;
            if (periodUs > 1000 && periodUs < 40000) {
                // Slow on purpose. This drives the gain rescale, so whatever it
                // tracks, the loop gain does too. At alpha 0.05 (20 frames, ~0.1s) it
                // was chasing jitter rather than the rate: on a real capture the
                // resulting scale had sd 0.036 and swung across 0.89 end to end,
                // i.e. the gains were being modulated ~18% several times a second.
                // 0.002 (500 frames, ~2s) puts that at sd 0.006 / span 0.06 while
                // still following a genuine rate change in a couple of seconds.
                //   alpha    0.05   0.02   0.01  0.005  0.002
                //   sd      0.036  0.021  0.016  0.012  0.006
                // The 1/n term makes it a running mean until the window fills, so
                // startup converges immediately instead of over the first 500 frames.
                ++framePeriodSamples;
                // Skip the first gaps outright: graph capture and first-frame
                // latency make them long, and a running mean that swallows them
                // reads slow for thousands of frames afterwards.
                if (framePeriodSamples > kFramePeriodWarmupSkip) {
                    const double n = static_cast<double>(framePeriodSamples - kFramePeriodWarmupSkip);
                    const double a = std::max(0.002, 1.0 / n);
                    framePeriodUsEma = (framePeriodUsEma <= 0.0)
                        ? static_cast<double>(periodUs)
                        : framePeriodUsEma + a * (periodUs - framePeriodUsEma);
                }
            }
        }

        // --- Frame-rate normalisation ---
        // Convert the authored gains to whatever rate we are actually getting. Until
        // there is an estimate this is a no-op, so startup behaves as configured.
        // Quantised to 1% so a jittering estimate does not re-upload the struct every
        // frame (uploadRuntimeAimConfig skips unchanged ones).
        if (cfg.aimRateAdaptive && framePeriodUsEma > 0.0 && cfg.aimTunedFps > 1.0f) {
            const double tunedPeriodUs = 1e6 / static_cast<double>(cfg.aimTunedFps);
            double r = std::clamp(framePeriodUsEma / tunedPeriodUs, kRateScaleMin, kRateScaleMax);
            r = std::round(r * 100.0) / 100.0;
            rescaleAimConfigForRate(frameAimConfig, r);
            // Dead time is a duration, not a frame count. Only the capture-sampling
            // part (~T/2) follows the rate; the rest (USB + the game's own render
            // pipeline) does not. Deriving it here is what keeps a rate change from
            // silently turning into an over- or under-compensation.
            const double periodMs = framePeriodUsEma / 1000.0;
            const double dtFrames = (static_cast<double>(cfg.inflightDeadtimeMs) + 0.5 * periodMs)
                                    / periodMs;
            // The in-flight ring is 4 frames deep, so a dead time longer than
            // that cannot be represented and is silently truncated - which reads
            // as under-compensation, i.e. exactly the overshoot the ring exists
            // to remove. At 244fps a 10.2ms dead time is already ~3 frames, so
            // this bound is close; say so once rather than let it hide.
            if (dtFrames > static_cast<double>(gpa::AimState::kInflightMax)) {
                static bool warnedDeadtimeClamp = false;
                if (!warnedDeadtimeClamp) {
                    warnedDeadtimeClamp = true;
                    std::cerr << "\n*** dead time resolves to " << dtFrames
                              << " frames at " << (1e6 / framePeriodUsEma)
                              << " fps, above the " << gpa::AimState::kInflightMax
                              << "-frame in-flight ring. It is being clamped, so the"
                                 " compensation is short by the remainder. Lower"
                                 " inflight_deadtime_ms or the capture rate. ***"
                              << std::endl;
                }
            }
            frameAimConfig.deadtime_frames = static_cast<float>(
                std::clamp(dtFrames, 0.0, static_cast<double>(gpa::AimState::kInflightMax)));
            frameAimConfig.ff_ego_lag = frameAimConfig.deadtime_frames;
            // Say what it resolved to, but only when it actually moves - a silent
            // rescale is exactly the kind of thing that hid the last two bugs.
            //
            // The threshold needs hysteresis and a floor on the interval. A plain
            // 0.05 band spammed: a normal session swings 226-246fps, which is a
            // range of r = 0.051, so it tripped on every crossing and then reset the
            // reference so the way back tripped too. 0.10 is about a 17% rate change
            // - a regime, not jitter - and the interval stops even that from
            // chattering at a boundary.
            static double loggedR = 0.0;
            static Clock::time_point loggedAt{};
            const auto nowTp = Clock::now();
            // Wait for the EMA to settle before the first line, or the warm-up
            // transient gets announced as a rate (it read 279fps on a 231fps stream).
            const bool warm = framePeriodSamples >= kFramePeriodLogAfter;
            const bool firstTime = (loggedR == 0.0);
            const bool moved = std::abs(r - loggedR) >= 0.10;
            const bool quietEnough =
                firstTime || (nowTp - loggedAt) >= std::chrono::seconds(10);
            if (warm && (firstTime || moved) && quietEnough) {
                loggedR = r;
                loggedAt = nowTp;
                std::cout << "[Rate] " << std::fixed << std::setprecision(0)
                          << (1e6 / framePeriodUsEma) << " fps (tuned for " << cfg.aimTunedFps
                          << ") -> gain x" << std::setprecision(2) << r
                          << ", deadtime " << frameAimConfig.deadtime_frames << " frames"
                          << std::defaultfloat << std::endl;
            }
        }

        if (cfg.deadtimeAdaptive && acquiredCaptureUnixMicros != 0) {
            bool offsetValid = false;
            const int64_t offset = udpCapture.GetClockOffsetMicros(&offsetValid);
            // Without the offset this block does nothing at all, and it did exactly
            // that for every non-measurement run because the sync ping was gated on
            // perf_stats. Silence is how that survived - so break the silence.
            if (!offsetValid) {
                static int quietFrames = 0;
                if (++quietFrames == 2000) {
                    std::cerr << "*** deadtime_adaptive is ON but the game clock offset is still "
                                 "unknown after 2000 frames - no sync pong from game_pc. Adaptive "
                                 "dead-time is inert and the gains assume it is not. ***"
                              << std::endl;
                }
            }
            if (offsetValid && prevCaptureUnixMicros != 0) {
                const int64_t ageUs =
                    (nowUnixMicros() + offset) - static_cast<int64_t>(acquiredCaptureUnixMicros);
                if (framePeriodUsEma > 0.0 && ageUs >= 0 && ageUs < 100000) {
                    const double ageFrames = ageUs / framePeriodUsEma;
                    ageFramesEma = (ageFramesEma <= 0.0)
                        ? ageFrames
                        : ageFramesEma + 0.02 * (ageFrames - ageFramesEma);
                    // Clamp the correction: a bad offset sample must not be able to
                    // swing the loop, and the in-flight ring only holds 4 frames.
                    double adj = ageFrames - ageFramesEma;
                    adj = std::clamp(adj, -kDeadtimeAdaptMax, kDeadtimeAdaptMax);
                    frameAimConfig.deadtime_frames = static_cast<float>(std::clamp(
                        static_cast<double>(frameAimConfig.deadtime_frames) + adj, 0.0, 4.0));
                    frameAimConfig.ff_ego_lag = frameAimConfig.deadtime_frames;
                }
            }
        }
        // Outside both blocks: the period estimate above needs this every frame, not
        // only when adaptive dead-time happens to be enabled.
        if (acquiredCaptureUnixMicros != 0) {
            prevCaptureUnixMicros = static_cast<int64_t>(acquiredCaptureUnixMicros);
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
        if (!graphReady && !graphFailedForThisShape) {
            // No graph for this shape yet. Ask the main loop to capture it (that
            // can take hundreds of ms, and this function may be on the receive
            // thread - see the comment above), but do NOT drop the frame: the
            // standard path runs the exact same pipeline, just without the graph
            // replay. Dropping it is what made the first frame after startup or a
            // resolution change cost an extra full round trip before the aim
            // reacted at all.
            pendingCaptureShape.store((static_cast<uint64_t>(width) << 32) | height,
                                      std::memory_order_relaxed);
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
        // Calibration needs this too - lat_us in the CSV is submit->completion, and
        // leaving submitTime at epoch would silently write -1 for every row.
        if (cfg.perfStatsEnabled || cfg.calibrationLogEnabled) {
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

    // Let the GPU-completion worker kick the next submit directly, once it has
    // released the finished frame's slot and in-flight counter (see
    // SimpleInference::setPostCompletionHook). This removes the wait for the main
    // loop's 1ms tick / next network frame to re-trigger submission. Registered
    // here - after trySubmitLatestFrame exists and before StartCapture()/the
    // first submit - so no completion can fire before the hook is set.
    auto trySubmitFromHotPath = [&]() { trySubmitLatestFrame(/*allowSlowWork=*/false); };
    inference.setPostCompletionHook(trySubmitFromHotPath);

    // Must be registered before StartCapture() spins up the receive thread.
    udpCapture.SetFrameReadyCallback(trySubmitFromHotPath);
    if (!udpCapture.StartCapture()) {
        std::cerr << "[Simple] Failed to start UDP capture" << std::endl;
        return 1;
    }
    std::cout << "[Simple] UDP capture started on port " << cfg.udpPort << std::endl;

    // Step-response dead-time injection state (aim OFF at a stationary target).
    Clock::time_point lastInject = Clock::now();
    bool injectFlip = false;
    if (cfg.calibrationStepPx > 0 && calibLogger) {
        std::cout << "[Calib] step-response: injecting +-" << cfg.calibrationStepPx
                  << " counts every " << cfg.calibrationStepPeriodMs
                  << "ms (aim OFF, stationary target, room to sweep)" << std::endl;
    }

    while (g_running) {
        // Step-response: inject a known mouse pulse so cx's delayed response
        // reveals the full emit->visible dead-time. Only when explicitly enabled,
        // the period is valid, AND aim is OFF - injecting while the controller is
        // also moving would corrupt the measurement and fight normal aim.
        if (cfg.calibrationStepPx > 0 && cfg.calibrationStepPeriodMs > 0 && calibLogger &&
            !(cfg.forceAimOn || controller::maskAiming(makcu.buttonMask())) &&
            elapsedUs(lastInject, Clock::now()) >= static_cast<int64_t>(cfg.calibrationStepPeriodMs) * 1000) {
            const int pulse = injectFlip ? cfg.calibrationStepPx : -cfg.calibrationStepPx;
            makcu.move(pulse, 0);
            g_injectCumX.fetch_add(pulse, std::memory_order_relaxed);
            injectFlip = !injectFlip;
            lastInject = Clock::now();
        }

        // Tick: wait briefly for a pipeline slot to free (the completion
        // callback notifies pipelineCv when a ticket is released) or simply
        // time out. Either way, fall through to drain below.
        {
            std::unique_lock<std::mutex> lock(pipelineCvMutex);
            pipelineCv.wait_for(lock, std::chrono::milliseconds(1));
        }

        // Keep the game<->inference clock offset fresh. Self-throttled to ~10/s,
        // so calling it every ~1ms tick is cheap (~400 B/s).
        //
        // This started out as E2E-latency instrumentation, hence the perf_stats
        // gate. deadtime_adaptive later came to depend on the same offset - it
        // needs the capture timestamp in our clock to know how stale a frame is -
        // and the gate did not move with it. The result was that the adaptive
        // dead-time block bailed on !offsetValid in every normal run and only
        // worked while measuring, which is the one time it does not matter. The
        // gains were retuned assuming it was on.
        if (cfg.perfStatsEnabled || cfg.deadtimeAdaptive) {
            udpCapture.SendClockSyncPing();
        }

        // Drain path: the hot path is the receive thread's frame-ready callback
        // (see trySubmitLatestFrame's comment); this call only matters when
        // that callback found the pipeline full, lost the submitMutex race, or
        // no frames are arriving other than this 1ms heartbeat.
        trySubmitLatestFrame(/*allowSlowWork=*/true);

        // A frame in trySubmitLatestFrame found no CUDA graph captured for its
        // shape - do the (possibly slow) capture here rather than on the receive
        // thread. Capture needs an idle GPU; frames now keep flowing through the
        // standard path meanwhile, so re-arm and retry instead of dropping the
        // request. Holding submitMutex blocks further submissions (the hot path
        // only try-locks), so the in-flight count drains within a frame or two.
        if (const uint64_t packedShape = pendingCaptureShape.exchange(0, std::memory_order_relaxed)) {
            const unsigned int shapeW = static_cast<unsigned int>(packedShape >> 32);
            const unsigned int shapeH = static_cast<unsigned int>(packedShape & 0xffffffffu);
            std::lock_guard<std::mutex> lock(submitMutex);
            // Capture needs an idle GPU, and frames now keep flowing through the
            // standard path while there is no graph - so we have to CREATE the
            // idle window rather than wait to be handed one. Holding submitMutex
            // across the drain does that: the hot path only try-locks, so nothing
            // new is submitted and the in-flight count reaches zero within one
            // inference time. Re-checking without holding the lock (the obvious
            // version) livelocks at a sustained frame rate - a fresh frame
            // re-arms the pipeline in the 1ms between ticks, every tick, and the
            // capture is postponed forever while the standard path it was meant
            // to be a brief bridge out of runs for the whole session.
            const auto drainDeadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
            while (inference.getCallbacksInFlight() != 0 &&
                   std::chrono::steady_clock::now() < drainDeadline) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            if (inference.getCallbacksInFlight() == 0) {
                (void)ensureFullGraphReady(shapeW, shapeH, "");
            } else {
                // GPU did not drain in 50ms - something is wedged. Retry next
                // tick rather than hold the loop here.
                pendingCaptureShape.store(packedShape, std::memory_order_relaxed);
            }
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
                    logline << " Stn:" << st.samples;
                    if (st.droppedSamples) {
                        // Untimeable completions are excluded from the averages
                        // rather than folded in as 0us. Say how many, or the
                        // breakdown silently describes fewer frames than ran.
                        logline << " StDrop:" << st.droppedSamples;
                    }
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

    // Drain the GPU **and** the host side of the pipeline before the stack-local
    // callback state (callbackTickets, callbackCtx, pipelineCv, the mutex behind
    // it) goes out of scope. cudaStreamSynchronize alone was not enough: the
    // completion callback and the post-completion hook run on SimpleInference's
    // worker thread AFTER the stream drains, and that thread is only joined much
    // later, in the destructor - so main could return, unwind this frame, and
    // leave the worker touching destroyed objects. quiesceCallbacks() refuses
    // further submissions, drains the stream, waits for every callback and hook
    // to finish, and drops the hook.
    const bool pipelineDrained = inference.quiesceCallbacks();

    if (!pipelineDrained) {
        // The wait timed out, so a callback may still be running against
        // callbackTickets / callbackCtx / pipelineCv. Returning normally would
        // unwind this frame and destroy exactly those objects underneath it, and
        // there is no way to cancel a callback that is already in progress. The
        // only honest option left is to stop the process without unwinding: no
        // destructor runs, so nothing the worker might touch is freed.
        // Everything worth keeping is flushed first.
        std::cerr << "\n*** Shutdown could not drain the inference pipeline. Exiting"
                     " without unwinding so the completion thread cannot touch freed"
                     " state. ***" << std::endl;
        if (calibLogger) calibLogger->dump();
        if (perfLog.is_open()) perfLog.flush();
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    }

    movementController.stop();

    return 0;
}
