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
#include <iomanip>
#include <random>
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

// Third-party JSON parser (header-only)
#include "needaimbot/modules/json.hpp"
using json = nlohmann::json;

std::atomic<bool> g_running{true};
std::atomic<int> g_frameCount{0};  // Completed inference callbacks per stat window
std::atomic<uint64_t> g_callbackLatencySamples{0};
std::atomic<int64_t> g_callbackLatencyTotalUs{0};
std::atomic<int64_t> g_callbackLatencyMaxUs{0};
std::atomic<uint64_t> g_moveQueueDropped{0};

namespace {
using Clock = std::chrono::steady_clock;

bool fileExists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

std::filesystem::path safeAbsolute(const std::filesystem::path& path) {
    std::error_code ec;
    auto absolutePath = std::filesystem::absolute(path, ec);
    return ec ? path : absolutePath.lexically_normal();
}

std::filesystem::path currentExecutablePath(const char* argv0) {
#ifndef _WIN32
    std::array<char, 4096> buffer{};
    const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length > 0) {
        buffer[static_cast<size_t>(length)] = '\0';
        return safeAbsolute(std::filesystem::path(buffer.data()));
    }
#endif
    return argv0 ? safeAbsolute(std::filesystem::path(argv0)) : std::filesystem::path();
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void addUniquePath(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path) {
    if (path.empty()) return;
    const auto normalized = safeAbsolute(path);
    if (std::find(paths.begin(), paths.end(), normalized) == paths.end()) {
        paths.push_back(normalized);
    }
}

std::optional<std::filesystem::path> findRepoRootFrom(std::filesystem::path start) {
    if (start.empty()) return std::nullopt;
    start = safeAbsolute(start);
    if (fileExists(start)) {
        start = start.parent_path();
    }

    for (auto current = start; !current.empty(); current = current.parent_path()) {
        if (fileExists(current / "inference_pc" / "simple_main.cpp") &&
            fileExists(current / "inference_pc" / "CMakeLists.txt")) {
            return current;
        }
        if (current == current.root_path()) break;
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> findRepoRoot(const std::filesystem::path& exeDir) {
    if (auto root = findRepoRootFrom(std::filesystem::current_path())) {
        return root;
    }
    return findRepoRootFrom(exeDir);
}

std::filesystem::path chooseConfigPath(const std::filesystem::path& exeDir) {
    return exeDir / "simple_config.json";
}

int engineScore(const std::filesystem::path& path) {
    const std::string name = toLower(path.filename().string());
    int score = 0;
    if (name.find("0.8.2") != std::string::npos) score += 80;
    if (name.find("256") != std::string::npos) score += 45;
    if (name.find("aux0") != std::string::npos) score += 5;
    if (name.find("320") != std::string::npos) score += 30;
    if (name.find("fp16io") != std::string::npos) score += 35;
    if (name.find("orin") != std::string::npos) score += 25;
    if (name.find("l5") != std::string::npos) score += 8;
    if (name.find("fp16") != std::string::npos) score += 20;
    if (name.find("trt") != std::string::npos) score += 10;
    if (name.find("dynamic") != std::string::npos) score -= 4;
    if (name.find("fp8") != std::string::npos) score -= 2;
    return score;
}

std::optional<std::filesystem::path> discoverEngine(
    const std::vector<std::filesystem::path>& searchDirs) {
    std::optional<std::filesystem::path> bestPath;
    int bestScore = std::numeric_limits<int>::min();

    for (const auto& dir : searchDirs) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) break;
            const auto candidate = entry.path();
            if (!fileExists(candidate) || candidate.extension() != ".engine") continue;

            const int score = engineScore(candidate);
            if (!bestPath || score > bestScore ||
                (score == bestScore && candidate.filename().string() > bestPath->filename().string())) {
                bestPath = candidate;
                bestScore = score;
            }
        }
    }
    return bestPath;
}

std::optional<std::filesystem::path> resolveEnginePath(
    const std::string& configuredPath,
    const std::filesystem::path& configPath,
    const std::filesystem::path& exeDir,
    const std::optional<std::filesystem::path>& repoRoot) {
    const std::filesystem::path enginePath(configuredPath);
    std::vector<std::filesystem::path> candidates;
    std::vector<std::filesystem::path> searchDirs;

    addUniquePath(searchDirs, std::filesystem::current_path());
    addUniquePath(searchDirs, configPath.parent_path());
    if (repoRoot) {
        addUniquePath(searchDirs, *repoRoot);
        addUniquePath(searchDirs, *repoRoot / "inference_pc");
    }
    addUniquePath(searchDirs, exeDir);

    if (enginePath.is_absolute()) {
        addUniquePath(candidates, enginePath);
    } else {
        for (const auto& dir : searchDirs) {
            addUniquePath(candidates, dir / enginePath);
        }
    }

    for (const auto& candidate : candidates) {
        if (fileExists(candidate)) return candidate;
    }
    return discoverEngine(searchDirs);
}

int64_t elapsedUs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

void atomicMax(std::atomic<int64_t>& target, int64_t value) {
    int64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

void writeU16LE(std::ostream& out, uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void writeU32LE(std::ostream& out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void writeI32LE(std::ostream& out, int32_t value) {
    writeU32LE(out, static_cast<uint32_t>(value));
}

bool writeRgbBmp(
    const std::filesystem::path& path,
    const uint8_t* rgbData,
    unsigned int width,
    unsigned int height) {
    if (!rgbData || width == 0 || height == 0) return false;

    const uint64_t rawRowBytes = static_cast<uint64_t>(width) * 3u;
    const uint64_t rowStride = (rawRowBytes + 3u) & ~uint64_t{3u};
    const uint64_t pixelBytes = rowStride * static_cast<uint64_t>(height);
    constexpr uint32_t kHeaderBytes = 14u + 40u;
    if (pixelBytes > std::numeric_limits<uint32_t>::max() - kHeaderBytes ||
        width > static_cast<unsigned int>(std::numeric_limits<int32_t>::max()) ||
        height > static_cast<unsigned int>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) return false;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    const uint32_t fileSize = kHeaderBytes + static_cast<uint32_t>(pixelBytes);

    out.write("BM", 2);
    writeU32LE(out, fileSize);
    writeU16LE(out, 0);
    writeU16LE(out, 0);
    writeU32LE(out, kHeaderBytes);

    writeU32LE(out, 40);  // BITMAPINFOHEADER
    writeI32LE(out, static_cast<int32_t>(width));
    writeI32LE(out, -static_cast<int32_t>(height));  // top-down BMP
    writeU16LE(out, 1);
    writeU16LE(out, 24);
    writeU32LE(out, 0);
    writeU32LE(out, static_cast<uint32_t>(pixelBytes));
    writeI32LE(out, 0);
    writeI32LE(out, 0);
    writeU32LE(out, 0);
    writeU32LE(out, 0);

    std::vector<uint8_t> row(static_cast<size_t>(rowStride), 0);
    const unsigned int centerX = width / 2u;
    const unsigned int centerY = height / 2u;
    const unsigned int markerRadius = std::max(8u, std::min(width, height) / 16u);
    for (unsigned int y = 0; y < height; ++y) {
        const uint8_t* src = rgbData + static_cast<size_t>(y) * static_cast<size_t>(width) * 3u;
        std::fill(row.begin(), row.end(), 0);
        for (unsigned int x = 0; x < width; ++x) {
            uint8_t r = src[static_cast<size_t>(x) * 3u + 0u];
            uint8_t g = src[static_cast<size_t>(x) * 3u + 1u];
            uint8_t b = src[static_cast<size_t>(x) * 3u + 2u];
            const unsigned int dx = (x > centerX) ? (x - centerX) : (centerX - x);
            const unsigned int dy = (y > centerY) ? (y - centerY) : (centerY - y);
            const bool onCenterDot = dx <= 1u && dy <= 1u;
            const bool onVerticalMarker = dx <= 1u && dy <= markerRadius;
            const bool onHorizontalMarker = dy <= 1u && dx <= markerRadius;
            if (onVerticalMarker || onHorizontalMarker) {
                r = 255;
                g = onCenterDot ? 255 : 0;
                b = 0;
            }
            row[static_cast<size_t>(x) * 3u + 0u] = b;
            row[static_cast<size_t>(x) * 3u + 1u] = g;
            row[static_cast<size_t>(x) * 3u + 2u] = r;
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        if (!out) return false;
    }

    return true;
}

struct RuntimeOptions {
    bool debugFrameDump = false;
    bool buttonTest = false;
    bool hasConfigPath = false;
    bool hasDebugDir = false;
    bool helpRequested = false;
    bool collectCalib = false;
    int collectCalibCount = 500;
    std::filesystem::path configPath;
    std::filesystem::path debugDir;
    std::filesystem::path collectCalibDir;
};

void printUsage(const char* exeName) {
    std::cout << "Usage: " << (exeName ? exeName : "simple_inference")
              << " [config.json] [--debug] [--debug-dir DIR] [--button-test]\n"
              << "  --debug              Save latest received RGB frame once per second\n"
              << "  --debug-dir DIR      Debug output directory (default: inference_pc/debug)\n"
              << "  --button-test        Print Makcu mouse button mask and exit with Ctrl+C\n"
              << "  --collect-calib DIR [N]  Save N received RGB frames to DIR for int8 calibration (default N=500)\n";
}

bool parseRuntimeOptions(int argc, char* argv[], RuntimeOptions& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--debug" || arg == "--debug-frame" || arg == "--debug-frames") {
            options.debugFrameDump = true;
        } else if (arg == "--button-test") {
            options.buttonTest = true;
        } else if (arg == "--collect-calib") {
            if (i + 1 >= argc) {
                std::cerr << "[Args] --collect-calib requires a directory" << std::endl;
                return false;
            }
            options.collectCalib = true;
            options.collectCalibDir = argv[++i];
            // Optional count: only consume the next arg if it is a positive integer.
            if (i + 1 < argc) {
                const std::string nxt = argv[i + 1] ? argv[i + 1] : "";
                if (!nxt.empty() && nxt.find_first_not_of("0123456789") == std::string::npos) {
                    options.collectCalibCount = std::max(1, std::atoi(nxt.c_str()));
                    ++i;
                }
            }
        } else if (arg == "--debug-dir") {
            if (i + 1 >= argc) {
                std::cerr << "[Args] --debug-dir requires a path" << std::endl;
                return false;
            }
            options.debugFrameDump = true;
            options.hasDebugDir = true;
            options.debugDir = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            options.helpRequested = true;
            return true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "[Args] Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        } else if (!options.hasConfigPath) {
            options.hasConfigPath = true;
            options.configPath = arg;
        } else {
            std::cerr << "[Args] Extra positional argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

struct DebugFrameDumper {
    bool enabled = false;
    std::filesystem::path outputPath;
    Clock::time_point nextCaptureTime{};
    uint64_t savedFrames = 0;
    bool reportedSaveError = false;

    bool due(Clock::time_point now) const {
        return enabled && now >= nextCaptureTime;
    }

    void scheduleNext(Clock::time_point now) {
        nextCaptureTime = now + std::chrono::seconds(1);
    }

    void scheduleRetry(Clock::time_point now) {
        nextCaptureTime = now + std::chrono::milliseconds(100);
    }

    void save(const void* rgbData, unsigned int width, unsigned int height, uint64_t frameId) {
        if (!enabled) return;

        if (writeRgbBmp(outputPath, static_cast<const uint8_t*>(rgbData), width, height)) {
            ++savedFrames;
            reportedSaveError = false;
            (void)frameId;
        } else if (!reportedSaveError) {
            std::cerr << "\n[Debug] Failed to save received frame to "
                      << outputPath.lexically_normal().string() << std::endl;
            reportedSaveError = true;
        }
    }
};

constexpr uint8_t kMakcuLeftMask = 0x01;
constexpr uint8_t kMakcuRightMask = 0x02;
constexpr uint8_t kMakcuMiddleMask = 0x04;
constexpr uint8_t kMakcuSide1Mask = 0x08;
constexpr uint8_t kMakcuSide2Mask = 0x10;

inline bool makcuMaskAiming(uint8_t mask) {
    return (mask & (kMakcuRightMask | kMakcuSide2Mask)) != 0;
}

inline bool makcuMaskThumbAiming(uint8_t mask) {
    return (mask & kMakcuSide2Mask) != 0;
}

inline bool makcuMaskShooting(uint8_t mask) {
    return (mask & kMakcuLeftMask) != 0;
}

// Pin the calling thread to a single CPU core. core < 0 is a no-op (leave the
// thread schedulable on any core).
void pinThreadToCore(int core) {
#ifdef __linux__
    if (core < 0) return;
    const long cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpuCount <= 1 || core >= cpuCount) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
    (void)core;
#endif
}

void applyRealtimeHint(const char* threadName, int priorityOffsetFromMax) {
#ifdef __linux__
    if (threadName && threadName[0] != '\0') {
        pthread_setname_np(pthread_self(), threadName);
    }

    const int maxPriority = sched_get_priority_max(SCHED_FIFO);
    const int minPriority = sched_get_priority_min(SCHED_FIFO);
    if (maxPriority < 0 || minPriority < 0) return;

    sched_param param{};
    param.sched_priority = std::clamp(maxPriority - priorityOffsetFromMax, minPriority, maxPriority);
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        pthread_setschedparam(pthread_self(), SCHED_RR, &param);
    }
#else
    (void)threadName;
    (void)priorityOffsetFromMax;
#endif
}

struct PerfWindowStats {
    uint64_t acquireSamples = 0;
    uint64_t acquireTimeouts = 0;
    uint64_t invalidFrames = 0;
    uint64_t submitSamples = 0;
    int64_t acquireTotalUs = 0;
    int64_t acquireMaxUs = 0;
    int64_t submitTotalUs = 0;
    int64_t submitMaxUs = 0;

    void recordAcquire(int64_t elapsed, bool gotFrame) {
        ++acquireSamples;
        acquireTotalUs += elapsed;
        acquireMaxUs = std::max(acquireMaxUs, elapsed);
        if (!gotFrame) ++acquireTimeouts;
    }

    void recordSubmit(int64_t elapsed) {
        ++submitSamples;
        submitTotalUs += elapsed;
        submitMaxUs = std::max(submitMaxUs, elapsed);
    }

    double averageAcquireMs() const {
        return acquireSamples == 0 ? 0.0 : static_cast<double>(acquireTotalUs) / acquireSamples / 1000.0;
    }

    double maxAcquireMs() const {
        return static_cast<double>(acquireMaxUs) / 1000.0;
    }

    double averageSubmitUs() const {
        return submitSamples == 0 ? 0.0 : static_cast<double>(submitTotalUs) / submitSamples;
    }

    void reset() {
        *this = PerfWindowStats{};
    }
};
}  // namespace

inline int fastRoundToInt(float value) {
    return static_cast<int>(value >= 0.0f ? (value + 0.5f) : (value - 0.5f));
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
    std::string enginePath = "sunxds_0.8.2_256_fp16io_orin_l5_aux0.engine";
    std::string makcuPort = "/dev/ttyACM0";
    int udpPort = 5007;

    // Detection
    float confThreshold = 0.35f;
    int headClassId = 1;      // Head class for headshot priority
    float headBonus = 0.15f;  // Bonus confidence for head shots
    int maxDetections = 100;  // Maximum detections per frame

    // Class filtering (max 32 classes)
    std::vector<bool> classAllowed = std::vector<bool>(32, true);  // Which classes to target
    int maxClasses = 32;

    // Aiming (0 = top, 1 = bottom of bbox)
    float headAimPoint = 1.0f;   // Head: aim at bottom (neck area)
    float bodyAimPoint = 0.15f;  // Body: aim near top (chest area)

    // Nonlinear P controller (GPU)
    float aimKpX = 0.55f;
    float aimKpY = 0.6f;
    // Softness can run lower now that One Euro de-noises the input center, so
    // the near-target damping no longer has to absorb detector jitter.
    float aimSoftnessX = 11.0f;
    float aimSoftnessY = 10.0f;
    // Derivative (damping) gain: brakes overshoot so kp can stay high without
    // oscillating. Split x/y like kp. 0 = pure P; default enables light damping.
    float aimKdX = 0.18f;
    float aimKdY = 0.22f;
    float thumbAimKpX = 0.6f;
    float thumbAimKpY = 0.62f;
    float thumbAimSoftnessX = 11.0f;
    float thumbAimSoftnessY = 10.0f;
    float thumbAimKdX = 0.25f;
    float thumbAimKdY = 0.35f;

    // IoU stickiness for target tracking
    float iouStickinessThreshold = 0.3f;
    // Same-target stickiness fallback: when IoU < threshold, a box whose
    // center is within (prev_diag * factor) is treated as the same enemy.
    // 0 = disabled, 0.4-0.6 typical for fast close targets.
    float distanceStickinessFactor = 0.5f;
    // Track persistence: after the tracked target stops matching this many
    // frames the system commits to a new target. During the gap the mouse
    // is held still (no movement) so the track can re-acquire if the same
    // enemy is briefly missed. 0 = disabled.
    int trackPersistenceFrames = 5;

    // Coast: bridge brief detection gaps by gliding from the last movement
    // (decayed) instead of freezing or shaking. Uses track_persistence_frames
    // as the gap window.
    bool coastEnabled = true;
    float coastDecay = 0.85f;           // per-missed-frame decay of the glide (0..1)
    // Velocity feedforward: tighter tracking of moving targets (0=off, ~1=cancel
    // P steady-state lag). Not lead/prediction - no overshoot on direction change.
    float feedforwardGain = 0.9f;

    // Per-frame max move (output px). Bounds the slew so a large error is crossed
    // in smooth bounded steps instead of one delayed leap that overshoots/rings -
    // overshoot the reactive D term cannot prevent. 0 = disabled (unbounded).
    float aimMaxStep = 30.0f;

    // One Euro adaptive low-pass on the target center (jitter suppression at the
    // source). Heavy smoothing at rest kills settle-shake; it relaxes as the
    // target moves so flicks are not lagged. Cutoffs are in cycles/frame.
    bool oneEuroEnabled = true;
    float oneEuroMinCutoff = 0.1f;  // base cutoff at rest (lower = smoother/more lag)
    float oneEuroBeta = 0.02f;      // speed coefficient (higher = less lag when fast)
    float oneEuroDCutoff = 0.5f;    // derivative cutoff for the speed estimate

    // No-recoil
    bool noRecoilEnabled = true;
    float recoilCompX = 0.0f;
    float recoilCompY = 0.8f;
    int recoilTickMs = 10;

    // Mouse rate limiting
    int mouseMinIntervalMs = 1;
    int frameWaitTimeoutMs = 16;  // UDP frame wait timeout per loop
    int maxInFlightFrames = 1;    // Keep latency low by avoiding stale queued frames
    int frameCreditDepth = 2;     // Allow the Game PC to pre-send the newest next frame
    bool directAimMoveInCallback = true;

    // WindMouse humanized movement (replaces the old gaussian noise).
    // Closed-loop adaptation: the GPU-computed step toward the target acts as
    // gravity; wind adds organic curvature that fades as the aim settles.
    bool windMouseEnabled = false;
    float windMouseGravity = 0.6f;      // fraction of the GPU step applied as pull (lower = smoother/slower)
    float windMouseWind = 3.0f;         // random wind magnitude (px) -> path curvature
    float windMouseInertia = 0.45f;     // velocity retention 0..0.9 (higher = more curve/overshoot)
    float windMouseMaxStep = 30.0f;     // per-frame movement clamp (px)
    float windMouseWindFalloff = 40.0f; // distance (px) under which wind fades so the aim settles

    // Shoot capture offset (applied when aiming+shooting)
    float shootOffsetX = 0.0f;
    float shootOffsetY = -13.0f;

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

    // Runtime diagnostics
    bool perfStatsEnabled = true;
    int perfStatsIntervalMs = 1000;
    bool realtimeThreadsEnabled = true;
    bool forceAimOn = false;  // Benchmark/testing override (keeps inference loop active)
    bool idleGraphPrecaptureEnabled = true;
    int idleGraphPrecaptureIntervalMs = 100;
    // Shapes whose CUDA graph should be captured at startup (before the first
    // frame arrives). Each entry is {sourceWidth, sourceHeight} in pixels.
    // Empty by default - capture happens lazily on first incoming frame.
    std::vector<std::pair<int, int>> preCaptureShapes;
    bool stageTimingEnabled = false;  // Per-stage CUDA event timings (opt-in)

    // Convert to GPU movement config
    static gpa::AimConfig makeGpuAimConfig(
        float kpX, float kpY, float softnessX, float softnessY,
        float kdX, float kdY,
        float distanceStickinessFactor, int trackPersistenceFrames) {
        gpa::AimConfig aim;
        aim.kp_x = kpX;
        aim.kp_y = kpY;
        aim.p_softness_x = softnessX;
        aim.p_softness_y = softnessY;
        aim.kd_x = kdX;
        aim.kd_y = kdY;
        aim.distance_stickiness_factor = distanceStickinessFactor;
        aim.track_persistence_frames = trackPersistenceFrames;
        return aim;
    }

    // Populate the runtime coast fields shared by both aim profiles.
    void applyCoast(gpa::AimConfig& aim) const {
        aim.coast_enabled = coastEnabled ? 1.0f : 0.0f;
        aim.coast_decay = coastDecay;
        aim.feedforward_gain = feedforwardGain;
    }

    // Populate the One Euro filter fields shared by both aim profiles.
    void applyOneEuro(gpa::AimConfig& aim) const {
        aim.oneeuro_enabled = oneEuroEnabled ? 1.0f : 0.0f;
        aim.oneeuro_min_cutoff = oneEuroMinCutoff;
        aim.oneeuro_beta = oneEuroBeta;
        aim.oneeuro_dcutoff = oneEuroDCutoff;
    }

    gpa::AimConfig toGpuAimConfig() const {
        gpa::AimConfig aim = makeGpuAimConfig(aimKpX, aimKpY, aimSoftnessX, aimSoftnessY,
                                              aimKdX, aimKdY,
                                              distanceStickinessFactor, trackPersistenceFrames);
        applyCoast(aim);
        applyOneEuro(aim);
        aim.max_step = aimMaxStep;
        return aim;
    }

    gpa::AimConfig toThumbGpuAimConfig() const {
        gpa::AimConfig aim = makeGpuAimConfig(
            thumbAimKpX, thumbAimKpY, thumbAimSoftnessX, thumbAimSoftnessY,
            thumbAimKdX, thumbAimKdY,
            distanceStickinessFactor, trackPersistenceFrames);
        applyCoast(aim);
        applyOneEuro(aim);
        aim.max_step = aimMaxStep;
        return aim;
    }

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
            if (j.contains("head_bonus")) headBonus = j["head_bonus"];
            if (j.contains("max_detections")) maxDetections = j["max_detections"];

            if (j.contains("head_aim_point")) headAimPoint = j["head_aim_point"];
            if (j.contains("body_aim_point")) bodyAimPoint = j["body_aim_point"];

            if (j.contains("aim_kp_x")) aimKpX = j["aim_kp_x"];
            if (j.contains("aim_kp_y")) aimKpY = j["aim_kp_y"];
            if (j.contains("aim_softness_x")) aimSoftnessX = j["aim_softness_x"];
            if (j.contains("aim_softness_y")) aimSoftnessY = j["aim_softness_y"];
            if (j.contains("aim_kd_x")) aimKdX = j["aim_kd_x"];
            if (j.contains("aim_kd_y")) aimKdY = j["aim_kd_y"];
            if (j.contains("thumb_aim_kp_x")) thumbAimKpX = j["thumb_aim_kp_x"];
            if (j.contains("thumb_aim_kp_y")) thumbAimKpY = j["thumb_aim_kp_y"];
            thumbAimSoftnessX = j.contains("thumb_aim_softness_x")
                                    ? j["thumb_aim_softness_x"].get<float>()
                                    : aimSoftnessX;
            thumbAimSoftnessY = j.contains("thumb_aim_softness_y")
                                    ? j["thumb_aim_softness_y"].get<float>()
                                    : aimSoftnessY;
            thumbAimKdX = j.contains("thumb_aim_kd_x")
                              ? j["thumb_aim_kd_x"].get<float>()
                              : aimKdX;
            thumbAimKdY = j.contains("thumb_aim_kd_y")
                              ? j["thumb_aim_kd_y"].get<float>()
                              : aimKdY;

            if (j.contains("iou_stickiness_threshold")) iouStickinessThreshold = j["iou_stickiness_threshold"];
            if (j.contains("distance_stickiness_factor")) distanceStickinessFactor = j["distance_stickiness_factor"];
            if (j.contains("track_persistence_frames")) {
                int v = j["track_persistence_frames"];
                trackPersistenceFrames = std::clamp(v, 0, 60);
            }

            if (j.contains("coast_enabled")) coastEnabled = j["coast_enabled"];
            if (j.contains("coast_decay")) coastDecay = j["coast_decay"];
            if (j.contains("feedforward_gain")) feedforwardGain = j["feedforward_gain"];
            if (j.contains("aim_max_step")) aimMaxStep = j["aim_max_step"];

            if (j.contains("oneeuro_enabled")) oneEuroEnabled = j["oneeuro_enabled"];
            if (j.contains("oneeuro_min_cutoff")) oneEuroMinCutoff = j["oneeuro_min_cutoff"];
            if (j.contains("oneeuro_beta")) oneEuroBeta = j["oneeuro_beta"];
            if (j.contains("oneeuro_dcutoff")) oneEuroDCutoff = j["oneeuro_dcutoff"];

            if (j.contains("no_recoil_enabled")) noRecoilEnabled = j["no_recoil_enabled"];
            if (j.contains("recoil_comp_x")) recoilCompX = j["recoil_comp_x"];
            if (j.contains("recoil_comp_y")) recoilCompY = j["recoil_comp_y"];
            if (j.contains("recoil_tick_ms")) recoilTickMs = j["recoil_tick_ms"];

            if (j.contains("mouse_min_interval_ms")) mouseMinIntervalMs = j["mouse_min_interval_ms"];
            if (j.contains("frame_wait_timeout_ms")) frameWaitTimeoutMs = j["frame_wait_timeout_ms"];
            if (j.contains("max_inflight_frames")) maxInFlightFrames = j["max_inflight_frames"];
            if (j.contains("frame_credit_depth")) frameCreditDepth = j["frame_credit_depth"];
            if (j.contains("direct_aim_move_in_callback")) directAimMoveInCallback = j["direct_aim_move_in_callback"];
            if (j.contains("makcu_baudrate")) makcuBaudrate = j["makcu_baudrate"];

            if (j.contains("windmouse_enabled")) windMouseEnabled = j["windmouse_enabled"];
            if (j.contains("windmouse_gravity")) windMouseGravity = j["windmouse_gravity"];
            if (j.contains("windmouse_wind")) windMouseWind = j["windmouse_wind"];
            if (j.contains("windmouse_inertia")) windMouseInertia = j["windmouse_inertia"];
            if (j.contains("windmouse_max_step")) windMouseMaxStep = j["windmouse_max_step"];
            if (j.contains("windmouse_wind_falloff")) windMouseWindFalloff = j["windmouse_wind_falloff"];

            if (j.contains("shoot_offset_x")) shootOffsetX = j["shoot_offset_x"];
            if (j.contains("shoot_offset_y")) shootOffsetY = j["shoot_offset_y"];
            if (j.contains("perf_stats_enabled")) perfStatsEnabled = j["perf_stats_enabled"];
            if (j.contains("perf_stats_interval_ms")) perfStatsIntervalMs = j["perf_stats_interval_ms"];
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
            j["head_bonus"] = headBonus;
            j["max_detections"] = maxDetections;

            j["head_aim_point"] = headAimPoint;
            j["body_aim_point"] = bodyAimPoint;

            j["aim_kp_x"] = aimKpX;
            j["aim_kp_y"] = aimKpY;
            j["aim_softness_x"] = aimSoftnessX;
            j["aim_softness_y"] = aimSoftnessY;
            j["aim_kd_x"] = aimKdX;
            j["aim_kd_y"] = aimKdY;
            j["thumb_aim_kp_x"] = thumbAimKpX;
            j["thumb_aim_kp_y"] = thumbAimKpY;
            j["thumb_aim_softness_x"] = thumbAimSoftnessX;
            j["thumb_aim_softness_y"] = thumbAimSoftnessY;
            j["thumb_aim_kd_x"] = thumbAimKdX;
            j["thumb_aim_kd_y"] = thumbAimKdY;

            j["iou_stickiness_threshold"] = iouStickinessThreshold;
            j["distance_stickiness_factor"] = distanceStickinessFactor;
            j["track_persistence_frames"] = trackPersistenceFrames;
            j["coast_enabled"] = coastEnabled;
            j["coast_decay"] = coastDecay;
            j["feedforward_gain"] = feedforwardGain;
            j["aim_max_step"] = aimMaxStep;

            j["_section_oneeuro"] = "===== One Euro center filter (jitter suppression) =====";
            j["oneeuro_enabled"] = oneEuroEnabled;
            j["oneeuro_min_cutoff"] = oneEuroMinCutoff;
            j["oneeuro_beta"] = oneEuroBeta;
            j["oneeuro_dcutoff"] = oneEuroDCutoff;

            j["no_recoil_enabled"] = noRecoilEnabled;
            j["recoil_comp_x"] = recoilCompX;
            j["recoil_comp_y"] = recoilCompY;
            j["recoil_tick_ms"] = recoilTickMs;

            j["mouse_min_interval_ms"] = mouseMinIntervalMs;
            j["frame_wait_timeout_ms"] = frameWaitTimeoutMs;
            j["max_inflight_frames"] = maxInFlightFrames;
            j["frame_credit_depth"] = frameCreditDepth;
            j["direct_aim_move_in_callback"] = directAimMoveInCallback;
            j["makcu_baudrate"] = makcuBaudrate;

            j["windmouse_enabled"] = windMouseEnabled;
            j["windmouse_gravity"] = windMouseGravity;
            j["windmouse_wind"] = windMouseWind;
            j["windmouse_inertia"] = windMouseInertia;
            j["windmouse_max_step"] = windMouseMaxStep;
            j["windmouse_wind_falloff"] = windMouseWindFalloff;

            j["shoot_offset_x"] = shootOffsetX;
            j["shoot_offset_y"] = shootOffsetY;
            j["perf_stats_enabled"] = perfStatsEnabled;
            j["perf_stats_interval_ms"] = perfStatsIntervalMs;
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
        std::cout << "[Config] Right-click P: Kp(" << aimKpX << "," << aimKpY
                  << ") Softness(" << aimSoftnessX << "," << aimSoftnessY
                  << ") Kd(" << aimKdX << "," << aimKdY << ")" << std::endl;
        std::cout << "[Config] Thumb P: Kp(" << thumbAimKpX << "," << thumbAimKpY
                  << ") Softness(" << thumbAimSoftnessX << "," << thumbAimSoftnessY
                  << ") Kd(" << thumbAimKdX << "," << thumbAimKdY << ")" << std::endl;
        std::cout << "[Config] IoU stickiness: " << iouStickinessThreshold << std::endl;
        std::cout << "[Config] Distance stickiness factor: " << distanceStickinessFactor
                  << (distanceStickinessFactor > 0.0f ? " (ON)" : " (OFF)") << std::endl;
        std::cout << "[Config] Coast (gap glide): " << (coastEnabled ? "ON" : "OFF")
                  << " (decay=" << coastDecay << ", window=" << trackPersistenceFrames << " frames)" << std::endl;
        std::cout << "[Config] Velocity feedforward: " << feedforwardGain << std::endl;
        std::cout << "[Config] Aim max step: " << aimMaxStep
                  << (aimMaxStep > 0.0f ? " px/frame" : " (disabled)") << std::endl;
        std::cout << "[Config] One Euro center filter: " << (oneEuroEnabled ? "ON" : "OFF")
                  << " (min_cutoff=" << oneEuroMinCutoff << ", beta=" << oneEuroBeta
                  << ", dcutoff=" << oneEuroDCutoff << ")" << std::endl;
        std::cout << "[Config] Track persistence: " << trackPersistenceFrames
                  << " frame(s)"
                  << (trackPersistenceFrames > 0 ? " (ON)" : " (OFF)") << std::endl;
        std::cout << "[Config] Max detections: " << maxDetections << std::endl;
        std::cout << "[Config] No-recoil: " << (noRecoilEnabled ? "ON" : "OFF")
                  << " (Y=" << recoilCompY << ", tick=" << recoilTickMs << "ms)" << std::endl;
        std::cout << "[Config] Frame wait timeout: " << frameWaitTimeoutMs << "ms" << std::endl;
        std::cout << "[Config] Max in-flight frames: " << maxInFlightFrames << std::endl;
        std::cout << "[Config] Frame credit depth: " << frameCreditDepth << std::endl;
        std::cout << "[Config] Direct aim move in callback: "
                  << (directAimMoveInCallback ? "ON" : "OFF") << std::endl;
        std::cout << "[Config] Perf stats: " << (perfStatsEnabled ? "ON" : "OFF")
                  << " (interval=" << perfStatsIntervalMs << "ms)" << std::endl;
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
        std::cout << "[Config] WindMouse: " << (windMouseEnabled ? "ON" : "OFF")
                  << " (gravity=" << windMouseGravity << ", wind=" << windMouseWind
                  << ", inertia=" << windMouseInertia << ", maxStep=" << windMouseMaxStep << ")" << std::endl;

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
// GPU Callback Context and Handler
// =============================================================================
// This callback runs from the completion worker when GPU inference finishes.
// It can either send aim movement directly or queue it to the sender thread.
//
// OPTIMIZATION: Cached config values eliminate pointer indirection in hot path.
// All frequently accessed values are copied to the context struct at init time.

struct CallbackContext {
    static constexpr size_t kWindLutSize = 1024;  // Must stay power-of-two.
    static_assert((kWindLutSize & (kWindLutSize - 1)) == 0, "Wind LUT size must be power-of-two");

    // Hardware reference (only thing we can't cache)
    MakcuConnection* makcu;
    UDPCapture* udpCapture;
    struct MoveQueue* moveQueue = nullptr;
    std::condition_variable* moveQueueCv = nullptr;
    std::atomic<uint64_t>* moveQueueDropped = nullptr;
    std::condition_variable* pipelineCv = nullptr;
    std::mutex* pipelineCvMutex = nullptr;

    // Cached config values (lock-free, no pointer chasing)
    bool forceAimOn = false;
    bool perfStatsEnabled = false;
    bool directAimMoveInCallback = false;
    float shootOffsetX;
    float shootOffsetY;

    // --- WindMouse humanized movement (closed-loop) ---
    bool windMouseEnabled = false;
    float wmGravity = 0.6f;
    float wmWind = 3.0f;
    float wmInertia = 0.45f;
    float wmMaxStep = 30.0f;
    float wmWindFalloff = 40.0f;
    // Persistent motion state (single consumer thread: callback or sender).
    float wmVelX = 0.0f, wmVelY = 0.0f;     // mouse velocity (inertia carrier)
    float wmWindX = 0.0f, wmWindY = 0.0f;   // current wind force
    float wmResidualX = 0.0f, wmResidualY = 0.0f;  // sub-pixel carry
    // Precomputed unit-gaussian LUT for wind (lock-free, no RNG in hot path).
    std::array<float, kWindLutSize> windLut{};
    size_t windCursor = 0;

    float nextWind() {
        const float v = windLut[windCursor];
        windCursor = (windCursor + 1) & (kWindLutSize - 1);
        return v;
    }

    // Reset motion state when the aim drops (no target / not aiming) so a
    // re-acquisition does not inherit stale velocity/wind.
    void resetWindMouse() {
        wmVelX = wmVelY = 0.0f;
        wmWindX = wmWindY = 0.0f;
        wmResidualX = wmResidualY = 0.0f;
    }

    // Initialize cached values from config
    void initFromConfig(const Config& cfg) {
        forceAimOn = cfg.forceAimOn;
        perfStatsEnabled = cfg.perfStatsEnabled;
        directAimMoveInCallback = cfg.directAimMoveInCallback && cfg.mouseMinIntervalMs <= 0;
        shootOffsetX = cfg.shootOffsetX;
        shootOffsetY = cfg.shootOffsetY;

        windMouseEnabled = cfg.windMouseEnabled;
        wmGravity = cfg.windMouseGravity;
        wmWind = cfg.windMouseWind;
        wmInertia = std::clamp(cfg.windMouseInertia, 0.0f, 0.95f);
        wmMaxStep = std::max(1.0f, cfg.windMouseMaxStep);
        wmWindFalloff = std::max(1.0f, cfg.windMouseWindFalloff);
        resetWindMouse();

        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < kWindLutSize; ++i) {
            windLut[i] = dist(gen);
        }
    }

    void processAimMovement(int rawDx, int rawDy, bool shooting, int& outDx, int& outDy) {
        float moveX = static_cast<float>(rawDx);
        float moveY = static_cast<float>(rawDy);

        if (windMouseEnabled) {
            // The GPU step (rawDx, rawDy) points toward the (predicted) target
            // and acts as gravity. Wind injects organic curvature that fades as
            // the aim closes in; velocity inertia turns it into a smooth curve.
            const float dist = std::sqrt(moveX * moveX + moveY * moveY);
            if (dist < 0.5f) {
                // Effectively on target: bleed off momentum so it settles.
                wmVelX *= 0.5f; wmVelY *= 0.5f;
                wmWindX *= 0.5f; wmWindY *= 0.5f;
            } else {
                const float windScale = wmWind * std::min(1.0f, dist / wmWindFalloff);
                wmWindX = wmWindX * 0.5f + nextWind() * windScale;
                wmWindY = wmWindY * 0.5f + nextWind() * windScale;
                wmVelX = wmVelX * wmInertia + moveX * wmGravity + wmWindX;
                wmVelY = wmVelY * wmInertia + moveY * wmGravity + wmWindY;
                // Clamp speed to max step.
                const float speed = std::sqrt(wmVelX * wmVelX + wmVelY * wmVelY);
                if (speed > wmMaxStep) {
                    const float s = wmMaxStep / speed;
                    wmVelX *= s; wmVelY *= s;
                }
            }
            moveX = wmVelX + wmResidualX;
            moveY = wmVelY + wmResidualY;
        }

        if (shooting) {
            moveX += shootOffsetX;
            moveY += shootOffsetY;
        }

        outDx = fastRoundToInt(moveX);
        outDy = fastRoundToInt(moveY);

        if (windMouseEnabled) {
            // Carry the sub-pixel remainder so slow drifts are not lost.
            wmResidualX = moveX - static_cast<float>(outDx);
            wmResidualY = moveY - static_cast<float>(outDy);
        }
    }
};

struct MoveCommand {
    enum class Kind : uint8_t {
        AimRaw = 0,
        Direct = 1
    };
    Kind kind = Kind::Direct;
    int dx = 0;
    int dy = 0;
    uint8_t shooting = 0;
};

struct MoveQueueSlot {
    std::atomic<uint64_t> sequence{0};
    MoveCommand command{};
};

struct MoveQueue {
    static constexpr uint32_t kCapacity = 4096;  // Must stay power-of-two.
    static_assert((kCapacity & (kCapacity - 1)) == 0, "MoveQueue capacity must be power-of-two");

    std::array<MoveQueueSlot, kCapacity> ring{};
    std::atomic<uint64_t> enqueuePos{0};
    std::atomic<uint64_t> dequeuePos{0};

    MoveQueue() {
        for (uint64_t i = 0; i < kCapacity; ++i) {
            ring[static_cast<size_t>(i)].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool tryPush(const MoveCommand& cmd) {
        uint64_t pos = enqueuePos.load(std::memory_order_relaxed);
        for (;;) {
            MoveQueueSlot& slot = ring[static_cast<size_t>(pos & (kCapacity - 1))];
            const uint64_t seq = slot.sequence.load(std::memory_order_acquire);
            const int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);

            if (diff == 0) {
                if (enqueuePos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    slot.command = cmd;
                    slot.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                pos = enqueuePos.load(std::memory_order_relaxed);
            }
        }
    }

    bool tryPop(MoveCommand& out) {
        uint64_t pos = dequeuePos.load(std::memory_order_relaxed);
        for (;;) {
            MoveQueueSlot& slot = ring[static_cast<size_t>(pos & (kCapacity - 1))];
            const uint64_t seq = slot.sequence.load(std::memory_order_acquire);
            const int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);

            if (diff == 0) {
                if (dequeuePos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    out = slot.command;
                    slot.sequence.store(pos + kCapacity, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;  // Queue empty
            } else {
                pos = dequeuePos.load(std::memory_order_relaxed);
            }
        }
    }

    bool hasPending() const {
        return dequeuePos.load(std::memory_order_acquire) !=
               enqueuePos.load(std::memory_order_acquire);
    }
};

struct CallbackTicket {
    CallbackContext* ctx = nullptr;
    int bufferIndex = -1;
    Clock::time_point submitTime{};
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
    }

    auto releaseTicket = [ticket, ctx]() {
        const int completedBuffer = ticket->bufferIndex;
        if (completedBuffer >= 0 && ctx->udpCapture) {
            ctx->udpCapture->ReleaseFrame(completedBuffer);
        }
        auto markReleased = [&]() {
            ticket->bufferIndex = -1;
            ticket->submitTime = Clock::time_point{};
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
    const bool aimingActive = ctx->forceAimOn || makcuMaskAiming(callbackButtonMask);
    if (!aimingActive) {
        ctx->resetWindMouse();
        releaseTicket();
        return;
    }

    if (!result.hasTarget) {
        ctx->resetWindMouse();
        releaseTicket();
        return;
    }
    
    const bool shooting = makcuMaskShooting(callbackButtonMask);
    if (ctx->directAimMoveInCallback) {
        int emitDx = 0;
        int emitDy = 0;
        ctx->processAimMovement(result.movement.dx, result.movement.dy, shooting, emitDx, emitDy);
        if (emitDx != 0 || emitDy != 0) {
            ctx->makcu->move(emitDx, emitDy);
        }
    } else if (ctx->moveQueue) {
        MoveCommand cmd;
        cmd.kind = MoveCommand::Kind::AimRaw;
        cmd.dx = result.movement.dx;
        cmd.dy = result.movement.dy;
        cmd.shooting = shooting ? 1u : 0u;
        const bool pushed = ctx->moveQueue->tryPush(cmd);
        if (pushed && ctx->moveQueueCv) {
            ctx->moveQueueCv->notify_one();
        } else if (!pushed && ctx->moveQueueDropped) {
            ctx->moveQueueDropped->fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        ctx->makcu->move(result.movement.dx, result.movement.dy);
    }

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
    const std::filesystem::path exePath = currentExecutablePath(argv[0]);
    const std::filesystem::path exeDir =
        exePath.has_parent_path() ? exePath.parent_path() : std::filesystem::current_path();
    const auto repoRoot = findRepoRoot(exeDir);
    std::filesystem::path configPath;
    if (runtimeOptions.hasConfigPath) {
        configPath = runtimeOptions.configPath;
    } else {
        configPath = chooseConfigPath(exeDir);
    }

    configPath = safeAbsolute(configPath);
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
                      << " L:" << ((mask & kMakcuLeftMask) ? "ON " : "OFF")
                      << " R:" << ((mask & kMakcuRightMask) ? "ON " : "OFF")
                      << " M:" << ((mask & kMakcuMiddleMask) ? "ON " : "OFF")
                      << " S1:" << ((mask & kMakcuSide1Mask) ? "ON " : "OFF")
                      << " S2:" << ((mask & kMakcuSide2Mask) ? "ON " : "OFF")
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
    auto resolvedEnginePath = resolveEnginePath(cfg.enginePath, configPath, exeDir, repoRoot);
    if (!resolvedEnginePath) {
        std::cerr << "[Config] Engine not found: " << requestedEnginePath << std::endl;
        std::cerr << "[Config] Place a .engine file under the repo root, inference_pc/, or set engine_path in "
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
        debugFrameDumper.outputPath = safeAbsolute(debugDir / "received_frame.bmp").lexically_normal();
        debugFrameDumper.nextCaptureTime = Clock::now();
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
    std::cout << "[Simple] Makcu connected at " << cfg.makcuBaudrate << " baud" << std::endl;

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
    const gpa::AimConfig rightGpuAimConfig = cfg.toGpuAimConfig();
    const gpa::AimConfig thumbGpuAimConfig = cfg.toThumbGpuAimConfig();
    auto selectGpuAimConfig = [&](uint8_t buttonMask) -> const gpa::AimConfig& {
        return makcuMaskThumbAiming(buttonMask) ? thumbGpuAimConfig : rightGpuAimConfig;
    };
    const uint32_t allowedClassMask = cfg.getAllowedClassMask();

    // Setup callback context with cached config values
    CallbackContext callbackCtx;
    callbackCtx.makcu = &makcu;
    callbackCtx.udpCapture = &udpCapture;
    MoveQueue moveQueue;
    std::condition_variable moveQueueCv;
    std::mutex moveQueueCvMutex;
    std::condition_variable pipelineCv;
    std::mutex pipelineCvMutex;
    callbackCtx.moveQueue = &moveQueue;
    callbackCtx.moveQueueCv = &moveQueueCv;
    callbackCtx.moveQueueDropped = &g_moveQueueDropped;
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
    const uint32_t frameWaitTimeoutMs = static_cast<uint32_t>(std::clamp(cfg.frameWaitTimeoutMs, 1, 100));
    const int senderMinIntervalMs = std::max(0, cfg.mouseMinIntervalMs);
    const int maxPipelineInFlight = std::clamp(cfg.maxInFlightFrames, 1, 4);
    const uint32_t frameCreditDepth = static_cast<uint32_t>(std::clamp(cfg.frameCreditDepth, 1, 4));
    const int perfStatsIntervalMs = std::clamp(cfg.perfStatsIntervalMs, 250, 10000);
    const int idleGraphPrecaptureIntervalMs = std::clamp(cfg.idleGraphPrecaptureIntervalMs, 20, 1000);

    auto lastStatTime = std::chrono::steady_clock::now();
    auto lastRecoilTime = std::chrono::steady_clock::now();
    // Sub-pixel carry for recoil compensation: queueMove only takes ints, so a
    // fractional comp (e.g. 1.3) would otherwise truncate to 1 and silently drop
    // the remainder every tick. Carry it so the emitted average equals the
    // configured value. Shared by both recoil emit sites (frame vs no-frame) and
    // zeroed at the loop top whenever a firing burst is not active (see guard).
    float recoilResidualX = 0.0f;
    float recoilResidualY = 0.0f;
    std::atomic<bool> moveSenderRunning{true};
    std::thread moveSenderThread([&]() {
        if (cfg.realtimeThreadsEnabled) {
            applyRealtimeHint("move-sender", 2);
        }
        if (cfg.cpuAffinityEnabled) {
            pinThreadToCore(cfg.affinityCoreSender);
        }

        MoveCommand cmd;
        // Accumulator may exceed the per-send MAKCU range (+-127); the excess
        // is carried into the next flush instead of being clamped away. Cap
        // the raw accumulator to a few frames' worth so a runaway producer
        // can't pile up unbounded.
        constexpr int kPendingAccumCap = 512;
        int pendingDx = 0;
        int pendingDy = 0;
        auto nextSendTime = std::chrono::steady_clock::now();

        auto hasPendingMove = [&]() { return pendingDx != 0 || pendingDy != 0; };
        auto flushMove = [&](std::chrono::steady_clock::time_point now) -> bool {
            if (!hasPendingMove()) return false;
            if (senderMinIntervalMs > 0 && now < nextSendTime) return false;
            const int sendDx = std::clamp(pendingDx, -127, 127);
            const int sendDy = std::clamp(pendingDy, -127, 127);
            makcu.move(sendDx, sendDy);
            // Carry any overflow into the next flush instead of dropping it.
            pendingDx = std::clamp(pendingDx - sendDx, -kPendingAccumCap, kPendingAccumCap);
            pendingDy = std::clamp(pendingDy - sendDy, -kPendingAccumCap, kPendingAccumCap);
            if (senderMinIntervalMs > 0) {
                nextSendTime = now + std::chrono::milliseconds(senderMinIntervalMs);
            }
            return true;
        };
        while (moveSenderRunning.load(std::memory_order_relaxed) || moveQueue.hasPending() || hasPendingMove()) {
            while (moveQueue.tryPop(cmd)) {
                int emitDx = cmd.dx;
                int emitDy = cmd.dy;
                if (cmd.kind == MoveCommand::Kind::AimRaw) {
                    callbackCtx.processAimMovement(
                        cmd.dx, cmd.dy, cmd.shooting != 0u, emitDx, emitDy);
                }
                if (emitDx != 0 || emitDy != 0) {
                    pendingDx = std::clamp(pendingDx + emitDx,
                                           -kPendingAccumCap, kPendingAccumCap);
                    pendingDy = std::clamp(pendingDy + emitDy,
                                           -kPendingAccumCap, kPendingAccumCap);
                }
            }

            const auto now = std::chrono::steady_clock::now();
            if (flushMove(now)) {
                continue;
            }

            if (moveSenderRunning.load(std::memory_order_relaxed) || moveQueue.hasPending() || hasPendingMove()) {
                std::unique_lock<std::mutex> lock(moveQueueCvMutex);
                if (hasPendingMove() && senderMinIntervalMs > 0 && now < nextSendTime) {
                    moveQueueCv.wait_until(lock, nextSendTime, [&]() {
                        return !moveSenderRunning.load(std::memory_order_relaxed) || moveQueue.hasPending();
                    });
                } else {
                    moveQueueCv.wait(lock, [&]() {
                        return !moveSenderRunning.load(std::memory_order_relaxed) || moveQueue.hasPending();
                    });
                }
            } else {
                break;
            }
        }
    });
    auto queueMove = [&](int dx, int dy) {
        if (dx == 0 && dy == 0) return;
        MoveCommand cmd;
        cmd.kind = MoveCommand::Kind::Direct;
        cmd.dx = dx;
        cmd.dy = dy;
        cmd.shooting = 0u;
        const bool pushed = moveQueue.tryPush(cmd);
        if (pushed) {
            moveQueueCv.notify_one();
        } else {
            makcu.move(dx, dy);
        }
    };

    // Emit one no-recoil tick, carrying the sub-pixel remainder so a fractional
    // comp value averages out instead of truncating to int every tick.
    auto emitRecoilTick = [&]() {
        recoilResidualX += cfg.recoilCompX;
        recoilResidualY += cfg.recoilCompY;
        const int dx = static_cast<int>(recoilResidualX);  // truncates toward zero
        const int dy = static_cast<int>(recoilResidualY);
        recoilResidualX -= static_cast<float>(dx);
        recoilResidualY -= static_cast<float>(dy);
        queueMove(dx, dy);  // no-op when both components are zero
    };

    std::cout << "[Simple] Full GPU pipeline: ENABLED (inference + decode + target + nonlinear P)" << std::endl;
    std::cout << "[Simple] GPU Callback API: ENABLED (lowest latency, no sync wait)" << std::endl;
    std::cout << "[Simple] Lock-free config cache: ENABLED" << std::endl;
    std::cout << "[Simple] Movement post-processing: WindMouse/shoot-offset" << std::endl;
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
                cfg.confThreshold, cfg.headClassId, cfg.headBonus,
                allowedClassMask, rightGpuAimConfig,
                cfg.iouStickinessThreshold, cfg.headAimPoint, cfg.bodyAimPoint)) {
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
              << (callbackCtx.directAimMoveInCallback ? "ENABLED" : "DISABLED")
              << std::endl;

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
            cfg.confThreshold, cfg.headClassId, cfg.headBonus,
            allowedClassMask, rightGpuAimConfig,
            cfg.iouStickinessThreshold, cfg.headAimPoint, cfg.bodyAimPoint);
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
                cfg.confThreshold, cfg.headClassId, cfg.headBonus,
                allowedClassMask, rightGpuAimConfig,
                cfg.iouStickinessThreshold, cfg.headAimPoint, cfg.bodyAimPoint)) {
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
        if (!(cfg.forceAimOn || makcuMaskAiming(idleButtonMask))) return;

        requestFrameCredits();

        void* pinnedRgbData = nullptr;
        unsigned int width = 0, height = 0;
        uint64_t acquiredFrameId = 0;
        int bufferIndex = -1;
        uint8_t bytesPerPixel = 3;
        uint8_t pixelFormat = UDP_PIXEL_FORMAT_RGB;

        Clock::time_point acquireStart{};
        if (cfg.perfStatsEnabled) {
            acquireStart = Clock::now();
        }
        // Non-blocking - see the function comment above.
        const bool gotFrame = udpCapture.AcquireFramePinned(
            &pinnedRgbData, &width, &height, &acquiredFrameId, &bufferIndex, /*timeoutMs=*/0,
            &bytesPerPixel, &pixelFormat);
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
        const bool aiming = cfg.forceAimOn || makcuMaskAiming(frameButtonMask);
        if (!aiming) {
            // Skip inference when not aiming (save power)
            udpCapture.ReleaseFrame(bufferIndex);
            return;
        }
        const gpa::AimConfig& frameAimConfig = selectGpuAimConfig(frameButtonMask);

        const bool graphReady = inference.isFullGraphReadyForShape(
            static_cast<int>(width), static_cast<int>(height),
            maxPipelineInFlight,
            cfg.confThreshold, cfg.headClassId, cfg.headBonus,
            allowedClassMask, rightGpuAimConfig,
            cfg.iouStickinessThreshold, cfg.headAimPoint, cfg.bodyAimPoint);
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
        bool submitted = inference.runInferenceWithCallback(
            pinnedRgbData, width, height,
            cfg.confThreshold, cfg.headClassId, cfg.headBonus,
            allowedClassMask,
            frameAimConfig,
            cfg.iouStickinessThreshold,
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
        {
            const uint8_t recoilGateMask = makcu.buttonMask();
            const bool recoilFiring =
                cfg.noRecoilEnabled && makcuMaskShooting(recoilGateMask) &&
                (cfg.forceAimOn || makcuMaskAiming(recoilGateMask));
            if (!recoilFiring) {
                // Drop any stale sub-pixel carry whenever a firing burst is not
                // active, so it can't survive into the next burst.
                recoilResidualX = 0.0f;
                recoilResidualY = 0.0f;
            } else if (now - lastRecoilTime >= std::chrono::milliseconds(cfg.recoilTickMs)) {
                emitRecoilTick();
                lastRecoilTime += std::chrono::milliseconds(cfg.recoilTickMs);
                if (lastRecoilTime < now - std::chrono::milliseconds(cfg.recoilTickMs)) {
                    lastRecoilTime = now;  // fell far behind -> resync, no burst
                }
            }
        }

        // Idle-only housekeeping: graph pre-capture and debug frame dump each
        // pull their own frame directly (the hot path only runs while aiming),
        // so both must serialize against trySubmitLatestFrame via submitMutex.
        const uint8_t idleButtonMask = makcu.buttonMask();
        const bool aimingActiveMain = cfg.forceAimOn || makcuMaskAiming(idleButtonMask);
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
                    &pinnedRgbData, &width, &height, &debugFrameId, &bufferIndex, frameWaitTimeoutMs,
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
            const uint64_t moveQueueDropped =
                g_moveQueueDropped.exchange(0, std::memory_order_relaxed);
            const double callbackAvgMs = callbackLatencySamples == 0
                ? 0.0
                : static_cast<double>(callbackLatencyTotalUs) / callbackLatencySamples / 1000.0;
            const double callbackMaxMs = static_cast<double>(callbackLatencyMaxUs) / 1000.0;
            uint64_t udpReceivedNow = udpCapture.GetReceivedFrameCount();
            uint64_t udpReceivedDelta = udpReceivedNow - lastUdpReceived;
            lastUdpReceived = udpReceivedNow;
            uint64_t udpDroppedNow = udpCapture.GetDroppedFrameCount();
            uint64_t udpDroppedDelta = udpDroppedNow - lastUdpDropped;
            lastUdpDropped = udpDroppedNow;
            const uint8_t statusButtonMask = makcu.buttonMask();

            std::ostringstream status;
            status << std::fixed << std::setprecision(1)
                   << "[Simple] R:" << (recvSnapshot * 1000.0f / elapsed)
                   << " S:" << (submittedSnapshot * 1000.0f / elapsed)
                   << " D:" << (completedFrames * 1000.0f / elapsed)
                   << " B:" << busyDropSnapshot
                   << " F:" << submitFailSnapshot
                   << " C:" << udpReceivedDelta
                   << " U:" << udpDroppedDelta
                   << " Cr:" << creditSnapshot
                   << " I:" << busyCallbackTicketCount()
                   << " A:" << ((cfg.forceAimOn || makcuMaskAiming(statusButtonMask)) ? "ON" : "OFF")
                   << " Sh:" << (makcuMaskShooting(statusButtonMask) ? "ON" : "OFF");
            if (cfg.perfStatsEnabled) {
                status << " Aw:" << perfSnapshot.averageAcquireMs() << "/" << perfSnapshot.maxAcquireMs() << "ms"
                       << " Su:" << perfSnapshot.averageSubmitUs() << "/" << perfSnapshot.submitMaxUs << "us"
                       << " Cb:" << callbackAvgMs << "/" << callbackMaxMs << "ms"
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
                    status << " St[h2d:" << avgUs(st.h2dUsTotal) << "/" << st.h2dUsMax
                           << " pre:" << avgUs(st.preprocessUsTotal) << "/" << st.preprocessUsMax
                           << " inf:" << avgUs(st.inferenceUsTotal) << "/" << st.inferenceUsMax
                           << " post:" << avgUs(st.postprocessUsTotal) << "/" << st.postprocessUsMax
                           << " d2h:" << avgUs(st.d2hUsTotal) << "/" << st.d2hUsMax << "us]";
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
    moveSenderRunning.store(false, std::memory_order_relaxed);
    moveQueueCv.notify_all();
    if (moveSenderThread.joinable()) {
        moveSenderThread.join();
    }

    return 0;
}
