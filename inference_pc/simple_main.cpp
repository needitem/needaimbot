// Simple aimbot - clean implementation
// UDP capture + TensorRT inference + Mouse control via Makcu
// Features: Full GPU pipeline (inference + postprocess + PID), No-recoil
// GPU Callback API for lowest latency (no cudaStreamSync wait)
// Minimal CPU usage - only frame receive, mouse send is done in GPU callback

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
#include <mutex>
#include <sstream>

#include "needaimbot/cuda/simple_inference.h"
#include "needaimbot/cuda/simple_postprocess.h"
#include "needaimbot/capture/udp_capture.h"
#include "needaimbot/mouse/input_drivers/MakcuConnection.h"

// Third-party JSON parser (header-only)
#include "needaimbot/modules/json.hpp"
using json = nlohmann::json;

std::atomic<bool> g_running{true};
std::atomic<int> g_frameCount{0};  // Completed inference callbacks per stat window

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
    std::string enginePath = "/home/hwan/needaimbot/sunxds_0.8.2_TRT_320_fp16.engine";
    std::string makcuPort = "/dev/ttyACM0";
    int udpPort = 5007;

    // Detection
    float confThreshold = 0.35f;
    int headClassId = 1;      // Head class for headshot priority
    float headBonus = 0.15f;  // Bonus confidence for head shots
    int maxDetections = 100;  // Maximum detections per frame

    // Class filtering (max 32 classes)
    std::vector<bool> classAllowed;  // Which classes to target
    int maxClasses = 32;

    // Aiming (0 = top, 1 = bottom of bbox)
    float headAimPoint = 1.0f;   // Head: aim at bottom (neck area)
    float bodyAimPoint = 0.15f;  // Body: aim near top (chest area)

    // PID controller (GPU)
    float pidKpX = 0.5f;
    float pidKpY = 0.5f;
    float pidKiX = 0.0f;
    float pidKiY = 0.0f;
    float pidKdX = 0.3f;
    float pidKdY = 0.3f;
    float pidIntegralMax = 50.0f;
    float pidDerivativeMax = 30.0f;

    // IoU stickiness for target tracking
    float iouStickinessThreshold = 0.3f;

    // No-recoil
    bool noRecoilEnabled = true;
    float recoilCompX = 0.0f;
    float recoilCompY = 0.8f;
    int recoilTickMs = 10;

    // Mouse rate limiting
    int mouseMinIntervalMs = 1;
    int frameWaitTimeoutMs = 16;  // UDP frame wait timeout per loop

    // Gaussian noise for humanization
    bool noiseEnabled = true;
    float noiseStddevX = 0.8f;  // Standard deviation for X axis
    float noiseStddevY = 0.8f;  // Standard deviation for Y axis

    // Shoot capture offset (applied when aiming+shooting)
    float shootOffsetX = 0.0f;
    float shootOffsetY = -13.0f;

    // Makcu settings
    int makcuBaudrate = 4000000;

    // Convert to GPU PID config
    gpa::PIDConfig toGpuPIDConfig() const {
        gpa::PIDConfig pid;
        pid.kp_x = pidKpX;
        pid.kp_y = pidKpY;
        pid.ki_x = pidKiX;
        pid.ki_y = pidKiY;
        pid.kd_x = pidKdX;
        pid.kd_y = pidKdY;
        pid.integral_max = pidIntegralMax;
        pid.derivative_max = pidDerivativeMax;
        return pid;
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

            if (j.contains("pid_kp_x")) pidKpX = j["pid_kp_x"];
            if (j.contains("pid_kp_y")) pidKpY = j["pid_kp_y"];
            if (j.contains("pid_ki_x")) pidKiX = j["pid_ki_x"];
            if (j.contains("pid_ki_y")) pidKiY = j["pid_ki_y"];
            if (j.contains("pid_kd_x")) pidKdX = j["pid_kd_x"];
            if (j.contains("pid_kd_y")) pidKdY = j["pid_kd_y"];
            if (j.contains("pid_integral_max")) pidIntegralMax = j["pid_integral_max"];
            if (j.contains("pid_derivative_max")) pidDerivativeMax = j["pid_derivative_max"];

            if (j.contains("iou_stickiness_threshold")) iouStickinessThreshold = j["iou_stickiness_threshold"];

            if (j.contains("no_recoil_enabled")) noRecoilEnabled = j["no_recoil_enabled"];
            if (j.contains("recoil_comp_x")) recoilCompX = j["recoil_comp_x"];
            if (j.contains("recoil_comp_y")) recoilCompY = j["recoil_comp_y"];
            if (j.contains("recoil_tick_ms")) recoilTickMs = j["recoil_tick_ms"];

            if (j.contains("mouse_min_interval_ms")) mouseMinIntervalMs = j["mouse_min_interval_ms"];
            if (j.contains("frame_wait_timeout_ms")) frameWaitTimeoutMs = j["frame_wait_timeout_ms"];
            if (j.contains("makcu_baudrate")) makcuBaudrate = j["makcu_baudrate"];

            if (j.contains("noise_enabled")) noiseEnabled = j["noise_enabled"];
            if (j.contains("noise_stddev_x")) noiseStddevX = j["noise_stddev_x"];
            if (j.contains("noise_stddev_y")) noiseStddevY = j["noise_stddev_y"];

            if (j.contains("shoot_offset_x")) shootOffsetX = j["shoot_offset_x"];
            if (j.contains("shoot_offset_y")) shootOffsetY = j["shoot_offset_y"];

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

            j["pid_kp_x"] = pidKpX;
            j["pid_kp_y"] = pidKpY;
            j["pid_ki_x"] = pidKiX;
            j["pid_ki_y"] = pidKiY;
            j["pid_kd_x"] = pidKdX;
            j["pid_kd_y"] = pidKdY;
            j["pid_integral_max"] = pidIntegralMax;
            j["pid_derivative_max"] = pidDerivativeMax;

            j["iou_stickiness_threshold"] = iouStickinessThreshold;

            j["no_recoil_enabled"] = noRecoilEnabled;
            j["recoil_comp_x"] = recoilCompX;
            j["recoil_comp_y"] = recoilCompY;
            j["recoil_tick_ms"] = recoilTickMs;

            j["mouse_min_interval_ms"] = mouseMinIntervalMs;
            j["frame_wait_timeout_ms"] = frameWaitTimeoutMs;
            j["makcu_baudrate"] = makcuBaudrate;

            j["noise_enabled"] = noiseEnabled;
            j["noise_stddev_x"] = noiseStddevX;
            j["noise_stddev_y"] = noiseStddevY;

            j["shoot_offset_x"] = shootOffsetX;
            j["shoot_offset_y"] = shootOffsetY;

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
        std::cout << "[Config] PID: Kp(" << pidKpX << "," << pidKpY << ") Ki(" << pidKiX << "," << pidKiY
                  << ") Kd(" << pidKdX << "," << pidKdY << ")" << std::endl;
        std::cout << "[Config] IoU stickiness: " << iouStickinessThreshold << std::endl;
        std::cout << "[Config] Max detections: " << maxDetections << std::endl;
        std::cout << "[Config] No-recoil: " << (noRecoilEnabled ? "ON" : "OFF")
                  << " (Y=" << recoilCompY << ", tick=" << recoilTickMs << "ms)" << std::endl;
        std::cout << "[Config] Frame wait timeout: " << frameWaitTimeoutMs << "ms" << std::endl;
        std::cout << "[Config] Noise: " << (noiseEnabled ? "ON" : "OFF")
                  << " (stddev X=" << noiseStddevX << ", Y=" << noiseStddevY << ")" << std::endl;

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
// This callback runs on CUDA's internal thread when GPU inference completes.
// It sends mouse movement immediately, eliminating cudaStreamSynchronize latency.
//
// OPTIMIZATION: Cached config values eliminate pointer indirection in hot path.
// All frequently accessed values are copied to the context struct at init time.

struct CallbackContext {
    static constexpr size_t kNoiseLutSize = 1024;  // Must stay power-of-two.
    static_assert((kNoiseLutSize & (kNoiseLutSize - 1)) == 0, "Noise LUT size must be power-of-two");

    // Hardware reference (only thing we can't cache)
    MakcuConnection* makcu;
    UDPCapture* udpCapture;
    struct MoveQueue* moveQueue = nullptr;
    std::condition_variable* moveQueueCv = nullptr;
    
    // Cached config values (lock-free, no pointer chasing)
    bool noiseEnabled;
    float shootOffsetX;
    float shootOffsetY;
    
    // Movement filter state (deadband + hysteresis)
    float deadbandX;          // Minimum movement threshold X
    float deadbandY;          // Minimum movement threshold Y
    int lastMoveX;            // Previous movement for sign-flip detection
    int lastMoveY;
    
    // Noise LUT (precomputed once, lock-free callback reads)
    std::array<float, kNoiseLutSize> noiseLutX{};
    std::array<float, kNoiseLutSize> noiseLutY{};
    size_t noiseCursor = 0;
    
    // Initialize cached values from config
    void initFromConfig(const Config& cfg) {
        noiseEnabled = cfg.noiseEnabled;
        shootOffsetX = cfg.shootOffsetX;
        shootOffsetY = cfg.shootOffsetY;
        
        // Movement filter defaults (can be made configurable)
        deadbandX = 0.3f;  // Ignore movements < 0.3 pixels
        deadbandY = 0.3f;
        lastMoveX = 0;
        lastMoveY = 0;
        noiseCursor = 0;

        if (noiseEnabled) {
            std::mt19937 gen(std::random_device{}());
            std::normal_distribution<float> distX(0.0f, cfg.noiseStddevX);
            std::normal_distribution<float> distY(0.0f, cfg.noiseStddevY);
            for (size_t i = 0; i < kNoiseLutSize; ++i) {
                noiseLutX[i] = distX(gen);
                noiseLutY[i] = distY(gen);
            }
        } else {
            noiseLutX.fill(0.0f);
            noiseLutY.fill(0.0f);
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
    auto releaseTicket = [ticket]() {
        ticket->bufferIndex = -1;
        ticket->busy.store(false, std::memory_order_release);
    };

    const int completedBuffer = ticket->bufferIndex;
    if (completedBuffer >= 0 && ctx->udpCapture) {
        ctx->udpCapture->ReleaseFrame(completedBuffer);
    }

    // Count every completed inference callback (target/no-target)
    g_frameCount.fetch_add(1, std::memory_order_relaxed);

    if (!ctx->makcu->aiming_active.load(std::memory_order_relaxed)) {
        releaseTicket();
        return;
    }

    if (!result.hasTarget) {
        releaseTicket();
        return;
    }
    
    // Keep callback as short as possible to avoid blocking CUDA stream work.
    if (ctx->moveQueue) {
        MoveCommand cmd;
        cmd.kind = MoveCommand::Kind::AimRaw;
        cmd.dx = result.movement.dx;
        cmd.dy = result.movement.dy;
        cmd.shooting = ctx->makcu->shooting_active.load(std::memory_order_relaxed) ? 1u : 0u;
        if (ctx->moveQueue->tryPush(cmd) && ctx->moveQueueCv) {
            ctx->moveQueueCv->notify_one();
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

    // Load config
    Config cfg;
    std::filesystem::path configPath;
    if (argc > 1) {
        configPath = argv[1];
    } else {
        std::filesystem::path exePath = argv[0] ? std::filesystem::path(argv[0]) : std::filesystem::path();
        std::filesystem::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : std::filesystem::current_path();
        configPath = exeDir / "simple_config.json";
    }

    const std::string configPathStr = configPath.lexically_normal().string();

    if (cfg.load(configPathStr)) {
        std::cout << "[Config] Loaded from " << configPathStr << std::endl;
    } else {
        std::cout << "[Config] Using defaults, saving to " << configPathStr << std::endl;
        cfg.save(configPathStr);
    }
    cfg.print();

    // 1. Load TensorRT engine
    gpa::SimpleInference inference;
    // Configure inference parameters before loading engine.
    inference.setBgraInput(true);
    inference.setMaxDetections(cfg.maxDetections);
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
    if (!udpCapture.Initialize(cfg.udpPort)) {
        std::cerr << "[Simple] Failed to initialize UDP capture" << std::endl;
        return 1;
    }
    if (!udpCapture.StartCapture()) {
        std::cerr << "[Simple] Failed to start UDP capture" << std::endl;
        return 1;
    }
    std::cout << "[Simple] UDP capture started on port " << cfg.udpPort << std::endl;

    // 4. State - all GPU now, minimal CPU state
    gpa::PIDConfig gpuPidConfig = cfg.toGpuPIDConfig();
    const uint32_t allowedClassMask = cfg.getAllowedClassMask();

    // Setup callback context with cached config values
    CallbackContext callbackCtx;
    callbackCtx.makcu = &makcu;
    callbackCtx.udpCapture = &udpCapture;
    MoveQueue moveQueue;
    std::condition_variable moveQueueCv;
    std::mutex moveQueueCvMutex;
    callbackCtx.moveQueue = &moveQueue;
    callbackCtx.moveQueueCv = &moveQueueCv;
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
    const uint32_t frameWaitTimeoutMs = static_cast<uint32_t>(std::clamp(cfg.frameWaitTimeoutMs, 1, 100));
    const int senderMinIntervalMs = std::max(0, cfg.mouseMinIntervalMs);

    auto lastStatTime = std::chrono::steady_clock::now();
    auto lastRecoilTime = std::chrono::steady_clock::now();
    std::atomic<bool> moveSenderRunning{true};
    std::thread moveSenderThread([&]() {
        MoveCommand cmd;
        int pendingDx = 0;
        int pendingDy = 0;
        auto nextSendTime = std::chrono::steady_clock::now();

        auto hasPendingMove = [&]() { return pendingDx != 0 || pendingDy != 0; };
        auto flushMove = [&](std::chrono::steady_clock::time_point now) -> bool {
            if (!hasPendingMove()) return false;
            if (senderMinIntervalMs > 0 && now < nextSendTime) return false;
            makcu.move(pendingDx, pendingDy);
            pendingDx = 0;
            pendingDy = 0;
            if (senderMinIntervalMs > 0) {
                nextSendTime = now + std::chrono::milliseconds(senderMinIntervalMs);
            }
            return true;
        };
        auto processAimMovement = [&](const MoveCommand& raw, int& outDx, int& outDy) {
            float moveX = static_cast<float>(raw.dx);
            float moveY = static_cast<float>(raw.dy);

            if (callbackCtx.noiseEnabled) {
                const size_t idx = callbackCtx.noiseCursor;
                moveX += callbackCtx.noiseLutX[idx];
                moveY += callbackCtx.noiseLutY[idx];
                callbackCtx.noiseCursor = (idx + 1) & (CallbackContext::kNoiseLutSize - 1);
            }

            if (raw.shooting != 0u) {
                moveX += callbackCtx.shootOffsetX;
                moveY += callbackCtx.shootOffsetY;
            }

            if (fabsf(moveX) < callbackCtx.deadbandX) moveX = 0.0f;
            if (fabsf(moveY) < callbackCtx.deadbandY) moveY = 0.0f;

            int finalX = fastRoundToInt(moveX);
            int finalY = fastRoundToInt(moveY);

            const bool signFlipX =
                (finalX != 0 && callbackCtx.lastMoveX != 0 &&
                 ((finalX > 0) != (callbackCtx.lastMoveX > 0)));
            const bool signFlipY =
                (finalY != 0 && callbackCtx.lastMoveY != 0 &&
                 ((finalY > 0) != (callbackCtx.lastMoveY > 0)));
            if (signFlipX || signFlipY) {
                if (signFlipX && finalX > -3 && finalX < 3) finalX = 0;
                if (signFlipY && finalY > -3 && finalY < 3) finalY = 0;
            }

            if (finalX != 0) callbackCtx.lastMoveX = finalX;
            if (finalY != 0) callbackCtx.lastMoveY = finalY;

            outDx = finalX;
            outDy = finalY;
        };

        while (moveSenderRunning.load(std::memory_order_relaxed) || moveQueue.hasPending() || hasPendingMove()) {
            while (moveQueue.tryPop(cmd)) {
                int emitDx = cmd.dx;
                int emitDy = cmd.dy;
                if (cmd.kind == MoveCommand::Kind::AimRaw) {
                    processAimMovement(cmd, emitDx, emitDy);
                }
                if (emitDx != 0 || emitDy != 0) {
                    pendingDx = std::clamp(pendingDx + emitDx, -127, 127);
                    pendingDy = std::clamp(pendingDy + emitDy, -127, 127);
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
        if (moveQueue.tryPush(cmd)) {
            moveQueueCv.notify_one();
        } else {
            makcu.move(dx, dy);
        }
    };

    std::cout << "[Simple] Full GPU pipeline: ENABLED (inference + decode + target + PID)" << std::endl;
    std::cout << "[Simple] GPU Callback API: ENABLED (lowest latency, no sync wait)" << std::endl;
    std::cout << "[Simple] Lock-free config cache: ENABLED" << std::endl;
    std::cout << "[Simple] Movement filter (deadband + sign-flip suppression): ENABLED" << std::endl;
    std::cout << "[Simple] IoU-based target stickiness: ENABLED" << std::endl;
    std::cout << "[Simple] Zero-copy pinned memory: " << (udpCapture.IsPinnedMemoryEnabled() ? "ENABLED" : "DISABLED") << std::endl;

    // 5. Capture full CUDA graph for maximum performance
    std::cout << "[Simple] Capturing full CUDA graph..." << std::endl;
    if (inference.captureFullGraph(
            cfg.confThreshold, cfg.headClassId, cfg.headBonus,
            allowedClassMask, gpuPidConfig,
            cfg.iouStickinessThreshold, cfg.headAimPoint, cfg.bodyAimPoint)) {
        std::cout << "[Simple] Full CUDA graph: ENABLED" << std::endl;
    } else {
        std::cout << "[Simple] Full CUDA graph: DISABLED (using standard execution)" << std::endl;
    }

    std::cout << "\n[Simple] Running... Press Ctrl+C to exit" << std::endl;
    std::cout << "[Simple] Right-click (or Side2) = AIM" << std::endl;
    std::cout << "[Simple] Left+Right = AIM + NO-RECOIL" << std::endl;

    // 6. Main loop with GPU callback API
    // - Frame acquisition runs on main thread
    // - Inference is queued to GPU
    // - Mouse movement is sent in GPU callback (no cudaStreamSync wait!)
    int recvFramesWindow = 0;
    int submittedFramesWindow = 0;
    int busyDropWindow = 0;
    int submitFailWindow = 0;
    uint64_t lastUdpReceived = udpCapture.GetReceivedFrameCount();
    uint64_t lastUdpDropped = udpCapture.GetDroppedFrameCount();
    size_t lastStatusLineLen = 0;
    int inFlightBackoff = 0;

    while (g_running) {
        // Wait for frame (returns pinned memory directly)
        void* pinnedRgbData = nullptr;
        unsigned int width = 0, height = 0;
        int bufferIndex = -1;

        // Stats every second (using atomic g_frameCount from callbacks)
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatTime).count();
        if (elapsed >= 1000) {
            int completedFrames = g_frameCount.exchange(0, std::memory_order_relaxed);  // Atomic read and reset
            uint64_t udpReceivedNow = udpCapture.GetReceivedFrameCount();
            uint64_t udpReceivedDelta = udpReceivedNow - lastUdpReceived;
            lastUdpReceived = udpReceivedNow;
            uint64_t udpDroppedNow = udpCapture.GetDroppedFrameCount();
            uint64_t udpDroppedDelta = udpDroppedNow - lastUdpDropped;
            lastUdpDropped = udpDroppedNow;

            std::ostringstream status;
            status << std::fixed << std::setprecision(1)
                   << "[Simple] R:" << (recvFramesWindow * 1000.0f / elapsed)
                   << " S:" << (submittedFramesWindow * 1000.0f / elapsed)
                   << " D:" << (completedFrames * 1000.0f / elapsed)
                   << " B:" << busyDropWindow
                   << " F:" << submitFailWindow
                   << " C:" << udpReceivedDelta
                   << " U:" << udpDroppedDelta
                   << " A:" << (makcu.aiming_active.load(std::memory_order_relaxed) ? "ON" : "OFF")
                   << " Sh:" << (makcu.shooting_active.load(std::memory_order_relaxed) ? "ON" : "OFF");

            const std::string statusLine = status.str();
            std::cout << '\r' << statusLine;
            if (lastStatusLineLen > statusLine.size()) {
                std::cout << std::string(lastStatusLineLen - statusLine.size(), ' ');
            }
            std::cout << std::flush;
            lastStatusLineLen = statusLine.size();

            recvFramesWindow = 0;
            submittedFramesWindow = 0;
            busyDropWindow = 0;
            submitFailWindow = 0;
            lastStatTime = now;
        }

        // Skip frame acquisition while not aiming to reduce idle CPU usage.
        if (!makcu.aiming_active.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Skip acquisition when inference queue is full.
        if (!inference.hasSubmissionCapacity()) {
            if (inFlightBackoff < 32) {
                ++inFlightBackoff;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            continue;
        }
        inFlightBackoff = 0;

        // Use pinned buffer API for zero-copy
        if (!udpCapture.AcquireFramePinned(&pinnedRgbData, &width, &height, nullptr, &bufferIndex, frameWaitTimeoutMs)) {
            // No frame, handle recoil if active (left+right click)
            auto recoilNow = std::chrono::steady_clock::now();
            if (cfg.noRecoilEnabled &&
                makcu.shooting_active.load(std::memory_order_relaxed) &&
                makcu.aiming_active.load(std::memory_order_relaxed)) {
                auto recoilElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(recoilNow - lastRecoilTime).count();
                if (recoilElapsed >= cfg.recoilTickMs) {
                    int recoilX = static_cast<int>(cfg.recoilCompX);
                    int recoilY = static_cast<int>(cfg.recoilCompY);
                    queueMove(recoilX, recoilY);
                    lastRecoilTime = recoilNow;
                }
            }
            continue;
        }

        if (!pinnedRgbData || width == 0 || height == 0) {
            if (bufferIndex >= 0) udpCapture.ReleaseFrame(bufferIndex);
            continue;
        }

        recvFramesWindow++;

        // Check button state
        bool aiming = makcu.aiming_active.load(std::memory_order_relaxed);
        bool shooting = makcu.shooting_active.load(std::memory_order_relaxed);

        // No-recoil compensation (runs every tick while left+right click)
        if (cfg.noRecoilEnabled && shooting && aiming) {
            auto recoilElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastRecoilTime).count();
            if (recoilElapsed >= cfg.recoilTickMs) {
                int recoilX = static_cast<int>(cfg.recoilCompX);
                int recoilY = static_cast<int>(cfg.recoilCompY);
                queueMove(recoilX, recoilY);
                lastRecoilTime = now;
            }
        }

        if (!aiming) {
            // Skip inference when not aiming (save power)
            udpCapture.ReleaseFrame(bufferIndex);
            continue;
        }

        CallbackTicket* ticket = acquireCallbackTicket();
        if (!ticket) {
            udpCapture.ReleaseFrame(bufferIndex);
            ++busyDropWindow;
            continue;
        }
        ticket->bufferIndex = bufferIndex;

        // GPU CALLBACK API: Queue inference, callback fires when GPU completes.
        // No cudaStreamSynchronize - mouse movement happens in callback thread.

        bool submitted = inference.runInferenceWithCallback(
            pinnedRgbData, width, height,
            cfg.confThreshold, cfg.headClassId, cfg.headBonus,
            allowedClassMask,
            gpuPidConfig,
            cfg.iouStickinessThreshold,
            cfg.headAimPoint, cfg.bodyAimPoint,
            inferenceCallback, ticket);

        if (submitted) {
            submittedFramesWindow++;
        } else {
            int failedBuffer = ticket->bufferIndex;
            if (failedBuffer >= 0) {
                udpCapture.ReleaseFrame(failedBuffer);
            }
            ticket->bufferIndex = -1;
            ticket->busy.store(false, std::memory_order_release);
            if (!inference.hasSubmissionCapacity()) {
                busyDropWindow++;
            } else {
                submitFailWindow++;
            }
        }

        // On success, buffer is released by callback after GPU work completes.

        // Main thread immediately loops back to get next frame
        // while GPU processes this one and callback handles mouse movement
    }

    // Wait for any pending GPU work before shutdown
    cudaStreamSynchronize(inference.getStream());
    moveSenderRunning.store(false, std::memory_order_relaxed);
    moveQueueCv.notify_all();
    if (moveSenderThread.joinable()) {
        moveSenderThread.join();
    }

    std::cout << "\n[Simple] Shutting down..." << std::endl;
    udpCapture.StopCapture();

    return 0;
}
