// Simple aimbot - clean implementation
// UDP capture + TensorRT inference + Mouse control via Makcu
// Features: Full GPU pipeline (inference + postprocess + PID), No-recoil
// Minimal CPU usage - only frame receive and mouse send

#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <random>

#include "needaimbot/cuda/simple_inference.h"
#include "needaimbot/cuda/simple_postprocess.h"
#include "needaimbot/capture/udp_capture.h"
#include "needaimbot/mouse/input_drivers/MakcuConnection.h"

// Third-party JSON parser (header-only)
#include "needaimbot/modules/json.hpp"
using json = nlohmann::json;

std::atomic<bool> g_running{true};

void signalHandler(int sig) {
    std::cout << "\n[Simple] Received signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

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


int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "=== Simple Aimbot (Full GPU Pipeline) ===" << std::endl;

    // Load config
    Config cfg;
    std::string configPath = "simple_config.json";
    if (argc > 1) configPath = argv[1];

    if (cfg.load(configPath)) {
        std::cout << "[Config] Loaded from " << configPath << std::endl;
    } else {
        std::cout << "[Config] Using defaults (no config file)" << std::endl;
    }
    cfg.print();

    // 1. Load TensorRT engine
    gpa::SimpleInference inference;
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
    gpa::MouseMovement mouseMovement;
    gpa::Detection bestTarget;

    // Gaussian noise generator for humanization
    std::random_device rd;
    std::mt19937 noiseGen(rd());
    std::normal_distribution<float> noiseDistX(0.0f, cfg.noiseStddevX);
    std::normal_distribution<float> noiseDistY(0.0f, cfg.noiseStddevY);

    int frameCount = 0;
    auto lastStatTime = std::chrono::steady_clock::now();
    auto lastRecoilTime = std::chrono::steady_clock::now();
    auto lastMouseTime = std::chrono::steady_clock::now();

    std::cout << "[Simple] Full GPU pipeline: ENABLED (inference + decode + target + PID)" << std::endl;
    std::cout << "[Simple] IoU-based target stickiness: ENABLED" << std::endl;

    std::cout << "\n[Simple] Running... Press Ctrl+C to exit" << std::endl;
    std::cout << "[Simple] Right-click (or Side2) = AIM" << std::endl;
    std::cout << "[Simple] Left+Right = AIM + NO-RECOIL" << std::endl;

    // 5. Main loop
    int loopCount = 0;
    int frameRecvCount = 0;
    while (g_running) {
        loopCount++;

        // Wait for frame
        void* rgbData = nullptr;
        unsigned int width = 0, height = 0;
        uint64_t frameId = 0;

        // Stats every second (even without frames)
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatTime).count();
        if (elapsed >= 1000) {
            std::cout << "\r[Simple] Recv: " << frameRecvCount
                      << " | FPS: " << std::fixed << std::setprecision(1) << (frameCount * 1000.0f / elapsed)
                      << " | Aim: " << (makcu.aiming_active ? "ON " : "OFF")
                      << " | Shoot: " << (makcu.shooting_active ? "ON " : "OFF")
                      << "    " << std::flush;
            frameCount = 0;
            lastStatTime = now;
        }

        if (!udpCapture.AcquireFrameSync(&rgbData, &width, &height, &frameId, 16)) {
            // No frame, handle recoil if active (left+right click)
            if (cfg.noRecoilEnabled && makcu.shooting_active && makcu.aiming_active) {
                auto recoilElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastRecoilTime).count();
                if (recoilElapsed >= cfg.recoilTickMs) {
                    int recoilX = static_cast<int>(cfg.recoilCompX);
                    int recoilY = static_cast<int>(cfg.recoilCompY);
                    if (recoilX != 0 || recoilY != 0) {
                        makcu.move(recoilX, recoilY);
                    }
                    lastRecoilTime = now;
                }
            }
            continue;
        }

        if (!rgbData || width == 0 || height == 0) {
            continue;
        }

        frameRecvCount++;

        // Check button state
        bool aiming = makcu.aiming_active;
        bool shooting = makcu.shooting_active;

        // No-recoil compensation (runs every tick while left+right click)
        if (cfg.noRecoilEnabled && shooting && aiming) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastRecoilTime).count();
            if (elapsed >= cfg.recoilTickMs) {
                int recoilX = static_cast<int>(cfg.recoilCompX);
                int recoilY = static_cast<int>(cfg.recoilCompY);
                if (recoilX != 0 || recoilY != 0) {
                    makcu.move(recoilX, recoilY);
                }
                lastRecoilTime = now;
            }
        }

        if (!aiming) {
            // Skip inference when not aiming (save power)
            continue;
        }

        // Run full GPU pipeline: inference + decode + target selection + PID
        // CPU only does: memcpy to pinned, call GPU, read mouse movement, send to Makcu
        bool hasTarget = inference.runInferenceFused(
            static_cast<const uint8_t*>(rgbData), width, height,
            cfg.confThreshold, cfg.headClassId, cfg.headBonus,
            cfg.getAllowedClassMask(),
            gpuPidConfig,
            cfg.iouStickinessThreshold,
            cfg.headAimPoint, cfg.bodyAimPoint,
            mouseMovement,
            &bestTarget);

        frameCount++;

        if (hasTarget) {
            // Mouse movement already calculated by GPU
            float moveX = static_cast<float>(mouseMovement.dx);
            float moveY = static_cast<float>(mouseMovement.dy);

            // Apply Gaussian noise for humanization
            if (cfg.noiseEnabled) {
                moveX += noiseDistX(noiseGen);
                moveY += noiseDistY(noiseGen);
            }

            // Apply shoot offset when aiming+shooting (shifts aim point)
            if (shooting) {
                moveX += cfg.shootOffsetX;
                moveY += cfg.shootOffsetY;
            }

            // Round to integer for mouse movement
            int finalX = static_cast<int>(std::round(moveX));
            int finalY = static_cast<int>(std::round(moveY));

            // Send mouse move immediately (no rate limiting)
            if (finalX != 0 || finalY != 0) {
                makcu.move(finalX, finalY);
            }
        }
    }

    std::cout << "\n[Simple] Shutting down..." << std::endl;
    udpCapture.StopCapture();

    return 0;
}
