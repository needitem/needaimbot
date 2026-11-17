#pragma once

// Standalone run modes (--button-test, --collect-calib) extracted from main().
// Header-only, following the existing needaimbot/app/ convention.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "config.hpp"
#include "runtime_options.hpp"
#include "pipeline_metrics.hpp"  // g_running
#include "../mouse/input_drivers/MakcuConnection.h"
#include "../mouse/controller.h"   // button masks
#include "../capture/udp_capture.h"

inline int runButtonTest(Config& cfg) {
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

inline int runCollectCalib(Config& cfg, const RuntimeOptions& opts) {
    const int target = opts.collectCalibCount;
    std::error_code ec;
    std::filesystem::create_directories(opts.collectCalibDir, ec);
    UDPCapture cap;
    if (!cap.Initialize(cfg.udpPort) || !cap.StartCapture()) {
        std::cerr << "[Collect] Failed to start UDP capture on port " << cfg.udpPort << std::endl;
        return 1;
    }
    std::cout << "[Collect] Saving up to " << target << " unique RGB frames to "
              << opts.collectCalibDir.string() << std::endl;
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
                std::ofstream of((opts.collectCalibDir / name.str()).string(),
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
              << opts.collectCalibDir.string() << std::endl;
    cap.StopCapture();
    return 0;
}
