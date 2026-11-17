#pragma once

// Pipeline-global atomic counters, wall-clock helpers and the signal handler.
// Header-only (C++17 inline variables).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <csignal>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "runtime_diagnostics.hpp"   // AtomicLatencyHistogram

inline std::atomic<bool> g_running{true};

// Step-response dead-time measurement: cumulative counts the main loop has
// injected into the mouse (aim OFF, stationary target). The callback logs it per
// detection so calibrate.py can cross-correlate cx against it to find the lag.
inline std::atomic<long long> g_injectCumX{0};
inline std::atomic<int> g_frameCount{0};  // Completed inference callbacks per stat window

inline std::atomic<uint64_t> g_callbackLatencySamples{0};
inline std::atomic<int64_t> g_callbackLatencyTotalUs{0};
inline std::atomic<int64_t> g_callbackLatencyMaxUs{0};
inline AtomicLatencyHistogram g_callbackLatencyHist;  // submit->completion latency percentiles

// Capture->inference-complete end-to-end latency (game-PC capture timestamp in
// the UDP header vs. this PC's completion time). Assumes NTP-synced wall clocks;
// clock-skew outliers are dropped at record time. Mouse actuation is NOT included.
inline std::atomic<uint64_t> g_e2eLatencySamples{0};
inline std::atomic<int64_t> g_e2eLatencyTotalUs{0};
inline std::atomic<int64_t> g_e2eLatencyMaxUs{0};
inline AtomicLatencyHistogram g_e2eLatencyHist;

// This PC's wall clock (system_clock) epoch microseconds.
inline int64_t nowUnixMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Wall-clock stamp for perf-log lines (local time, millisecond resolution) so a
// window's metrics can be correlated with events on the game PC.
inline std::string formatWallClock() {
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

inline void signalHandler(int sig) {
    std::cout << "\n[Simple] Received signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}
