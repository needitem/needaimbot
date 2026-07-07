#pragma once

// Runtime/thread diagnostics shared by main()'s worker threads: CPU affinity
// pinning, realtime scheduling hints, lock-free running-max, and a small
// per-stat-window accumulator for capture/submit latency.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

inline int64_t elapsedUs(std::chrono::steady_clock::time_point begin,
                          std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

inline void atomicMax(std::atomic<int64_t>& target, int64_t value) {
    int64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

// Pin the calling thread to a single CPU core. core < 0 is a no-op (leave the
// thread schedulable on any core).
inline void pinThreadToCore(int core) {
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

inline void applyRealtimeHint(const char* threadName, int priorityOffsetFromMax) {
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

// Per-stat-window accumulator for frame-acquire and GPU-submit latency
// (reset every time main() prints its status line).
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
