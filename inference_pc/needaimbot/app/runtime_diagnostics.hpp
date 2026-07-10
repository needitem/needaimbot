#pragma once

// Runtime/thread diagnostics shared by main()'s worker threads: CPU affinity
// pinning, realtime scheduling hints, lock-free running-max, and a small
// per-stat-window accumulator for capture/submit latency.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

// Fixed-bucket latency histogram for percentile (p50/p95/p99) estimation over a
// stat window. 100us buckets up to ~51ms plus an overflow bucket - cheap to
// record, drained once per status print. Avg/max are tracked separately; this
// only exists to expose tail behavior the mean hides.
struct LatencyHistogram {
    static constexpr int kBucketCount = 512;
    static constexpr int64_t kBucketWidthUs = 100;

    uint64_t buckets[kBucketCount] = {};
    uint64_t overflow = 0;

    void record(int64_t us) {
        if (us < 0) us = 0;
        const int64_t idx = us / kBucketWidthUs;
        if (idx >= kBucketCount) ++overflow;
        else ++buckets[static_cast<int>(idx)];
    }

    // Latency (microseconds) at fraction [0,1] by the nearest-rank method:
    // returns the upper edge of the bucket holding the ceil(fraction*N)-th
    // sample (1-based rank). Overflow samples report as the top edge. 0 if
    // no samples.
    int64_t percentileUs(double fraction) const {
        uint64_t total = overflow;
        for (int i = 0; i < kBucketCount; ++i) total += buckets[i];
        if (total == 0) return 0;
        uint64_t rank = static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(total)));
        if (rank == 0) rank = 1;            // fraction ~0 -> first sample
        if (rank > total) rank = total;     // fraction >=1 -> last sample
        uint64_t cumulative = 0;
        for (int i = 0; i < kBucketCount; ++i) {
            cumulative += buckets[i];
            if (cumulative >= rank) return static_cast<int64_t>(i + 1) * kBucketWidthUs;
        }
        return static_cast<int64_t>(kBucketCount) * kBucketWidthUs;  // rank lands in overflow
    }
};

// Lock-free counterpart recorded from the GPU completion callback thread and
// drained (reset) once per status print by main(). Same bucket layout as
// LatencyHistogram so drain() hands back a plain snapshot for percentileUs().
struct AtomicLatencyHistogram {
    std::atomic<uint64_t> buckets[LatencyHistogram::kBucketCount];
    std::atomic<uint64_t> overflow{0};

    AtomicLatencyHistogram() {
        for (auto& b : buckets) b.store(0, std::memory_order_relaxed);
    }

    void record(int64_t us) {
        if (us < 0) us = 0;
        const int64_t idx = us / LatencyHistogram::kBucketWidthUs;
        if (idx >= LatencyHistogram::kBucketCount)
            overflow.fetch_add(1, std::memory_order_relaxed);
        else
            buckets[static_cast<int>(idx)].fetch_add(1, std::memory_order_relaxed);
    }

    LatencyHistogram drain() {
        LatencyHistogram out;
        for (int i = 0; i < LatencyHistogram::kBucketCount; ++i) {
            out.buckets[i] = buckets[i].exchange(0, std::memory_order_relaxed);
        }
        out.overflow = overflow.exchange(0, std::memory_order_relaxed);
        return out;
    }
};

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
    // Percentiles for real frame-wait only (timeouts excluded so the tail
    // reflects actual capture latency, not the acquire poll timeout).
    LatencyHistogram acquireHist;

    void recordAcquire(int64_t elapsed, bool gotFrame) {
        ++acquireSamples;
        acquireTotalUs += elapsed;
        acquireMaxUs = std::max(acquireMaxUs, elapsed);
        if (gotFrame) acquireHist.record(elapsed);
        else ++acquireTimeouts;
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
