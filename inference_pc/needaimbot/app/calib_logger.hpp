#pragma once

// Calibration logger: records the raw per-frame detection so
// bench/calibrate.py can measure this rig's real detector noise,
// head-selection rate, dropout and frame timing.  Buffered in memory
// (no hot-path file I/O), written once on shutdown.
//
// Header-only, following the existing needaimbot/app/ convention.

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

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
