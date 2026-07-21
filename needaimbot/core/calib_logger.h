// Calibration CSV logger (ported from the 2pc branch).
//
// Buffers one row per completed inference frame in memory (no hot-path file
// I/O) and writes the CSV once on shutdown. Feeds bench/calibrate.py: with the
// aim key held on a STATIONARY target the logged center variance is the pure
// detector-noise sigma, and lat_us is the capture->complete dead-time - the two
// numbers needed to tune the nonlinear P+D + One Euro controller for THIS rig.
//
// Enable via GlobalSettings::calib_logging_enabled; path from calib_log_path.
// Off by default: when disabled nothing is allocated and nothing is written.

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace gpa {

struct CalibRecord {
    int64_t t_us;       // since logger start
    int64_t lat_us;     // capture(present)->inference-complete, -1 if unknown
    uint8_t aiming;
    uint8_t hasTarget;
    int     classId;
    float   conf;
    float   cx, cy, w, h;   // selected target box center + size (detection px)
    int     emitDx, emitDy; // this frame's emitted mouse move (output counts)
};

class CalibLogger {
public:
    explicit CalibLogger(std::string path)
        : path_(std::move(path)), start_(std::chrono::steady_clock::now()) {
        recs_.reserve(200000);   // ~11 min @300fps; bounded, no realloc on hot path
    }
    ~CalibLogger() { dump(); }

    std::chrono::steady_clock::time_point start() const { return start_; }

    void record(const CalibRecord& r) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (recs_.size() < recs_.capacity()) recs_.push_back(r);
    }

    void dump() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (dumped_ || path_.empty()) return;
        dumped_ = true;
        std::ofstream f(path_);
        if (!f) { std::cerr << "[Calib] cannot open " << path_ << std::endl; return; }
        // Header matches bench/calibrate.py. movement_scale is 1.0 on 1pc (moves
        // are already in output px) and step-response injection is unused here.
        f << "t_us,lat_us,aiming,hasTarget,classId,conf,cx,cy,w,h,"
             "emit_dx,emit_dy,mscale_x,mscale_y,inject_cum_x\n";
        for (const auto& r : recs_) {
            f << r.t_us << ',' << r.lat_us << ',' << int(r.aiming) << ','
              << int(r.hasTarget) << ',' << r.classId << ',' << r.conf << ','
              << r.cx << ',' << r.cy << ',' << r.w << ',' << r.h << ','
              << r.emitDx << ',' << r.emitDy << ",1,1,0\n";
        }
        std::cout << "[Calib] wrote " << recs_.size() << " rows to " << path_ << std::endl;
    }

private:
    std::string path_;
    std::vector<CalibRecord> recs_;
    std::mutex mtx_;
    std::chrono::steady_clock::time_point start_;
    bool dumped_ = false;
};

}  // namespace gpa
