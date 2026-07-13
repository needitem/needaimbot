#pragma once

// Warped-replay flick generator: instead of synthesizing a stroke from a
// hand-tuned motor model (a formula-based generator, now removed), this
// replays a REAL recorded human stroke, rigid-transformed to hit the target.
//
// Why: against a strong mouse-dynamics classifier, every *synthetic* flick
// (formula-based or learned) is detectable (~0.83-0.99 accuracy), because it
// can't reproduce the full joint micro-structure of real motion. A real stroke
// rotated/scaled onto the aim vector keeps that structure exactly, so it reads
// as human (~0.50 - indistinguishable). See the mouse-bot-detector study /
// SCRAP (ACM AISec 2020): domain-knowledge replay beats adversarial ML.
//
// The trajectory database (flick_trajectories.json, produced by
// mouse-bot-detector/scripts/export_flick_db.py) holds STRAIGHT, low-lateral-
// deviation human strokes, canonicalized to start at the origin with their
// endpoint on the +x axis, each with its real distance and real (irregular)
// timestamps. generate() picks a stroke whose recorded reach is close to the
// required reach (so the scale warp is ~1 and doesn't distort speed), rotates
// it to the aim direction, scales it to the exact reach, and returns the same
// absolute (x, y, t_ms) point vector the playback layer already consumes.
//
// Namespace is `warped_replay`. simple_main.cpp holds a warped_replay::config
// and warped_replay::FlickPlayback.

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <optional>
#include <iostream>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <atomic>
#include <unistd.h>

#include "json.hpp"

namespace warped_replay {

struct trajectory_point {
    double x, y;
    double t;   // ms, first point 0, monotonic non-decreasing
};

struct config {
    // Path to the recorded-stroke database. Resolved relative to the CWD and,
    // failing that, next to the executable (see detail::load_db).
    std::string replay_db_path = "flick_trajectories.json";
    // Pick a stroke whose recorded reach is within +-tolerance of the needed
    // reach, so scaling stays near 1x (large stretches distort speed/tremor).
    double distance_tolerance = 0.15;
    // Human-variability perturbation. 0 = PURE REPLAY (default, best for a
    // single session): each flick is a real stroke warped onto the target, no
    // perturbation, and no-repeat source selection already avoids duplicates -
    // this leaves no session-distribution trace. Set > 0 (e.g. 0.07) only if
    // you allow source reuse and need to break near-duplicates via
    // mag*(shape_a - shape_b), a direction humans genuinely vary along
    // (mouse-bot-detector/attack_sweet_spot.py) - but that adds a faint trace.
    double variability_mag = 0.0;
    // Residual per-point Gaussian jitter (px). 0 = pure replay (default): the
    // stroke is reproduced unmodified. AVOID > 0: white per-point jitter adds
    // high-frequency jerk, the single strongest tell a mouse-dynamics detector
    // reads (synthetic-noise generators peak the detector at ~0.85). Use
    // elastic_amp instead to break duplicates without adding jerk.
    double position_jitter = 0.0;
    // Elastic deformation amplitude (fraction of the reach). After warping a real
    // stroke onto the aim vector, bend it by a SMOOTH low-frequency lateral
    // displacement (sum of elastic_modes sine modes perpendicular to the local
    // direction, zero at both endpoints so start/target are preserved). This
    // changes the canonical SHAPE - breaking finite-pool near-duplicates, so a
    // small DB yields unlimited distinct flicks and no-repeat never has to reuse -
    // while preserving the fine kinematics, so a strong detector still reads it as
    // human. Measured: amp~0.03 -> ~0.60 strong-detector accuracy with near-dups
    // broken, vs 0.85 for synthetic generators and 0.51-but-finite for pure
    // replay (mouse-bot-detector/elastic_replay.py). 0 disables (pure replay).
    double elastic_amp = 0.03;
    int elastic_modes = 3;
    // Below this reach, just emit a 2-point straight segment.
    double min_reach = 5.0;

    // flick_-prefixed keys sit alongside pd_controller's aim_-prefixed keys.
    void load(const nlohmann::json& j) {
        if (j.contains("flick_replay_db_path")) replay_db_path = j["flick_replay_db_path"];
        if (j.contains("flick_distance_tolerance")) distance_tolerance = j["flick_distance_tolerance"];
        if (j.contains("flick_variability_mag")) variability_mag = j["flick_variability_mag"];
        if (j.contains("flick_position_jitter")) position_jitter = j["flick_position_jitter"];
        if (j.contains("flick_elastic_amp")) elastic_amp = j["flick_elastic_amp"];
        if (j.contains("flick_elastic_modes")) elastic_modes = j["flick_elastic_modes"];
        if (j.contains("flick_min_reach")) min_reach = j["flick_min_reach"];
    }
    void save(nlohmann::json& j) const {
        j["flick_replay_db_path"] = replay_db_path;
        j["flick_distance_tolerance"] = distance_tolerance;
        j["flick_variability_mag"] = variability_mag;
        j["flick_position_jitter"] = position_jitter;
        j["flick_elastic_amp"] = elastic_amp;
        j["flick_elastic_modes"] = elastic_modes;
        j["flick_min_reach"] = min_reach;
    }
    void print() const {
        std::cout << "[Config] Flick generator: warped-replay, db=" << replay_db_path
                  << ", dist tol=" << distance_tolerance
                  << ", variability=" << variability_mag
                  << ", jitter=" << position_jitter << "px"
                  << ", elastic=" << elastic_amp << " x" << elastic_modes << std::endl;
    }
};

namespace detail {

constexpr int kNumPoints = 48;            // must match export_flick_db.py N_PTS

struct Stroke {
    double d;                             // recorded reach distance (px)
    double sx[kNumPoints], sy[kNumPoints];// UNIT canonical shape: origin -> (1, 0)
    double t[kNumPoints];                 // real (irregular) timestamps, ms
};

inline std::string exe_dir() {
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = '\0';
    std::string p(buf);
    auto pos = p.find_last_of('/');
    return pos == std::string::npos ? "" : p.substr(0, pos);
}

// Lazily loaded once, sorted by distance. Thread-safe (start() runs on the
// callback thread but a cancel poll can race the first load).
inline const std::vector<Stroke>& load_db(const std::string& path) {
    static std::mutex mtx;
    static std::vector<Stroke> db;
    static bool tried = false;
    std::lock_guard<std::mutex> lk(mtx);
    if (tried) return db;
    tried = true;

    std::vector<std::string> cands = {path};
    const std::string ed = exe_dir();
    if (!ed.empty()) {
        auto pos = path.find_last_of("/\\");
        cands.push_back(ed + "/" + (pos == std::string::npos ? path : path.substr(pos + 1)));
    }
    for (const auto& c : cands) {
        std::ifstream f(c);
        if (!f.good()) continue;
        try {
            nlohmann::json j;
            f >> j;
            for (const auto& tr : j.at("traj")) {
                const auto& sh = tr.at("s");
                const auto& tt = tr.at("t");
                if ((int)sh.size() != kNumPoints || (int)tt.size() != kNumPoints) continue;
                Stroke s;
                s.d = tr.at("d").get<double>();
                for (int k = 0; k < kNumPoints; ++k) {
                    s.sx[k] = sh[k][0].get<double>();
                    s.sy[k] = sh[k][1].get<double>();
                    s.t[k]  = tt[k].get<double>();
                }
                db.push_back(s);
            }
            std::sort(db.begin(), db.end(),
                      [](const Stroke& a, const Stroke& b) { return a.d < b.d; });
            std::cout << "[warped-replay] loaded " << db.size() << " strokes from " << c << std::endl;
            break;
        } catch (const std::exception& e) {
            std::cerr << "[warped-replay] failed to parse " << c << ": " << e.what() << std::endl;
            db.clear();
        }
    }
    if (db.empty()) {
        std::cerr << "[warped-replay] WARNING: no trajectory DB loaded (tried "
                  << cands.size() << " path(s)); flicks fall back to straight lines" << std::endl;
    }
    return db;
}

}  // namespace detail

// Preload the trajectory DB at startup so the first flick's ~130ms JSON parse
// doesn't land on the hot callback path. Call once after config is loaded.
inline void warmup(const config& cfg) { detail::load_db(cfg.replay_db_path); }

// Rigid-transform-replays a recorded human stroke from (x0,y0) to (x1,y1).
// Same signature/return as the previous generator so FlickPlayback
// is untouched: absolute (x,y,t_ms) points, first t=0, monotonic timestamps.
inline std::vector<trajectory_point> generate(
    double x0, double y0, double x1, double y1,
    const config& cfg = {}, uint64_t seed = 0) {

    const double dx = x1 - x0, dy = y1 - y0;
    const double D = std::hypot(dx, dy);
    std::vector<trajectory_point> out;

    if (D < cfg.min_reach) {
        out.push_back({x0, y0, 0.0});
        out.push_back({x1, y1, 16.0});
        return out;
    }

    const auto& db = detail::load_db(cfg.replay_db_path);
    if (db.empty()) {
        out.push_back({x0, y0, 0.0});
        out.push_back({x1, y1, 60.0 + 0.25 * D});  // plausible straight-line fallback
        return out;
    }

    // seed 0 -> reuse a thread-local RNG (no per-flick random_device syscall);
    // nonzero seed -> reproducible local RNG for tests.
    static thread_local std::mt19937_64 tls_rng(std::random_device{}());
    std::mt19937_64 seeded;
    std::mt19937_64& rng = seed ? (seeded.seed(seed), seeded) : tls_rng;

    // distance-matched window so the scale warp stays near 1x
    const double loD = D * (1.0 - cfg.distance_tolerance);
    const double hiD = D * (1.0 + cfg.distance_tolerance);
    auto lo = std::lower_bound(db.begin(), db.end(), loD,
                               [](const detail::Stroke& s, double v) { return s.d < v; });
    auto hi = std::upper_bound(db.begin(), db.end(), hiD,
                               [](double v, const detail::Stroke& s) { return v < s.d; });
    // No-repeat source selection: within a session, prefer strokes not used
    // yet, so the same real stroke never replays twice (defeats near-duplicate
    // and session-distribution detection). When the distance window is
    // exhausted, reset it and reuse. State is per-thread (the callback thread).
    const detail::Stroke* s = nullptr;
    if (hi > lo) {
        static thread_local std::unordered_set<const detail::Stroke*> used;
        std::vector<const detail::Stroke*> avail;
        avail.reserve(static_cast<size_t>(hi - lo));
        for (auto it = lo; it != hi; ++it)
            if (!used.count(&*it)) avail.push_back(&*it);
        if (avail.empty()) {                       // window exhausted -> reset
            for (auto it = lo; it != hi; ++it) { used.erase(&*it); avail.push_back(&*it); }
        }
        std::uniform_int_distribution<size_t> pick(0, avail.size() - 1);
        s = avail[pick(rng)];
        used.insert(s);
    } else {
        auto it = std::lower_bound(db.begin(), db.end(), D,
                                   [](const detail::Stroke& a, double v) { return a.d < v; });
        if (it == db.end()) it = std::prev(db.end());
        s = &*it;
    }

    const double theta = std::atan2(dy, dx);
    const double c = std::cos(theta), sn = std::sin(theta);
    std::normal_distribution<double> jit(0.0, cfg.position_jitter);

    // Pure replay (mag == 0): warp the source stroke straight onto the aim
    // vector. Optional human-variability perturbation when mag > 0: two random
    // strokes give a real difference vector, added to break near-duplicates if
    // source reuse is ever allowed (see config comment).
    const double mag = cfg.variability_mag;
    std::uniform_int_distribution<size_t> anyStroke(0, db.size() - 1);
    const detail::Stroke* A = mag > 0.0 ? &db[anyStroke(rng)] : nullptr;
    const detail::Stroke* B = mag > 0.0 ? &db[anyStroke(rng)] : nullptr;

    // Elastic deformation: one coefficient per sine mode, drawn once per flick.
    // Higher modes get smaller amplitude (a_j ~ N(0, amp/j)) so the bend stays
    // low-frequency (kinematically cheap). Applied in the unit-shape frame,
    // perpendicular to the local direction; sin(j*pi*u) is 0 at u=0 and u=1 so
    // the start and target are never moved.
    constexpr int kMaxModes = 8;
    const int EM = cfg.elastic_amp > 0.0
                 ? std::min(std::max(cfg.elastic_modes, 1), kMaxModes) : 0;
    double ecoef[kMaxModes] = {0.0};
    for (int j = 0; j < EM; ++j) {
        std::normal_distribution<double> ej(0.0, cfg.elastic_amp / (j + 1));
        ecoef[j] = ej(rng);
    }

    out.reserve(detail::kNumPoints);
    for (int k = 0; k < detail::kNumPoints; ++k) {
        double bx = s->sx[k], by = s->sy[k];       // unit canonical point
        if (EM > 0) {
            const double u = static_cast<double>(k) / (detail::kNumPoints - 1);
            double disp = 0.0;
            for (int j = 0; j < EM; ++j)
                disp += ecoef[j] * std::sin((j + 1) * M_PI * u);
            const int kp = std::min(k + 1, detail::kNumPoints - 1);
            const int km = std::max(k - 1, 0);
            const double tx = s->sx[kp] - s->sx[km], ty = s->sy[kp] - s->sy[km];
            const double tl = std::hypot(tx, ty);
            if (tl > 1e-9) { bx += disp * (-ty / tl); by += disp * (tx / tl); }
        }
        double ux = bx * D, uy = by * D;
        if (mag > 0.0) {
            ux += mag * (A->sx[k] - B->sx[k]) * D;
            uy += mag * (A->sy[k] - B->sy[k]) * D;
        }
        double rx = ux * c - uy * sn;
        double ry = ux * sn + uy * c;
        if (cfg.position_jitter > 0.0) { rx += jit(rng); ry += jit(rng); }
        out.push_back({x0 + rx, y0 + ry, s->t[k]});
    }
    // land exactly on the target, and normalize timestamps
    out.back().x = x1;
    out.back().y = y1;
    out.front().t = 0.0;
    for (size_t i = 1; i < out.size(); ++i)
        if (out[i].t < out[i - 1].t) out[i].t = out[i - 1].t;
    return out;
}

struct FlickDelta {
    int dx = 0;
    int dy = 0;
};

// One-shot playback of a stroke, sampled incrementally as (dx, dy) mouse
// deltas by wall-clock elapsed time - stands in for pd_controller's per-frame
// output for the first stretch of a freshly-locked target. start()/sample()
// are callback-thread-only; cancel()/active() use an atomic so the main loop
// can cancel an in-progress flick without racing the callback thread.
class FlickPlayback {
public:
    using Clock = std::chrono::steady_clock;

    // errorX/errorY: target - screen-center in model-input space px (the aim
    // reach vector; start is always the origin). movementScaleX/Y: that space
    // -> output px. detectedTargetWidth: unused by warped replay (a stroke is
    // chosen by reach distance, not a Fitts width) - kept for call-site
    // compatibility. Replaces any in-progress playback outright.
    void start(double errorX, double errorY,
               double movementScaleX, double movementScaleY,
               double detectedTargetWidth,
               const config& cfg, uint64_t seed = 0) {
        (void)detectedTargetWidth;
        path_ = generate(0.0, 0.0, errorX, errorY, cfg, seed);
        movementScaleX_ = movementScaleX;
        movementScaleY_ = movementScaleY;
        startTime_ = Clock::now();
        emittedScaledX_ = 0.0;
        emittedScaledY_ = 0.0;
        sampleIndex_ = 0;
        active_.store(path_.size() >= 2, std::memory_order_relaxed);
    }

    void cancel() { active_.store(false, std::memory_order_relaxed); }
    bool active() const { return active_.load(std::memory_order_relaxed); }

    // Interpolates the path at `now`'s elapsed time and returns the incremental
    // (dx, dy) since the last sample, in output px (carrying sub-pixel
    // remainder). nullopt once the full path has been delivered - caller falls
    // back to pd_controller's own output from then on.
    std::optional<FlickDelta> sample(Clock::time_point now) {
        if (!active_.load(std::memory_order_relaxed)) return std::nullopt;

        const double elapsedMs =
            std::chrono::duration<double, std::milli>(now - startTime_).count();

        if (elapsedMs >= path_.back().t) {
            active_.store(false, std::memory_order_relaxed);
            return emit(path_.back().x, path_.back().y);
        }

        while (sampleIndex_ + 1 < path_.size() && path_[sampleIndex_ + 1].t <= elapsedMs) {
            ++sampleIndex_;
        }

        double x = path_.back().x;
        double y = path_.back().y;
        if (sampleIndex_ + 1 < path_.size()) {
            const auto& a = path_[sampleIndex_];
            const auto& b = path_[sampleIndex_ + 1];
            const double span = b.t - a.t;
            const double frac = (span > 0.0) ? (elapsedMs - a.t) / span : 0.0;
            x = a.x + (b.x - a.x) * frac;
            y = a.y + (b.y - a.y) * frac;
        }

        return emit(x, y);
    }

private:
    // targetScaled - emittedScaled carries rounding remainder into the next
    // emit, so many small steps sum to the same total as one big one.
    FlickDelta emit(double x, double y) {
        const double targetScaledX = x * movementScaleX_;
        const double targetScaledY = y * movementScaleY_;
        const int dx = static_cast<int>(std::lround(targetScaledX - emittedScaledX_));
        const int dy = static_cast<int>(std::lround(targetScaledY - emittedScaledY_));
        emittedScaledX_ += dx;
        emittedScaledY_ += dy;
        return FlickDelta{dx, dy};
    }

    std::vector<trajectory_point> path_;
    double movementScaleX_ = 1.0, movementScaleY_ = 1.0;
    Clock::time_point startTime_{};
    double emittedScaledX_ = 0.0, emittedScaledY_ = 0.0;
    size_t sampleIndex_ = 0;
    std::atomic<bool> active_{false};
};

}  // namespace warped_replay
