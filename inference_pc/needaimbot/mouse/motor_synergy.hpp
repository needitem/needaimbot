#pragma once

// Host-side motor-synergy trajectory generator: draws the per-call random
// shape (Fitts' law timing, primary submovement, corrections, curvature,
// tremor) with a CPU RNG, integrates the one genuinely sequential piece (the
// Ornstein-Uhlenbeck path noise, each step depends on the last) here, and
// hands the rest - the embarrassingly-parallel per-sample position/tremor/SDN
// math - to the GPU kernel in needaimbot/cuda/motor_synergy.cuh. This mirrors
// how pd_controller.hpp assembles config host-side for the GPU aim
// controller in pd_controller.cuh: tuning/RNG/config assembly here, the
// actual per-element math on the GPU.

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numbers>
#include <numeric>
#include <chrono>
#include <optional>
#include <iostream>
#include <atomic>

#include "motor_synergy.h"
#include "json.hpp"

namespace motor_synergy {

// Sample cap shared with gpu::TrajectoryGenerator's default capacity - a
// stroke's sample count (~total_t / sample_dt_mean) stays well under this
// even for cross-screen distances, but generate() truncates defensively if a
// pathological config ever exceeds it.
constexpr int kMaxSamples = 512;

// Floor for FlickPlayback's per-target detected width override (model-input
// space px) - a near-zero detection box would blow up the Fitts index
// (log2(distance/width + 1)) into a runaway movement time, so this bounds
// how "small" a target is allowed to make the flick.
constexpr double kMinTargetWidth = 4.0;

struct trajectory_point {
    double x, y;
    double t;
};

struct config {
    double fitts_a = 50.0;
    double fitts_b = 150.0;
    double target_width = 20.0;

    double undershoot_min = 0.92;
    double undershoot_max = 0.97;
    double peak_time_ratio = 0.35;
    double primary_sigma_min = 0.18;
    double primary_sigma_max = 0.28;

    double overshoot_prob = 0.15;
    double overshoot_min = 1.02;
    double overshoot_max = 1.08;
    double correction_sigma_min = 0.12;
    double correction_sigma_max = 0.20;
    double second_correction_prob = 0.25;

    double curvature_scale = 0.025;

    double ou_theta = 3.5;
    double ou_sigma = 1.2;

    double tremor_freq_min = 8.0;
    double tremor_freq_max = 12.0;
    double tremor_amp_min = 0.15;
    double tremor_amp_max = 0.55;

    double sdn_k = 0.04;

    double sample_dt_mean = 7.8;
    double gamma_shape = 3.5;

    // JSON keys are "flick_"-prefixed so they sit alongside pd_controller's
    // "aim_"-prefixed keys in the same config file without colliding.
    void load(const nlohmann::json& j) {
        if (j.contains("flick_fitts_a")) fitts_a = j["flick_fitts_a"];
        if (j.contains("flick_fitts_b")) fitts_b = j["flick_fitts_b"];
        if (j.contains("flick_target_width")) target_width = j["flick_target_width"];

        if (j.contains("flick_undershoot_min")) undershoot_min = j["flick_undershoot_min"];
        if (j.contains("flick_undershoot_max")) undershoot_max = j["flick_undershoot_max"];
        if (j.contains("flick_peak_time_ratio")) peak_time_ratio = j["flick_peak_time_ratio"];
        if (j.contains("flick_primary_sigma_min")) primary_sigma_min = j["flick_primary_sigma_min"];
        if (j.contains("flick_primary_sigma_max")) primary_sigma_max = j["flick_primary_sigma_max"];

        if (j.contains("flick_overshoot_prob")) overshoot_prob = j["flick_overshoot_prob"];
        if (j.contains("flick_overshoot_min")) overshoot_min = j["flick_overshoot_min"];
        if (j.contains("flick_overshoot_max")) overshoot_max = j["flick_overshoot_max"];
        if (j.contains("flick_correction_sigma_min")) correction_sigma_min = j["flick_correction_sigma_min"];
        if (j.contains("flick_correction_sigma_max")) correction_sigma_max = j["flick_correction_sigma_max"];
        if (j.contains("flick_second_correction_prob")) second_correction_prob = j["flick_second_correction_prob"];

        if (j.contains("flick_curvature_scale")) curvature_scale = j["flick_curvature_scale"];

        if (j.contains("flick_ou_theta")) ou_theta = j["flick_ou_theta"];
        if (j.contains("flick_ou_sigma")) ou_sigma = j["flick_ou_sigma"];

        if (j.contains("flick_tremor_freq_min")) tremor_freq_min = j["flick_tremor_freq_min"];
        if (j.contains("flick_tremor_freq_max")) tremor_freq_max = j["flick_tremor_freq_max"];
        if (j.contains("flick_tremor_amp_min")) tremor_amp_min = j["flick_tremor_amp_min"];
        if (j.contains("flick_tremor_amp_max")) tremor_amp_max = j["flick_tremor_amp_max"];

        if (j.contains("flick_sdn_k")) sdn_k = j["flick_sdn_k"];

        if (j.contains("flick_sample_dt_mean")) sample_dt_mean = j["flick_sample_dt_mean"];
        if (j.contains("flick_gamma_shape")) gamma_shape = j["flick_gamma_shape"];
    }

    void save(nlohmann::json& j) const {
        j["flick_fitts_a"] = fitts_a;
        j["flick_fitts_b"] = fitts_b;
        j["flick_target_width"] = target_width;

        j["flick_undershoot_min"] = undershoot_min;
        j["flick_undershoot_max"] = undershoot_max;
        j["flick_peak_time_ratio"] = peak_time_ratio;
        j["flick_primary_sigma_min"] = primary_sigma_min;
        j["flick_primary_sigma_max"] = primary_sigma_max;

        j["flick_overshoot_prob"] = overshoot_prob;
        j["flick_overshoot_min"] = overshoot_min;
        j["flick_overshoot_max"] = overshoot_max;
        j["flick_correction_sigma_min"] = correction_sigma_min;
        j["flick_correction_sigma_max"] = correction_sigma_max;
        j["flick_second_correction_prob"] = second_correction_prob;

        j["flick_curvature_scale"] = curvature_scale;

        j["flick_ou_theta"] = ou_theta;
        j["flick_ou_sigma"] = ou_sigma;

        j["flick_tremor_freq_min"] = tremor_freq_min;
        j["flick_tremor_freq_max"] = tremor_freq_max;
        j["flick_tremor_amp_min"] = tremor_amp_min;
        j["flick_tremor_amp_max"] = tremor_amp_max;

        j["flick_sdn_k"] = sdn_k;

        j["flick_sample_dt_mean"] = sample_dt_mean;
        j["flick_gamma_shape"] = gamma_shape;
    }

    void print() const {
        std::cout << "[Config] Flick target width: " << target_width
                  << " (smaller -> longer, more corrected movements)" << std::endl;
        std::cout << "[Config] Flick overshoot prob: " << overshoot_prob
                  << ", curvature scale: " << curvature_scale
                  << ", OU drift sigma: " << ou_sigma << std::endl;
        std::cout << "[Config] Flick tremor amp max: " << tremor_amp_max
                  << ", SDN k: " << sdn_k
                  << ", sample dt mean: " << sample_dt_mean << "ms" << std::endl;
    }
};

struct metrics {
    double movement_time;
    double path_length;
    double straight_distance;
    double path_efficiency;
    double peak_speed;
    double time_to_peak;
    int num_submovements;
    double endpoint_error;
    double fitts_predicted_mt;
};

namespace detail {

// vertical movements produce more curvature due to wrist/forearm geometry
inline double direction_factor(double angle) {
    double sa = std::abs(std::sin(angle));
    double ca = std::abs(std::cos(angle));
    return 0.5 + 0.8 * sa - 0.15 * ca;
}

}  // namespace detail

inline std::vector<trajectory_point> generate(
    double x0, double y0, double x1, double y1,
    const config& cfg = {}, uint64_t seed = 0)
{
    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    auto uniform = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };
    auto normal = [&](double m, double s) {
        return std::normal_distribution<double>(m, s)(rng);
    };
    auto gamma_dist = [&](double shape, double scale) {
        return std::gamma_distribution<double>(shape, scale)(rng);
    };

    double dx = x1 - x0, dy = y1 - y0;
    double distance = std::hypot(dx, dy);
    double direction = std::atan2(dy, dx);

    if (distance < 1.0)
        return {{x0, y0, 0.0}, {x1, y1, 50.0}};

    double tx = dx / distance, ty = dy / distance;
    double nx = -ty, ny = tx;

    double id = std::log2(distance / cfg.target_width + 1.0);
    double mt = (cfg.fitts_a + cfg.fitts_b * id) * std::exp(normal(0.0, 0.08));
    mt = std::max(mt, 80.0);

    bool overshoot = uniform(0.0, 1.0) < cfg.overshoot_prob;
    double reach = overshoot
        ? uniform(cfg.overshoot_min, cfg.overshoot_max)
        : uniform(cfg.undershoot_min, cfg.undershoot_max);

    double primary_D = distance * reach;
    double primary_sigma = uniform(cfg.primary_sigma_min, cfg.primary_sigma_max);

    // mu derived from mode = exp(mu - sigma^2) so that peak velocity lands at peak_t
    double peak_t = mt * uniform(cfg.peak_time_ratio - 0.03, cfg.peak_time_ratio + 0.03);
    double primary_mu = std::log(peak_t) + primary_sigma * primary_sigma;

    gpu::TrajectoryParams params;
    params.x0 = static_cast<float>(x0);
    params.y0 = static_cast<float>(y0);
    params.tx = static_cast<float>(tx);
    params.ty = static_cast<float>(ty);
    params.nx = static_cast<float>(nx);
    params.ny = static_cast<float>(ny);
    params.primary_D = static_cast<float>(primary_D);
    params.primary_mu = static_cast<float>(primary_mu);
    params.primary_sigma = static_cast<float>(primary_sigma);
    params.sdn_k = static_cast<float>(cfg.sdn_k);
    params.seed = seed ? seed : rng();

    double remaining = distance - primary_D;
    if (std::abs(remaining) > 0.5) {
        double dir = remaining > 0.0 ? 1.0 : -1.0;
        double cD = std::abs(remaining) * uniform(0.88, 1.02);
        double cS = uniform(cfg.correction_sigma_min, cfg.correction_sigma_max);
        double cPeak = mt * uniform(0.12, 0.18);
        gpu::Correction corr;
        corr.D = static_cast<float>(cD);
        corr.t0 = static_cast<float>(mt * uniform(0.55, 0.68));
        corr.sigma = static_cast<float>(cS);
        corr.mu = static_cast<float>(std::log(cPeak) + cS * cS);
        corr.dir_x = static_cast<float>(tx * dir);
        corr.dir_y = static_cast<float>(ty * dir);
        params.corrections[params.num_corrections++] = corr;

        double left = remaining - cD * dir;
        if (std::abs(left) > 0.3 && uniform(0.0, 1.0) < cfg.second_correction_prob
            && params.num_corrections < gpu::kMaxCorrections) {
            double d2 = left > 0.0 ? 1.0 : -1.0;
            double cD2 = std::abs(left) * uniform(0.85, 1.05);
            double cS2 = uniform(0.10, 0.16);
            double cP2 = mt * uniform(0.08, 0.12);
            gpu::Correction corr2;
            corr2.D = static_cast<float>(cD2);
            corr2.t0 = static_cast<float>(mt * uniform(0.78, 0.88));
            corr2.sigma = static_cast<float>(cS2);
            corr2.mu = static_cast<float>(std::log(cP2) + cS2 * cS2);
            corr2.dir_x = static_cast<float>(tx * d2);
            corr2.dir_y = static_cast<float>(ty * d2);
            params.corrections[params.num_corrections++] = corr2;
        }
    }

    params.curv_amp = static_cast<float>(distance * cfg.curvature_scale
        * detail::direction_factor(direction) * normal(0.0, 1.0));

    params.tremor_freq = static_cast<float>(uniform(cfg.tremor_freq_min, cfg.tremor_freq_max));
    params.tremor_amp = static_cast<float>(uniform(cfg.tremor_amp_min, cfg.tremor_amp_max));
    params.tremor_phase_x = static_cast<float>(uniform(0.0, 2.0 * std::numbers::pi));
    params.tremor_phase_y = static_cast<float>(uniform(0.0, 2.0 * std::numbers::pi));

    double total_t = mt * 1.15;
    double g_scale = cfg.sample_dt_mean / cfg.gamma_shape;

    std::vector<double> times = {0.0};
    for (double t = 0.0; t < total_t && static_cast<int>(times.size()) < kMaxSamples;) {
        double dt = std::clamp(gamma_dist(cfg.gamma_shape, g_scale), 2.0, 25.0);
        t += dt;
        if (t <= total_t + 15.0) times.push_back(t);
    }

    // The OU path is a sequential recurrence (each step needs the last), so
    // it's integrated here rather than per-thread on the GPU.
    const int num_samples = static_cast<int>(times.size());
    std::vector<float> times_f(num_samples);
    std::vector<float> ou_x(num_samples), ou_y(num_samples);
    double oux = 0.0, ouy = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        double dt_ms = (i > 0) ? (times[i] - times[i - 1]) : cfg.sample_dt_mean;
        double dt_s = dt_ms / 1000.0;
        oux += -cfg.ou_theta * oux * dt_s + cfg.ou_sigma * std::sqrt(dt_s) * normal(0.0, 1.0);
        ouy += -cfg.ou_theta * ouy * dt_s + cfg.ou_sigma * std::sqrt(dt_s) * normal(0.0, 1.0);
        times_f[i] = static_cast<float>(times[i]);
        ou_x[i] = static_cast<float>(oux);
        ou_y[i] = static_cast<float>(ouy);
    }

    // One persistent device-buffer generator per calling thread - avoids a
    // cudaMalloc/cudaFree round trip on every movement.
    thread_local gpu::TrajectoryGenerator generator(kMaxSamples);

    std::vector<gpu::TrajectoryPoint> gpu_points(num_samples);
    cudaError_t err = generator.compute(
        params, times_f.data(), ou_x.data(), ou_y.data(),
        num_samples, gpu_points.data());
    if (err != cudaSuccess) {
        // GPU path failed (no device / OOM) - trajectory generation is a
        // fire-and-forget input-humanization aid, not safety-critical, so
        // fall back to the straight-line endpoints rather than throwing.
        return {{x0, y0, 0.0}, {x1, y1, total_t}};
    }

    std::vector<trajectory_point> result(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        result[i] = {gpu_points[i].x, gpu_points[i].y, gpu_points[i].t};
    }
    return result;
}

inline metrics compute_metrics(
    const std::vector<trajectory_point>& path,
    double target_x, double target_y,
    double target_width, double straight_dist)
{
    metrics m{};
    if (path.size() < 2) return m;

    m.movement_time = path.back().t - path.front().t;
    m.straight_distance = straight_dist;

    double max_speed = 0.0;
    m.path_length = 0.0;
    std::vector<double> speeds(path.size(), 0.0);

    for (size_t i = 1; i < path.size(); ++i) {
        double dx = path[i].x - path[i - 1].x;
        double dy = path[i].y - path[i - 1].y;
        double dt = path[i].t - path[i - 1].t;
        double seg = std::hypot(dx, dy);
        m.path_length += seg;
        double spd = (dt > 0.0) ? seg / dt : 0.0;
        speeds[i] = spd;
        if (spd > max_speed) { max_speed = spd; m.time_to_peak = path[i].t; }
    }

    m.peak_speed = max_speed;
    m.path_efficiency = (m.path_length > 0.0) ? m.straight_distance / m.path_length : 1.0;

    // peaks above 15% of max in the speed signal → sub-movement count
    double threshold = max_speed * 0.15;
    int peaks = 0;
    for (size_t i = 2; i + 1 < speeds.size(); ++i) {
        if (speeds[i] > threshold && speeds[i] > speeds[i - 1] && speeds[i] > speeds[i + 1])
            ++peaks;
    }
    m.num_submovements = std::max(peaks, 1);

    m.endpoint_error = std::hypot(path.back().x - target_x, path.back().y - target_y);

    double id = std::log2(straight_dist / target_width + 1.0);
    m.fitts_predicted_mt = 50.0 + 150.0 * id;

    return m;
}

struct FlickDelta {
    int dx = 0;
    int dy = 0;
};

// One-shot playback of a motor_synergy trajectory, sampled incrementally as
// (dx, dy) mouse deltas by wall-clock elapsed time. Meant to stand in for
// pd_controller's own per-frame output for the first stretch of a
// freshly-locked target (see stage2FinalizeKernel's freshAcquire flag and
// simple_main.cpp's inference callback, which does the hand-off).
// start()/sample() are callback-thread-only (not thread-safe with each
// other), but cancel()/active() use an atomic flag so a second thread (e.g.
// simple_main.cpp's main loop, which keeps polling aim-button state even
// while the inference callback isn't firing) can cancel an in-progress flick
// without racing the callback thread.
class FlickPlayback {
public:
    using Clock = std::chrono::steady_clock;

    // errorX/errorY: target - screen-center in model-input space (same units
    // pd_controller's own error_x/error_y use, e.g. InferenceResult::errorX/Y).
    // movementScaleX/Y: converts that space to output px (InferenceResult::
    // movementScaleX/Y). detectedTargetWidth: the actual acquired target's
    // size in that same model-input space (e.g. averaged from its bbox) -
    // overrides cfg.target_width for just this flick so Fitts' law timing
    // scales with the REAL target instead of a fixed guess (a small target
    // should take a longer, more-corrected movement than a large one, same
    // as a human). <= 0 keeps cfg.target_width as-is (e.g. no box available).
    // Floored so a degenerate near-zero box can't blow up the Fitts index
    // (id = log2(distance/width + 1)) into a runaway movement time.
    // Replaces any in-progress playback outright - a fresh acquire always
    // wins immediately rather than waiting for the old flick to finish.
    void start(double errorX, double errorY,
               double movementScaleX, double movementScaleY,
               double detectedTargetWidth,
               const config& cfg, uint64_t seed = 0) {
        config effectiveCfg = cfg;
        if (detectedTargetWidth > 0.0) {
            effectiveCfg.target_width = std::max(detectedTargetWidth, kMinTargetWidth);
        }
        path_ = generate(0.0, 0.0, errorX, errorY, effectiveCfg, seed);
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

    // Interpolates the path at `now`'s elapsed time and returns the
    // incremental (dx, dy) since the last sample, in output px (carrying
    // sub-pixel remainder the same way pd_controller's emitMouseDelta does).
    // Returns nullopt once playback has finished delivering the full path -
    // the caller should fall back to pd_controller's own output from then on.
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
    // targetScaled - emittedScaled naturally carries any rounding remainder
    // from the previous emit into the next one, so many small steps sum to
    // the same total as one big one (no truncation drift).
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

}  // namespace motor_synergy
