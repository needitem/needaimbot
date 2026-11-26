#!/usr/bin/env python3
"""Verify combined optimal parameters from 5 parallel parameter sweeps.

Runs all trajectories with 5 seeds comparing:
  - baseline:  PD + One Euro + dead-time comp (no novel algorithms)
  - sem_pags:  baseline + SEM + PAGS (current defaults)
  - optimal:   all optimized parameters + adaptive kp
"""

import math
import json
import sys
import os
import copy
import dataclasses

# Import the simulation module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

# ---------------------------------------------------------------------------
# Optimal constants (override sim.py K_* values for optimal config)
# ---------------------------------------------------------------------------
OPTIMAL_CONSTANTS = {
    "K_FF_V_EMA": 0.05,
    "K_SEM_RATE": 0.01,
    "K_SEM_DECAY": 0.97,
    "K_PAGS_OSC_MULT": 0.4,
    "K_PAGS_OSC_THRESHOLD": 0.4,
}

# Adaptive kp constants
ADAPTIVE_KP_LARGE_THRESHOLD = 15.0
ADAPTIVE_KP_SMALL_THRESHOLD = 4.0
ADAPTIVE_KP_LARGE_MULT = 1.3
ADAPTIVE_KP_SMALL_MULT = 0.7


def config_optimal():
    """Optimal config: SEM + PAGS + lead terms + tuned One Euro."""
    c = sim.AimConfig()
    c.sem_enabled = 1.0
    c.pags_enabled = 1.0
    # Optimized One Euro
    c.oneeuro_min_cutoff = 0.08
    c.oneeuro_beta = 0.001
    # Optimized lead terms
    c.deadtime_frames = 0.5
    c.ff_gain = 0.5
    c.predict_frames = 1.5
    return c


def compute_aim_movement_optimal(raw_cx, raw_cy, screen_cx, screen_cy,
                                  scale_x, scale_y, cfg, s, target_class,
                                  detection_confidence):
    """Port of compute_aim_movement with optimal constants and adaptive kp."""
    fresh_track = (s.has_track == 0)

    class_changed = (not fresh_track and s.prev_class >= 0
                     and target_class != s.prev_class)
    if class_changed:
        s.prev_raw_x = raw_cx
        s.prev_raw_y = raw_cy
    s.prev_class = target_class

    inf_x, inf_y = 0.0, 0.0
    use_comp = (cfg.inflight_comp > 0 and cfg.deadtime_frames > 0)
    if use_comp:
        inf_x, inf_y = sim.inflight_sum(s, cfg.deadtime_frames)

    target_cx, target_cy = raw_cx, raw_cy
    if cfg.oneeuro_enabled != 0.0:
        in_x, in_y = raw_cx, raw_cy
        carry_x, carry_y = s.filt_x, s.filt_y
        if s.has_track:
            ad = sim.K_DERIV_ALPHA
            de_x = in_x - s.filt_x
            s.dfilt_x = ad * de_x + (1 - ad) * s.dfilt_x
            cutoff_x = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_x)
            ax = sim.one_euro_alpha(cutoff_x)
            s.filt_x = ax * in_x + (1 - ax) * carry_x

            de_y = in_y - s.filt_y
            s.dfilt_y = ad * de_y + (1 - ad) * s.dfilt_y
            cutoff_y = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_y)
            ay = sim.one_euro_alpha(cutoff_y)
            s.filt_y = ay * in_y + (1 - ay) * carry_y
        else:
            s.filt_x = in_x
            s.filt_y = in_y
            s.dfilt_x = 0.0
            s.dfilt_y = 0.0
        target_cx = s.filt_x
        target_cy = s.filt_y

    # JCAD displacement
    if cfg.jcad_enabled != 0.0:
        if fresh_track:
            s.jcad_ew_mean = 0.0
            s.jcad_ew_var = 0.0
        else:
            jdx = raw_cx - s.prev_raw_x
            jdy = raw_cy - s.prev_raw_y
            jcad_disp = math.sqrt(jdx * jdx + jdy * jdy)
            ja = sim.K_JCAD_ALPHA
            s.jcad_ew_mean = ja * jcad_disp + (1 - ja) * s.jcad_ew_mean
            jdiff = jcad_disp - s.jcad_ew_mean
            s.jcad_ew_var = ja * jdiff * jdiff + (1 - ja) * s.jcad_ew_var

    # Ego-corrected velocity (use OPTIMAL kFfVEma)
    K_FF_V_EMA_OPT = OPTIMAL_CONSTANTS["K_FF_V_EMA"]
    lead_active = (cfg.ff_gain > 0 or cfg.predict_frames > 0)
    if lead_active:
        med_x, med_y = sim.apply_median_prefilter(raw_cx, raw_cy, s)
        if s.has_track:
            ax_v, ay_v = sim.inflight_at(s, sim.K_FF_EGO_LAG)
            max_drift = 60.0
            dvx = (med_x - s.prev_raw_x) + (ax_v / scale_x if scale_x != 0 else 0)
            dvy = (med_y - s.prev_raw_y) + (ay_v / scale_y if scale_y != 0 else 0)
            dvx = min(max(dvx, -max_drift), max_drift)
            dvy = min(max(dvy, -max_drift), max_drift)
            a = K_FF_V_EMA_OPT
            s.vel_x = (1 - a) * s.vel_x + a * dvx
            s.vel_y = (1 - a) * s.vel_y + a * dvy
        else:
            s.vel_x = s.vel_y = 0.0
            s.med_initialized = 0
            s.accel_x = s.accel_y = 0.0
            s.prev_vel_x = s.prev_vel_y = 0.0

        a_ema = sim.K_ACCEL_EMA
        dax = s.vel_x - s.prev_vel_x
        day = s.vel_y - s.prev_vel_y
        max_accel = 15.0
        s.accel_x = (1 - a_ema) * s.accel_x + a_ema * min(max(dax, -max_accel), max_accel)
        s.accel_y = (1 - a_ema) * s.accel_y + a_ema * min(max(day, -max_accel), max_accel)
        s.prev_vel_x = s.vel_x
        s.prev_vel_y = s.vel_y
        s.prev_raw_x = med_x
        s.prev_raw_y = med_y

    s.has_track = 1

    error_x = target_cx - screen_cx - (cfg.shoot_offset_x / scale_x if scale_x else 0)
    error_y = target_cy - screen_cy - (cfg.shoot_offset_y / scale_y if scale_y else 0)

    if use_comp:
        if scale_x != 0:
            error_x -= cfg.inflight_comp * inf_x / scale_x
        if scale_y != 0:
            error_y -= cfg.inflight_comp * inf_y / scale_y

    # JCAD deadband
    if cfg.jcad_enabled != 0.0:
        mean_sq = s.jcad_ew_mean ** 2
        J = s.jcad_ew_var / (mean_sq + 1e-6)
        sat_J = min(J / sim.K_JCAD_JITTER_THRESHOLD, 1.0)
        db = sim.K_JCAD_BASE_DB + sim.K_JCAD_JITTER_DB * sat_J
        err_mag = math.sqrt(error_x ** 2 + error_y ** 2)
        if err_mag < db:
            error_x = error_y = 0.0

    # predict_frames setpoint shift
    if cfg.predict_frames > 0:
        g = sim.lead_gate(s, error_x, error_y)
        t = cfg.predict_frames
        error_x += g * (t * s.vel_x + 0.5 * t * t * s.accel_x)
        error_y += g * (t * s.vel_y + 0.5 * t * t * s.accel_y)

    # D term
    max_derr = 150.0
    de_x = min(max(error_x - s.prev_err_x, -max_derr), max_derr)
    de_y = min(max(error_y - s.prev_err_y, -max_derr), max_derr)
    if fresh_track:
        s.derr_x = s.derr_y = 0.0
    else:
        s.derr_x = 0.6 * de_x + 0.4 * s.derr_x
        s.derr_y = 0.6 * de_y + 0.4 * s.derr_y
    s.prev_err_x = error_x
    s.prev_err_y = error_y

    # ff output term
    ff_x, ff_y = 0.0, 0.0
    if cfg.ff_gain > 0:
        g = sim.lead_gate(s, error_x, error_y)
        ff_x = cfg.ff_gain * g * s.vel_x
        ff_y = cfg.ff_gain * g * s.vel_y

    # --- Adaptive kp: scale gain based on error magnitude ---
    err_mag = math.sqrt(error_x ** 2 + error_y ** 2)
    kp_scale = 1.0
    if err_mag > ADAPTIVE_KP_LARGE_THRESHOLD:
        kp_scale = ADAPTIVE_KP_LARGE_MULT   # ACQUIRE boost
    elif err_mag < ADAPTIVE_KP_SMALL_THRESHOLD:
        kp_scale = ADAPTIVE_KP_SMALL_MULT   # Fine-track damping

    movement_x = (sim.nonlinear_p_move(error_x, cfg.kp_x * kp_scale, cfg.p_softness_x,
                                        sim.K_FINE_TRACK_BOOST, sim.K_FINE_TRACK_RANGE)
                  + cfg.kd_x * s.derr_x + ff_x) * scale_x
    movement_y = (sim.nonlinear_p_move(error_y, cfg.kp_y * kp_scale, cfg.p_softness_y,
                                        sim.K_FINE_TRACK_BOOST, sim.K_FINE_TRACK_RANGE)
                  + cfg.kd_y * s.derr_y + ff_y) * scale_y

    # SEM (use optimal constants)
    K_SEM_RATE_OPT = OPTIMAL_CONSTANTS["K_SEM_RATE"]
    K_SEM_DECAY_OPT = OPTIMAL_CONSTANTS["K_SEM_DECAY"]
    if cfg.sem_enabled != 0.0:
        if fresh_track:
            s.sem_x = s.sem_y = 0.0
        s.sem_x *= K_SEM_DECAY_OPT
        s.sem_y *= K_SEM_DECAY_OPT
        if error_x * s.sem_x < 0:
            s.sem_x = 0.0
        s.sem_x += K_SEM_RATE_OPT * error_x
        s.sem_x = min(max(s.sem_x, -sim.K_SEM_MAX), sim.K_SEM_MAX)
        if error_y * s.sem_y < 0:
            s.sem_y = 0.0
        s.sem_y += K_SEM_RATE_OPT * error_y
        s.sem_y = min(max(s.sem_y, -sim.K_SEM_MAX), sim.K_SEM_MAX)
        movement_x += s.sem_x * scale_x
        movement_y += s.sem_y * scale_y

    # PAGS (use optimal constants)
    K_PAGS_OSC_MULT_OPT = OPTIMAL_CONSTANTS["K_PAGS_OSC_MULT"]
    K_PAGS_OSC_THRESHOLD_OPT = OPTIMAL_CONSTANTS["K_PAGS_OSC_THRESHOLD"]
    if cfg.pags_enabled != 0.0:
        if fresh_track:
            s.pags_flip_rate_x = s.pags_flip_rate_y = 0.0
            s.pags_prev_sign_x = s.pags_prev_sign_y = 0
            s.pags_gain_mult = 1.0
        sign_x = 1 if error_x > 0 else (-1 if error_x < 0 else 0)
        sign_y = 1 if error_y > 0 else (-1 if error_y < 0 else 0)
        flip_x = 1.0 if (s.pags_prev_sign_x != 0 and sign_x != 0
                         and sign_x != s.pags_prev_sign_x) else 0.0
        flip_y = 1.0 if (s.pags_prev_sign_y != 0 and sign_y != 0
                         and sign_y != s.pags_prev_sign_y) else 0.0
        s.pags_flip_rate_x = sim.K_PAGS_ALPHA * flip_x + (1 - sim.K_PAGS_ALPHA) * s.pags_flip_rate_x
        s.pags_flip_rate_y = sim.K_PAGS_ALPHA * flip_y + (1 - sim.K_PAGS_ALPHA) * s.pags_flip_rate_y
        if sign_x != 0: s.pags_prev_sign_x = sign_x
        if sign_y != 0: s.pags_prev_sign_y = sign_y
        err_mag_pags = math.sqrt(error_x ** 2 + error_y ** 2)
        flip_rate = max(s.pags_flip_rate_x, s.pags_flip_rate_y)
        if err_mag_pags > sim.K_PAGS_ACQUIRE_THRESHOLD:
            target_mult = sim.K_PAGS_ACQUIRE_MULT
        elif flip_rate > K_PAGS_OSC_THRESHOLD_OPT:
            target_mult = K_PAGS_OSC_MULT_OPT
        else:
            target_mult = 1.0
        s.pags_gain_mult = 0.3 * target_mult + 0.7 * s.pags_gain_mult
        movement_x *= s.pags_gain_mult
        movement_y *= s.pags_gain_mult

    movement_x, movement_y = sim.clamp_max_step(movement_x, movement_y, cfg.max_step)
    movement_y *= cfg.aim_y_scale
    movement_x, movement_y = sim.clamp_max_step(movement_x, movement_y, cfg.max_step)

    # CGA
    if cfg.cga_enabled != 0.0:
        speed = math.sqrt(s.vel_x ** 2 + s.vel_y ** 2)
        speed_bypass = speed / (speed + sim.K_CGA_SPEED_THRESHOLD + 1e-7)
        conf_gate = sim.gpa_smoothstep(sim.K_CGA_CONF_LOW, sim.K_CGA_CONF_HIGH,
                                        detection_confidence)
        gate = max(conf_gate, speed_bypass)
        movement_x *= gate
        movement_y *= gate

    if cfg.aim_output_enabled == 0.0:
        s.residual_x = s.residual_y = 0.0
        sim.push_inflight(s, 0.0, 0.0)
        return 0, 0

    out_dx, s.residual_x = sim.emit_mouse_delta(movement_x, s.residual_x)
    out_dy, s.residual_y = sim.emit_mouse_delta(movement_y, s.residual_y)
    sim.push_inflight(s, float(out_dx), float(out_dy))
    return out_dx, out_dy


def run_sim_optimal(traj_name, seed=42, n_frames=300, deadtime_delay=1):
    """Run simulation with optimal parameters and adaptive kp."""
    cfg = config_optimal()
    state = sim.AimState()
    camera = sim.FPSCamera()
    noise = sim.NoiseModel(seed=seed)
    traj_fn = sim.TRAJECTORIES[traj_name]
    result = sim.SimResult(traj_name, "optimal", seed, n_frames)

    model_size = 320
    screen_cx = model_size / 2.0
    screen_cy = model_size / 2.0
    scale_x = 1.0
    scale_y = 1.0
    target_class = 0
    prev_sign_x = 0
    prev_sign_y = 0
    detection_buffer = []

    for frame in range(n_frames):
        world_x, world_y = traj_fn(frame)
        det_x, det_y = camera.world_to_screen(world_x, world_y)
        jx, jy = noise.jitter()
        det_x += jx
        det_y += jy
        conf = noise.confidence()
        target_class, flip_offset = noise.class_flip(target_class)
        det_y += flip_offset
        missed = noise.is_missed()
        detection_buffer.append((det_x, det_y, conf, target_class, missed))

        if len(detection_buffer) > deadtime_delay:
            d = detection_buffer[-1 - deadtime_delay]
            d_x, d_y, d_conf, d_class, d_missed = d
            if d_missed:
                result.misses += 1
                state.frames_since_seen += 1
                sim.push_inflight(state, 0.0, 0.0)
                dx, dy = 0, 0
            else:
                state.frames_since_seen = 0
                dx, dy = compute_aim_movement_optimal(
                    d_x, d_y, screen_cx, screen_cy,
                    scale_x, scale_y, cfg, state,
                    d_class, d_conf)
        else:
            dx, dy = 0, 0

        camera.apply_mouse(dx, dy)
        result.total_movement += math.sqrt(dx * dx + dy * dy)
        actual_x, actual_y = camera.world_to_screen(world_x, world_y)
        err = math.sqrt((actual_x - screen_cx) ** 2 + (actual_y - screen_cy) ** 2)
        result.errors.append(err)

        cur_sign_x = 1 if (actual_x - screen_cx) > 0 else -1
        cur_sign_y = 1 if (actual_y - screen_cy) > 0 else -1
        if prev_sign_x != 0 and cur_sign_x != prev_sign_x:
            result.sign_flips_x += 1
        if prev_sign_y != 0 and cur_sign_y != prev_sign_y:
            result.sign_flips_y += 1
        prev_sign_x = cur_sign_x
        prev_sign_y = cur_sign_y

    return result


def main():
    seeds = [42, 137, 256, 999, 1337]
    n_frames = 300

    configs_to_run = {
        "baseline": lambda tn, s: sim.run_sim(tn, "baseline", seed=s, n_frames=n_frames),
        "sem_pags": lambda tn, s: sim.run_sim(tn, "sem_pags", seed=s, n_frames=n_frames),
        "optimal":  lambda tn, s: run_sim_optimal(tn, seed=s, n_frames=n_frames),
    }

    print("=" * 80)
    print("OPTIMAL PARAMETER VERIFICATION")
    print("=" * 80)
    print(f"Frames: {n_frames} | Seeds: {seeds} | Dead-time: 1 frame")
    print()
    print("Optimal parameters under test:")
    print("  One Euro: min_cutoff=0.08, beta=0.001")
    print("  SEM: rate=0.01, decay=0.97")
    print("  PAGS: osc_mult=0.4, osc_threshold=0.4")
    print("  Lead: deadtime=0.5, ff_gain=0.5, predict=1.5")
    print("  Velocity EMA: 0.05")
    print("  Adaptive kp: large>15px -> 1.3x, small<4px -> 0.7x")
    print()

    all_results = {}

    header = f"{'Trajectory':<16} {'Config':<12} {'MAE':>7} {'RMS':>7} {'P95':>7} {'Settle':>7} {'Osc':>5}"
    print(header)
    print("-" * len(header))

    for traj_name in sim.TRAJECTORIES:
        for cfg_name, run_fn in configs_to_run.items():
            seed_results = []
            for seed in seeds:
                r = run_fn(traj_name, seed)
                seed_results.append(r)

            avg_mae = sum(r.mae for r in seed_results) / len(seed_results)
            avg_rms = sum(r.rms for r in seed_results) / len(seed_results)
            avg_p95 = sum(r.p95 for r in seed_results) / len(seed_results)
            avg_settle = sum(r.settle_frame for r in seed_results) / len(seed_results)
            avg_osc = sum(r.oscillation_count for r in seed_results) / len(seed_results)

            key = f"{traj_name}|{cfg_name}"
            all_results[key] = {
                "trajectory": traj_name,
                "config": cfg_name,
                "mae": round(avg_mae, 3),
                "rms": round(avg_rms, 3),
                "p95": round(avg_p95, 3),
                "settle": round(avg_settle, 1),
                "oscillation": round(avg_osc, 1),
            }

            print(f"{traj_name:<16} {cfg_name:<12} {avg_mae:7.3f} {avg_rms:7.3f} "
                  f"{avg_p95:7.3f} {avg_settle:7.1f} {avg_osc:5.1f}")
        print()

    # Summary: overall MAE across all trajectories
    print("=" * 80)
    print("OVERALL SUMMARY (mean MAE across all trajectories)")
    print("=" * 80)

    for cfg_name in configs_to_run:
        maes = [all_results[f"{t}|{cfg_name}"]["mae"] for t in sim.TRAJECTORIES]
        overall_mae = sum(maes) / len(maes)
        print(f"  {cfg_name:<12}: MAE = {overall_mae:.3f}")

    # Relative comparison
    print()
    print("=" * 80)
    print("RELATIVE PERFORMANCE vs BASELINE (% change, negative = better)")
    print("=" * 80)
    header2 = f"{'Trajectory':<16} {'Config':<12} {'dMAE':>8} {'dRMS':>8} {'dP95':>8} {'dOsc':>8}"
    print(header2)
    print("-" * len(header2))

    for traj_name in sim.TRAJECTORIES:
        base = all_results.get(f"{traj_name}|baseline")
        if not base:
            continue
        for cfg_name in ["sem_pags", "optimal"]:
            r = all_results.get(f"{traj_name}|{cfg_name}")
            if not r:
                continue
            def pct(val, ref):
                if ref == 0: return 0.0
                return ((val - ref) / ref) * 100
            d_mae = pct(r["mae"], base["mae"])
            d_rms = pct(r["rms"], base["rms"])
            d_p95 = pct(r["p95"], base["p95"])
            d_osc = pct(r["oscillation"], base["oscillation"])
            print(f"{traj_name:<16} {cfg_name:<12} {d_mae:+7.1f}% {d_rms:+7.1f}% "
                  f"{d_p95:+7.1f}% {d_osc:+7.1f}%")
        print()

    # Check criteria
    print("=" * 80)
    print("ACCEPTANCE CRITERIA")
    print("=" * 80)

    stat_mae_opt = all_results.get("stationary|optimal", {}).get("mae", 999)
    stat_mae_base = all_results.get("stationary|baseline", {}).get("mae", 999)

    moving_trajs = ["slow_strafe", "fast_strafe", "sine_wave", "diagonal"]
    opt_moving_mae = sum(all_results[f"{t}|optimal"]["mae"] for t in moving_trajs) / len(moving_trajs)
    base_moving_mae = sum(all_results[f"{t}|baseline"]["mae"] for t in moving_trajs) / len(moving_trajs)

    all_trajs = list(sim.TRAJECTORIES.keys())
    opt_overall = sum(all_results[f"{t}|optimal"]["mae"] for t in all_trajs) / len(all_trajs)
    sem_overall = sum(all_results[f"{t}|sem_pags"]["mae"] for t in all_trajs) / len(all_trajs)
    base_overall = sum(all_results[f"{t}|baseline"]["mae"] for t in all_trajs) / len(all_trajs)

    c1 = stat_mae_opt <= 1.53
    c2 = opt_moving_mae < base_moving_mae
    c3 = opt_overall <= min(sem_overall, base_overall)

    print(f"  [{'PASS' if c1 else 'FAIL'}] Stationary MAE <= 1.53: {stat_mae_opt:.3f}")
    print(f"  [{'PASS' if c2 else 'FAIL'}] Moving MAE better than baseline: {opt_moving_mae:.3f} vs {base_moving_mae:.3f}")
    print(f"  [{'PASS' if c3 else 'FAIL'}] Overall MAE best: optimal={opt_overall:.3f} vs sem_pags={sem_overall:.3f} vs baseline={base_overall:.3f}")

    all_pass = c1 and c2 and c3
    print(f"\n  {'ALL CRITERIA MET' if all_pass else 'SOME CRITERIA FAILED'}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimal_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "parameters": {
                "oneeuro_min_cutoff": 0.08,
                "oneeuro_beta": 0.001,
                "sem_rate": 0.01,
                "sem_decay": 0.97,
                "pags_osc_mult": 0.4,
                "pags_osc_threshold": 0.4,
                "deadtime_frames": 0.5,
                "ff_gain": 0.5,
                "predict_frames": 1.5,
                "ff_v_ema": 0.05,
                "adaptive_kp_large_threshold": 15.0,
                "adaptive_kp_small_threshold": 4.0,
                "adaptive_kp_large_mult": 1.3,
                "adaptive_kp_small_mult": 0.7,
            },
            "results": all_results,
            "summary": {
                "stationary_mae": stat_mae_opt,
                "moving_mae": opt_moving_mae,
                "overall_mae": opt_overall,
                "baseline_overall_mae": base_overall,
                "sem_pags_overall_mae": sem_overall,
                "all_criteria_met": all_pass,
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
