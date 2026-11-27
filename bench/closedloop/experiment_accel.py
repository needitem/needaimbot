#!/usr/bin/env python3
"""Experiment: acceleration feedforward output term.

Tests whether adding a direct acceleration feedforward to the controller output
(separate from the predict_frames setpoint shift that already uses acceleration)
improves tracking MAE.

New parameters:
  - ff_accel_gain: gain for accel feedforward output [0.0, 2.0]
  - accel_smooth:  EMA alpha for acceleration estimation [0.05, 0.5]

The existing predict_frames setpoint shift already uses accel:
  error_x += g * (t * vel_x + 0.5*t*t*accel_x)

This experiment adds a *separate* output-side accel feedforward:
  ff_accel_x = ff_accel_gain * lead_gate * accel_x
added to movement alongside the velocity feedforward (ff_x).
"""

import math
import json
import time
import random
import dataclasses
import sys
import os
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import (
    AimConfig, AimState, TRAJECTORIES, CONFIGS,
    SimResult, FPSCamera, NoiseModel,
    PipelineModel, NetworkModel, TargetTrackingModel,
    rescale_aim_config_for_rate, push_inflight,
    # Controller helpers
    lead_gate, nonlinear_p_move, clamp_max_step, emit_mouse_delta,
    inflight_sum, inflight_at, apply_median_prefilter, one_euro_alpha,
    gpa_smoothstep,
    # Constants
    INFLIGHT_MAX, K_DERIV_ALPHA,
    K_FF_EGO_LAG, K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE,
    K_SEM_RATE, K_SEM_MAX, K_SEM_DECAY,
    K_CGA_CONF_LOW, K_CGA_CONF_HIGH, K_CGA_SPEED_THRESHOLD,
    K_JCAD_ALPHA, K_JCAD_BASE_DB, K_JCAD_JITTER_DB, K_JCAD_JITTER_THRESHOLD,
    K_PAGS_ACQUIRE_THRESHOLD, K_PAGS_OSC_THRESHOLD,
    K_PAGS_ACQUIRE_MULT, K_PAGS_OSC_MULT, K_PAGS_ALPHA,
    # run_sim for baseline
    run_sim as _run_sim_orig,
    compute_aim_movement,
)

TRAJECTORIES_LIST = list(TRAJECTORIES.keys())
SEEDS = [42, 137, 256, 999, 1337, 7777, 31415]
N_FRAMES = 300

# ---------------------------------------------------------------------------
# Parameter space: 18 existing + 2 new (ff_accel_gain, accel_smooth)
# ---------------------------------------------------------------------------
@dataclass
class ParamDef:
    name: str
    lo: float
    hi: float
    init: float

PARAM_DEFS = [
    # -- existing 18 (init values from latest CMA-ES round) --
    ParamDef("kp_x",              0.3,   1.5,   0.991),
    ParamDef("kp_y",              0.1,   1.2,   0.389),
    ParamDef("p_softness_x",      2.0,  25.0,   7.39),
    ParamDef("p_softness_y",      2.0,  25.0,  20.0),
    ParamDef("kd_x",              0.0,   0.3,   0.035),
    ParamDef("kd_y",              0.0,   0.3,   0.025),
    ParamDef("ff_gain",           0.0,   5.0,   2.08),
    ParamDef("predict_frames",    0.0,   6.0,   4.30),
    ParamDef("deadtime_frames",   1.0,   4.0,   3.90),
    ParamDef("oneeuro_min_cutoff",0.001, 0.5,   0.03),
    ParamDef("oneeuro_beta",      0.001, 0.5,   0.05),
    ParamDef("max_step",          3.0,  30.0,  14.55),
    ParamDef("akp_large_thresh",  5.0,  50.0,  26.67),
    ParamDef("akp_small_thresh",  0.5,  10.0,   1.79),
    ParamDef("akp_large_mult",    1.0,   2.0,   1.28),
    ParamDef("akp_small_mult",    0.5,   1.0,   0.97),
    ParamDef("ff_v_ema",          0.01,  0.5,   0.097),
    ParamDef("lead_vgate",        1.0,  20.0,   5.82),
    # -- new 2 --
    ParamDef("ff_accel_gain",     0.0,   2.0,   0.3),
    ParamDef("accel_smooth",      0.05,  0.5,   0.15),
]

N_PARAMS = len(PARAM_DEFS)
assert N_PARAMS == 20, f"Expected 20 params, got {N_PARAMS}"

# ---------------------------------------------------------------------------
# Sigmoid transform for bounded parameters (same as opt_cmaes.py)
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def inv_sigmoid(p: float) -> float:
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))

def to_bounded(z: np.ndarray) -> np.ndarray:
    result = np.empty_like(z)
    for i, pd in enumerate(PARAM_DEFS):
        s = sigmoid(z[i])
        result[i] = pd.lo + s * (pd.hi - pd.lo)
    return result

def from_bounded(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    for i, pd in enumerate(PARAM_DEFS):
        p = (x[i] - pd.lo) / (pd.hi - pd.lo)
        result[i] = inv_sigmoid(p)
    return result

# ---------------------------------------------------------------------------
# Modified compute_aim_movement with acceleration feedforward output
# ---------------------------------------------------------------------------
def compute_aim_movement_accel(raw_cx, raw_cy, screen_cx, screen_cy,
                               scale_x, scale_y, cfg: AimConfig,
                               s: AimState, target_class: int,
                               detection_confidence: float,
                               ff_accel_gain: float = 0.0,
                               accel_smooth: float = 0.15):
    """Controller with acceleration feedforward output term.

    Identical to sim.compute_aim_movement except:
      1. accel_smooth replaces K_ACCEL_EMA for acceleration estimation
      2. ff_accel_gain * lead_gate * accel is added to the movement output
    """
    fresh_track = (s.has_track == 0)

    class_changed = (not fresh_track and s.prev_class >= 0
                     and target_class != s.prev_class)
    if class_changed:
        s.prev_raw_x = raw_cx
        s.prev_raw_y = raw_cy
    s.prev_class = target_class

    inf_x, inf_y = 0.0, 0.0
    use_comp = (cfg.deadtime_frames > 0)
    if use_comp:
        inf_x, inf_y = inflight_sum(s, cfg.deadtime_frames)

    target_cx, target_cy = raw_cx, raw_cy
    if cfg.oneeuro_enabled != 0.0:
        in_x, in_y = raw_cx, raw_cy
        carry_x, carry_y = s.filt_x, s.filt_y
        if s.has_track:
            ad = K_DERIV_ALPHA
            de_x = in_x - s.filt_x
            s.dfilt_x = ad * de_x + (1 - ad) * s.dfilt_x
            cutoff_x = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_x)
            ax = one_euro_alpha(cutoff_x)
            s.filt_x = ax * in_x + (1 - ax) * carry_x

            de_y = in_y - s.filt_y
            s.dfilt_y = ad * de_y + (1 - ad) * s.dfilt_y
            cutoff_y = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_y)
            ay = one_euro_alpha(cutoff_y)
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
            ja = K_JCAD_ALPHA
            s.jcad_ew_mean = ja * jcad_disp + (1 - ja) * s.jcad_ew_mean
            jdiff = jcad_disp - s.jcad_ew_mean
            s.jcad_ew_var = ja * jdiff * jdiff + (1 - ja) * s.jcad_ew_var

    # Ego-corrected velocity
    lead_active = (cfg.ff_gain > 0 or cfg.predict_frames > 0 or ff_accel_gain > 0)
    if lead_active:
        med_x, med_y = apply_median_prefilter(raw_cx, raw_cy, s)
        if s.has_track:
            ax_v, ay_v = inflight_at(s, K_FF_EGO_LAG)
            max_drift = 60.0
            dvx = (med_x - s.prev_raw_x) + (ax_v / scale_x if scale_x != 0 else 0)
            dvy = (med_y - s.prev_raw_y) + (ay_v / scale_y if scale_y != 0 else 0)
            dvx = min(max(dvx, -max_drift), max_drift)
            dvy = min(max(dvy, -max_drift), max_drift)
            a = cfg.ff_v_ema
            s.vel_x = (1 - a) * s.vel_x + a * dvx
            s.vel_y = (1 - a) * s.vel_y + a * dvy
        else:
            s.vel_x = s.vel_y = 0.0
            s.med_initialized = 0
            s.accel_x = s.accel_y = 0.0
            s.prev_vel_x = s.prev_vel_y = 0.0

        # >>> CHANGE 1: use accel_smooth instead of K_ACCEL_EMA <<<
        a_ema = accel_smooth
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
            error_x -= inf_x / scale_x
        if scale_y != 0:
            error_y -= inf_y / scale_y

    # JCAD deadband
    if cfg.jcad_enabled != 0.0:
        mean_sq = s.jcad_ew_mean ** 2
        J = s.jcad_ew_var / (mean_sq + 1e-6)
        sat_J = min(J / K_JCAD_JITTER_THRESHOLD, 1.0)
        db = K_JCAD_BASE_DB + K_JCAD_JITTER_DB * sat_J
        err_mag = math.sqrt(error_x ** 2 + error_y ** 2)
        if err_mag < db:
            error_x = error_y = 0.0

    # predict_frames setpoint shift (unchanged -- still uses accel)
    if cfg.predict_frames > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
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

    # Compute lead gate once for both ff terms
    g_ff = 0.0
    if cfg.ff_gain > 0 or ff_accel_gain > 0:
        g_ff = lead_gate(s, error_x, error_y, cfg.lead_vgate)

    # ff output term (velocity)
    ff_x = cfg.ff_gain * g_ff * s.vel_x if cfg.ff_gain > 0 else 0.0
    ff_y = cfg.ff_gain * g_ff * s.vel_y if cfg.ff_gain > 0 else 0.0

    # >>> CHANGE 2: acceleration feedforward output <<<
    ff_accel_x = ff_accel_gain * g_ff * s.accel_x
    ff_accel_y = ff_accel_gain * g_ff * s.accel_y

    # Adaptive kp: scale gain based on error magnitude
    err_mag_kp = math.sqrt(error_x ** 2 + error_y ** 2)
    kp_scale = 1.0
    if err_mag_kp > cfg.akp_large_thresh:
        kp_scale = cfg.akp_large_mult
    elif err_mag_kp < cfg.akp_small_thresh:
        kp_scale = cfg.akp_small_mult

    movement_x = (nonlinear_p_move(error_x, cfg.kp_x * kp_scale, cfg.p_softness_x,
                                    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + cfg.kd_x * s.derr_x + ff_x + ff_accel_x) * scale_x
    movement_y = (nonlinear_p_move(error_y, cfg.kp_y * kp_scale, cfg.p_softness_y,
                                    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + cfg.kd_y * s.derr_y + ff_y + ff_accel_y) * scale_y

    # SEM
    if cfg.sem_enabled != 0.0:
        if fresh_track:
            s.sem_x = s.sem_y = 0.0
        s.sem_x *= K_SEM_DECAY
        s.sem_y *= K_SEM_DECAY
        if error_x * s.sem_x < 0:
            s.sem_x = 0.0
        s.sem_x += K_SEM_RATE * error_x
        s.sem_x = min(max(s.sem_x, -K_SEM_MAX), K_SEM_MAX)
        if error_y * s.sem_y < 0:
            s.sem_y = 0.0
        s.sem_y += K_SEM_RATE * error_y
        s.sem_y = min(max(s.sem_y, -K_SEM_MAX), K_SEM_MAX)
        movement_x += s.sem_x * scale_x
        movement_y += s.sem_y * scale_y

    # PAGS
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
        s.pags_flip_rate_x = K_PAGS_ALPHA * flip_x + (1 - K_PAGS_ALPHA) * s.pags_flip_rate_x
        s.pags_flip_rate_y = K_PAGS_ALPHA * flip_y + (1 - K_PAGS_ALPHA) * s.pags_flip_rate_y
        if sign_x != 0:
            s.pags_prev_sign_x = sign_x
        if sign_y != 0:
            s.pags_prev_sign_y = sign_y
        err_mag = math.sqrt(error_x ** 2 + error_y ** 2)
        flip_rate = max(s.pags_flip_rate_x, s.pags_flip_rate_y)
        if err_mag > K_PAGS_ACQUIRE_THRESHOLD:
            target_mult = K_PAGS_ACQUIRE_MULT
        elif flip_rate > K_PAGS_OSC_THRESHOLD:
            target_mult = K_PAGS_OSC_MULT
        else:
            target_mult = 1.0
        s.pags_gain_mult = 0.3 * target_mult + 0.7 * s.pags_gain_mult
        movement_x *= s.pags_gain_mult
        movement_y *= s.pags_gain_mult

    movement_x, movement_y = clamp_max_step(movement_x, movement_y, cfg.max_step)
    movement_y *= cfg.aim_y_scale
    movement_x, movement_y = clamp_max_step(movement_x, movement_y, cfg.max_step)

    # CGA
    if cfg.cga_enabled != 0.0:
        speed = math.sqrt(s.vel_x ** 2 + s.vel_y ** 2)
        speed_bypass = speed / (speed + K_CGA_SPEED_THRESHOLD + 1e-7)
        conf_gate = gpa_smoothstep(K_CGA_CONF_LOW, K_CGA_CONF_HIGH,
                                    detection_confidence)
        gate = max(conf_gate, speed_bypass)
        movement_x *= gate
        movement_y *= gate

    if cfg.aim_output_enabled == 0.0:
        s.residual_x = s.residual_y = 0.0
        push_inflight(s, 0.0, 0.0)
        return 0, 0

    out_dx, s.residual_x = emit_mouse_delta(movement_x, s.residual_x)
    out_dy, s.residual_y = emit_mouse_delta(movement_y, s.residual_y)
    push_inflight(s, float(out_dx), float(out_dy))
    return out_dx, out_dy


# ---------------------------------------------------------------------------
# Simulation loop using the modified controller
# ---------------------------------------------------------------------------
def run_sim_accel(cfg: AimConfig, traj_name: str, seed: int,
                  ff_accel_gain: float, accel_smooth: float) -> SimResult:
    """Same as opt_cmaes.run_sim_custom but calls compute_aim_movement_accel."""
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "accel_ff", seed, N_FRAMES)

    model_size = 160
    screen_cx = model_size / 2.0
    screen_cy = model_size / 2.0
    scale_x = 1.0
    scale_y = 1.0
    target_class = 0
    prev_sign_x = 0
    prev_sign_y = 0

    for frame in range(N_FRAMES):
        frame_period_ms = pipeline.sample_frame_period_ms()
        world_x, world_y = traj_fn(frame)
        det_x, det_y = camera.world_to_screen(world_x, world_y)
        jx, jy = noise.jitter()
        det_x += jx
        det_y += jy
        conf = noise.confidence()
        target_class, flip_offset = noise.class_flip(target_class)
        det_y += flip_offset
        missed = noise.is_missed()
        if not missed and network.is_frame_lost():
            missed = True
            result.udp_losses += 1
        track_break = tracker.update(det_x, det_y, missed)
        if track_break and state.has_track:
            result.track_breaks += 1
            state.has_track = 0

        dx, dy = 0, 0
        if missed:
            result.misses += 1
            state.frames_since_seen += 1
            push_inflight(state, 0.0, 0.0)
        else:
            state.frames_since_seen = 0
            frame_cfg = dataclasses.replace(cfg)
            rate_scale = pipeline.compute_rate_scale(frame_period_ms)
            if rate_scale != 1.0:
                rescale_aim_config_for_rate(frame_cfg, rate_scale)
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(frame_period_ms)
            dx, dy = compute_aim_movement_accel(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf,
                ff_accel_gain=ff_accel_gain,
                accel_smooth=accel_smooth)

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


def run_sim_baseline(cfg: AimConfig, traj_name: str, seed: int) -> SimResult:
    """Baseline sim using original compute_aim_movement (no accel ff)."""
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "baseline", seed, N_FRAMES)

    model_size = 160
    screen_cx = model_size / 2.0
    screen_cy = model_size / 2.0
    scale_x = 1.0
    scale_y = 1.0
    target_class = 0
    prev_sign_x = 0
    prev_sign_y = 0

    for frame in range(N_FRAMES):
        frame_period_ms = pipeline.sample_frame_period_ms()
        world_x, world_y = traj_fn(frame)
        det_x, det_y = camera.world_to_screen(world_x, world_y)
        jx, jy = noise.jitter()
        det_x += jx
        det_y += jy
        conf = noise.confidence()
        target_class, flip_offset = noise.class_flip(target_class)
        det_y += flip_offset
        missed = noise.is_missed()
        if not missed and network.is_frame_lost():
            missed = True
            result.udp_losses += 1
        track_break = tracker.update(det_x, det_y, missed)
        if track_break and state.has_track:
            result.track_breaks += 1
            state.has_track = 0

        dx, dy = 0, 0
        if missed:
            result.misses += 1
            state.frames_since_seen += 1
            push_inflight(state, 0.0, 0.0)
        else:
            state.frames_since_seen = 0
            frame_cfg = dataclasses.replace(cfg)
            rate_scale = pipeline.compute_rate_scale(frame_period_ms)
            if rate_scale != 1.0:
                rescale_aim_config_for_rate(frame_cfg, rate_scale)
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(frame_period_ms)
            dx, dy = compute_aim_movement(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf)

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


# ---------------------------------------------------------------------------
# Config builder (all 20 params properly wired)
# ---------------------------------------------------------------------------
def make_config(params: np.ndarray):
    """Build AimConfig from param vector, return (cfg, ff_accel_gain, accel_smooth).

    Params 0-17 map to AimConfig fields; 18-19 are experiment-specific.
    Algorithm flags fixed to production: SEM+JCAD+PAGS, no CGA.
    """
    cfg = AimConfig()
    cfg.kp_x = float(params[0])
    cfg.kp_y = float(params[1])
    cfg.p_softness_x = float(params[2])
    cfg.p_softness_y = float(params[3])
    cfg.kd_x = float(params[4])
    cfg.kd_y = float(params[5])
    cfg.ff_gain = float(params[6])
    cfg.predict_frames = float(params[7])
    cfg.deadtime_frames = float(params[8])
    cfg.oneeuro_min_cutoff = float(params[9])
    cfg.oneeuro_beta = float(params[10])
    cfg.max_step = float(params[11])
    cfg.akp_large_thresh = float(params[12])
    cfg.akp_small_thresh = float(params[13])
    cfg.akp_large_mult = float(params[14])
    cfg.akp_small_mult = float(params[15])
    cfg.ff_v_ema = float(params[16])
    cfg.lead_vgate = float(params[17])
    # Production algorithm flags
    cfg.sem_enabled = 1.0
    cfg.cga_enabled = 0.0
    cfg.jcad_enabled = 1.0
    cfg.pags_enabled = 1.0
    # Experiment-specific params (not in AimConfig)
    ff_accel_gain = float(params[18])
    accel_smooth = float(params[19])
    return cfg, ff_accel_gain, accel_smooth


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------
def compute_baseline_maes() -> Dict[str, float]:
    """Compute per-trajectory MAE for the production config (no accel ff)."""
    baseline_cfg = CONFIGS['production']()
    maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_baseline(baseline_cfg, traj, seed)
            errors.extend(r.errors)
        maes[traj] = sum(errors) / len(errors)
    return maes


# ---------------------------------------------------------------------------
# Evaluation (fitness function for CMA-ES)
# ---------------------------------------------------------------------------
def evaluate(params: np.ndarray, baseline_maes: dict) -> float:
    cfg, ff_accel_gain, accel_smooth = make_config(params)
    traj_maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_accel(cfg, traj, seed, ff_accel_gain, accel_smooth)
            errors.extend(r.errors)
        traj_maes[traj] = sum(errors) / len(errors)

    avg_mae = sum(traj_maes.values()) / len(traj_maes)

    # Strict no-regression penalty: each trajectory must not worsen vs baseline
    penalty = 0.0
    for traj in TRAJECTORIES_LIST:
        if traj_maes[traj] > baseline_maes[traj] * 1.001:
            excess = (traj_maes[traj] - baseline_maes[traj]) / baseline_maes[traj]
            penalty += excess * 50.0

    return avg_mae + penalty


# ---------------------------------------------------------------------------
# CMA-ES implementation (pure numpy, from opt_cmaes.py)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: float = 0.5,
                 popsize: int = 16):
        self.dim = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma0

        self.lam = popsize
        self.mu = self.lam // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)
        self.eigenvectors = np.eye(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.gen = 0
        self.eigen_update_gap = max(1, int(self.lam / (10 * self.dim)))
        self.chiN = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

    def ask(self) -> list:
        samples = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * (self.eigenvectors @ (self.eigenvalues * z))
            samples.append(x)
        return samples

    def tell(self, solutions: list, fitnesses: list):
        idx = np.argsort(fitnesses)
        old_mean = self.mean.copy()

        self.mean = np.zeros(self.dim)
        for i in range(self.mu):
            self.mean += self.weights[i] * solutions[idx[i]]

        diff = self.mean - old_mean
        self.ps = (1 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  (self.invsqrt_C @ diff) / self.sigma

        hsig = (np.linalg.norm(self.ps) /
                math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) /
                self.chiN) < (1.4 + 2 / (self.dim + 1))
        hsig = float(hsig)

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  diff / self.sigma

        artmp = np.zeros((self.dim, self.mu))
        for i in range(self.mu):
            artmp[:, i] = (solutions[idx[i]] - old_mean) / self.sigma

        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) +
                            (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp @ np.diag(self.weights) @ artmp.T)

        self.sigma *= math.exp((self.cs / self.damps) *
                               (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(self.sigma, 10.0)

        self.gen += 1
        if self.gen % self.eigen_update_gap == 0:
            self._update_eigen()

    def _update_eigen(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        try:
            D2, B = np.linalg.eigh(self.C)
            D2 = np.maximum(D2, 1e-20)
            self.eigenvalues = np.sqrt(D2)
            self.eigenvectors = B
            self.invsqrt_C = B @ np.diag(1.0 / np.sqrt(D2)) @ B.T
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.invsqrt_C = np.eye(self.dim)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    POPSIZE = 16
    MAX_GENS = 100
    STAGNATION_LIMIT = 50

    print("=" * 70)
    print("Experiment: Acceleration Feedforward Output Term")
    print("=" * 70)
    print(f"Params: {N_PARAMS} (18 existing + ff_accel_gain + accel_smooth)")
    print(f"Seeds: {len(SEEDS)} | Trajectories: {len(TRAJECTORIES_LIST)}")
    print(f"CMA-ES: popsize={POPSIZE}, max_gens={MAX_GENS}, "
          f"stagnation={STAGNATION_LIMIT}")
    print(f"Algo: SEM+JCAD+PAGS (production, CGA disabled)")
    print()

    # 1. Compute baseline MAEs
    print("Computing baseline MAEs (production config, no accel ff)...")
    t0 = time.time()
    baseline_maes = compute_baseline_maes()
    baseline_avg = sum(baseline_maes.values()) / len(baseline_maes)
    baseline_time = time.time() - t0
    print(f"Baseline avg MAE: {baseline_avg:.4f} ({baseline_time:.1f}s)")
    for t, m in baseline_maes.items():
        print(f"  {t:15s}: {m:.4f}")
    print()

    # 2. Set up CMA-ES
    x0_bounded = np.array([pd.init for pd in PARAM_DEFS])
    x0_z = from_bounded(x0_bounded)

    es = CMAES(x0_z, sigma0=0.5, popsize=POPSIZE)
    best_fitness = float('inf')
    best_params = None
    stagnation = 0
    t_start = time.time()

    # 3. Optimization loop
    print("Starting CMA-ES optimization...")
    print(f"{'Gen':>5} {'Best':>9} {'GenBest':>9} {'Improv':>8} "
          f"{'Sigma':>8} {'Stag':>5} {'Time':>7}")
    print("-" * 60)

    for gen in range(MAX_GENS):
        solutions = es.ask()
        fitnesses = []

        for sol in solutions:
            bounded = to_bounded(sol)
            f = evaluate(bounded, baseline_maes)
            fitnesses.append(f)

        es.tell(solutions, fitnesses)

        gen_best = min(fitnesses)
        gen_best_idx = fitnesses.index(gen_best)

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_params = to_bounded(solutions[gen_best_idx]).copy()
            stagnation = 0
        else:
            stagnation += 1

        elapsed = time.time() - t_start
        improv = (best_fitness - baseline_avg) / baseline_avg * 100

        if gen % 5 == 0 or stagnation == 0:
            print(f"{gen:5d} {best_fitness:9.4f} {gen_best:9.4f} "
                  f"{improv:+7.2f}% {es.sigma:8.4f} {stagnation:5d} "
                  f"{elapsed:6.0f}s")

        if stagnation >= STAGNATION_LIMIT:
            print(f"\nEarly stop: {stagnation} generations without improvement")
            break

    total_gens = gen + 1
    wall_clock = time.time() - t_start

    # 4. Final evaluation with best params
    print(f"\n{'='*70}")
    print("Final evaluation with best parameters")
    print(f"{'='*70}")

    cfg, ff_accel_gain, accel_smooth = make_config(best_params)
    per_trajectory = {}
    any_regression = False

    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_accel(cfg, traj, seed, ff_accel_gain, accel_smooth)
            errors.extend(r.errors)
        mae = sum(errors) / len(errors)
        bl = baseline_maes[traj]
        ti = (mae - bl) / bl * 100
        status = "OK" if mae <= bl * 1.001 else "REGRESS"
        if status == "REGRESS":
            any_regression = True
        per_trajectory[traj] = {
            "mae": round(mae, 4),
            "baseline_mae": round(bl, 4),
            "improvement_pct": round(ti, 2),
            "status": status,
        }
        print(f"  {traj:15s}: {mae:.4f} (baseline {bl:.4f}, {ti:+.1f}%) [{status}]")

    final_avg_mae = sum(v["mae"] for v in per_trajectory.values()) / len(per_trajectory)
    improvement = (final_avg_mae - baseline_avg) / baseline_avg * 100

    print(f"\n  Average MAE: {final_avg_mae:.4f} (baseline {baseline_avg:.4f}, "
          f"{improvement:+.2f}%)")
    if any_regression:
        print("  WARNING: Some trajectories regressed vs baseline")
    print(f"  ff_accel_gain = {ff_accel_gain:.4f}")
    print(f"  accel_smooth  = {accel_smooth:.4f}")
    print(f"  Generations: {total_gens}, Wall clock: {wall_clock:.1f}s")

    # 5. Build results dict
    best_params_dict = {pd.name: round(float(best_params[i]), 4)
                        for i, pd in enumerate(PARAM_DEFS)}

    results = {
        "approach": "acceleration_feedforward",
        "baseline_avg_mae": round(baseline_avg, 4),
        "best_mae": round(final_avg_mae, 4),
        "improvement_pct": round(improvement, 2),
        "best_params": best_params_dict,
        "per_trajectory": per_trajectory,
        "generations_used": total_gens,
        "wall_clock_seconds": round(wall_clock, 1),
        "config": {
            "popsize": POPSIZE,
            "max_gens": MAX_GENS,
            "stagnation_limit": STAGNATION_LIMIT,
            "n_seeds": len(SEEDS),
            "n_frames": N_FRAMES,
            "algo": "SEM+JCAD+PAGS",
        },
    }

    # 6. Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "experiment_accel_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
