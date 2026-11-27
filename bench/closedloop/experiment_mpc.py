#!/usr/bin/env python3
"""MPC/LQR optimal gain experiment for aimbot controller.

Replaces the PD gain structure (kp_x, kp_y, kd_x, kd_y) with an optimal
linear controller derived from discrete-time LQR (Linear Quadratic Regulator).

Key insight: the LQR gain computation happens ONCE (offline) via the
Discrete Algebraic Riccati Equation (DARE). At runtime, the controller is
just u = K[0,0]*error + K[0,1]*velocity — same cost as the current PD but
with optimally balanced position/velocity gains.

The nonlinear envelope (softness curve, adaptive kp, SEM, JCAD, PAGS,
One-Euro filter, dead-time compensation) is kept intact on top of the
LQR-derived gains.

CMA-ES optimizes 17 parameters:
  - 3 MPC cost weights: Q_pos, Q_vel, R_ctrl
  - 14 existing params (minus kp/kd which are replaced by K)
"""

import math
import json
import time
import random
import dataclasses
import sys
import os
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sim import (AimConfig, AimState, TRAJECTORIES, CONFIGS,
                 SimResult, FPSCamera, NoiseModel,
                 PipelineModel, NetworkModel, TargetTrackingModel,
                 compute_aim_movement, rescale_aim_config_for_rate,
                 push_inflight, nonlinear_p_move, one_euro_alpha,
                 clamp_max_step, inflight_sum, inflight_at,
                 lead_gate, apply_median_prefilter, emit_mouse_delta,
                 gpa_smoothstep,
                 K_DERIV_ALPHA, K_FF_EGO_LAG, K_ACCEL_EMA,
                 K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE,
                 K_SEM_RATE, K_SEM_MAX, K_SEM_DECAY,
                 K_CGA_CONF_LOW, K_CGA_CONF_HIGH, K_CGA_SPEED_THRESHOLD,
                 K_JCAD_ALPHA, K_JCAD_BASE_DB, K_JCAD_JITTER_DB,
                 K_JCAD_JITTER_THRESHOLD,
                 K_PAGS_ACQUIRE_THRESHOLD, K_PAGS_OSC_THRESHOLD,
                 K_PAGS_ACQUIRE_MULT, K_PAGS_OSC_MULT, K_PAGS_ALPHA,
                 K_LEAD_ERR_GATE, INFLIGHT_MAX,
                 run_sim)

TRAJECTORIES_LIST = list(TRAJECTORIES.keys())
SEEDS = [42, 137, 256, 999, 1337, 7777, 31415]
N_FRAMES = 300

# ---------------------------------------------------------------------------
# DARE solver (iterative, no scipy dependency for Jetson compatibility)
# ---------------------------------------------------------------------------
def solve_dare_iterative(A, B, Q, R, max_iter=200):
    """Solve the Discrete Algebraic Riccati Equation iteratively.

    Finds P such that:
        P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q

    Returns the optimal gain K = (R + B^T P B)^{-1} B^T P A
    """
    P = Q.copy().astype(np.float64)
    for _ in range(max_iter):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_new = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_new - P)) < 1e-10:
            break
        P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def compute_lqr_gain(Q_pos, Q_vel, R_ctrl, dt=1.0):
    """Compute the optimal LQR gain for a 1D error-velocity system.

    State: x = [error, velocity]
    Dynamics: error_next = error - u + vel*dt
              vel_next = vel  (constant velocity assumption)
    Control: u reduces error (negative B on control)

    Returns K of shape (1, 2):
      K[0, 0] = position gain (replaces kp)
      K[0, 1] = velocity gain (replaces/complements kd + ff)
    """
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[-1.0],
                  [0.0]])
    Q = np.diag([Q_pos, Q_vel])
    R = np.array([[R_ctrl]])

    K = solve_dare_iterative(A, B, Q, R)

    # LQR convention: u* = -K @ x, so K values are negative for stabilising
    # control. Negate to get positive gains for the controller:
    #   K[0,0] -> position gain (like kp, positive)
    #   K[0,1] -> velocity gain (positive: compensate for target motion)
    return -K


# ---------------------------------------------------------------------------
# MPC-augmented controller
# ---------------------------------------------------------------------------
def compute_aim_movement_mpc(raw_cx, raw_cy, screen_cx, screen_cy,
                             scale_x, scale_y, cfg: AimConfig,
                             s: AimState, target_class: int,
                             detection_confidence: float,
                             lqr_kp: float, lqr_kv: float):
    """Controller with LQR-derived gains replacing PD structure.

    lqr_kp: K[0,0] from DARE — replaces cfg.kp_x/kp_y as the base position gain
    lqr_kv: K[0,1] from DARE — velocity-proportional term added to feedforward

    Everything else (softness, One-Euro, JCAD, dead-time comp, SEM, PAGS,
    adaptive kp, predict_frames, ff_gain) works identically to the original.
    """
    fresh_track = (s.has_track == 0)

    class_changed = (not fresh_track and s.prev_class >= 0
                     and target_class != s.prev_class)
    if class_changed:
        s.prev_raw_x = raw_cx
        s.prev_raw_y = raw_cy
    s.prev_class = target_class

    # Dead-time compensation
    inf_x, inf_y = 0.0, 0.0
    use_comp = (cfg.deadtime_frames > 0)
    if use_comp:
        inf_x, inf_y = inflight_sum(s, cfg.deadtime_frames)

    # One-Euro filter
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

    # JCAD displacement tracking
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

    # Ego-corrected velocity estimation
    lead_active = (cfg.ff_gain > 0 or cfg.predict_frames > 0)
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

        a_ema = K_ACCEL_EMA
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

    # Error computation
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

    # Predictive setpoint shift
    if cfg.predict_frames > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
        t = cfg.predict_frames
        error_x += g * (t * s.vel_x + 0.5 * t * t * s.accel_x)
        error_y += g * (t * s.vel_y + 0.5 * t * t * s.accel_y)

    # D term (kept for state tracking, but not used in movement — LQR replaces it)
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

    # Feedforward output term (original ff_gain-based)
    ff_x, ff_y = 0.0, 0.0
    if cfg.ff_gain > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
        ff_x = cfg.ff_gain * g * s.vel_x
        ff_y = cfg.ff_gain * g * s.vel_y

    # LQR velocity feedforward term (K[0,1] * velocity)
    # This complements the existing ff_gain by adding the optimal velocity
    # coupling from the DARE solution
    lqr_ff_x = lqr_kv * s.vel_x
    lqr_ff_y = lqr_kv * s.vel_y

    # Adaptive kp: scale gain based on error magnitude
    err_mag_kp = math.sqrt(error_x ** 2 + error_y ** 2)
    kp_scale = 1.0
    if err_mag_kp > cfg.akp_large_thresh:
        kp_scale = cfg.akp_large_mult
    elif err_mag_kp < cfg.akp_small_thresh:
        kp_scale = cfg.akp_small_mult

    # --- MPC/LQR gain replaces PD structure ---
    # Use lqr_kp (from DARE K[0,0]) as the effective kp in the nonlinear
    # softness envelope, instead of cfg.kp_x / cfg.kp_y.
    # The D term (kd * derr) is dropped — the LQR velocity gain (lqr_kv)
    # provides optimal velocity coupling.
    movement_x = (nonlinear_p_move(error_x, lqr_kp * kp_scale, cfg.p_softness_x,
                                   K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + ff_x + lqr_ff_x) * scale_x
    movement_y = (nonlinear_p_move(error_y, lqr_kp * kp_scale, cfg.p_softness_y,
                                   K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + ff_y + lqr_ff_y) * scale_y

    # SEM
    if cfg.sem_enabled != 0.0:
        if fresh_track:
            s.sem_x = s.sem_y = 0.0
        s.sem_x *= K_SEM_DECAY
        s.sem_y *= K_SEM_DECAY
        if error_x * s.sem_x < 0:
            s.sem_x = 0.0
        if error_y * s.sem_y < 0:
            s.sem_y = 0.0
        s.sem_x += K_SEM_RATE * error_x
        s.sem_x = min(max(s.sem_x, -K_SEM_MAX), K_SEM_MAX)
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
# Parameter space definition (17 params: 14 existing + 3 MPC)
# ---------------------------------------------------------------------------
@dataclass
class ParamDef:
    name: str
    lo: float
    hi: float
    init: float


PARAM_DEFS = [
    # MPC cost weights (new)
    ParamDef("Q_pos",              0.1,  100.0,  10.0),
    ParamDef("Q_vel",              0.01,  50.0,   1.0),
    ParamDef("R_ctrl",             0.01,  50.0,   1.0),
    # Existing params (kp_x, kp_y, kd_x, kd_y removed — replaced by LQR K)
    ParamDef("p_softness_x",       1.0,   25.0,   1.66),
    ParamDef("p_softness_y",       1.0,   25.0,  11.50),
    ParamDef("ff_gain",            0.0,    5.0,   0.653),
    ParamDef("predict_frames",     0.0,    6.0,   5.78),
    ParamDef("deadtime_frames",    1.0,    4.0,   3.97),
    ParamDef("oneeuro_min_cutoff", 0.001,  0.5,   0.039),
    ParamDef("oneeuro_beta",       0.001,  0.5,   0.009),
    ParamDef("max_step",           3.0,   30.0,   9.71),
    ParamDef("akp_large_thresh",   5.0,   50.0,  24.50),
    ParamDef("akp_small_thresh",   0.5,   10.0,   1.51),
    ParamDef("akp_large_mult",     1.0,    2.0,   1.19),
    ParamDef("akp_small_mult",     0.5,    1.0,   0.97),
    ParamDef("ff_v_ema",           0.01,   0.5,   0.126),
    ParamDef("lead_vgate",         1.0,   20.0,   5.52),
]

assert len(PARAM_DEFS) == 17, f"Expected 17 params, got {len(PARAM_DEFS)}"


# ---------------------------------------------------------------------------
# Sigmoid transform for bounded parameters
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
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
# Config builder from MPC parameter vector
# ---------------------------------------------------------------------------
def make_mpc_config(params: np.ndarray):
    """Build AimConfig + LQR gains from the 17-parameter vector.

    Returns (cfg, lqr_kp, lqr_kv) where:
      cfg:    AimConfig with all non-gain params set
      lqr_kp: K[0,0] — optimal position gain (replaces kp_x/kp_y)
      lqr_kv: K[0,1] — optimal velocity gain (additional feedforward)
    """
    Q_pos = float(params[0])
    Q_vel = float(params[1])
    R_ctrl = float(params[2])

    # Compute LQR gain (offline, once per parameter set)
    K = compute_lqr_gain(Q_pos, Q_vel, R_ctrl, dt=1.0)
    lqr_kp = float(K[0, 0])  # position gain
    lqr_kv = float(K[0, 1])  # velocity gain

    cfg = AimConfig()
    # kp_x/kp_y are unused by compute_aim_movement_mpc (lqr_kp is used instead)
    # but set them to lqr_kp for rescale_aim_config_for_rate compatibility
    cfg.kp_x = lqr_kp
    cfg.kp_y = lqr_kp
    cfg.kd_x = 0.0  # D term replaced by LQR velocity coupling
    cfg.kd_y = 0.0
    cfg.p_softness_x = float(params[3])
    cfg.p_softness_y = float(params[4])
    cfg.ff_gain = float(params[5])
    cfg.predict_frames = float(params[6])
    cfg.deadtime_frames = float(params[7])
    cfg.oneeuro_min_cutoff = float(params[8])
    cfg.oneeuro_beta = float(params[9])
    cfg.max_step = float(params[10])
    cfg.akp_large_thresh = float(params[11])
    cfg.akp_small_thresh = float(params[12])
    cfg.akp_large_mult = float(params[13])
    cfg.akp_small_mult = float(params[14])
    cfg.ff_v_ema = float(params[15])
    cfg.lead_vgate = float(params[16])

    # Enable all novel algorithms (production config)
    cfg.sem_enabled = 1.0
    cfg.cga_enabled = 0.0
    cfg.jcad_enabled = 1.0
    cfg.pags_enabled = 1.0

    return cfg, lqr_kp, lqr_kv


# ---------------------------------------------------------------------------
# Simulation with MPC controller
# ---------------------------------------------------------------------------
def run_sim_mpc(cfg: AimConfig, lqr_kp: float, lqr_kv: float,
                traj_name: str, seed: int) -> SimResult:
    """Run closed-loop simulation using the MPC/LQR controller."""
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "mpc", seed, N_FRAMES)

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
            # Scale the LQR gains the same way kp is scaled for rate
            frame_lqr_kp = lqr_kp
            frame_lqr_kv = lqr_kv
            if rate_scale != 1.0:
                rescale_aim_config_for_rate(frame_cfg, rate_scale)
                # kp is scaled by r in rescale, so scale lqr_kp the same way
                frame_lqr_kp *= rate_scale
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(frame_period_ms)
            dx, dy = compute_aim_movement_mpc(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf,
                frame_lqr_kp, frame_lqr_kv)

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
# Run simulation with original PD controller (for baseline)
# ---------------------------------------------------------------------------
def run_sim_baseline(cfg: AimConfig, traj_name: str, seed: int) -> SimResult:
    """Run with original compute_aim_movement (PD controller)."""
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
# Evaluation function
# ---------------------------------------------------------------------------
def compute_baseline_maes() -> Dict[str, float]:
    """Compute per-trajectory MAE for the production baseline config."""
    baseline_cfg = CONFIGS['production']()
    maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_baseline(baseline_cfg, traj, seed)
            errors.extend(r.errors)
        maes[traj] = sum(errors) / len(errors)
    return maes


def evaluate(params: np.ndarray, baseline_maes: dict) -> float:
    """Evaluate a parameter set: average MAE across all trajectories + seeds,
    with a penalty for trajectory-level regressions vs baseline."""
    cfg, lqr_kp, lqr_kv = make_mpc_config(params)

    traj_maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_mpc(cfg, lqr_kp, lqr_kv, traj, seed)
            errors.extend(r.errors)
        traj_maes[traj] = sum(errors) / len(errors)

    avg_mae = sum(traj_maes.values()) / len(traj_maes)

    # Penalty for regression on any trajectory
    penalty = 0.0
    for traj in TRAJECTORIES_LIST:
        if traj_maes[traj] > baseline_maes[traj] * 1.001:
            excess = (traj_maes[traj] - baseline_maes[traj]) / baseline_maes[traj]
            penalty += excess * 50.0

    return avg_mae + penalty


# ---------------------------------------------------------------------------
# CMA-ES implementation (pure numpy, same as opt_cmaes.py)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: float = 0.5,
                 popsize: int = 20):
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

    def ask(self) -> List[np.ndarray]:
        samples = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * (self.eigenvectors @ (self.eigenvalues * z))
            samples.append(x)
        return samples

    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]):
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
# Main optimization
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("MPC/LQR Optimal Gain Experiment")
    print("=" * 70)
    print(f"Params: {len(PARAM_DEFS)} (3 MPC cost weights + 14 controller params)")
    print(f"Seeds: {len(SEEDS)} | Trajectories: {len(TRAJECTORIES_LIST)}")
    print(f"Evals per fitness: {len(SEEDS) * len(TRAJECTORIES_LIST)} sim runs")
    print(f"Controller: LQR gain (DARE) replaces PD kp/kd structure")
    print()

    # Show initial LQR gain for reference
    K_init = compute_lqr_gain(10.0, 1.0, 1.0, dt=1.0)
    print(f"Initial LQR gain (Q_pos=10, Q_vel=1, R=1):")
    print(f"  K[0,0] (position) = {K_init[0, 0]:.4f}  (cf. baseline kp_x=0.638)")
    print(f"  K[0,1] (velocity) = {K_init[0, 1]:.4f}  (cf. baseline kd_x=0.0025)")
    print()

    # Compute baseline
    print("Computing baseline MAEs (production config)...")
    t0 = time.time()
    baseline_maes = compute_baseline_maes()
    baseline_avg = sum(baseline_maes.values()) / len(baseline_maes)
    print(f"Baseline avg MAE: {baseline_avg:.4f} ({time.time() - t0:.1f}s)")
    for t, m in baseline_maes.items():
        print(f"  {t:15s}: {m:.4f}")
    print()

    # Evaluate initial MPC config (no optimization yet)
    x0_bounded = np.array([pd.init for pd in PARAM_DEFS])
    init_fitness = evaluate(x0_bounded, baseline_maes)
    init_improv = (init_fitness - baseline_avg) / baseline_avg * 100
    print(f"Initial MPC fitness (before CMA-ES): {init_fitness:.4f} ({init_improv:+.1f}%)")
    print()

    # CMA-ES optimization
    x0_z = from_bounded(x0_bounded)
    POPSIZE = 16
    MAX_GENS = 100
    STAGNATION_LIMIT = 50

    print(f"Starting CMA-ES: popsize={POPSIZE}, max_gens={MAX_GENS}, "
          f"early_stop={STAGNATION_LIMIT} gens stagnation")
    print("-" * 70)

    es = CMAES(x0_z, sigma0=0.5, popsize=POPSIZE)
    best_fitness = float('inf')
    best_params = None
    stagnation = 0
    history = []
    t_start = time.time()

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
        gen_median = sorted(fitnesses)[len(fitnesses) // 2]

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_params = to_bounded(solutions[gen_best_idx]).copy()
            stagnation = 0
        else:
            stagnation += 1

        history.append({
            "gen": gen,
            "best": float(best_fitness),
            "gen_best": float(gen_best),
            "gen_median": float(gen_median),
            "sigma": float(es.sigma),
        })

        if gen % 5 == 0 or gen_best < best_fitness + 0.001:
            elapsed = time.time() - t_start
            improv = (best_fitness - baseline_avg) / baseline_avg * 100
            print(f"  Gen {gen:3d} | best={best_fitness:.4f} "
                  f"({improv:+.1f}%) | gen_best={gen_best:.4f} "
                  f"| median={gen_median:.4f} | sigma={es.sigma:.4f} "
                  f"| stag={stagnation} | {elapsed:.0f}s")

        if stagnation >= STAGNATION_LIMIT:
            print(f"  Early stop: {stagnation} gens without improvement")
            break

    total_time = time.time() - t_start
    total_gens = gen + 1
    print(f"\nOptimization complete: {total_gens} generations in {total_time:.0f}s")

    # ---------------------------------------------------------------------------
    # Final evaluation with best params
    # ---------------------------------------------------------------------------
    if best_params is None:
        print("ERROR: No valid solution found")
        return

    cfg, lqr_kp, lqr_kv = make_mpc_config(best_params)
    K_final = compute_lqr_gain(float(best_params[0]), float(best_params[1]),
                               float(best_params[2]), dt=1.0)

    print(f"\n{'=' * 70}")
    print("BEST MPC/LQR PARAMETERS")
    print(f"{'=' * 70}")
    print(f"LQR gains:")
    print(f"  Q_pos    = {best_params[0]:.4f}")
    print(f"  Q_vel    = {best_params[1]:.4f}")
    print(f"  R_ctrl   = {best_params[2]:.4f}")
    print(f"  K[0,0]   = {K_final[0, 0]:.4f} (effective kp, baseline=0.638)")
    print(f"  K[0,1]   = {K_final[0, 1]:.4f} (velocity gain)")
    print(f"\nController params:")
    for i, pd in enumerate(PARAM_DEFS):
        if i < 3:
            continue  # Already printed
        print(f"  {pd.name:22s} = {best_params[i]:.4f} (init {pd.init:.3f})")

    # Detailed per-trajectory comparison
    print(f"\n{'=' * 70}")
    print("PER-TRAJECTORY COMPARISON: MPC vs BASELINE")
    print(f"{'=' * 70}")

    header = (f"{'Trajectory':<16} {'BL MAE':>8} {'MPC MAE':>8} {'Delta':>8} "
              f"{'%':>7} {'BL Osc':>7} {'MPC Osc':>7}")
    print(header)
    print("-" * len(header))

    traj_details = {}
    total_bl_mae = 0.0
    total_mpc_mae = 0.0

    for traj in TRAJECTORIES_LIST:
        bl_errors = []
        mpc_errors = []
        bl_osc = 0
        mpc_osc = 0
        for seed in SEEDS:
            bl_r = run_sim_baseline(CONFIGS['production'](), traj, seed)
            bl_errors.extend(bl_r.errors)
            bl_osc += bl_r.oscillation_count

            mpc_r = run_sim_mpc(cfg, lqr_kp, lqr_kv, traj, seed)
            mpc_errors.extend(mpc_r.errors)
            mpc_osc += mpc_r.oscillation_count

        bl_mae = sum(bl_errors) / len(bl_errors)
        mpc_mae = sum(mpc_errors) / len(mpc_errors)
        delta = mpc_mae - bl_mae
        pct = delta / bl_mae * 100 if bl_mae > 0 else 0

        total_bl_mae += bl_mae
        total_mpc_mae += mpc_mae

        status = "BETTER" if delta < -0.001 else ("WORSE" if delta > 0.001 else "SAME")
        print(f"{traj:<16} {bl_mae:8.4f} {mpc_mae:8.4f} {delta:+8.4f} "
              f"{pct:+6.1f}% {bl_osc / len(SEEDS):7.1f} {mpc_osc / len(SEEDS):7.1f}  "
              f"[{status}]")

        traj_details[traj] = {
            "baseline_mae": round(bl_mae, 4),
            "mpc_mae": round(mpc_mae, 4),
            "delta": round(delta, 4),
            "improvement_pct": round(-pct, 2),
            "baseline_osc": round(bl_osc / len(SEEDS), 1),
            "mpc_osc": round(mpc_osc / len(SEEDS), 1),
        }

    avg_bl = total_bl_mae / len(TRAJECTORIES_LIST)
    avg_mpc = total_mpc_mae / len(TRAJECTORIES_LIST)
    avg_delta = avg_mpc - avg_bl
    avg_pct = avg_delta / avg_bl * 100

    print("-" * len(header))
    print(f"{'AVERAGE':<16} {avg_bl:8.4f} {avg_mpc:8.4f} {avg_delta:+8.4f} "
          f"{avg_pct:+6.1f}%")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    output = {
        "experiment": "MPC/LQR optimal gain",
        "description": ("DARE-derived LQR gain replaces PD kp/kd. "
                        "Softness, adaptive kp, SEM, JCAD, PAGS, One-Euro, "
                        "dead-time comp remain intact."),
        "baseline_avg_mae": round(avg_bl, 4),
        "mpc_avg_mae": round(avg_mpc, 4),
        "improvement_pct": round(-avg_pct, 2),
        "lqr_gains": {
            "Q_pos": round(float(best_params[0]), 4),
            "Q_vel": round(float(best_params[1]), 4),
            "R_ctrl": round(float(best_params[2]), 4),
            "K_position": round(float(K_final[0, 0]), 4),
            "K_velocity": round(float(K_final[0, 1]), 4),
        },
        "best_params": {pd.name: round(float(best_params[i]), 4)
                        for i, pd in enumerate(PARAM_DEFS)},
        "per_trajectory": traj_details,
        "optimization": {
            "method": "CMA-ES",
            "popsize": POPSIZE,
            "generations": total_gens,
            "max_gens": MAX_GENS,
            "stagnation_limit": STAGNATION_LIMIT,
            "total_time_s": round(total_time, 1),
            "seeds": SEEDS,
            "n_frames": N_FRAMES,
        },
        "convergence_history": history,
    }

    out_path = os.path.join(os.path.dirname(__file__), "experiment_mpc_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")
    print(f"Overall: MPC MAE={avg_mpc:.4f} vs Baseline MAE={avg_bl:.4f} "
          f"({avg_pct:+.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
