#!/usr/bin/env python3
"""LQR state-space controller experiment.

Tests whether replacing PD gains with DARE-optimal feedback gains
improves closed-loop tracking performance.

State-space model:
  State: x = [ex, ey, vx, vy]  (position error + velocity in model px/frame)
  Input: u = [dx, dy]           (mouse movement in model px)

  x_{k+1} = A*x_k + B*u_k
  A = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]  (constant velocity model)
  B = [[-1,0],[0,-1],[0,0],[0,0]]                   (control reduces error)

DARE solution yields K (2x4).  Since u = -K @ x and B has negative entries,
the actual control gains K_ctrl = -K are positive:
  - K_ctrl[0,0] -> kp_x (through nonlinear softness curve)
  - K_ctrl[1,1] -> kp_y
  - K_ctrl[0,2], K_ctrl[1,3] -> velocity feedforward gains
  - Cross-terms add coupling that PD cannot express (zero for diagonal Q/R)

Keeps: softness curve, adaptive kp scaling, SEM, JCAD, PAGS, One-Euro,
       dead-time compensation, predict_frames setpoint shift, ff_gain.
Removes: kd (LQR handles derivative through velocity state).

CMA-ES optimizes 18 params: 6 LQR weights (Q_ex/ey/vx/vy, R_x/y) +
12 remaining controller params.
"""

import math
import json
import time
import random
import dataclasses
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import (
    AimConfig, AimState, TRAJECTORIES, CONFIGS, INFLIGHT_MAX,
    SimResult, FPSCamera, NoiseModel, PipelineModel, NetworkModel,
    TargetTrackingModel, rescale_aim_config_for_rate,
    nonlinear_p_move, one_euro_alpha, clamp_max_step,
    lead_gate, push_inflight, inflight_sum, inflight_at,
    apply_median_prefilter, emit_mouse_delta, gpa_smoothstep,
    K_DERIV_ALPHA, K_FF_EGO_LAG, K_ACCEL_EMA,
    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE,
    K_SEM_RATE, K_SEM_MAX, K_SEM_DECAY,
    K_CGA_CONF_LOW, K_CGA_CONF_HIGH, K_CGA_SPEED_THRESHOLD,
    K_JCAD_ALPHA, K_JCAD_BASE_DB, K_JCAD_JITTER_DB, K_JCAD_JITTER_THRESHOLD,
    K_PAGS_ALPHA, K_PAGS_ACQUIRE_THRESHOLD, K_PAGS_OSC_THRESHOLD,
    K_PAGS_ACQUIRE_MULT, K_PAGS_OSC_MULT,
    K_LEAD_ERR_GATE,
)
from opt_cmaes import CMAES, sigmoid, inv_sigmoid, run_sim_custom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAJECTORIES_LIST = list(TRAJECTORIES.keys())
SEEDS = [42, 137, 256, 999, 1337, 7777, 31415]
N_FRAMES = 300

# Fixed adaptive kp thresholds (from CMA-ES round 2 optimals)
FIXED_AKP_LARGE_THRESH = 26.67
FIXED_AKP_SMALL_THRESH = 1.79

# ---------------------------------------------------------------------------
# DARE solver (no scipy dependency)
# ---------------------------------------------------------------------------
def solve_dare(A, B, Q, R, max_iter=500):
    """Solve discrete algebraic Riccati equation iteratively.

    Returns (K, P) where K is the optimal feedback gain matrix and P is
    the DARE solution.  Returns (None, None) on convergence failure.
    """
    n = A.shape[0]
    P = Q.copy().astype(float)
    for _ in range(max_iter):
        BtP = B.T @ P
        S = R + BtP @ B
        try:
            K = np.linalg.solve(S, BtP @ A)
        except np.linalg.LinAlgError:
            return None, None
        P_new = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_new - P)) < 1e-12:
            break
        P = P_new
    # Final K with converged P
    try:
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    except np.linalg.LinAlgError:
        return None, None
    return K, P


def compute_lqr_gains(Q_ex, Q_ey, Q_vx, Q_vy, R_x, R_y):
    """Compute LQR feedback control gains from cost weights.

    Returns K_ctrl (2x4), the NEGATED DARE gain so that all entries used
    as proportional / velocity gains are positive.  Returns None on
    failure (non-convergence or degenerate parameters).
    """
    dt = 1.0  # one frame step; velocity is in px/frame
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)
    B = np.array([
        [-1,  0],
        [ 0, -1],
        [ 0,  0],
        [ 0,  0],
    ], dtype=float)
    Q = np.diag([Q_ex, Q_ey, Q_vx, Q_vy])
    R = np.diag([R_x, R_y])

    K, P = solve_dare(A, B, Q, R)
    if K is None:
        return None
    # Negate: DARE gives u = -K x, but B already has -1, so actual
    # control gains (positive for same-direction response) are -K.
    K_ctrl = -K
    if np.any(np.isnan(K_ctrl)) or np.any(np.abs(K_ctrl) > 100):
        return None
    # Sanity: position gains must be positive
    if K_ctrl[0, 0] <= 0 or K_ctrl[1, 1] <= 0:
        return None
    return K_ctrl


# ---------------------------------------------------------------------------
# LQR-modified controller (derived from sim.py compute_aim_movement)
# ---------------------------------------------------------------------------
def compute_aim_movement_lqr(raw_cx, raw_cy, screen_cx, screen_cy,
                              scale_x, scale_y, cfg: AimConfig,
                              s: AimState, target_class: int,
                              detection_confidence: float,
                              K_ctrl: np.ndarray):
    """Controller with LQR gains replacing PD.

    Identical to sim.py compute_aim_movement except:
      - cfg.kp_x / kp_y contain K_ctrl position gains (rate-scaled externally)
      - K_ctrl[0,2]*vel_x + K_ctrl[0,3]*vel_y added as LQR velocity terms
      - kd terms removed (LQR handles derivative through velocity state)
      - Velocity estimation always active (LQR velocity terms need it)
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

    # Ego-corrected velocity -- ALWAYS active for LQR (needs vel state)
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

    # predict_frames setpoint shift (kept)
    if cfg.predict_frames > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
        t = cfg.predict_frames
        error_x += g * (t * s.vel_x + 0.5 * t * t * s.accel_x)
        error_y += g * (t * s.vel_y + 0.5 * t * t * s.accel_y)

    # D-term state update (kept for state consistency, NOT used in movement)
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

    # ff output term (kept, CMA-ES can zero ff_gain)
    ff_x, ff_y = 0.0, 0.0
    if cfg.ff_gain > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
        ff_x = cfg.ff_gain * g * s.vel_x
        ff_y = cfg.ff_gain * g * s.vel_y

    # Adaptive kp scaling
    err_mag_kp = math.sqrt(error_x ** 2 + error_y ** 2)
    kp_scale = 1.0
    if err_mag_kp > cfg.akp_large_thresh:
        kp_scale = cfg.akp_large_mult
    elif err_mag_kp < cfg.akp_small_thresh:
        kp_scale = cfg.akp_small_mult

    # === LQR-modified movement computation ===
    # Position term: cfg.kp_x/kp_y contain K_ctrl position gains, already
    # rate-scaled by rescale_aim_config_for_rate in the caller.
    # Velocity term: K_ctrl velocity gains, NOT rate-scaled (analogous to
    # ff_gain which is divided by r, while vel itself scales with r).
    lqr_vel_x = K_ctrl[0, 2] * s.vel_x + K_ctrl[0, 3] * s.vel_y
    lqr_vel_y = K_ctrl[1, 2] * s.vel_x + K_ctrl[1, 3] * s.vel_y

    movement_x = (nonlinear_p_move(error_x, cfg.kp_x * kp_scale, cfg.p_softness_x,
                                    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + ff_x + lqr_vel_x) * scale_x
    movement_y = (nonlinear_p_move(error_y, cfg.kp_y * kp_scale, cfg.p_softness_y,
                                    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + ff_y + lqr_vel_y) * scale_y

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

    # CGA (disabled in production but kept for completeness)
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
# LQR simulation runner
# ---------------------------------------------------------------------------
def run_sim_lqr(cfg: AimConfig, K_ctrl: np.ndarray,
                traj_name: str, seed: int) -> SimResult:
    """Run one simulation episode with the LQR controller."""
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "lqr", seed, N_FRAMES)

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
            # cfg.kp_x/kp_y already contain K_ctrl position gains;
            # rescale_aim_config_for_rate will scale them by r (like normal kp).
            frame_cfg = dataclasses.replace(cfg)
            rate_scale = pipeline.compute_rate_scale(frame_period_ms)
            if rate_scale != 1.0:
                rescale_aim_config_for_rate(frame_cfg, rate_scale)
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(frame_period_ms)
            dx, dy = compute_aim_movement_lqr(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf, K_ctrl)

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
# Parameter space definition (18 params total)
# ---------------------------------------------------------------------------
@dataclass
class ParamDef:
    name: str
    lo: float
    hi: float
    init: float

PARAM_DEFS = [
    # LQR cost weights (6 params, indices 0-5)
    ParamDef("Q_ex",               0.1,  100.0,  10.0),
    ParamDef("Q_ey",               0.1,  100.0,  10.0),
    ParamDef("Q_vx",               0.01,  50.0,   1.0),
    ParamDef("Q_vy",               0.01,  50.0,   1.0),
    ParamDef("R_x",                0.01,  50.0,   1.0),
    ParamDef("R_y",                0.01,  50.0,   1.0),
    # Controller params (12 params, indices 6-17)
    ParamDef("p_softness_x",       2.0,   25.0,   7.39),
    ParamDef("p_softness_y",       2.0,   25.0,  20.0),
    ParamDef("ff_gain",            0.0,    5.0,   2.08),
    ParamDef("predict_frames",     0.0,    6.0,   4.30),
    ParamDef("deadtime_frames",    1.0,    4.0,   3.90),
    ParamDef("oneeuro_min_cutoff", 0.001,  0.5,   0.03),
    ParamDef("oneeuro_beta",       0.001,  0.5,   0.05),
    ParamDef("max_step",           3.0,   30.0,  14.55),
    ParamDef("akp_large_mult",     1.0,    2.0,   1.28),
    ParamDef("akp_small_mult",     0.5,    1.0,   0.97),
    ParamDef("ff_v_ema",           0.01,   0.5,   0.097),
    ParamDef("lead_vgate",         1.0,   20.0,   5.82),
]

assert len(PARAM_DEFS) == 18, f"Expected 18 params, got {len(PARAM_DEFS)}"


# ---------------------------------------------------------------------------
# Sigmoid transform for bounded optimisation
# ---------------------------------------------------------------------------
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
# Config builder
# ---------------------------------------------------------------------------
def make_config_lqr(params: np.ndarray, K_ctrl: np.ndarray) -> AimConfig:
    """Build AimConfig from optimisation parameters and LQR gains.

    Sets cfg.kp_x/kp_y to K_ctrl position gains (so rate scaling works),
    sets kd to 0 (LQR handles derivative through velocity state),
    and fills remaining params from the CMA-ES vector.
    """
    cfg = AimConfig()
    # LQR position gains replace kp
    cfg.kp_x = float(K_ctrl[0, 0])
    cfg.kp_y = float(K_ctrl[1, 1])
    # No kd -- LQR velocity state handles derivative
    cfg.kd_x = 0.0
    cfg.kd_y = 0.0
    # Remaining params from CMA-ES vector (indices 6-17)
    cfg.p_softness_x       = float(params[6])
    cfg.p_softness_y       = float(params[7])
    cfg.ff_gain             = float(params[8])
    cfg.predict_frames      = float(params[9])
    cfg.deadtime_frames     = float(params[10])
    cfg.oneeuro_min_cutoff  = float(params[11])
    cfg.oneeuro_beta        = float(params[12])
    cfg.max_step            = float(params[13])
    # Adaptive kp: thresholds fixed, multipliers from CMA-ES
    cfg.akp_large_thresh    = FIXED_AKP_LARGE_THRESH
    cfg.akp_small_thresh    = FIXED_AKP_SMALL_THRESH
    cfg.akp_large_mult      = float(params[14])
    cfg.akp_small_mult      = float(params[15])
    cfg.ff_v_ema            = float(params[16])
    cfg.lead_vgate          = float(params[17])
    # Algorithm flags: production combo
    cfg.sem_enabled  = 1.0
    cfg.cga_enabled  = 0.0
    cfg.jcad_enabled = 1.0
    cfg.pags_enabled = 1.0
    return cfg


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------
def compute_baseline_maes() -> Dict[str, float]:
    """Compute per-trajectory MAE for the production baseline."""
    baseline_cfg = CONFIGS["production"]()
    maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_custom(baseline_cfg, traj, seed)
            errors.extend(r.errors)
        maes[traj] = sum(errors) / len(errors)
    return maes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(params: np.ndarray, baseline_maes: Dict[str, float]) -> float:
    """Evaluate one parameter vector.  Returns fitness (lower = better).

    Computes K_ctrl from Q/R weights, runs sim across all trajectories
    and seeds, and applies a no-regression penalty.
    """
    # Decode LQR weights
    Q_ex, Q_ey, Q_vx, Q_vy = params[0], params[1], params[2], params[3]
    R_x, R_y = params[4], params[5]

    K_ctrl = compute_lqr_gains(Q_ex, Q_ey, Q_vx, Q_vy, R_x, R_y)
    if K_ctrl is None:
        return 1e6  # infeasible

    cfg = make_config_lqr(params, K_ctrl)
    traj_maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_lqr(cfg, K_ctrl, traj, seed)
            errors.extend(r.errors)
        traj_maes[traj] = sum(errors) / len(errors)

    avg_mae = sum(traj_maes.values()) / len(traj_maes)

    # Strict no-regression penalty
    penalty = 0.0
    for traj in TRAJECTORIES_LIST:
        if traj_maes[traj] > baseline_maes[traj] * 1.001:
            excess = (traj_maes[traj] - baseline_maes[traj]) / baseline_maes[traj]
            penalty += excess * 50.0

    return avg_mae + penalty


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LQR State-Space Controller Experiment")
    print("=" * 70)
    print(f"Params: {len(PARAM_DEFS)} (6 LQR weights + 12 controller)")
    print(f"Seeds: {len(SEEDS)} | Trajectories: {len(TRAJECTORIES_LIST)}")
    print(f"Algo: SEM+JCAD+PAGS (production combo)")
    print(f"Evals per fitness: {len(SEEDS) * len(TRAJECTORIES_LIST)} sim runs")
    print()

    # --- Baseline ---
    print("Computing baseline MAEs (production config)...")
    t0 = time.time()
    baseline_maes = compute_baseline_maes()
    baseline_avg = sum(baseline_maes.values()) / len(baseline_maes)
    print(f"Baseline avg MAE: {baseline_avg:.4f} ({time.time() - t0:.1f}s)")
    for t, m in baseline_maes.items():
        print(f"  {t:15s}: {m:.4f}")
    print()

    # --- Verify initial LQR gains ---
    K_init = compute_lqr_gains(10.0, 10.0, 1.0, 1.0, 1.0, 1.0)
    if K_init is not None:
        print("Initial K_ctrl (Q=diag(10,10,1,1), R=diag(1,1)):")
        print(f"  Position gains: kp_x={K_init[0,0]:.4f}, kp_y={K_init[1,1]:.4f}")
        print(f"  Velocity gains: vx={K_init[0,2]:.4f}, vy={K_init[1,3]:.4f}")
        print(f"  Cross terms:    K[0,1]={K_init[0,1]:.6f}, K[1,0]={K_init[1,0]:.6f}")
        print(f"                  K[0,3]={K_init[0,3]:.6f}, K[1,2]={K_init[1,2]:.6f}")
    else:
        print("WARNING: Initial LQR gains failed to compute!")
    print()

    # --- CMA-ES optimisation ---
    x0_bounded = np.array([pd.init for pd in PARAM_DEFS])
    x0_z = from_bounded(x0_bounded)

    POPSIZE = 16
    MAX_GENS = 100
    EARLY_STOP = 50

    es = CMAES(x0_z, sigma0=0.5, popsize=POPSIZE)
    best_fitness = float("inf")
    best_params = None
    best_K_ctrl = None
    stagnation = 0
    t_start = time.time()

    print(f"CMA-ES: popsize={POPSIZE}, max_gens={MAX_GENS}, "
          f"early_stop={EARLY_STOP}")
    print("-" * 70)

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
            best_K_ctrl = compute_lqr_gains(
                best_params[0], best_params[1],
                best_params[2], best_params[3],
                best_params[4], best_params[5])
            stagnation = 0
        else:
            stagnation += 1

        if gen % 5 == 0 or gen_best < best_fitness + 0.001:
            elapsed = time.time() - t_start
            improv = (best_fitness - baseline_avg) / baseline_avg * 100
            print(f"  Gen {gen:3d} | best={best_fitness:.4f} "
                  f"({improv:+.1f}%) | gen_best={gen_best:.4f} "
                  f"| sigma={es.sigma:.4f} | {elapsed:.0f}s")

        if stagnation > EARLY_STOP:
            print(f"  Early stop: {stagnation} gens without improvement")
            break

    total_time = time.time() - t_start
    print(f"\nOptimisation complete: {gen + 1} generations, {total_time:.0f}s")

    # --- Results ---
    if best_params is None or best_K_ctrl is None:
        print("ERROR: No valid solution found.")
        return

    improv = (best_fitness - baseline_avg) / baseline_avg * 100
    print(f"\n{'='*70}")
    print(f"Best LQR MAE: {best_fitness:.4f} ({improv:+.1f}% vs baseline {baseline_avg:.4f})")
    print(f"{'='*70}")

    # K gains
    print(f"\nK_ctrl gains:")
    print(f"  Position: kp_x={best_K_ctrl[0,0]:.4f}, kp_y={best_K_ctrl[1,1]:.4f}")
    print(f"  Velocity: vx={best_K_ctrl[0,2]:.4f}, vy={best_K_ctrl[1,3]:.4f}")

    # Per-trajectory breakdown
    cfg = make_config_lqr(best_params, best_K_ctrl)
    traj_detail = {}
    print(f"\nPer-trajectory results:")
    print(f"  {'Trajectory':<16} {'LQR MAE':>8} {'Baseline':>8} {'Delta':>8} {'Status'}")
    print(f"  {'-'*56}")
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_lqr(cfg, best_K_ctrl, traj, seed)
            errors.extend(r.errors)
        mae = sum(errors) / len(errors)
        traj_detail[traj] = mae
        b = baseline_maes[traj]
        ti = (mae - b) / b * 100
        status = "OK" if mae <= b * 1.001 else "REGRESS"
        print(f"  {traj:<16} {mae:8.4f} {b:8.4f} {ti:+7.1f}% [{status}]")

    # Compare K gains to PD baseline gains
    prod_cfg = CONFIGS["production"]()
    print(f"\nGain comparison (LQR vs PD baseline):")
    print(f"  kp_x: LQR={best_K_ctrl[0,0]:.4f} vs PD={prod_cfg.kp_x:.4f}")
    print(f"  kp_y: LQR={best_K_ctrl[1,1]:.4f} vs PD={prod_cfg.kp_y:.4f}")
    print(f"  kd_x: LQR=0 (velocity state) vs PD={prod_cfg.kd_x:.4f}")
    print(f"  kd_y: LQR=0 (velocity state) vs PD={prod_cfg.kd_y:.4f}")
    print(f"  LQR velocity ff: vx={best_K_ctrl[0,2]:.4f}, vy={best_K_ctrl[1,3]:.4f}")
    print(f"  PD ff_gain:      {prod_cfg.ff_gain:.4f}")

    # --- Save results ---
    param_dict = {pd.name: float(best_params[i]) for i, pd in enumerate(PARAM_DEFS)}
    output = {
        "experiment": "lqr_state_space",
        "best_mae": float(best_fitness),
        "baseline_avg_mae": float(baseline_avg),
        "improvement_pct": float(improv),
        "K_ctrl": best_K_ctrl.tolist(),
        "lqr_weights": {
            "Q_ex": float(best_params[0]),
            "Q_ey": float(best_params[1]),
            "Q_vx": float(best_params[2]),
            "Q_vy": float(best_params[3]),
            "R_x": float(best_params[4]),
            "R_y": float(best_params[5]),
        },
        "params": param_dict,
        "algo": {"sem": 1, "cga": 0, "jcad": 1, "pags": 1},
        "per_trajectory": {
            t: {
                "mae": float(traj_detail[t]),
                "baseline": float(baseline_maes[t]),
                "improv_pct": float((traj_detail[t] - baseline_maes[t]) / baseline_maes[t] * 100),
            }
            for t in TRAJECTORIES_LIST
        },
        "generations": gen + 1,
        "time_seconds": round(total_time, 1),
        "config": {
            "popsize": POPSIZE,
            "max_gens": MAX_GENS,
            "early_stop": EARLY_STOP,
            "seeds": SEEDS,
            "n_frames": N_FRAMES,
            "fixed_akp_large_thresh": FIXED_AKP_LARGE_THRESH,
            "fixed_akp_small_thresh": FIXED_AKP_SMALL_THRESH,
        },
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "experiment_lqr_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
