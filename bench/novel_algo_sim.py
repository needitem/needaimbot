#!/usr/bin/env python3
"""Synthetic closed-loop simulation of the PD controller with 4 novel algorithms.

Replicates the computeAimMovement() math from pd_controller.cuh in pure Python,
runs 6 scenarios x 8 configurations, and reports a comparison table.

Scenarios:
  A. Stationary target (step response from 50px offset)
  B. Linear motion (2 px/frame constant velocity)
  C. Direction change (right 100 frames, reverse left)
  D. Jittery stationary (gaussian noise ±5px on fixed position)
  E. Low confidence (linear motion, conf drops to 0.3-0.5 randomly)
  F. Oscillation-inducing (high kp to trigger ringing)

Configurations:
  1. Baseline (all novel OFF)
  2. SEM only
  3. CGA only
  4. JCAD only
  5. PAGS only
  6. All ON
  7. SEM + PAGS
  8. JCAD + CGA
"""

import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

INFLIGHT_MAX = 4
NUM_FRAMES = 300
SCREEN_CENTER = 160.0  # model-space center (320/2)


@dataclass
class AimState:
    residual_x: float = 0.0
    residual_y: float = 0.0
    has_track: int = 0
    prev_raw_x: float = 0.0
    prev_raw_y: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    prev_vel_x: float = 0.0
    prev_vel_y: float = 0.0
    prev_class: int = -1
    filt_x: float = 0.0
    filt_y: float = 0.0
    dfilt_x: float = 0.0
    dfilt_y: float = 0.0
    prev_err_x: float = 0.0
    prev_err_y: float = 0.0
    derr_x: float = 0.0
    derr_y: float = 0.0
    inflight_x: list = field(default_factory=lambda: [0.0] * INFLIGHT_MAX)
    inflight_y: list = field(default_factory=lambda: [0.0] * INFLIGHT_MAX)
    inflight_head: int = 0
    med_raw_x: list = field(default_factory=lambda: [0.0, 0.0])
    med_raw_y: list = field(default_factory=lambda: [0.0, 0.0])
    med_initialized: int = 0
    # SEM
    sem_x: float = 0.0
    sem_y: float = 0.0
    # JCAD
    jcad_ew_mean: float = 0.0
    jcad_ew_var: float = 0.0
    # PAGS
    pags_flip_rate_x: float = 0.0
    pags_flip_rate_y: float = 0.0
    pags_prev_sign_x: int = 0
    pags_prev_sign_y: int = 0
    pags_gain_mult: float = 1.0


@dataclass
class AimConfig:
    kp_x: float = 0.55
    kp_y: float = 0.6
    p_softness_x: float = 11.0
    p_softness_y: float = 10.0
    kd_x: float = 0.18
    kd_y: float = 0.22
    fine_track_boost: float = 1.3
    fine_track_range: float = 0.5
    max_step: float = 30.0
    inflight_comp: float = 0.75
    deadtime_frames: float = 1.0
    ff_gain: float = 2.924
    ff_ego_lag: float = 2.85
    ff_v_ema: float = 0.2
    lead_vgate: float = 9.0
    lead_err_gate: float = 18.0
    predict_frames: float = 5.47
    accel_ema: float = 0.15
    class_switch_reject: float = 1.0
    oneeuro_enabled: float = 1.0
    oneeuro_min_cutoff: float = 0.1
    oneeuro_beta: float = 0.02
    aim_output_enabled: float = 1.0
    aim_y_scale: float = 1.0
    shoot_offset_x: float = 0.0
    shoot_offset_y: float = 0.0
    movement_scale: float = 1.0
    # SEM
    sem_enabled: float = 0.0
    sem_rate: float = 0.02
    sem_max: float = 3.0
    sem_decay: float = 0.95
    # CGA
    cga_enabled: float = 0.0
    cga_conf_low: float = 0.4
    cga_conf_high: float = 0.8
    cga_speed_threshold: float = 3.0
    # JCAD
    jcad_enabled: float = 0.0
    jcad_alpha: float = 0.15
    jcad_base_db: float = 0.5
    jcad_jitter_db: float = 3.0
    jcad_jitter_threshold: float = 2.0
    # PAGS
    pags_enabled: float = 0.0
    pags_acquire_threshold: float = 15.0
    pags_osc_threshold: float = 0.5
    pags_acquire_mult: float = 1.1
    pags_osc_mult: float = 0.6
    pags_alpha: float = 0.2
    # Kalman (off for sim)


def one_euro_alpha(cutoff):
    w = 2.0 * math.pi * max(cutoff, 1e-4)
    return w / (w + 1.0)

DERIV_ALPHA = one_euro_alpha(1.0)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_max_step(mx, my, max_step):
    if max_step > 0:
        step = math.sqrt(mx*mx + my*my)
        if step > max_step:
            s = max_step / step
            mx *= s
            my *= s
    return mx, my

def nonlinear_p_move(error, kp, softness, fine_boost=1.0, fine_range=0.0):
    abs_error = abs(error)
    safe_softness = max(softness, 1.0)
    effective_kp = max(kp, 0.0)
    if fine_boost > 1.0 and fine_range > 0.0:
        threshold = safe_softness * fine_range
        if abs_error < threshold:
            t = abs_error / threshold
            effective_kp *= fine_boost * (1.0 - t) + t
    gain = effective_kp * (abs_error / (abs_error + safe_softness))
    return error * gain

def push_inflight(s, dx, dy):
    i = s.inflight_head
    s.inflight_x[i] = dx
    s.inflight_y[i] = dy
    s.inflight_head = (i + 1) % INFLIGHT_MAX

def inflight_sum(s, frames):
    sx, sy = 0.0, 0.0
    w = clamp(frames, 0.0, float(INFLIGHT_MAX))
    n = int(math.floor(w))
    for k in range(1, n + 1):
        i = (s.inflight_head - k + INFLIGHT_MAX) % INFLIGHT_MAX
        sx += s.inflight_x[i]
        sy += s.inflight_y[i]
    frac = w - float(n)
    if frac > 0.0 and (n + 1) <= INFLIGHT_MAX:
        i = (s.inflight_head - (n + 1) + INFLIGHT_MAX) % INFLIGHT_MAX
        sx += frac * s.inflight_x[i]
        sy += frac * s.inflight_y[i]
    return sx, sy

def inflight_at(s, lag):
    w = clamp(lag, 1.0, float(INFLIGHT_MAX))
    n = int(math.floor(w))
    frac = w - float(n)
    i = (s.inflight_head - n + INFLIGHT_MAX) % INFLIGHT_MAX
    ax = s.inflight_x[i]
    ay = s.inflight_y[i]
    if frac > 0.0 and (n + 1) <= INFLIGHT_MAX:
        j = (s.inflight_head - (n + 1) + INFLIGHT_MAX) % INFLIGHT_MAX
        ax = (1.0 - frac) * ax + frac * s.inflight_x[j]
        ay = (1.0 - frac) * ay + frac * s.inflight_y[j]
    return ax, ay

def lead_gate(cfg, s, error_x, error_y):
    g = 1.0
    if cfg.lead_vgate > 0:
        sp = math.sqrt(s.vel_x**2 + s.vel_y**2)
        g *= sp / (sp + cfg.lead_vgate)
    if cfg.lead_err_gate > 0:
        e0 = cfg.lead_err_gate
        e2 = error_x**2 + error_y**2
        g *= (e0*e0) / (e0*e0 + e2)
    return g

def median3(a, b, c):
    return max(min(a, b), min(max(a, b), c))

def apply_median_prefilter(raw_x, raw_y, s):
    if s.med_initialized < 2:
        s.med_raw_x[s.med_initialized] = raw_x
        s.med_raw_y[s.med_initialized] = raw_y
        s.med_initialized += 1
        return raw_x, raw_y
    out_x = median3(s.med_raw_x[0], s.med_raw_x[1], raw_x)
    out_y = median3(s.med_raw_y[0], s.med_raw_y[1], raw_y)
    s.med_raw_x[0] = s.med_raw_x[1]
    s.med_raw_x[1] = raw_x
    s.med_raw_y[0] = s.med_raw_y[1]
    s.med_raw_y[1] = raw_y
    return out_x, out_y

def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0 + 1e-7), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def emit_mouse_delta(movement, residual):
    carried = residual
    if movement * carried < 0:
        carried = 0.0
    value = movement + carried
    emit = int(value)  # truncate toward zero
    if emit > 127:
        new_res = min(256.0, value - 127.0)
        return 127, new_res
    if emit < -127:
        new_res = max(-256.0, value + 127.0)
        return -127, new_res
    new_res = value - float(emit)
    return emit, new_res


def compute_aim_movement(raw_cx, raw_cy, screen_cx, screen_cy,
                          ms_x, ms_y, cfg, s, target_class, det_conf):
    fresh_track = (s.has_track == 0)
    class_changed = (cfg.class_switch_reject != 0.0 and not fresh_track and
                     s.prev_class >= 0 and target_class != s.prev_class)
    if class_changed:
        s.prev_raw_x = raw_cx
        s.prev_raw_y = raw_cy
    s.prev_class = target_class

    # Inflight sum for dead-time comp
    inf_x, inf_y = 0.0, 0.0
    use_comp = (cfg.inflight_comp > 0 and cfg.deadtime_frames > 0)
    if use_comp:
        inf_x, inf_y = inflight_sum(s, cfg.deadtime_frames)

    # One Euro filter
    target_cx, target_cy = raw_cx, raw_cy
    if cfg.oneeuro_enabled != 0:
        in_x, in_y = raw_cx, raw_cy
        if s.has_track:
            ad = DERIV_ALPHA
            de_x = in_x - s.filt_x
            s.dfilt_x = ad * de_x + (1 - ad) * s.dfilt_x
            cutoff_x = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_x)
            ax = one_euro_alpha(cutoff_x)
            s.filt_x = ax * in_x + (1 - ax) * s.filt_x
            de_y = in_y - s.filt_y
            s.dfilt_y = ad * de_y + (1 - ad) * s.dfilt_y
            cutoff_y = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_y)
            ay = one_euro_alpha(cutoff_y)
            s.filt_y = ay * in_y + (1 - ay) * s.filt_y
        else:
            s.filt_x = in_x
            s.filt_y = in_y
            s.dfilt_x = 0.0
            s.dfilt_y = 0.0
        target_cx = s.filt_x
        target_cy = s.filt_y

    # JCAD: capture displacement
    if cfg.jcad_enabled != 0:
        if fresh_track:
            s.jcad_ew_mean = 0.0
            s.jcad_ew_var = 0.0
        else:
            jdx = raw_cx - s.prev_raw_x
            jdy = raw_cy - s.prev_raw_y
            jcad_disp = math.sqrt(jdx*jdx + jdy*jdy)
            ja = cfg.jcad_alpha
            s.jcad_ew_mean = ja * jcad_disp + (1 - ja) * s.jcad_ew_mean
            jdiff = jcad_disp - s.jcad_ew_mean
            s.jcad_ew_var = ja * jdiff * jdiff + (1 - ja) * s.jcad_ew_var

    # Velocity estimation (EMA path)
    lead_active = (cfg.ff_gain > 0 or cfg.predict_frames > 0)
    if lead_active:
        med_x, med_y = apply_median_prefilter(raw_cx, raw_cy, s)
        if s.has_track:
            ego_x, ego_y = inflight_at(s, cfg.ff_ego_lag)
            max_drift = 60.0
            dvx = (med_x - s.prev_raw_x) + (ego_x / ms_x if ms_x != 0 else 0)
            dvy = (med_y - s.prev_raw_y) + (ego_y / ms_y if ms_y != 0 else 0)
            dvx = clamp(dvx, -max_drift, max_drift)
            dvy = clamp(dvy, -max_drift, max_drift)
            a = clamp(cfg.ff_v_ema, 0, 1)
            s.vel_x = (1 - a) * s.vel_x + a * dvx
            s.vel_y = (1 - a) * s.vel_y + a * dvy
        else:
            s.vel_x = 0.0
            s.vel_y = 0.0
            s.med_initialized = 0
            s.accel_x = 0.0
            s.accel_y = 0.0
            s.prev_vel_x = 0.0
            s.prev_vel_y = 0.0

        if cfg.accel_ema > 0:
            a_ema = clamp(cfg.accel_ema, 0, 1)
            dax = s.vel_x - s.prev_vel_x
            day = s.vel_y - s.prev_vel_y
            s.accel_x = (1 - a_ema) * s.accel_x + a_ema * clamp(dax, -15, 15)
            s.accel_y = (1 - a_ema) * s.accel_y + a_ema * clamp(day, -15, 15)
        s.prev_vel_x = s.vel_x
        s.prev_vel_y = s.vel_y
        s.prev_raw_x = med_x
        s.prev_raw_y = med_y

    s.has_track = 1

    # Error
    error_x = target_cx - screen_cx
    error_y = target_cy - screen_cy

    # Dead-time compensation
    if use_comp:
        if ms_x != 0:
            error_x -= cfg.inflight_comp * inf_x / ms_x
        if ms_y != 0:
            error_y -= cfg.inflight_comp * inf_y / ms_y

    # JCAD: adaptive deadband
    if cfg.jcad_enabled != 0:
        mean_sq = s.jcad_ew_mean * s.jcad_ew_mean
        J = s.jcad_ew_var / (mean_sq + 1e-6)
        sat_J = min(J / cfg.jcad_jitter_threshold, 1.0)
        db = cfg.jcad_base_db + cfg.jcad_jitter_db * sat_J
        err_mag = math.sqrt(error_x**2 + error_y**2)
        if err_mag < db:
            error_x = 0.0
            error_y = 0.0

    # Target prediction (setpoint shift)
    if cfg.predict_frames > 0:
        g = lead_gate(cfg, s, error_x, error_y)
        t = cfg.predict_frames
        error_x += g * (t * s.vel_x + 0.5 * t * t * s.accel_x)
        error_y += g * (t * s.vel_y + 0.5 * t * t * s.accel_y)

    # Derivative term
    max_derr = 150.0
    de_x = clamp(error_x - s.prev_err_x, -max_derr, max_derr)
    de_y = clamp(error_y - s.prev_err_y, -max_derr, max_derr)
    if fresh_track:
        s.derr_x = 0.0
        s.derr_y = 0.0
    else:
        s.derr_x = 0.6 * de_x + 0.4 * s.derr_x
        s.derr_y = 0.6 * de_y + 0.4 * s.derr_y
    s.prev_err_x = error_x
    s.prev_err_y = error_y

    # Feedforward
    ff_x, ff_y = 0.0, 0.0
    if cfg.ff_gain > 0:
        g = lead_gate(cfg, s, error_x, error_y)
        ff_x = cfg.ff_gain * g * s.vel_x
        ff_y = cfg.ff_gain * g * s.vel_y

    # Movement
    movement_x = (nonlinear_p_move(error_x, cfg.kp_x, cfg.p_softness_x,
                                    cfg.fine_track_boost, cfg.fine_track_range)
                  + cfg.kd_x * s.derr_x + ff_x) * ms_x
    movement_y = (nonlinear_p_move(error_y, cfg.kp_y, cfg.p_softness_y,
                                    cfg.fine_track_boost, cfg.fine_track_range)
                  + cfg.kd_y * s.derr_y + ff_y) * ms_y

    # SEM
    if cfg.sem_enabled != 0:
        if fresh_track:
            s.sem_x = 0.0
            s.sem_y = 0.0
        s.sem_x *= cfg.sem_decay
        s.sem_y *= cfg.sem_decay
        if error_x * s.sem_x < 0:
            s.sem_x = 0.0
        s.sem_x += cfg.sem_rate * error_x
        s.sem_x = clamp(s.sem_x, -cfg.sem_max, cfg.sem_max)
        if error_y * s.sem_y < 0:
            s.sem_y = 0.0
        s.sem_y += cfg.sem_rate * error_y
        s.sem_y = clamp(s.sem_y, -cfg.sem_max, cfg.sem_max)
        movement_x += s.sem_x * ms_x
        movement_y += s.sem_y * ms_y

    # PAGS
    if cfg.pags_enabled != 0:
        if fresh_track:
            s.pags_flip_rate_x = 0.0
            s.pags_flip_rate_y = 0.0
            s.pags_prev_sign_x = 0
            s.pags_prev_sign_y = 0
            s.pags_gain_mult = 1.0
        sign_x = 1 if error_x > 0 else (-1 if error_x < 0 else 0)
        sign_y = 1 if error_y > 0 else (-1 if error_y < 0 else 0)
        flip_x = 1.0 if (s.pags_prev_sign_x != 0 and sign_x != 0 and
                         sign_x != s.pags_prev_sign_x) else 0.0
        flip_y = 1.0 if (s.pags_prev_sign_y != 0 and sign_y != 0 and
                         sign_y != s.pags_prev_sign_y) else 0.0
        pa = cfg.pags_alpha
        s.pags_flip_rate_x = pa * flip_x + (1 - pa) * s.pags_flip_rate_x
        s.pags_flip_rate_y = pa * flip_y + (1 - pa) * s.pags_flip_rate_y
        if sign_x != 0:
            s.pags_prev_sign_x = sign_x
        if sign_y != 0:
            s.pags_prev_sign_y = sign_y
        err_mag = math.sqrt(error_x**2 + error_y**2)
        flip_rate = max(s.pags_flip_rate_x, s.pags_flip_rate_y)
        if err_mag > cfg.pags_acquire_threshold:
            target_mult = cfg.pags_acquire_mult
        elif flip_rate > cfg.pags_osc_threshold:
            target_mult = cfg.pags_osc_mult
        else:
            target_mult = 1.0
        s.pags_gain_mult = 0.3 * target_mult + 0.7 * s.pags_gain_mult
        movement_x *= s.pags_gain_mult
        movement_y *= s.pags_gain_mult

    # Max step clamp
    movement_x, movement_y = clamp_max_step(movement_x, movement_y, cfg.max_step)
    movement_y *= cfg.aim_y_scale
    movement_x, movement_y = clamp_max_step(movement_x, movement_y, cfg.max_step)

    # CGA
    if cfg.cga_enabled != 0:
        speed = math.sqrt(s.vel_x**2 + s.vel_y**2)
        speed_bypass = speed / (speed + cfg.cga_speed_threshold + 1e-7)
        conf_gate = smoothstep(cfg.cga_conf_low, cfg.cga_conf_high, det_conf)
        gate = max(conf_gate, speed_bypass)
        movement_x *= gate
        movement_y *= gate

    # Output gate
    if cfg.aim_output_enabled == 0:
        s.residual_x = 0.0
        s.residual_y = 0.0
        push_inflight(s, 0, 0)
        return 0, 0

    out_dx, s.residual_x = emit_mouse_delta(movement_x, s.residual_x)
    out_dy, s.residual_y = emit_mouse_delta(movement_y, s.residual_y)
    push_inflight(s, float(out_dx), float(out_dy))
    return out_dx, out_dy


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_stationary(frame, _rng):
    """Fixed target at (210, 160), crosshair starts at (160, 160)."""
    return 210.0, 160.0, 0.85

def scenario_linear(frame, _rng):
    """Target moves right at 2 px/frame from (160, 160)."""
    return 160.0 + 2.0 * frame, 160.0, 0.85

def scenario_direction_change(frame, _rng):
    """Right 100 frames at 2px/f, then reverses left at 2px/f."""
    if frame < 100:
        return 160.0 + 2.0 * frame, 160.0, 0.85
    else:
        return 160.0 + 2.0 * 100 - 2.0 * (frame - 100), 160.0, 0.85

def scenario_jittery(frame, rng):
    """Fixed at (210, 160) + gaussian noise N(0, 5)."""
    return 210.0 + rng.gauss(0, 5), 160.0 + rng.gauss(0, 5), 0.85

def scenario_low_confidence(frame, rng):
    """Linear motion, but confidence randomly drops to 0.3-0.5."""
    conf = 0.85 if rng.random() > 0.4 else rng.uniform(0.3, 0.5)
    return 160.0 + 1.5 * frame, 160.0, conf

def scenario_high_gain_osc(frame, _rng):
    """Stationary target, but config uses high kp to induce oscillation."""
    return 210.0, 160.0, 0.85

SCENARIOS = {
    "A_stationary":    scenario_stationary,
    "B_linear":        scenario_linear,
    "C_dir_change":    scenario_direction_change,
    "D_jittery":       scenario_jittery,
    "E_low_conf":      scenario_low_confidence,
    "F_high_gain_osc": scenario_high_gain_osc,
}


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def make_config(name, scenario_name):
    cfg = AimConfig()
    cfg.movement_scale = 1.0

    # For the oscillation scenario, crank up kp
    if scenario_name == "F_high_gain_osc":
        cfg.kp_x = 1.2
        cfg.kp_y = 1.2
        cfg.kd_x = 0.05
        cfg.kd_y = 0.05

    if name == "baseline":
        pass
    elif name == "SEM":
        cfg.sem_enabled = 1.0
    elif name == "CGA":
        cfg.cga_enabled = 1.0
    elif name == "JCAD":
        cfg.jcad_enabled = 1.0
    elif name == "PAGS":
        cfg.pags_enabled = 1.0
    elif name == "ALL":
        cfg.sem_enabled = 1.0
        cfg.cga_enabled = 1.0
        cfg.jcad_enabled = 1.0
        cfg.pags_enabled = 1.0
    elif name == "SEM+PAGS":
        cfg.sem_enabled = 1.0
        cfg.pags_enabled = 1.0
    elif name == "JCAD+CGA":
        cfg.jcad_enabled = 1.0
        cfg.cga_enabled = 1.0
    return cfg

CONFIGS = ["baseline", "SEM", "CGA", "JCAD", "PAGS", "ALL", "SEM+PAGS", "JCAD+CGA"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    settling_time: int = -1       # frames to reach <1px error (or -1)
    max_overshoot: float = 0.0    # max error magnitude AFTER first settling
    ss_rms_error: float = 0.0     # RMS error over last 50 frames
    osc_count: int = 0            # error sign changes in last 100 frames
    mean_abs_error: float = 0.0   # mean |error| over all frames

def compute_metrics(errors_x, errors_y):
    n = len(errors_x)
    m = Metrics()

    mags = [math.sqrt(ex**2 + ey**2) for ex, ey in zip(errors_x, errors_y)]
    m.mean_abs_error = sum(mags) / n if n > 0 else 0

    # Settling time: first frame where |error| < 1 and stays below for 10 frames
    settled_frame = -1
    for i in range(n - 10):
        if all(mags[j] < 1.0 for j in range(i, min(i + 10, n))):
            settled_frame = i
            break
    m.settling_time = settled_frame

    # Max overshoot: largest error AFTER first settling
    if settled_frame >= 0 and settled_frame < n:
        m.max_overshoot = max(mags[settled_frame:])

    # Steady-state RMS (last 50 frames)
    tail = mags[-50:]
    m.ss_rms_error = math.sqrt(sum(e**2 for e in tail) / len(tail))

    # Oscillation count (sign changes in last 100 frames, X axis)
    tail_ex = errors_x[-100:]
    sign_changes = 0
    for i in range(1, len(tail_ex)):
        if tail_ex[i] * tail_ex[i-1] < 0:
            sign_changes += 1
    m.osc_count = sign_changes

    return m


# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_experiment(scenario_name, config_name, seed=42):
    rng = random.Random(seed)
    cfg = make_config(config_name, scenario_name)
    s = AimState()
    scenario_fn = SCENARIOS[scenario_name]

    # Crosshair position (starts at screen center, moves by emitted deltas)
    cx, cy = SCREEN_CENTER, SCREEN_CENTER

    errors_x = []
    errors_y = []

    for frame in range(NUM_FRAMES):
        # Target position (in model space)
        tx, ty, conf = scenario_fn(frame, rng)

        ms = cfg.movement_scale
        dx, dy = compute_aim_movement(tx, ty, cx, cy, ms, ms, cfg, s, 0, conf)

        # Update crosshair position
        cx += dx * ms
        cy += dy * ms

        # Record error AFTER movement
        errors_x.append(tx - cx)
        errors_y.append(ty - cy)

    return compute_metrics(errors_x, errors_y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Run multiple seeds for stability
    seeds = [42, 123, 456, 789, 1024]

    print(f"{'Scenario':<18} {'Config':<12} {'Settle':>7} {'Overshoot':>10} "
          f"{'SS_RMS':>8} {'OscCnt':>7} {'MAE':>8}")
    print("-" * 76)

    results = {}
    for scenario_name in SCENARIOS:
        for config_name in CONFIGS:
            all_metrics = []
            for seed in seeds:
                m = run_experiment(scenario_name, config_name, seed)
                all_metrics.append(m)

            # Average metrics
            avg = Metrics()
            valid_settles = [m.settling_time for m in all_metrics if m.settling_time >= 0]
            avg.settling_time = int(sum(valid_settles) / len(valid_settles)) if valid_settles else -1
            avg.max_overshoot = sum(m.max_overshoot for m in all_metrics) / len(all_metrics)
            avg.ss_rms_error = sum(m.ss_rms_error for m in all_metrics) / len(all_metrics)
            avg.osc_count = int(sum(m.osc_count for m in all_metrics) / len(all_metrics))
            avg.mean_abs_error = sum(m.mean_abs_error for m in all_metrics) / len(all_metrics)

            settle_str = str(avg.settling_time) if avg.settling_time >= 0 else "never"
            print(f"{scenario_name:<18} {config_name:<12} {settle_str:>7} "
                  f"{avg.max_overshoot:>10.2f} {avg.ss_rms_error:>8.3f} "
                  f"{avg.osc_count:>7} {avg.mean_abs_error:>8.3f}")

            results[(scenario_name, config_name)] = avg

        print()  # blank line between scenarios

    # Summary: compare each config vs baseline
    print("\n" + "=" * 76)
    print("SUMMARY: % improvement vs baseline (negative = worse)")
    print("=" * 76)
    print(f"{'Config':<12} {'MAE_avg':>10} {'SS_RMS_avg':>12} {'Osc_avg':>10}")
    print("-" * 48)

    for config_name in CONFIGS:
        if config_name == "baseline":
            continue
        mae_pcts = []
        rms_pcts = []
        osc_pcts = []
        for scenario_name in SCENARIOS:
            bl = results[(scenario_name, "baseline")]
            cfg_r = results[(scenario_name, config_name)]
            if bl.mean_abs_error > 0.001:
                mae_pcts.append((bl.mean_abs_error - cfg_r.mean_abs_error) / bl.mean_abs_error * 100)
            if bl.ss_rms_error > 0.001:
                rms_pcts.append((bl.ss_rms_error - cfg_r.ss_rms_error) / bl.ss_rms_error * 100)
            if bl.osc_count > 0:
                osc_pcts.append((bl.osc_count - cfg_r.osc_count) / bl.osc_count * 100)

        avg_mae = sum(mae_pcts) / len(mae_pcts) if mae_pcts else 0
        avg_rms = sum(rms_pcts) / len(rms_pcts) if rms_pcts else 0
        avg_osc = sum(osc_pcts) / len(osc_pcts) if osc_pcts else 0
        print(f"{config_name:<12} {avg_mae:>+9.1f}% {avg_rms:>+11.1f}% {avg_osc:>+9.1f}%")


if __name__ == "__main__":
    main()
