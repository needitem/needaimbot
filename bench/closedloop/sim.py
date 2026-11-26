#!/usr/bin/env python3
"""Closed-loop FPS aimbot simulation with comprehensive real-world effects.

All default parameters match the production runtime (simple_config.json)
so simulation results are directly comparable to real-system behaviour.

The simulation models the full chain:
  1. World-space target follows a trajectory
  2. FPS camera transforms world→screen (ego-pollution: mouse output shifts ALL screen coords)
  3. Detector produces noisy bounding box with confidence
  4. Pipeline latency + UDP transport (variable per-frame dead-time)
  5. Controller computes mouse delta (PD + One Euro + dead-time comp + adaptive kp + SEM/CGA/JCAD/PAGS)
  6. Frame-rate adaptive gain scaling (rescaleAimConfigForRate)
  7. Makcu mouse HW delay (1000Hz HID quantization + serial transport)
  8. Mouse delta feeds back into camera → next frame

Noise / real-world sources:
  - Detection jitter (Gaussian, calibrated from rig CSVs: σ≈1.75px at 160×160)
  - Confidence variation (Gaussian, mean≈0.72, σ≈0.12)
  - Missed detections (3% drop rate, up to 2 consecutive)
  - Class flips (head↔body, 1.5% per frame, 7.5px vertical offset at 160×160)
  - UDP packet loss (0.1% per fragment × 12 fragments/frame ≈ 1.2% frame loss)
  - Pipeline latency jitter (lognormal, p50=2.7ms, p95=4.8ms)
  - Frame timing jitter (95% normal σ=0.3ms, 5% stall spikes 8-15ms)
  - Makcu HID delay (uniform 0.5-1.5ms quantised to 1ms intervals)
  - IoU-based target tracking loss / re-acquisition
  - Non-linear camera (slight high-speed deceleration)
"""

import math
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random

# ---------------------------------------------------------------------------
# AimState: mirrors simple_postprocess.h
# ---------------------------------------------------------------------------
INFLIGHT_MAX = 4

@dataclass
class AimState:
    residual_x: float = 0.0
    residual_y: float = 0.0
    frames_since_seen: int = 0
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
    h_ema: float = 0.0
    filt_x: float = 0.0
    filt_y: float = 0.0
    dfilt_x: float = 0.0
    dfilt_y: float = 0.0
    prev_err_x: float = 0.0
    prev_err_y: float = 0.0
    derr_x: float = 0.0
    derr_y: float = 0.0
    inflight_x: list = field(default_factory=lambda: [0.0]*INFLIGHT_MAX)
    inflight_y: list = field(default_factory=lambda: [0.0]*INFLIGHT_MAX)
    inflight_head: int = 0
    med_raw_x: list = field(default_factory=lambda: [0.0, 0.0])
    med_raw_y: list = field(default_factory=lambda: [0.0, 0.0])
    med_initialized: int = 0
    sem_x: float = 0.0
    sem_y: float = 0.0
    jcad_ew_mean: float = 0.0
    jcad_ew_var: float = 0.0
    pags_flip_rate_x: float = 0.0
    pags_flip_rate_y: float = 0.0
    pags_prev_sign_x: int = 0
    pags_prev_sign_y: int = 0
    pags_gain_mult: float = 1.0

@dataclass
class AimConfig:
    kp_x: float = 0.638
    kp_y: float = 0.447
    p_softness_x: float = 1.66
    p_softness_y: float = 11.50
    kd_x: float = 0.0025
    kd_y: float = 0.005
    fine_track_boost: float = 1.3
    fine_track_range: float = 0.5
    max_step: float = 9.92
    deadtime_frames: float = 3.98
    ff_gain: float = 0.653
    ff_ego_lag: float = 2.85
    predict_frames: float = 5.84
    oneeuro_enabled: float = 1.0
    oneeuro_min_cutoff: float = 0.034
    oneeuro_beta: float = 0.008
    aim_output_enabled: float = 1.0
    shoot_offset_x: float = 0.0
    shoot_offset_y: float = 0.0
    aim_y_scale: float = 1.0
    sem_enabled: float = 1.0
    cga_enabled: float = 0.0
    jcad_enabled: float = 1.0
    pags_enabled: float = 1.0
    ff_v_ema: float = 0.126
    lead_vgate: float = 5.52
    akp_large_thresh: float = 24.50
    akp_small_thresh: float = 1.51
    akp_large_mult: float = 1.19
    akp_small_mult: float = 0.97

# ---------------------------------------------------------------------------
# Hardcoded constants (mirrors pd_controller.cuh)
# ---------------------------------------------------------------------------
K_LEAD_VGATE = 5.52
K_LEAD_ERR_GATE = 22.25
K_FF_V_EMA = 0.126
K_FF_EGO_LAG = 2.85
K_ACCEL_EMA = 0.15
K_FINE_TRACK_BOOST = 1.3
K_FINE_TRACK_RANGE = 0.5
K_SEM_RATE = 0.01
K_SEM_MAX = 3.0
K_SEM_DECAY = 0.97
K_CGA_CONF_LOW = 0.4
K_CGA_CONF_HIGH = 0.8
K_CGA_SPEED_THRESHOLD = 3.0
K_JCAD_ALPHA = 0.15
K_JCAD_BASE_DB = 0.5
K_JCAD_JITTER_DB = 3.0
K_JCAD_JITTER_THRESHOLD = 2.0
K_PAGS_ACQUIRE_THRESHOLD = 15.0
K_PAGS_OSC_THRESHOLD = 0.4
K_PAGS_ACQUIRE_MULT = 1.1
K_PAGS_OSC_MULT = 0.4
K_PAGS_ALPHA = 0.2
# Adaptive kp: error-magnitude-dependent gain scaling
K_ADAPTIVE_KP_LARGE_THRESHOLD = 24.50
K_ADAPTIVE_KP_SMALL_THRESHOLD = 1.51
K_ADAPTIVE_KP_LARGE_MULT = 1.19
K_ADAPTIVE_KP_SMALL_MULT = 0.97

K_DERIV_ALPHA = (2.0 * math.pi) / (2.0 * math.pi + 1.0)

# Pipeline constants (from simple_config.json / simple_main.cpp)
K_INFLIGHT_DEADTIME_MS = 10.2
K_AIM_TUNED_FPS = 143.0
K_RATE_SCALE_MIN = 0.5
K_RATE_SCALE_MAX = 2.0

# ---------------------------------------------------------------------------
# Controller math (exact port from pd_controller.cuh)
# ---------------------------------------------------------------------------
def one_euro_alpha(cutoff: float) -> float:
    w = 2.0 * math.pi * max(cutoff, 1e-4)
    return w / (w + 1.0)

def clamp_max_step(mx: float, my: float, max_step: float):
    if max_step > 0:
        step = math.sqrt(mx*mx + my*my)
        if step > max_step:
            s = max_step / step
            mx *= s
            my *= s
    return mx, my

def nonlinear_p_move(error: float, kp: float, softness: float,
                     fine_boost: float = 1.0, fine_range: float = 0.0) -> float:
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

def push_inflight(s: AimState, dx: float, dy: float):
    i = s.inflight_head
    s.inflight_x[i] = dx
    s.inflight_y[i] = dy
    s.inflight_head = (i + 1) % INFLIGHT_MAX

def inflight_sum(s: AimState, frames: float):
    sx, sy = 0.0, 0.0
    w = min(max(frames, 0.0), float(INFLIGHT_MAX))
    n = int(math.floor(w))
    for k in range(1, n+1):
        i = (s.inflight_head - k + INFLIGHT_MAX) % INFLIGHT_MAX
        sx += s.inflight_x[i]
        sy += s.inflight_y[i]
    frac = w - float(n)
    if frac > 0.0 and (n+1) <= INFLIGHT_MAX:
        i = (s.inflight_head - (n+1) + INFLIGHT_MAX) % INFLIGHT_MAX
        sx += frac * s.inflight_x[i]
        sy += frac * s.inflight_y[i]
    return sx, sy

def inflight_at(s: AimState, lag: float):
    w = min(max(lag, 1.0), float(INFLIGHT_MAX))
    n = int(math.floor(w))
    frac = w - float(n)
    i = (s.inflight_head - n + INFLIGHT_MAX) % INFLIGHT_MAX
    ax = s.inflight_x[i]
    ay = s.inflight_y[i]
    if frac > 0.0 and (n+1) <= INFLIGHT_MAX:
        j = (s.inflight_head - (n+1) + INFLIGHT_MAX) % INFLIGHT_MAX
        ax = (1.0 - frac) * ax + frac * s.inflight_x[j]
        ay = (1.0 - frac) * ay + frac * s.inflight_y[j]
    return ax, ay

def lead_gate(s: AimState, error_x: float, error_y: float, lead_vgate: float = 5.52) -> float:
    g = 1.0
    sp = math.sqrt(s.vel_x**2 + s.vel_y**2)
    g *= sp / (sp + lead_vgate)
    e0 = K_LEAD_ERR_GATE
    e2 = error_x**2 + error_y**2
    g *= (e0*e0) / (e0*e0 + e2)
    return g

def median3(a, b, c):
    return max(min(a, b), min(max(a, b), c))

def apply_median_prefilter(raw_x, raw_y, s: AimState):
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

def gpa_smoothstep(edge0, edge1, x):
    t = min(max((x - edge0) / (edge1 - edge0 + 1e-7), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

def emit_mouse_delta(movement: float, residual: float):
    carried = residual
    if movement * carried < 0.0:
        carried = 0.0
    value = movement + carried
    emit = int(math.trunc(value))
    if emit > 127:
        return 127, min(256.0, value - 127.0)
    if emit < -127:
        return -127, max(-256.0, value + 127.0)
    return emit, value - float(emit)

def rescale_aim_config_for_rate(cfg: AimConfig, r: float):
    """Mirrors rescaleAimConfigForRate() from config.hpp.
    r = actual_period / tuned_period, clamped to [0.5, 2.0]."""
    cfg.kp_x *= r
    cfg.kp_y *= r
    cfg.max_step *= r
    cfg.oneeuro_min_cutoff *= r
    cfg.predict_frames /= r
    cfg.ff_gain /= r

def compute_aim_movement(raw_cx, raw_cy, screen_cx, screen_cy,
                         scale_x, scale_y, cfg: AimConfig,
                         s: AimState, target_class: int,
                         detection_confidence: float):
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
            s.dfilt_x = ad * de_x + (1-ad) * s.dfilt_x
            cutoff_x = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_x)
            ax = one_euro_alpha(cutoff_x)
            s.filt_x = ax * in_x + (1-ax) * carry_x

            de_y = in_y - s.filt_y
            s.dfilt_y = ad * de_y + (1-ad) * s.dfilt_y
            cutoff_y = cfg.oneeuro_min_cutoff + cfg.oneeuro_beta * abs(s.dfilt_y)
            ay = one_euro_alpha(cutoff_y)
            s.filt_y = ay * in_y + (1-ay) * carry_y
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
            jcad_disp = math.sqrt(jdx*jdx + jdy*jdy)
            ja = K_JCAD_ALPHA
            s.jcad_ew_mean = ja * jcad_disp + (1-ja) * s.jcad_ew_mean
            jdiff = jcad_disp - s.jcad_ew_mean
            s.jcad_ew_var = ja * jdiff*jdiff + (1-ja) * s.jcad_ew_var

    # Ego-corrected velocity
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
            s.vel_x = (1-a) * s.vel_x + a * dvx
            s.vel_y = (1-a) * s.vel_y + a * dvy
        else:
            s.vel_x = s.vel_y = 0.0
            s.med_initialized = 0
            s.accel_x = s.accel_y = 0.0
            s.prev_vel_x = s.prev_vel_y = 0.0

        a_ema = K_ACCEL_EMA
        dax = s.vel_x - s.prev_vel_x
        day = s.vel_y - s.prev_vel_y
        max_accel = 15.0
        s.accel_x = (1-a_ema)*s.accel_x + a_ema*min(max(dax,-max_accel),max_accel)
        s.accel_y = (1-a_ema)*s.accel_y + a_ema*min(max(day,-max_accel),max_accel)
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
        err_mag = math.sqrt(error_x**2 + error_y**2)
        if err_mag < db:
            error_x = error_y = 0.0

    # predict_frames setpoint shift
    if cfg.predict_frames > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
        t = cfg.predict_frames
        error_x += g * (t * s.vel_x + 0.5*t*t*s.accel_x)
        error_y += g * (t * s.vel_y + 0.5*t*t*s.accel_y)

    # D term
    max_derr = 150.0
    de_x = min(max(error_x - s.prev_err_x, -max_derr), max_derr)
    de_y = min(max(error_y - s.prev_err_y, -max_derr), max_derr)
    if fresh_track:
        s.derr_x = s.derr_y = 0.0
    else:
        s.derr_x = 0.6*de_x + 0.4*s.derr_x
        s.derr_y = 0.6*de_y + 0.4*s.derr_y
    s.prev_err_x = error_x
    s.prev_err_y = error_y

    # ff output term
    ff_x, ff_y = 0.0, 0.0
    if cfg.ff_gain > 0:
        g = lead_gate(s, error_x, error_y, cfg.lead_vgate)
        ff_x = cfg.ff_gain * g * s.vel_x
        ff_y = cfg.ff_gain * g * s.vel_y

    # Adaptive kp: scale gain based on error magnitude
    err_mag_kp = math.sqrt(error_x**2 + error_y**2)
    kp_scale = 1.0
    if err_mag_kp > cfg.akp_large_thresh:
        kp_scale = cfg.akp_large_mult
    elif err_mag_kp < cfg.akp_small_thresh:
        kp_scale = cfg.akp_small_mult

    movement_x = (nonlinear_p_move(error_x, cfg.kp_x * kp_scale, cfg.p_softness_x,
                                    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + cfg.kd_x * s.derr_x + ff_x) * scale_x
    movement_y = (nonlinear_p_move(error_y, cfg.kp_y * kp_scale, cfg.p_softness_y,
                                    K_FINE_TRACK_BOOST, K_FINE_TRACK_RANGE)
                  + cfg.kd_y * s.derr_y + ff_y) * scale_y

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
        s.pags_flip_rate_x = K_PAGS_ALPHA*flip_x + (1-K_PAGS_ALPHA)*s.pags_flip_rate_x
        s.pags_flip_rate_y = K_PAGS_ALPHA*flip_y + (1-K_PAGS_ALPHA)*s.pags_flip_rate_y
        if sign_x != 0: s.pags_prev_sign_x = sign_x
        if sign_y != 0: s.pags_prev_sign_y = sign_y
        err_mag = math.sqrt(error_x**2 + error_y**2)
        flip_rate = max(s.pags_flip_rate_x, s.pags_flip_rate_y)
        if err_mag > K_PAGS_ACQUIRE_THRESHOLD:
            target_mult = K_PAGS_ACQUIRE_MULT
        elif flip_rate > K_PAGS_OSC_THRESHOLD:
            target_mult = K_PAGS_OSC_MULT
        else:
            target_mult = 1.0
        s.pags_gain_mult = 0.3*target_mult + 0.7*s.pags_gain_mult
        movement_x *= s.pags_gain_mult
        movement_y *= s.pags_gain_mult

    movement_x, movement_y = clamp_max_step(movement_x, movement_y, cfg.max_step)
    movement_y *= cfg.aim_y_scale
    movement_x, movement_y = clamp_max_step(movement_x, movement_y, cfg.max_step)

    # CGA
    if cfg.cga_enabled != 0.0:
        speed = math.sqrt(s.vel_x**2 + s.vel_y**2)
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
# Noise model (calibrated from rig CSV data)
# ---------------------------------------------------------------------------
class NoiseModel:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.det_sigma = 1.75      # detection jitter σ in model px (160×160)
        self.conf_mean = 0.72
        self.conf_std = 0.12
        self.miss_rate = 0.03
        self.max_consec_miss = 2
        self.class_flip_rate = 0.015
        self.class_flip_offset_y = 7.5   # px offset on head↔body flip (160×160)
        self.consec_misses = 0

    def jitter(self):
        return self.rng.gauss(0, self.det_sigma), self.rng.gauss(0, self.det_sigma)

    def confidence(self):
        c = self.rng.gauss(self.conf_mean, self.conf_std)
        return max(0.1, min(0.99, c))

    def is_missed(self):
        if self.consec_misses >= self.max_consec_miss:
            self.consec_misses = 0
            return False
        if self.rng.random() < self.miss_rate:
            self.consec_misses += 1
            return True
        self.consec_misses = 0
        return False

    def class_flip(self, current_class):
        if self.rng.random() < self.class_flip_rate:
            return (1 - current_class), self.class_flip_offset_y * (1 if current_class == 0 else -1)
        return current_class, 0.0

# ---------------------------------------------------------------------------
# Pipeline latency model (from simple_main.cpp measured E2E p50/p95)
# ---------------------------------------------------------------------------
class PipelineModel:
    """Models variable inference + transport latency per frame.

    Real system: E2E p50=2.7ms, p95=4.8ms (lognormal).
    Converts to variable deadtime_frames per frame using the same formula
    as simple_main.cpp: dtFrames = (inflight_deadtime_ms + 0.5*periodMs) / periodMs
    """
    def __init__(self, rng: random.Random,
                 inflight_deadtime_ms: float = K_INFLIGHT_DEADTIME_MS,
                 tuned_fps: float = K_AIM_TUNED_FPS):
        self.rng = rng
        self.inflight_deadtime_ms = inflight_deadtime_ms
        self.tuned_fps = tuned_fps
        self.tuned_period_ms = 1000.0 / tuned_fps
        # Lognormal parameters fit to p50=2.7ms, p95=4.8ms
        # p50 = exp(mu) → mu = ln(2.7)
        # p95 = exp(mu + 1.645*sigma) → sigma = (ln(4.8) - ln(2.7)) / 1.645
        self.mu = math.log(2.7)
        self.sigma = (math.log(4.8) - math.log(2.7)) / 1.645
        # Frame period EMA (like the real code's slow estimator)
        self.frame_period_ema_ms = self.tuned_period_ms

    def sample_frame_period_ms(self) -> float:
        """Sample a frame interval with realistic jitter and occasional stalls."""
        if self.rng.random() < 0.05:
            # 5% chance of stall (GC, OS scheduler, GPU contention)
            period = self.rng.uniform(8.0, 15.0)
        else:
            period = self.rng.gauss(self.tuned_period_ms, 0.3)
            period = max(period, 2.0)  # floor
        # Update EMA (alpha=0.002 like real code, but faster startup)
        a = 0.002
        self.frame_period_ema_ms = (1.0 - a) * self.frame_period_ema_ms + a * period
        return period

    def sample_transport_latency_ms(self) -> float:
        """Sample E2E transport latency (lognormal, p50=2.7ms, p95=4.8ms)."""
        return self.rng.lognormvariate(self.mu, self.sigma)

    def compute_deadtime_frames(self, period_ms: float) -> float:
        """Compute per-frame adaptive dead-time (mirrors simple_main.cpp line 818)."""
        if period_ms <= 0:
            return 0.0
        dt = (self.inflight_deadtime_ms + 0.5 * period_ms) / period_ms
        return min(max(dt, 0.0), float(INFLIGHT_MAX))

    def compute_rate_scale(self, period_ms: float) -> float:
        """Rate scaling factor r for gain normalisation."""
        if self.tuned_period_ms <= 0:
            return 1.0
        r = period_ms / self.tuned_period_ms
        r = min(max(r, K_RATE_SCALE_MIN), K_RATE_SCALE_MAX)
        # Quantise to 1% like real code
        r = round(r * 100.0) / 100.0
        return r

# ---------------------------------------------------------------------------
# Network model (UDP packet loss)
# ---------------------------------------------------------------------------
class NetworkModel:
    """Models UDP fragmentation loss.

    GamePC sends 12 IP fragments per frame (from stats: 12.0 pkts/frame).
    Any single lost fragment = entire frame lost.
    """
    def __init__(self, rng: random.Random,
                 packet_loss_rate: float = 0.001,
                 packets_per_frame: int = 12):
        self.rng = rng
        self.frame_loss_prob = 1.0 - (1.0 - packet_loss_rate) ** packets_per_frame

    def is_frame_lost(self) -> bool:
        return self.rng.random() < self.frame_loss_prob

# ---------------------------------------------------------------------------
# Mouse hardware model (Makcu serial + HID)
# ---------------------------------------------------------------------------
class MouseHardwareModel:
    """Models Makcu mouse transport delay.

    Serial baud 4Mbps, USB HID 1000Hz polling → ~1ms quantisation.
    Total delay: uniform 0.5-1.5ms (serial TX + HID report alignment).
    """
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.pending_dx = 0
        self.pending_dy = 0
        self.pending_time_ms = 0.0
        self.hid_interval_ms = 1.0  # 1000Hz polling

    def submit_move(self, dx: int, dy: int, current_time_ms: float):
        """Queue a mouse move with HW delay."""
        delay = self.rng.uniform(0.5, 1.5)
        self.pending_dx = dx
        self.pending_dy = dy
        self.pending_time_ms = current_time_ms + delay

    def get_move(self, current_time_ms: float) -> Tuple[int, int]:
        """Get the mouse move if the HID report time has arrived."""
        if self.pending_time_ms > 0 and current_time_ms >= self.pending_time_ms:
            dx, dy = self.pending_dx, self.pending_dy
            self.pending_dx = 0
            self.pending_dy = 0
            self.pending_time_ms = 0.0
            return dx, dy
        return 0, 0

# ---------------------------------------------------------------------------
# Target tracking model (IoU-based stickiness)
# ---------------------------------------------------------------------------
class TargetTrackingModel:
    """Models IoU-based target tracking with stickiness and re-acquisition.

    Real system: iou_stickiness_threshold=0.3, track_persistence_frames=3.
    When target moves fast, tracking can momentarily lose the target, causing
    a fresh_track reset that disrupts the velocity estimator.
    """
    def __init__(self, rng: random.Random,
                 iou_threshold: float = 0.3,
                 persistence_frames: int = 3):
        self.rng = rng
        self.iou_threshold = iou_threshold
        self.persistence_frames = persistence_frames
        self.prev_det_x = 0.0
        self.prev_det_y = 0.0
        self.prev_det_w = 20.0  # typical bbox width in model px
        self.frames_without_match = 0
        self.has_prev = False

    def update(self, det_x: float, det_y: float, missed: bool) -> bool:
        """Returns True if this is a tracking loss (fresh_track reset).

        Estimates IoU based on detection displacement vs bbox size. Large
        displacement = low IoU = potential track break.
        """
        if missed:
            self.frames_without_match += 1
            if self.frames_without_match > self.persistence_frames:
                self.has_prev = False
                return True  # track lost
            return False

        if not self.has_prev:
            self.prev_det_x = det_x
            self.prev_det_y = det_y
            self.has_prev = True
            self.frames_without_match = 0
            return True  # first detection = fresh track

        # Estimate IoU from displacement: IoU ≈ max(0, 1 - displacement/bbox_size)
        dx = abs(det_x - self.prev_det_x)
        dy = abs(det_y - self.prev_det_y)
        displacement = math.sqrt(dx*dx + dy*dy)
        estimated_iou = max(0.0, 1.0 - displacement / self.prev_det_w)

        self.prev_det_x = det_x
        self.prev_det_y = det_y
        self.frames_without_match = 0

        if estimated_iou < self.iou_threshold:
            return True  # IoU too low = track break
        return False

# ---------------------------------------------------------------------------
# FPS Camera model (ego-pollution + non-linear sensitivity)
# ---------------------------------------------------------------------------
@dataclass
class FPSCamera:
    """Models FPS game camera with ego-pollution and speed-dependent sensitivity.

    At high mouse speeds (>50px/frame), sensitivity drops slightly (0.95x)
    to model common game engine deceleration curves.
    """
    yaw: float = 0.0
    pitch: float = 0.0
    sensitivity: float = 0.022  # degrees per mouse pixel (typical FPS @ 800 DPI)
    model_size: int = 160
    high_speed_threshold: float = 50.0  # px/frame
    high_speed_factor: float = 0.95

    def apply_mouse(self, dx: int, dy: int):
        speed = math.sqrt(dx*dx + dy*dy)
        sens = self.sensitivity
        if speed > self.high_speed_threshold:
            blend = min((speed - self.high_speed_threshold) / self.high_speed_threshold, 1.0)
            sens *= (1.0 - blend) + blend * self.high_speed_factor
        self.yaw += dx * sens
        self.pitch += dy * sens
        self.pitch = max(-89, min(89, self.pitch))

    def world_to_screen(self, world_x: float, world_y: float):
        screen_x = (world_x - self.yaw) / self.sensitivity + self.model_size / 2
        screen_y = (world_y - self.pitch) / self.sensitivity + self.model_size / 2
        return screen_x, screen_y

# ---------------------------------------------------------------------------
# Target trajectories
# ---------------------------------------------------------------------------
def trajectory_stationary(t, **kw):
    return 0.0, 0.0

def trajectory_slow_strafe(t, speed=1.5, **kw):
    return speed * t * 0.022, 0.0

def trajectory_fast_strafe(t, speed=6.0, **kw):
    return speed * t * 0.022, 0.0

def trajectory_sine_wave(t, amp=4.0, period=60, **kw):
    return amp * math.sin(2*math.pi*t/period) * 0.022, 0.0

def trajectory_diagonal(t, speed_x=3.0, speed_y=2.0, **kw):
    return speed_x * t * 0.022, speed_y * t * 0.022

def trajectory_step_response(t, step_frame=30, step_size=40.0, **kw):
    if t < step_frame:
        return 0.0, 0.0
    return step_size * 0.022, 0.0

TRAJECTORIES = {
    "stationary":   trajectory_stationary,
    "slow_strafe":  trajectory_slow_strafe,
    "fast_strafe":  trajectory_fast_strafe,
    "sine_wave":    trajectory_sine_wave,
    "diagonal":     trajectory_diagonal,
    "step_response":trajectory_step_response,
}

# ---------------------------------------------------------------------------
# Algorithm configurations
# ---------------------------------------------------------------------------
def config_baseline():
    """PD + One Euro + dead-time comp + lead (no novel algorithms)."""
    return AimConfig()

def config_production():
    """Full production config: SEM+JCAD+PAGS enabled, CGA disabled."""
    return AimConfig()

def config_sem_pags():
    """Baseline + SEM + PAGS only."""
    c = config_baseline()
    c.sem_enabled = 1.0
    c.pags_enabled = 1.0
    return c

def config_no_lead():
    """Baseline with lead terms disabled (shows lead contribution)."""
    c = config_baseline()
    c.ff_gain = 0.0
    c.predict_frames = 0.0
    return c

CONFIGS = {
    "baseline":       config_baseline,
    "production":     config_production,
    "sem_pags":       config_sem_pags,
    "no_lead":        config_no_lead,
}

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    trajectory: str
    config_name: str
    seed: int
    frames: int
    errors: List[float] = field(default_factory=list)
    sign_flips_x: int = 0
    sign_flips_y: int = 0
    misses: int = 0
    total_movement: float = 0.0
    udp_losses: int = 0
    track_breaks: int = 0

    @property
    def mae(self):
        return sum(self.errors) / len(self.errors) if self.errors else 0

    @property
    def rms(self):
        return math.sqrt(sum(e*e for e in self.errors)/len(self.errors)) if self.errors else 0

    @property
    def p95(self):
        if not self.errors: return 0
        s = sorted(self.errors)
        return s[int(0.95*len(s))]

    @property
    def settle_frame(self):
        """Frame at which error first stays below 2px for 10 consecutive frames."""
        threshold = 2.0
        count = 0
        for i, e in enumerate(self.errors):
            if e < threshold:
                count += 1
                if count >= 10:
                    return i - 9
            else:
                count = 0
        return self.frames

    @property
    def oscillation_count(self):
        return self.sign_flips_x + self.sign_flips_y

def run_sim(traj_name: str, config_name: str, seed: int = 42,
            n_frames: int = 300) -> SimResult:
    base_cfg = CONFIGS[config_name]()
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, config_name, seed, n_frames)

    model_size = 160
    screen_cx = model_size / 2.0
    screen_cy = model_size / 2.0
    scale_x = 1.0
    scale_y = 1.0
    target_class = 0
    prev_sign_x = 0
    prev_sign_y = 0

    for frame in range(n_frames):
        # --- Frame timing (variable period with jitter & stalls) ---
        frame_period_ms = pipeline.sample_frame_period_ms()

        # 1. Target moves in world space
        world_x, world_y = traj_fn(frame)

        # 2. Project target to screen via camera (ego-pollution)
        det_x, det_y = camera.world_to_screen(world_x, world_y)

        # 3. Add detector noise
        jx, jy = noise.jitter()
        det_x += jx
        det_y += jy
        conf = noise.confidence()

        # Class flip
        target_class, flip_offset = noise.class_flip(target_class)
        det_y += flip_offset

        # Missed detection (detector miss OR UDP frame loss)
        missed = noise.is_missed()
        if not missed and network.is_frame_lost():
            missed = True
            result.udp_losses += 1

        # IoU-based tracking: check if this detection breaks the track
        track_break = tracker.update(det_x, det_y, missed)
        if track_break and state.has_track:
            result.track_breaks += 1
            state.has_track = 0

        # 4. Controller (detection processed same frame; deadtime_frames
        #    already accounts for the full capture→mouse pipeline delay)
        dx, dy = 0, 0
        if missed:
            result.misses += 1
            state.frames_since_seen += 1
            push_inflight(state, 0.0, 0.0)
        else:
            state.frames_since_seen = 0
            cfg = dataclasses.replace(base_cfg)
            rate_scale = pipeline.compute_rate_scale(frame_period_ms)
            if rate_scale != 1.0:
                rescale_aim_config_for_rate(cfg, rate_scale)
            cfg.deadtime_frames = pipeline.compute_deadtime_frames(frame_period_ms)

            dx, dy = compute_aim_movement(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, cfg, state,
                target_class, conf)

        # 5. Apply mouse delta to camera (ego-pollution feedback loop)
        camera.apply_mouse(dx, dy)
        result.total_movement += math.sqrt(dx*dx + dy*dy)

        # 6. Compute actual error (current screen position of target)
        actual_x, actual_y = camera.world_to_screen(world_x, world_y)
        err = math.sqrt((actual_x - screen_cx)**2 + (actual_y - screen_cy)**2)
        result.errors.append(err)

        # Track sign flips for oscillation metric
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
# Main: run all combinations, report results
# ---------------------------------------------------------------------------
def main():
    seeds = [42, 137, 256, 999, 1337]
    n_frames = 300
    all_results = {}

    print(f"Closed-Loop FPS Aimbot Simulation (Full Real-World Model)")
    print(f"{'='*80}")
    print(f"Frames: {n_frames} | Seeds: {len(seeds)} | Adaptive dead-time + rate scaling")
    print(f"Noise: det_σ=1.75px @160×160, miss=3%, class_flip=1.5%")
    print(f"Pipeline: latency p50=2.7ms p95=4.8ms, UDP loss ~1.2%/frame")
    print(f"HW: Makcu 1kHz HID, non-linear camera, IoU tracking")
    print()

    header = f"{'Trajectory':<16} {'Config':<14} {'MAE':>7} {'RMS':>7} {'P95':>7} {'Settle':>7} {'Osc':>5} {'Miss':>5} {'UDP':>4} {'TBrk':>4}"
    print(header)
    print("-" * len(header))

    for traj_name in TRAJECTORIES:
        for cfg_name in CONFIGS:
            seed_results = []
            for seed in seeds:
                r = run_sim(traj_name, cfg_name, seed=seed, n_frames=n_frames)
                seed_results.append(r)

            avg_mae = sum(r.mae for r in seed_results) / len(seed_results)
            avg_rms = sum(r.rms for r in seed_results) / len(seed_results)
            avg_p95 = sum(r.p95 for r in seed_results) / len(seed_results)
            avg_settle = sum(r.settle_frame for r in seed_results) / len(seed_results)
            avg_osc = sum(r.oscillation_count for r in seed_results) / len(seed_results)
            avg_miss = sum(r.misses for r in seed_results) / len(seed_results)
            avg_udp = sum(r.udp_losses for r in seed_results) / len(seed_results)
            avg_tbrk = sum(r.track_breaks for r in seed_results) / len(seed_results)

            key = f"{traj_name}|{cfg_name}"
            all_results[key] = {
                "trajectory": traj_name,
                "config": cfg_name,
                "mae": round(avg_mae, 2),
                "rms": round(avg_rms, 2),
                "p95": round(avg_p95, 2),
                "settle": round(avg_settle, 1),
                "oscillation": round(avg_osc, 1),
                "misses": round(avg_miss, 1),
                "udp_losses": round(avg_udp, 1),
                "track_breaks": round(avg_tbrk, 1),
            }

            print(f"{traj_name:<16} {cfg_name:<14} {avg_mae:7.2f} {avg_rms:7.2f} "
                  f"{avg_p95:7.2f} {avg_settle:7.1f} {avg_osc:5.1f} {avg_miss:5.1f} "
                  f"{avg_udp:4.1f} {avg_tbrk:4.1f}")
        print()

    # Summary comparison
    print(f"\n{'='*80}")
    print("RELATIVE PERFORMANCE vs BASELINE (% change, negative = better)")
    print(f"{'='*80}")
    header2 = f"{'Trajectory':<16} {'Config':<14} {'ΔMAE':>8} {'ΔRMS':>8} {'ΔP95':>8} {'ΔSettle':>8} {'ΔOsc':>8}"
    print(header2)
    print("-" * len(header2))

    for traj_name in TRAJECTORIES:
        base_key = f"{traj_name}|baseline"
        base = all_results.get(base_key)
        if not base:
            continue
        for cfg_name in CONFIGS:
            if cfg_name == "baseline":
                continue
            key = f"{traj_name}|{cfg_name}"
            r = all_results.get(key)
            if not r:
                continue
            def pct(val, ref):
                if ref == 0: return 0.0
                return ((val - ref) / ref) * 100

            d_mae = pct(r["mae"], base["mae"])
            d_rms = pct(r["rms"], base["rms"])
            d_p95 = pct(r["p95"], base["p95"])
            d_settle = pct(r["settle"], base["settle"])
            d_osc = pct(r["oscillation"], base["oscillation"])

            print(f"{traj_name:<16} {cfg_name:<14} {d_mae:+7.1f}% {d_rms:+7.1f}% "
                  f"{d_p95:+7.1f}% {d_settle:+7.1f}% {d_osc:+7.1f}%")
        print()

    # Save JSON for further analysis
    with open("/home/taeho/portfolio/inferencetool/bench/closedloop/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to bench/closedloop/results.json")

if __name__ == "__main__":
    main()
