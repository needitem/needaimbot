#!/usr/bin/env python3
"""Ringing-vs-speed tuner for the SHIPPED controller (pd_controller.cuh).

Unlike the older simulators that used to live here (they modelled an ff/coast
pipeline and do NOT model dead-time compensation), this ports
computeAimMovement() line-for-line INCLUDING inflight_comp (the Smith-predictor
dead-time subtraction that is the actual anti-ringing lever) and closes the loop
with a real emit->visible dead time D_true frames.

Goal (user): minimise RINGING, maximise SPEED.
  RINGING  = step overshoot (px past target) + post-settle oscillation count
             + reversal post-turn peak.
  SPEED    = step rise time (frames to first reach the target band)
             + step settle time (frames to stay inside the band).

Detector noise is calibrated to the measured rig (memory: v3 aim-off white
X~9.9 / Y~7.4 px @320, correlated drift, head/body switch, outliers, dropouts).
Ringing/speed are read on a low-noise step (controller's intrinsic transient);
noise scenarios only guard against picking a config that buzzes or lags.
"""
import math
import random

# ------------------------------------------------------------ ported controller
# One Euro alpha, nonlinear P, max-step clamp, integer emit with residual carry,
# and the in-flight ring — all identical to pd_controller.cuh.
K_INFLIGHT_MAX = 4

def one_euro_alpha(cutoff):
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
    return 1.0 / (1.0 + tau)

def nonlinear_p(error, kp, softness):
    ae = abs(error)
    safe = max(softness, 1.0)
    return error * (max(kp, 0.0) * (ae / (ae + safe)))

def clamp_max_step(mx, my, max_step):
    if max_step > 0.0:
        step = math.hypot(mx, my)
        if step > max_step:
            s = max_step / step
            return mx * s, my * s
    return mx, my

def emit_mouse_delta(movement, residual):
    carried = residual
    if movement * carried < 0.0:
        carried = 0.0
    value = movement + carried
    emit = int(value)  # __float2int_rz (trunc toward zero)
    if emit > 127:
        return 127, min(256.0, value - 127.0)
    if emit < -127:
        return -127, max(-256.0, value + 127.0)
    return emit, value - float(emit)


class Cfg:
    """AimConfig subset that matters for ringing/speed."""
    def __init__(self, kp_x=0.75, kp_y=0.82, soft_x=9.0, soft_y=8.0,
                 kd_x=0.05, kd_y=0.06, max_step=25.0,
                 inflight_comp=1.0, deadtime_frames=1,
                 oneeuro_enabled=1.0, min_cutoff=0.1, beta=0.02, dcutoff=0.5,
                 coast_enabled=1.0, coast_decay=0.85,
                 iou_stickiness_threshold=0.3, distance_stickiness_factor=0.4,
                 track_persistence_frames=3,
                 class_switch_penalty=0.0, head_min_box_px=0.0):
        self.kp_x, self.kp_y = kp_x, kp_y
        self.soft_x, self.soft_y = soft_x, soft_y
        self.kd_x, self.kd_y = kd_x, kd_y
        self.max_step = max_step
        self.inflight_comp = inflight_comp
        self.deadtime_frames = deadtime_frames
        self.oneeuro_enabled = oneeuro_enabled
        self.min_cutoff, self.beta, self.dcutoff = min_cutoff, beta, dcutoff
        self.coast_enabled, self.coast_decay = coast_enabled, coast_decay
        # target-selection knobs (used by aim_sim_select.py; ignored by the
        # controller-only scenarios in this file)
        self.iou_stickiness_threshold = iou_stickiness_threshold
        self.distance_stickiness_factor = distance_stickiness_factor
        self.track_persistence_frames = track_persistence_frames
        # candidate anti-flip-flop knobs (0 = shipped behaviour)
        self.class_switch_penalty = class_switch_penalty
        self.head_min_box_px = head_min_box_px


class Ctrl:
    """State machine == AimState + computeAimMovement()/computeCoastMovement()."""
    def __init__(self, cfg: Cfg, scale=1.0):
        self.c = cfg
        self.scale = scale
        self.reset()

    def reset(self):
        self.has_track = 0
        self.filt_x = self.filt_y = 0.0
        self.dfilt_x = self.dfilt_y = 0.0
        self.prev_cx = self.prev_cy = 0.0
        self.vel_x = self.vel_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.derr_x = self.derr_y = 0.0
        self.res_x = self.res_y = 0.0
        self.inflight_x = [0.0] * K_INFLIGHT_MAX
        self.inflight_y = [0.0] * K_INFLIGHT_MAX
        self.inflight_head = 0

    def _push_inflight(self, dx, dy):
        i = self.inflight_head
        self.inflight_x[i] = dx
        self.inflight_y[i] = dy
        self.inflight_head = (i + 1) % K_INFLIGHT_MAX

    def _inflight_sum(self, frames):
        sx = sy = 0.0
        n = min(max(frames, 0), K_INFLIGHT_MAX)
        for k in range(1, n + 1):
            i = (self.inflight_head - k + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
            sx += self.inflight_x[i]
            sy += self.inflight_y[i]
        return sx, sy

    def coast(self, missed):
        c = self.c
        factor = 1.0
        for _ in range(missed):
            factor *= c.coast_decay
        mx = self.vel_x * self.scale * factor
        my = self.vel_y * self.scale * factor
        mx, my = clamp_max_step(mx, my, c.max_step)
        dx, self.res_x = emit_mouse_delta(mx, self.res_x)
        dy, self.res_y = emit_mouse_delta(my, self.res_y)
        self._push_inflight(float(dx), float(dy))
        return dx, dy

    def step(self, raw_x, raw_y, sc_x, sc_y):
        c = self.c
        scale = self.scale
        fresh = (self.has_track == 0)

        tcx, tcy = raw_x, raw_y
        if c.oneeuro_enabled != 0.0:
            if self.has_track:
                ad = one_euro_alpha(c.dcutoff)
                de_x = raw_x - self.filt_x
                self.dfilt_x = ad * de_x + (1.0 - ad) * self.dfilt_x
                ax = one_euro_alpha(c.min_cutoff + c.beta * abs(self.dfilt_x))
                self.filt_x = ax * raw_x + (1.0 - ax) * self.filt_x
                de_y = raw_y - self.filt_y
                self.dfilt_y = ad * de_y + (1.0 - ad) * self.dfilt_y
                ay = one_euro_alpha(c.min_cutoff + c.beta * abs(self.dfilt_y))
                self.filt_y = ay * raw_y + (1.0 - ay) * self.filt_y
            else:
                self.filt_x, self.filt_y = raw_x, raw_y
                self.dfilt_x = self.dfilt_y = 0.0
            tcx, tcy = self.filt_x, self.filt_y

        if self.has_track:
            maxDrift = 60.0
            nvx = max(-maxDrift, min(maxDrift, raw_x - self.prev_cx))
            nvy = max(-maxDrift, min(maxDrift, raw_y - self.prev_cy))
            self.vel_x = 0.6 * self.vel_x + 0.4 * nvx
            self.vel_y = 0.6 * self.vel_y + 0.4 * nvy
        else:
            self.vel_x = self.vel_y = 0.0
        self.prev_cx, self.prev_cy = raw_x, raw_y
        self.has_track = 1

        error_x = tcx - sc_x
        error_y = tcy - sc_y

        if c.inflight_comp > 0.0 and c.deadtime_frames > 0:
            inf_x, inf_y = self._inflight_sum(c.deadtime_frames)
            if scale != 0.0:
                error_x -= c.inflight_comp * inf_x / scale
                error_y -= c.inflight_comp * inf_y / scale

        maxDErr = 150.0
        de_x = max(-maxDErr, min(maxDErr, error_x - self.prev_err_x))
        de_y = max(-maxDErr, min(maxDErr, error_y - self.prev_err_y))
        if fresh:
            self.derr_x = self.derr_y = 0.0
        else:
            self.derr_x = 0.6 * de_x + 0.4 * self.derr_x
            self.derr_y = 0.6 * de_y + 0.4 * self.derr_y
        self.prev_err_x, self.prev_err_y = error_x, error_y

        mx = (nonlinear_p(error_x, c.kp_x, c.soft_x) + c.kd_x * self.derr_x) * scale
        my = (nonlinear_p(error_y, c.kp_y, c.soft_y) + c.kd_y * self.derr_y) * scale
        mx, my = clamp_max_step(mx, my, c.max_step)
        dx, self.res_x = emit_mouse_delta(mx, self.res_x)
        dy, self.res_y = emit_mouse_delta(my, self.res_y)
        self._push_inflight(float(dx), float(dy))
        return dx, dy


# ------------------------------------------------------------ detector noise
SC = 160.0

class Detector:
    """Realistic rig noise (memory: white X~9.9 / Y~7.4 @320, correlated drift,
    head/body switch, fat-tailed outliers, dropouts). white=0 -> clean step."""
    def __init__(self, rng, white_x=9.9, white_y=7.4, drift=1.2, drift_rho=0.9,
                 p_head=0.09, hb_offset=25.0, p_drop=0.015,
                 p_outlier=0.035, out_lo=15.0, out_hi=60.0):
        self.rng = rng
        self.wx, self.wy = white_x, white_y
        self.drift, self.rho = drift, drift_rho
        self.p_head, self.hb = p_head, hb_offset
        self.p_drop = p_drop
        self.p_out, self.olo, self.ohi = p_outlier, out_lo, out_hi
        self.dx = self.dy = 0.0
        self.last = None

    def _ar1(self, prev):
        return self.rho * prev + self.rng.gauss(0.0, self.drift * math.sqrt(max(1e-9, 1 - self.rho**2)))

    def measure(self, body_cx, body_cy):
        if self.rng.random() < self.p_drop:
            return None
        self.dx = self._ar1(self.dx)
        self.dy = self._ar1(self.dy)
        # head selected only p_head of the time (small box); else body point HB below
        if self.rng.random() < self.p_head:
            cx, cy = body_cx, body_cy - self.hb
        else:
            cx, cy = body_cx, body_cy
        mx = cx + self.dx + self.rng.gauss(0.0, self.wx)
        my = cy + self.dy + self.rng.gauss(0.0, self.wy)
        if self.rng.random() < self.p_out:
            ang = self.rng.random() * 6.283
            mag = self.olo + self.rng.random() * (self.ohi - self.olo)
            mx += mag * math.cos(ang)
            my += mag * math.sin(ang)
        self.last = (mx, my)
        return (mx, my)


# ------------------------------------------------------------ closed loop
def run(cfg, scenario, frames=140, seed=0, d_true=1, noise=False,
        render_q=0, scale=1.0, scale_err=0.0,
        occl_period=0, occl_len=0, coast_window=3):
    """Closed loop. Measurement at frame f reflects the world (T and C) from
    d_true frames ago (emit->visible dead time). render_q>0 quantises the dead
    time to whole render frames of that many loop-steps. scale_err = emit->screen
    mismatch (C moves emit*(1+scale_err); controller only knows its own emit)."""
    rng = random.Random(seed)
    ctrl = Ctrl(cfg, scale=scale)
    ctrl.reset()
    det = Detector(rng) if noise else None

    C = [SC, SC]
    # scenario target setup (absolute coords)
    if scenario == "step":
        T = [SC + 60.0, SC + 22.0]; v = [0.0, 0.0]
    elif scenario == "step_small":
        T = [SC + 20.0, SC + 8.0]; v = [0.0, 0.0]
    elif scenario == "step_big":
        T = [SC + 100.0, SC + 40.0]; v = [0.0, 0.0]
    elif scenario == "hold":
        T = [SC, SC]; v = [0.0, 0.0]
    elif scenario == "track":
        T = [SC - 40.0, SC]; v = [1.1, 0.25]
    elif scenario == "reversal":
        T = [SC, SC]; v = [1.3, 0.0]
    else:
        T = [SC, SC]; v = [0.0, 0.0]

    d0 = math.hypot(T[0] - C[0], T[1] - C[1])
    ax_unit = ((T[0] - C[0]) / d0, (T[1] - C[1]) / d0) if d0 > 1e-9 else (1.0, 0.0)

    hist = []
    errs = []
    emits = []
    par = []          # signed progress toward target along approach axis (for step)
    missed = 0
    turn_frame = None
    post_turn_peak = 0.0

    for f in range(frames):
        # world update
        if scenario == "reversal" and f > 0 and f % 60 == 0:
            v[0] = -v[0]; turn_frame = f
        T[0] += v[0]; T[1] += v[1]
        hist.append((T[0], T[1], C[0], C[1]))

        # capture: measurement reflects snapshot d_true frames ago
        lat = d_true
        if render_q > 0:
            lat = max(render_q, (d_true // render_q) * render_q)
        idx = max(0, f - lat)
        sT0, sT1, sC0, sC1 = hist[idx]

        body_cx = sT0 - sC0 + SC
        body_cy = sT1 - sC1 + SC
        # deterministic occlusion burst (target hidden occl_len frames every occl_period)
        forced_gap = occl_period > 0 and (f % occl_period) < occl_len
        if noise:
            meas = det.measure(body_cx, body_cy)
            gap = (meas is None) or forced_gap
        else:
            gap = forced_gap
            meas = (body_cx, body_cy)

        if gap:
            missed += 1
            if cfg.coast_enabled != 0.0 and missed <= coast_window:
                dx, dy = ctrl.coast(missed)
            else:
                dx, dy = 0, 0
                ctrl.res_x = ctrl.res_y = 0.0
        else:
            dx, dy = ctrl.step(meas[0], meas[1], SC, SC)
            missed = 0

        C[0] += dx * (1.0 + scale_err)
        C[1] += dy * (1.0 + scale_err)

        # metrics vs the TRUE current target (head point in noise mode already folded)
        tgt_y = T[1] - (25.0 if (noise) else 0.0) if False else T[1]
        e = math.hypot(T[0] - C[0], T[1] - C[1])
        errs.append(e)
        emits.append((dx, dy))
        # signed progress along approach axis
        prog = (C[0] - (T[0] - d0 * ax_unit[0])) * ax_unit[0] + \
               (C[1] - (T[1] - d0 * ax_unit[1])) * ax_unit[1]
        par.append(prog)  # 0 at start, d0 at target, >d0 = overshoot
        if scenario == "reversal" and turn_frame is not None and 0 <= f - turn_frame < 45:
            post_turn_peak = max(post_turn_peak, e)

    out = {}
    if scenario in ("step", "step_small", "step_big"):
        # overshoot: max distance past the target along the axis
        overshoot = max(0.0, max(par) - d0)
        # oscillation count: sign changes of (par - d0) after first reaching d0
        rise = None
        for i, p in enumerate(par):
            if p >= d0 - 2.0:
                rise = i; break
        n_osc = 0
        if rise is not None:
            sgn = 0
            for p in par[rise:]:
                s = 1 if (p - d0) > 0.5 else (-1 if (p - d0) < -0.5 else 0)
                if s != 0 and s != sgn and sgn != 0:
                    n_osc += 1
                if s != 0:
                    sgn = s
        # settle: last frame error exceeded 2px band, +1
        settle = 0
        for i, e in enumerate(errs):
            if e > 2.0:
                settle = i + 1
        out.update(d0=d0, overshoot=overshoot, n_osc=n_osc,
                   rise=(rise if rise is not None else frames), settle=settle,
                   ss_rms=math.sqrt(sum(x*x for x in errs[settle:]) / max(1, len(errs) - settle)))
    elif scenario == "reversal":
        w = errs[70:]
        out.update(rms=math.sqrt(sum(x*x for x in w) / len(w)), turn_peak=post_turn_peak)
    else:  # hold / track
        w = errs[40:]
        out.update(rms=math.sqrt(sum(x*x for x in w) / len(w)),
                   max_err=max(w),
                   twitch=100.0 * sum(1 for d in emits[40:] if d != (0, 0)) / len(emits[40:]),
                   emit=sum(math.hypot(*d) for d in emits[40:]) / len(emits[40:]))
    return out


def avg(cfg, scenario, n=6, **kw):
    acc = {}
    for s in range(n):
        m = run(cfg, scenario, seed=s, **kw)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}
