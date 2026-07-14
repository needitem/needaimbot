#!/usr/bin/env python3
"""Closed-loop aim-controller simulation: OLD pipeline (One Euro + velocity
feedforward + coast + nonlinear P+D) vs NEW pipeline (gated alpha-beta
estimator + hold deadband + nonlinear P+D), both ported line-for-line from
needaimbot/cuda/pd_controller.cuh (new = current HEAD, old = pre-rewrite).

Frame-coordinate closed loop: the measured target center is
    raw = T_abs - C + SC + noise
(the frame shifts when the mouse moves, exactly like the real capture), the
controller emits an integer (dx, dy), and the crosshair C absorbs it before
the next capture. movement_scale = 1 (model px == output px).

Scenarios (300 detections/s -> units are px/frame):
  step        stationary target acquired 80 px away          -> settle + overshoot
  still       stationary target, center noise                -> stationary RMS + emit jitter
  cv          constant velocity 1.0 px/frame                 -> steady tracking RMS
  zigzag      +/-1.5 px/frame, reverses every 150 frames     -> RMS + post-turn peak
  glitch      stationary + 40 px measurement outlier (1%)    -> max deflection
  gap         cv 1.0 with 3-frame detection gaps every 60    -> RMS incl. recovery

Metrics are TRUE error |T - C| (not the controller's own belief).

Shipped defaults chosen from these sweeps: estimator_alpha=0.35,
estimator_gate_px=50, aim_hold_deadband_px=1.0, aim_track_ff=0.8,
softness 8/7 (still: 0.30 px RMS / 1% moving frames; cv: 1.30 px;
zigzag: 1.74 px; glitch: 2.12 px; step overshoot 0.7 px).
Old pipeline for comparison - still: 0.89 px with 36% moving frames,
cv: 5.06 px, zigzag: 5.31 px, glitch: 3.29 px.
"""
import math
import random

MAX_STEP = 30.0

# ---------------------------------------------------------------- shared bits
def emit_mouse_delta(movement, residual):
    carried = residual
    if movement * carried < 0.0:
        carried = 0.0
    value = movement + carried
    emit = int(value)  # trunc toward zero, same as __float2int_rz
    if emit > 127:
        return 127, min(256.0, value - 127.0)
    if emit < -127:
        return -127, max(-256.0, value + 127.0)
    return emit, value - float(emit)

def nonlinear_p(error, kp, softness):
    ae = abs(error)
    return error * (max(kp, 0.0) * (ae / (ae + max(softness, 1.0))))

def clamp_max_step(mx, my):
    step = math.hypot(mx, my)
    if MAX_STEP > 0.0 and step > MAX_STEP:
        s = MAX_STEP / step
        return mx * s, my * s
    return mx, my

class Gains:
    def __init__(self, kp_x=0.55, kp_y=0.6, soft_x=8.0, soft_y=7.0,
                 kd_x=0.18, kd_y=0.22):
        self.kp_x, self.kp_y = kp_x, kp_y
        self.soft_x, self.soft_y = soft_x, soft_y
        self.kd_x, self.kd_y = kd_x, kd_y

# ------------------------------------------------------------- OLD controller
def one_euro_alpha(cutoff):
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
    return 1.0 / (1.0 + tau)

class OldController:
    """Pre-rewrite pipeline: One Euro pre-filter, raw-delta velocity EMA,
    P + D + feedforward, coast glide on detection gaps."""
    def __init__(self, g: Gains, ff=0.9, min_cutoff=0.1, beta=0.02, dcutoff=0.5,
                 coast_decay=0.85):
        self.g, self.ff = g, ff
        self.min_cutoff, self.beta, self.dcutoff = min_cutoff, beta, dcutoff
        self.coast_decay = coast_decay
        self.reset()

    def reset(self):
        self.has_track = 0
        self.filt = [0.0, 0.0]; self.dfilt = [0.0, 0.0]
        self.prev_center = [0.0, 0.0]; self.vel = [0.0, 0.0]
        self.prev_err = [0.0, 0.0]; self.derr = [0.0, 0.0]
        self.residual = [0.0, 0.0]

    def coast(self, missed):
        factor = self.coast_decay ** missed
        mx, my = clamp_max_step(self.vel[0] * factor, self.vel[1] * factor)
        dx, self.residual[0] = emit_mouse_delta(mx, self.residual[0])
        dy, self.residual[1] = emit_mouse_delta(my, self.residual[1])
        return dx, dy

    def step(self, raw, sc):
        fresh = (self.has_track == 0)
        tc = [raw[0], raw[1]]
        if not fresh:
            ad = one_euro_alpha(self.dcutoff)
            for i in range(2):
                de = raw[i] - self.filt[i]
                self.dfilt[i] = ad * de + (1.0 - ad) * self.dfilt[i]
                cutoff = self.min_cutoff + self.beta * abs(self.dfilt[i])
                a = one_euro_alpha(cutoff)
                self.filt[i] = a * raw[i] + (1.0 - a) * self.filt[i]
                tc[i] = self.filt[i]
        else:
            self.filt = [raw[0], raw[1]]; self.dfilt = [0.0, 0.0]

        if not fresh:
            for i in range(2):
                nv = max(-60.0, min(60.0, raw[i] - self.prev_center[i]))
                self.vel[i] = 0.6 * self.vel[i] + 0.4 * nv
        else:
            self.vel = [0.0, 0.0]
        self.prev_center = [raw[0], raw[1]]
        self.has_track = 1

        err = [tc[0] - sc[0], tc[1] - sc[1]]
        for i in range(2):
            de = max(-150.0, min(150.0, err[i] - self.prev_err[i]))
            if fresh:
                self.derr[i] = 0.0
            else:
                self.derr[i] = 0.6 * de + 0.4 * self.derr[i]
            self.prev_err[i] = err[i]

        mx = nonlinear_p(err[0], self.g.kp_x, self.g.soft_x) \
            + self.g.kd_x * self.derr[0] + self.ff * self.vel[0]
        my = nonlinear_p(err[1], self.g.kp_y, self.g.soft_y) \
            + self.g.kd_y * self.derr[1] + self.ff * self.vel[1]
        mx, my = clamp_max_step(mx, my)
        dx, self.residual[0] = emit_mouse_delta(mx, self.residual[0])
        dy, self.residual[1] = emit_mouse_delta(my, self.residual[1])
        return dx, dy

# ------------------------------------------------------------- NEW controller
class NewController:
    """Current pipeline: gated alpha-beta estimator + hold deadband + P + D.

    ego_comp: subtract our own emitted delta from the estimator state before
    the next fuse (the frame shifts by exactly -emit when the mouse moves, and
    we know emit precisely). With it, est/est_v live in target-only terms:
    est_v is the TARGET's true screen drift, not polluted by our approach.
    track_ff: add track_ff * est_v to the move - cancels the P controller's
    own steady-state chase lag so the crosshair sits ON the target's current
    position while it moves. NOT prediction (no future lead); with ego_comp
    the velocity is clean, and the hold deadband still pins stationary targets.
    """
    def __init__(self, g: Gains, alpha=0.35, gate_px=50.0, deadband=1.0,
                 ego_comp=True, track_ff=0.8, ego_bug=False):
        self.g, self.alpha, self.gate_px, self.deadband = g, alpha, gate_px, deadband
        self.ego_comp, self.track_ff = ego_comp, track_ff
        # ego_bug reproduces the REVIEWED defect: subtract this kernel's own
        # emitted delta (guess) instead of the host's real applied delta. When
        # a move is discarded/dropped, est desyncs. Off = the shipped closed-
        # loop feedback (subtract set_applied()).
        self.ego_bug = ego_bug
        self.reset()

    def reset(self):
        self.has_track = 0
        self.est = [0.0, 0.0]; self.est_v = [0.0, 0.0]; self.gate_count = 0
        self.prev_err = [0.0, 0.0]; self.derr = [0.0, 0.0]
        self.residual = [0.0, 0.0]
        # Ego-motion feedback: the delta the host applied LAST frame, subtracted
        # at the top of THIS frame (mirrors AimConfig::applied_dx/dy - the kernel
        # closes on real output, not its own guess). set_applied() is the host
        # telling us what actually reached the mouse.
        self.applied = [0.0, 0.0]

    def set_applied(self, dx, dy):
        if self.ego_bug:
            return  # defect path ignores real feedback; it self-subtracts
        self.applied = [float(dx), float(dy)]

    def step(self, raw, sc, frames_elapsed=1):
        fresh = (self.has_track == 0)
        alpha = min(max(self.alpha, 0.01), 1.0)
        beta = alpha * alpha / (2.0 - alpha)
        dt = max(float(frames_elapsed), 1.0)

        if fresh:
            self.est = [raw[0], raw[1]]; self.est_v = [0.0, 0.0]; self.gate_count = 0
        else:
            # Ego-motion compensation FIRST, using the host's real applied delta.
            if self.ego_comp and not self.ego_bug:
                self.est[0] -= self.applied[0]; self.est[1] -= self.applied[1]
            p = [self.est[0] + self.est_v[0] * dt, self.est[1] + self.est_v[1] * dt]
            r = [raw[0] - p[0], raw[1] - p[1]]
            rn = math.hypot(r[0], r[1])
            if self.gate_px > 0.0 and rn > self.gate_px:
                self.gate_count += 1
                if self.gate_count >= 3:
                    p = [raw[0], raw[1]]; self.est_v = [0.0, 0.0]; self.gate_count = 0
            else:
                for i in range(2):
                    p[i] += alpha * r[i]
                    self.est_v[i] += (beta / dt) * r[i]
                self.gate_count = 0
            self.est = p
        self.has_track = 1

        err = [self.est[0] - sc[0], self.est[1] - sc[1]]
        for i in range(2):
            de = max(-150.0, min(150.0, err[i] - self.prev_err[i]))
            if fresh:
                self.derr[i] = 0.0
            else:
                self.derr[i] = 0.6 * de + 0.4 * self.derr[i]
            self.prev_err[i] = err[i]

        db = self.deadband
        if db > 0.0 and err[0]**2 + err[1]**2 < db*db and \
           self.est_v[0]**2 + self.est_v[1]**2 < db*db:
            self.residual = [0.0, 0.0]
            return 0, 0

        mx = nonlinear_p(err[0], self.g.kp_x, self.g.soft_x) \
            + self.g.kd_x * self.derr[0] + self.track_ff * self.est_v[0]
        my = nonlinear_p(err[1], self.g.kp_y, self.g.soft_y) \
            + self.g.kd_y * self.derr[1] + self.track_ff * self.est_v[1]
        mx, my = clamp_max_step(mx, my)
        dx, self.residual[0] = emit_mouse_delta(mx, self.residual[0])
        dy, self.residual[1] = emit_mouse_delta(my, self.residual[1])
        # Shipped path: NOT ego-compensated here - the host tells us the real
        # applied delta via set_applied() before the next step. ego_bug path:
        # subtract our own guess now (the reviewed defect) to show it desyncs.
        if self.ego_bug and self.ego_comp and self.has_track:
            self.est[0] -= dx; self.est[1] -= dy
            self.applied = [0.0, 0.0]  # ignore the host's real feedback
        return dx, dy

# ------------------------------------------------------------------ scenarios
SC = (160.0, 160.0)   # screen center (320 model)
NOISE = 1.0           # detector center noise sigma, px/axis

def run(ctrl, scenario, frames=900, seed=0):
    """Returns dict of metrics. Target position T is ABSOLUTE; crosshair C
    starts at SC; frame-coord measurement raw = T - C + SC + noise."""
    rng = random.Random(seed)
    ctrl.reset()
    C = [SC[0], SC[1]]
    T = [SC[0] + 80.0, SC[1] + 40.0] if scenario == "step" else [SC[0], SC[1]]
    v = [0.0, 0.0]
    if scenario in ("cv", "gap", "drop"):
        v = [1.0, 0.3]
    if scenario == "zigzag":
        v = [1.5, 0.0]

    errs, emits, settle_frame, max_overshoot = [], [], None, 0.0
    post_turn_peak = 0.0
    turn_frames = set()
    init_dist = math.hypot(T[0]-C[0], T[1]-C[1])
    missed = 0

    for f in range(frames):
        # --- world update
        if scenario == "zigzag" and f > 0 and f % 150 == 0:
            v[0] = -v[0]; turn_frames.add(f)
        T[0] += v[0]; T[1] += v[1]

        # --- detection
        gap = (scenario == "gap" and f % 60 in (20, 21, 22))
        if gap:
            missed += 1
            if isinstance(ctrl, OldController):
                dx, dy = ctrl.coast(missed)      # old: coast glide
            else:
                dx, dy = 0, 0                    # new: hold
                ctrl.residual = [0.0, 0.0]
        else:
            nx = rng.gauss(0.0, NOISE); ny = rng.gauss(0.0, NOISE)
            if scenario == "glitch" and rng.random() < 0.01:
                nx += 40.0
            raw = (T[0] - C[0] + SC[0] + nx, T[1] - C[1] + SC[1] + ny)
            if isinstance(ctrl, NewController):
                dx, dy = ctrl.step(raw, SC, frames_elapsed=missed + 1)
            else:
                dx, dy = ctrl.step(raw, SC)
            missed = 0
        # 'drop': the host discards the emitted move (flick override / queue
        # full / aiming dropped between kernel and callback). The kernel already
        # ran, but the mouse does NOT move. Applied delta fed back is 0.
        applied_dx, applied_dy = dx, dy
        if scenario == "drop" and not gap and rng.random() < 0.15:
            applied_dx, applied_dy = 0, 0
        C[0] += applied_dx; C[1] += applied_dy
        dx, dy = applied_dx, applied_dy  # metrics reflect real mouse motion
        # Host feeds the ACTUALLY-applied delta back for next frame's ego comp.
        if isinstance(ctrl, NewController):
            ctrl.set_applied(applied_dx, applied_dy)

        e = math.hypot(T[0] - C[0], T[1] - C[1])
        errs.append(e); emits.append((dx, dy))
        if scenario == "step":
            if settle_frame is None and e < 2.0:
                settle_frame = f
            # overshoot: crosshair travelled past the target along the approach
            ax, ay = T[0] - SC[0], T[1] - SC[1]
            proj = ((C[0]-SC[0])*ax + (C[1]-SC[1])*ay) / max(init_dist, 1e-9)
            max_overshoot = max(max_overshoot, proj - init_dist)
        if scenario == "zigzag":
            for tf in turn_frames:
                if tf <= f < tf + 60:
                    post_turn_peak = max(post_turn_peak, e)

    w = errs[120:]  # steady-state window
    rms = math.sqrt(sum(x*x for x in w) / len(w))
    move_frac = sum(1 for d in emits[120:] if d != (0, 0)) / len(emits[120:])
    mean_emit = sum(math.hypot(*d) for d in emits[120:]) / len(emits[120:])
    out = {"rms": rms, "move%": 100.0*move_frac, "emit": mean_emit,
           "max_err": max(w)}
    if scenario == "step":
        out["settle"] = settle_frame if settle_frame is not None else -1
        out["overshoot"] = max_overshoot
    if scenario == "zigzag":
        out["turn_peak"] = post_turn_peak
    return out

def avg_runs(make_ctrl, scenario, n=8):
    acc = {}
    for s in range(n):
        m = run(make_ctrl(), scenario, seed=s)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v)/len(v) for k, v in acc.items()}

def flick_handover(ctrl, flick_frames=30, seed=0):
    """The applied != emitted case the review flagged, and every real
    acquisition in needaimbot: for the first flick_frames the host plays a
    scripted acquisition flick (moves the crosshair ~2.7 px/frame toward the
    target) and DISCARDS the controller's emitted move; the controller still
    fuses measurements. Then PD takes over. Returns (handover_peak_px,
    settle_frames_after_flick). A controller that ego-compensates its own
    emitted guess instead of the host's applied delta desyncs by the whole
    flick distance and lurches at handover; one closed on applied_dx/dy does
    not. Kept as a regression oracle for AimConfig::applied_dx/dy."""
    ctrl.reset()
    SC = (160.0, 160.0)
    rng = random.Random(seed)
    C = [SC[0], SC[1]]; T = [SC[0] + 80.0, SC[1] + 20.0]
    flick_per = [(T[0]-C[0])/flick_frames, (T[1]-C[1])/flick_frames]
    peak, settle = 0.0, None
    for f in range(300):
        nx = rng.gauss(0.0, NOISE); ny = rng.gauss(0.0, NOISE)
        raw = (T[0]-C[0]+SC[0]+nx, T[1]-C[1]+SC[1]+ny)
        edx, edy = ctrl.step(raw, SC)
        adx, ady = (flick_per if f < flick_frames else (edx, edy))
        C[0] += adx; C[1] += ady
        ctrl.set_applied(adx, ady)
        e = math.hypot(T[0]-C[0], T[1]-C[1])
        if f >= flick_frames:
            peak = max(peak, e)
            if settle is None and e < 2.0:
                settle = f - flick_frames
    return peak, (settle if settle is not None else -1)

SCENARIOS = ["step", "still", "cv", "zigzag", "glitch", "gap", "drop"]

def row(name, make_ctrl):
    cells = [f"{name:26s}"]
    for sc in SCENARIOS:
        m = avg_runs(make_ctrl, sc)
        extra = ""
        if sc == "step":
            extra = f" st={m['settle']:.0f}f ov={m['overshoot']:.1f}"
        if sc == "zigzag":
            extra = f" pk={m['turn_peak']:.1f}"
        if sc == "still":
            extra = f" mv={m['move%']:.0f}%"
        cells.append(f"{m['rms']:5.2f}{extra}")
    print(" | ".join(cells))

if __name__ == "__main__":
    hdr = f"{'controller':26s} | " + " | ".join(f"{s}(rms px)" for s in SCENARIOS)
    print(hdr); print("-" * len(hdr))
    g = Gains()
    row("OLD (oneeuro+ff+coast)", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)))
    print()
    print("-- ego-motion compensation off vs on (track_ff=0) --")
    row("NEW a=.35 no-ego", lambda: NewController(g, alpha=0.35, ego_comp=False))
    for a in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        row(f"NEW a={a:.2f} ego", lambda a=a: NewController(g, alpha=a))
    print()
    print("-- ego + track_ff sweep (alpha=0.35): cancel P chase lag, no future lead --")
    for ff in (0.0, 0.5, 0.8, 1.0):
        row(f"NEW a=.35 ego ff={ff:.1f}", lambda ff=ff: NewController(g, alpha=0.35, track_ff=ff))
    print()
    print("-- ego + ff=0.8, alpha sweep --")
    for a in (0.20, 0.25, 0.30, 0.35, 0.40):
        row(f"NEW a={a:.2f} ego ff=.8", lambda a=a: NewController(g, alpha=a, track_ff=0.8))
    print()
    print("-- ego + ff=0.8 a=0.30, tighter softness --")
    for sx, sy in ((11, 10), (8, 7), (6, 5)):
        gg = Gains(soft_x=float(sx), soft_y=float(sy))
        row(f"NEW soft={sx}/{sy}", lambda gg=gg: NewController(gg, alpha=0.30, track_ff=0.8))
    print()
    print("-- flick handover: ego closed on applied delta vs on emitted guess --")
    pk_s, st_s = flick_handover(NewController(Gains()))
    pk_b, st_b = flick_handover(NewController(Gains(), ego_bug=True))
    print(f"  shipped (applied_dx/dy) : handover_peak={pk_s:6.2f}px  settle={st_s}f")
    print(f"  bug (self-subtract emit): handover_peak={pk_b:6.2f}px  settle={st_b}f")
