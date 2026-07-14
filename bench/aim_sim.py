#!/usr/bin/env python3
"""Closed-loop aim-controller simulation for OFFLINE design comparison. Both
controllers are ported into Python here (not imported from C++):
  SHIPPED (this branch, "OLD" below): One Euro pre-filter + raw-delta velocity
      feedforward + coast + nonlinear P+D - what needaimbot/cuda/pd_controller.cuh
      actually implements on 2pc.
  EXPERIMENTAL / ABANDONED ("NEW" below): gated alpha-beta estimator + ego-motion
      compensation (AimConfig::applied_dx/dy) + hold deadband + nonlinear P+D.
      Tried on a since-deleted branch and dropped - it wins this sim but felt
      worse on hardware (ego-comp is sensitive to capture->move latency this sim
      models as perfect). NOT in the shipped pd_controller.cuh (no applied_dx/dy
      there); kept only as a research/ceiling model.

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

The "NEW" rows below quantify the ABANDONED design's ceiling under idealized
(perfect-latency) ego-comp - they do NOT validate the shipped code. The "OLD"
rows are the shipped One Euro pipeline.

Reference numbers (idealized sim):
  ABANDONED alpha-beta+ego (alpha=0.35, gate=50, deadband=1.0, track_ff=0.8,
    softness 8/7): still 0.30 px / 1% moving; cv 1.30; zigzag 1.74; glitch 2.12;
    step overshoot 0.7.
  SHIPPED One Euro+ff+coast: still 0.89 px / 36% moving; cv 5.06; zigzag 5.31;
    glitch 3.29.
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
    """SHIPPED pipeline (this branch): One Euro pre-filter, raw-delta velocity EMA,
    P + D + feedforward, coast glide on detection gaps."""
    def __init__(self, g: Gains, ff=0.9, min_cutoff=0.1, beta=0.02, dcutoff=0.5,
                 coast_decay=0.85, ego=False):
        self.g, self.ff = g, ff
        self.min_cutoff, self.beta, self.dcutoff = min_cutoff, beta, dcutoff
        self.coast_decay = coast_decay
        # ego: HYBRID - add ONLY ego-motion comp to One Euro. The velocity that
        # feeds the feedforward is (raw_delta + applied) = the target's TRUE
        # screen drift instead of (drift - our own motion), so ff cancels the
        # chase lag against a moving target. The One Euro position filter (what
        # gives the quiet, sticky feel) and everything else are untouched.
        self.ego = ego
        self.reset()

    def reset(self):
        self.has_track = 0
        self.filt = [0.0, 0.0]; self.dfilt = [0.0, 0.0]
        self.prev_center = [0.0, 0.0]; self.vel = [0.0, 0.0]
        self.prev_err = [0.0, 0.0]; self.derr = [0.0, 0.0]
        self.residual = [0.0, 0.0]
        # SUM of the host's applied deltas since the last accepted detection.
        # Accumulated (not overwritten) so a multi-frame coast/gap - where the
        # crosshair keeps moving while prev_center is frozen - still gets its full
        # motion added back. Consumed and zeroed by step() on the next detection.
        self.applied_accum = [0.0, 0.0]

    def set_applied(self, dx, dy):
        self.applied_accum[0] += float(dx)
        self.applied_accum[1] += float(dy)

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
                # raw_delta = dTarget - our_applied_motion. With ego, add applied
                # back to recover the target's TRUE screen drift (dTarget); the
                # feedforward then cancels chase lag. Without ego (base One Euro),
                # once we track well raw_delta -> 0 and ff does nothing -> lag.
                d = raw[i] - self.prev_center[i]
                if self.ego:
                    d += self.applied_accum[i]   # full motion since last detection
                nv = max(-60.0, min(60.0, d))
                self.vel[i] = 0.6 * self.vel[i] + 0.4 * nv
        else:
            self.vel = [0.0, 0.0]
        self.applied_accum = [0.0, 0.0]   # consumed; restart accumulation
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
        # a move is discarded/dropped, est desyncs. Off = the experimental
        # design's closed-loop feedback (subtract set_applied()). This whole ego
        # path is the ABANDONED design, not shipped on this branch.
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
        # Correct (experimental-design) path: NOT ego-compensated here - the host
        # tells us the real applied delta via set_applied() before the next step.
        # ego_bug path: subtract our own guess now (the reviewed defect) to desync.
        if self.ego_bug and self.ego_comp and self.has_track:
            self.est[0] -= dx; self.est[1] -= dy
            self.applied = [0.0, 0.0]  # ignore the host's real feedback
        return dx, dy

# ------------------------------------------------------------------ scenarios
SC = (160.0, 160.0)   # screen center (320 model)
NOISE = 1.0           # detector center noise sigma, px/axis (clean model)
FRAME_MS = 1000.0 / 300.0   # 300 detections/s -> 3.33 ms/frame


class DetectorNoise:
    """Every perturbation that can hit the measured target center in this
    YOLO-nano (mAP95~0.5) + two-PC UDP pipeline. Stateful (correlated) sources:

      white    iid sub-pixel decode jitter
      drift    AR(1) low-frequency wander - the box 'breathes', frame-correlated
      boxjit   bbox-edge jitter -> center shift, Y >> X: a head box's top/bottom
               edge is far less stable than its sides, so the CENTER wobbles
               vertically more than horizontally (the real vertical-shake source)
      quant    model-input grid quantization of the decoded center
      dropout  missed detection this frame (no box -> controller holds/coasts)
      stale    UDP frame stall: the previous detection repeats (frame not updated)
      flicker  target-selection flip (head<->body / candidate switch): the center
               jumps - mostly in Y - for a few frames, then back = the 'phantom'
      outlier  gross false box for a single frame

    measure(cx, cy) -> (mx, my) measured center, or None on a dropout.
    (Latency jitter and host move-discard are modelled in run(), not here.)
    """
    def __init__(self, rng, white=0.8, drift=1.5, drift_rho=0.92,
                 box_x=0.5, box_y=2.5, box_rho=0.6, quant=0.75,
                 p_dropout=0.04, p_stale=0.03, p_flicker=0.015, flicker_y=20.0,
                 p_outlier=0.008, outlier=35.0):
        self.rng = rng
        self.white, self.drift, self.drift_rho = white, drift, drift_rho
        self.box_x, self.box_y, self.box_rho = box_x, box_y, box_rho
        self.quant = quant
        self.p_dropout, self.p_stale = p_dropout, p_stale
        self.p_flicker, self.flicker_y = p_flicker, flicker_y
        self.p_outlier, self.outlier = p_outlier, outlier
        self.dx = self.dy = 0.0          # AR(1) drift state
        self.bx = self.by = 0.0          # AR(1) box-jitter state
        self.flick_left = 0
        self.flick_off = (0.0, 0.0)
        self.last = None                 # last emitted measurement (for stale)

    def _ar1(self, prev, rho, sig):
        return rho * prev + self.rng.gauss(0.0, sig * math.sqrt(max(1e-9, 1.0 - rho*rho)))

    def measure(self, cx, cy):
        if self.rng.random() < self.p_dropout:
            return None                                  # missed detection
        if self.last is not None and self.rng.random() < self.p_stale:
            return self.last                             # UDP frame stall
        self.dx = self._ar1(self.dx, self.drift_rho, self.drift)
        self.dy = self._ar1(self.dy, self.drift_rho, self.drift)
        self.bx = self._ar1(self.bx, self.box_rho, self.box_x)
        self.by = self._ar1(self.by, self.box_rho, self.box_y)
        mx = cx + self.dx + self.bx + self.rng.gauss(0.0, self.white)
        my = cy + self.dy + self.by + self.rng.gauss(0.0, self.white)
        if self.flick_left > 0:                          # phantom, held a few frames
            mx += self.flick_off[0]; my += self.flick_off[1]
            self.flick_left -= 1
        elif self.rng.random() < self.p_flicker:
            self.flick_left = self.rng.randint(1, 4) - 1
            self.flick_off = (self.rng.gauss(0.0, 3.0),
                              self.rng.choice((-1.0, 1.0)) * self.flicker_y)
            mx += self.flick_off[0]; my += self.flick_off[1]
        if self.rng.random() < self.p_outlier:           # gross false box
            mx += self.rng.gauss(0.0, self.outlier)
            my += self.rng.gauss(0.0, self.outlier)
        if self.quant > 0.0:
            mx = round(mx / self.quant) * self.quant
            my = round(my / self.quant) * self.quant
        self.last = (mx, my)
        return (mx, my)

def run(ctrl, scenario, frames=900, seed=0, latency_ms=None, noise_model="clean"):
    """Returns dict of metrics. Target position T is ABSOLUTE; crosshair C
    starts at SC; frame-coord measurement raw = T - C + SC + noise.

    latency_ms: None = perfect (measurement reflects the world NOW, so the
    ~1-frame ego-comp is exact). A (lo, hi) tuple models a random per-frame
    capture->process latency in ms: the detection instead reflects a snapshot of
    the WHOLE frame (target AND crosshair) from lat frames ago, interpolated. The
    ego-comp still assumes ~1 frame, so the (lat-1) frames of crosshair motion it
    cannot predict leak in as error - the real-hardware failure mode.

    noise_model: 'clean' = iid gaussian center noise + the scenario's own
    gap/glitch injections. 'real' = the full DetectorNoise (correlated drift,
    Y-heavy box-edge jitter, quantization, random dropouts, UDP stale frames,
    target-selection flicker/phantom, outliers); the scenario supplies only the
    target motion (its gap/glitch injections are superseded by the detector)."""
    rng = random.Random(seed)
    detector = DetectorNoise(rng) if noise_model == "real" else None
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
    world_hist = []  # (Tx, Ty, Cx, Cy) per frame, for latency-delayed snapshots

    for f in range(frames):
        # --- world update
        if scenario == "zigzag" and f > 0 and f % 150 == 0:
            v[0] = -v[0]; turn_frames.add(f)
        T[0] += v[0]; T[1] += v[1]
        world_hist.append((T[0], T[1], C[0], C[1]))

        # --- what the camera actually captured this frame: a snapshot of the
        # whole scene (target AND crosshair) delayed by a random 1-10ms latency
        if latency_ms is not None:
            lat_fr = rng.uniform(latency_ms[0], latency_ms[1]) / FRAME_MS
            idx = max(0.0, f - lat_fr)
            i0 = int(idx); i1 = min(i0 + 1, len(world_hist) - 1); fr = idx - i0
            sT0 = world_hist[i0][0]*(1-fr) + world_hist[i1][0]*fr
            sT1 = world_hist[i0][1]*(1-fr) + world_hist[i1][1]*fr
            sC0 = world_hist[i0][2]*(1-fr) + world_hist[i1][2]*fr
            sC1 = world_hist[i0][3]*(1-fr) + world_hist[i1][3]*fr
        else:
            sT0, sT1, sC0, sC1 = T[0], T[1], C[0], C[1]

        # --- detection: clean gaussian (+ scenario gap/glitch) or full DetectorNoise
        true_cx, true_cy = sT0 - sC0 + SC[0], sT1 - sC1 + SC[1]
        if noise_model == "real":
            meas = detector.measure(true_cx, true_cy)    # None on a real dropout
            gap = (meas is None)
        else:
            gap = (scenario == "gap" and f % 60 in (20, 21, 22))
            meas = None
            if not gap:
                nx = rng.gauss(0.0, NOISE); ny = rng.gauss(0.0, NOISE)
                if scenario == "glitch" and rng.random() < 0.01:
                    nx += 40.0
                meas = (true_cx + nx, true_cy + ny)

        if gap:
            missed += 1
            if isinstance(ctrl, OldController):
                dx, dy = ctrl.coast(missed)      # old: coast glide
            else:
                dx, dy = 0, 0                    # new: hold
                ctrl.residual = [0.0, 0.0]
        else:
            if isinstance(ctrl, NewController):
                dx, dy = ctrl.step(meas, SC, frames_elapsed=missed + 1)
            else:
                dx, dy = ctrl.step(meas, SC)
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
        # (Both controllers have set_applied; it is a no-op unless ego is on.)
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

def avg_runs(make_ctrl, scenario, n=8, latency_ms=None, noise_model="clean"):
    acc = {}
    for s in range(n):
        m = run(make_ctrl(), scenario, seed=s, latency_ms=latency_ms, noise_model=noise_model)
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

def row(name, make_ctrl, latency_ms=None, noise_model="clean"):
    cells = [f"{name:26s}"]
    for sc in SCENARIOS:
        m = avg_runs(make_ctrl, sc, latency_ms=latency_ms, noise_model=noise_model)
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
    print("-- REALISTIC: random 1-10 ms pipeline latency (frame = 3.33 ms) --")
    print("   whole frame (target+crosshair) delayed a random 1-10ms each frame;")
    print("   ego-comp still assumes ~1 frame, so the jitter it cannot predict")
    print("   leaks in. Compare each row against its perfect-latency twin above.")
    LAT = (1.0, 10.0)
    row("OLD (oneeuro) +lat", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)), latency_ms=LAT)
    row("NEW a=.35 no-ego +lat", lambda: NewController(g, alpha=0.35, ego_comp=False), latency_ms=LAT)
    for a in (0.30, 0.35, 0.40):
        row(f"NEW a={a:.2f} ego +lat", lambda a=a: NewController(g, alpha=a, track_ff=0.8), latency_ms=LAT)
    print()
    print("-- REALISTIC detector+pipeline noise (correlated drift, Y-heavy box")
    print("   jitter, quantization, dropouts, UDP stale frames, target flicker/")
    print("   phantom, outliers). 'still'/'cv'/'zigzag' are the ones to read. --")
    row("OLD (oneeuro) real", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)), noise_model="real")
    row("NEW a=.35 no-ego real", lambda: NewController(g, alpha=0.35, ego_comp=False), noise_model="real")
    for a in (0.25, 0.30, 0.35):
        row(f"NEW a={a:.2f} ego real", lambda a=a: NewController(g, alpha=a, track_ff=0.8), noise_model="real")
    print("   -- track_ff sweep under real noise (suspected Y-jitter amplifier) --")
    for ff in (0.0, 0.4, 0.8):
        row(f"NEW a=.30 ff={ff:.1f} real", lambda ff=ff: NewController(g, alpha=0.30, track_ff=ff), noise_model="real")
    print("   -- FULL: real noise + 1-10ms latency --")
    row("OLD real+lat", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)), latency_ms=(1.0, 10.0), noise_model="real")
    row("NEW a=.30 ego real+lat", lambda: NewController(g, alpha=0.30, track_ff=0.8), latency_ms=(1.0, 10.0), noise_model="real")
    row("NEW a=.30 ff=0 real+lat", lambda: NewController(g, alpha=0.30, track_ff=0.0), latency_ms=(1.0, 10.0), noise_model="real")
    print()
    print("== HYBRID: One Euro + ego-comp ONLY (no alpha-beta, no track_ff) ==")
    print("   Does adding just ego to the shipped filter cancel chase lag WITHOUT")
    print("   adding rest jitter? (still/mv% = feel, cv/zigzag = tracking)")
    print("   clean:")
    row("OneEuro base", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)))
    row("OneEuro + ego", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0), ego=True))
    print("   real detector+pipeline noise:")
    row("OneEuro base real", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)), noise_model="real")
    row("OneEuro+ego real", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0), ego=True), noise_model="real")
    print("   real noise + 1-10ms latency:")
    row("OneEuro base real+lat", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0)), latency_ms=(1.0,10.0), noise_model="real")
    row("OneEuro+ego real+lat", lambda: OldController(Gains(soft_x=11.0, soft_y=10.0), ego=True), latency_ms=(1.0,10.0), noise_model="real")
    print()
    print("-- flick handover: ego closed on applied delta vs on emitted guess --")
    pk_s, st_s = flick_handover(NewController(Gains()))
    pk_b, st_b = flick_handover(NewController(Gains(), ego_bug=True))
    print(f"  applied_dx/dy (correct) : handover_peak={pk_s:6.2f}px  settle={st_s}f")
    print(f"  bug (self-subtract emit): handover_peak={pk_b:6.2f}px  settle={st_b}f")
