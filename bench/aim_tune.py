#!/usr/bin/env python3
"""Search filter+controller combinations for the best feel, on the validated
3D view-rotation model (144 Hz, ~1-frame/5-7 ms latency, perspective lens,
head/body dual detection, detector jitter, +-1 mouse quantization).

Evaluated on three regimes and reported per-regime, because no single number is
"feel":
  HOLD      stationary target  -> settled RMS (stickiness) + twitch% (buzz)
  TRACK     constant strafe     -> tracking RMS (lag)
  REVERSAL  sharp counter-strafe-> post-turn peak (overshoot / lurch)

Controllers are the shipped math, parameterised. Read the trade-offs, then A/B
the top pick on hardware (the final judge)."""
import math, random

FOVx = 100 * math.pi / 180
WU = 800.0
FOCAL = (WU * 0.5) / math.tan(FOVx * 0.5)
SENS0 = 0.0026
HEADH, PERSON = 0.28, 1.8

# --- Calibrated from a clean 17s / 2276-frame rig capture that has BOTH aim-OFF
# (static view, no controller) and aim-ON segments (bench/calibrate.py) ---
#   aim-OFF baseline white 11.5/8.5 px, aim-ON 10.5/9.3 px, motion-blur factor
#   ~1.0 (blur barely adds noise) -> the DETECTOR itself is the ~10px noise floor,
#   not motion blur. dropout 1.5%, head 8.9%, mildly X-heavy (Y/X ~0.85). This is
#   ~2.5x noisier than the first (short, bursty) capture suggested. WHITE_* are the
#   per-frame white sigma at the reference box (nsc=1.5); they scale with nsc.
WHITE_X, WHITE_Y = 7.0, 6.2        # -> ~10.5/9.3 px at nsc=1.5 (matches aim-ON)
P_HEAD = 0.09                      # real head-selection rate 8.9%
P_DROP = 0.015                     # real dropout under continuous aim (was mis-measured 0.40)
# Fat-tailed detector outliers (measured: kurtosis ~24, ~3-4% of frames spike
# 15-80px in a single frame, reverting the next). This is what a low-pass filter
# smears into a multi-frame excursion ("phantom shake") but an outlier-rejection
# gate removes cleanly.
P_OUTLIER = 0.035
OUTLIER_MIN, OUTLIER_MAX = 15.0, 60.0

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def nlp(e, kp, s):
    ae = abs(e); return e * (max(kp, 0) * (ae / (ae + max(s, 1))))
def oea(c): return 1.0 / (1.0 + 1.0 / (2 * math.pi * max(c, 1e-4)))
def trunc0(v): return math.floor(v) if v >= 0 else math.ceil(v)

class R:
    __slots__ = ("v")
    def __init__(self): self.v = 0.0
def emit_int(m, r):
    c = r.v
    if m * c < 0: c = 0.0
    val = m + c; e = trunc0(val); r.v = val - e; return e

# ------------------------------------------------------------- controllers
class OneEuro:
    """One Euro low-pass (min_cutoff/beta) + nonlinear P (kp/soft) + D (kd) +
    velocity feedforward (ff). filt=False disables the low-pass. ego adds the
    crosshair's own (believed) motion back into the feedforward velocity."""
    def __init__(self, p):
        self.p = p; self.filt = p.get("filt", True); self.ego = p.get("ego", False)
        self.reset()
    def reset(self):
        self.has = False; self.fx = self.fy = self.dfx = self.dfy = 0.0
        self.px = self.py = self.vx = self.vy = 0.0
        self.pex = self.pey = self.dex = self.dey = self.ax = self.ay = 0.0
    def set_applied(self, dx, dy): self.ax += dx; self.ay += dy
    def step(self, rx, ry, dt=1.0):
        p = self.p; fresh = not self.has; tcx, tcy = rx, ry
        if self.filt:
            if not fresh:
                ad = oea(p["dcut"])
                de = rx - self.fx; self.dfx = ad * de + (1 - ad) * self.dfx
                a = oea(p["mincut"] + p["beta"] * abs(self.dfx)); self.fx = a * rx + (1 - a) * self.fx
                de = ry - self.fy; self.dfy = ad * de + (1 - ad) * self.dfy
                a = oea(p["mincut"] + p["beta"] * abs(self.dfy)); self.fy = a * ry + (1 - a) * self.fy
                tcx, tcy = self.fx, self.fy
            else:
                self.fx, self.fy, self.dfx, self.dfy = rx, ry, 0.0, 0.0
        if not fresh:
            dx, dy = rx - self.px, ry - self.py
            if self.ego: dx += self.ax; dy += self.ay
            dx = clamp(dx, -60, 60); dy = clamp(dy, -60, 60)
            self.vx = 0.6 * self.vx + 0.4 * dx; self.vy = 0.6 * self.vy + 0.4 * dy
        else:
            self.vx = self.vy = 0.0
        self.ax = self.ay = 0.0; self.px, self.py = rx, ry; self.has = True
        ex, ey = tcx, tcy
        dex = clamp(ex - self.pex, -150, 150); dey = clamp(ey - self.pey, -150, 150)
        if fresh: self.dex = self.dey = 0.0
        else: self.dex = 0.6 * dex + 0.4 * self.dex; self.dey = 0.6 * dey + 0.4 * self.dey
        self.pex, self.pey = ex, ey
        mx = nlp(ex, p["kp"], p["soft"]) + p["kd"] * self.dex + p["ff"] * self.vx
        my = nlp(ey, p["kp"] * 1.1, p["soft"] * 0.9) + p["kd"] * 1.2 * self.dey + p["ff"] * self.vy
        st = math.hypot(mx, my)
        if st > 30: k = 30 / st; mx *= k; my *= k
        return mx, my

class AlphaBeta:
    def __init__(self, p):
        self.p = p; self.alpha = p["alpha"]; self.ego = p.get("ego", True)
        self.tf = p["ff"]; self.gate = 50.0; self.reset()
    def reset(self):
        self.has = False; self.ex = self.ey = self.vx = self.vy = 0.0; self.gc = 0
        self.pex = self.pey = self.dex = self.dey = self.ax = self.ay = 0.0
    def set_applied(self, dx, dy): self.ax += dx; self.ay += dy
    def step(self, rx, ry, dt=1.0):
        p = self.p; fresh = not self.has; a = clamp(self.alpha, 0.01, 1); b = a * a / (2 - a)
        if fresh:
            self.ex, self.ey, self.vx, self.vy, self.gc = rx, ry, 0.0, 0.0, 0
        else:
            if self.ego: self.ex -= self.ax; self.ey -= self.ay
            px = self.ex + self.vx * dt; py = self.ey + self.vy * dt
            rrx, rry = rx - px, ry - py; rn = math.hypot(rrx, rry)
            if rn > self.gate:
                self.gc += 1
                if self.gc >= 3: px, py, self.vx, self.vy, self.gc = rx, ry, 0.0, 0.0, 0
            else:
                px += a * rrx; py += a * rry; self.vx += (b / dt) * rrx; self.vy += (b / dt) * rry; self.gc = 0
            self.ex, self.ey = px, py
        self.ax = self.ay = 0.0; self.has = True
        ex, ey = self.ex, self.ey
        dex = clamp(ex - self.pex, -150, 150); dey = clamp(ey - self.pey, -150, 150)
        if fresh: self.dex = self.dey = 0.0
        else: self.dex = 0.6 * dex + 0.4 * self.dex; self.dey = 0.6 * dey + 0.4 * self.dey
        self.pex, self.pey = ex, ey
        mx = nlp(ex, p["kp"], p["soft"]) + p["kd"] * self.dex + self.tf * self.vx
        my = nlp(ey, p["kp"] * 1.1, p["soft"] * 0.9) + p["kd"] * 1.2 * self.dey + self.tf * self.vy
        st = math.hypot(mx, my)
        if st > 30: k = 30 / st; mx *= k; my *= k
        return mx, my


class OneEuroX(OneEuro):
    """One Euro + PD with three new experimental levers on top of the shipped
    architecture, each attacking the dead-time-vs-noise tension differently:

      gate_ff : confidence-gated feedforward. Instead of a fixed ff, scale it by
                how steady the velocity estimate is: g = |v|/(|v|+vgate). Near
                rest / under jitter |v| is small -> ff suppressed (kills the
                ff0.9 buzz); when the target genuinely moves -> ff engages.
      predict : dead-time predictor (Smith-flavored). Aim at where the target
                will be `horizon` frames ahead = filtered_pos + v*horizon, so
                the command lands on the future target instead of the stale one.
                Also confidence-gated so noise isn't extrapolated.
      axis    : X-heavy noise (real calibration) -> filter/damp X harder than Y.
                xcut_k<1 lowers X's min-cutoff (smoother X); ykd_k adjusts Y kd.
    """
    def __init__(self, p):
        super().__init__(p)
        self.gate_ff = p.get("gate_ff", False)
        self.predict = p.get("predict", 0.0)   # frames of lead (0 = off)
        self.vgate = p.get("vgate", 6.0)       # px/frame speed at which ff is ~half
        self.xcut_k = p.get("xcut_k", 1.0)     # <1 = smoother X (lower min-cutoff)
        self.ykd_k = p.get("ykd_k", 1.0)
        self.jclamp = p.get("jump_clamp", 0.0)   # max px a detection can move the estimate/frame (0=off)
        self.jlock = p.get("jump_lock", 0.0)      # only clamp when |error| < jlock (locked); 0 = always

    def step(self, rx, ry, dt=1.0):
        p = self.p; fresh = not self.has; tcx, tcy = rx, ry
        # Jump clamp (rate limiter on the filter input): the detector noise is
        # fat-tailed (rare 15-60px single-frame spikes), but real target motion is
        # <~10px/frame. So limit how far ONE detection can move the estimate: normal
        # noise and real motion (< jclamp) pass unchanged, outliers get capped. Unlike
        # a hard reject gate this never holds stale (no false-positives), just
        # attenuates the spike. Beats a low-pass, which smears a spike over frames.
        if self.jclamp > 0.0 and not fresh:
            # Only clamp when LOCKED (filtered estimate near the target). While
            # acquiring (large error) a big jump is real motion -> don't clamp, so
            # fast flicks/target-switches stay crisp. Once locked, a big jump is an
            # outlier -> clamp it (kills phantom shake without lagging acquisition).
            locked = (self.jlock <= 0.0) or (math.hypot(self.fx, self.fy) < self.jlock)
            if locked:
                rx = self.fx + clamp(rx - self.fx, -self.jclamp, self.jclamp)
                ry = self.fy + clamp(ry - self.fy, -self.jclamp, self.jclamp)
        if self.filt:
            if not fresh:
                ad = oea(p["dcut"])
                de = rx - self.fx; self.dfx = ad * de + (1 - ad) * self.dfx
                ax_ = oea(p["mincut"] * self.xcut_k + p["beta"] * abs(self.dfx))
                self.fx = ax_ * rx + (1 - ax_) * self.fx
                de = ry - self.fy; self.dfy = ad * de + (1 - ad) * self.dfy
                ay_ = oea(p["mincut"] + p["beta"] * abs(self.dfy))
                self.fy = ay_ * ry + (1 - ay_) * self.fy
                tcx, tcy = self.fx, self.fy
            else:
                self.fx, self.fy, self.dfx, self.dfy = rx, ry, 0.0, 0.0
        if not fresh:
            dx, dy = rx - self.px, ry - self.py
            if self.ego: dx += self.ax; dy += self.ay
            dx = clamp(dx, -60, 60); dy = clamp(dy, -60, 60)
            self.vx = 0.6 * self.vx + 0.4 * dx; self.vy = 0.6 * self.vy + 0.4 * dy
        else:
            self.vx = self.vy = 0.0
        self.ax = self.ay = 0.0; self.px, self.py = rx, ry; self.has = True
        # velocity-confidence gate: ~0 at rest/jitter, ->1 when steadily moving
        spd = math.hypot(self.vx, self.vy)
        gconf = spd / (spd + self.vgate)
        ex, ey = tcx, tcy
        if self.predict > 0.0:                 # aim ahead of the (filtered) target
            ex += self.vx * self.predict * gconf
            ey += self.vy * self.predict * gconf
        dex = clamp(ex - self.pex, -150, 150); dey = clamp(ey - self.pey, -150, 150)
        if fresh: self.dex = self.dey = 0.0
        else: self.dex = 0.6 * dex + 0.4 * self.dex; self.dey = 0.6 * dey + 0.4 * self.dey
        self.pex, self.pey = ex, ey
        ffx = p["ff"] * self.vx; ffy = p["ff"] * self.vy
        if self.gate_ff: ffx *= gconf; ffy *= gconf
        mx = nlp(ex, p["kp"], p["soft"]) + p["kd"] * self.dex + ffx
        my = nlp(ey, p["kp"] * 1.1, p["soft"] * 0.9) + p["kd"] * self.ykd_k * 1.2 * self.dey + ffy
        st = math.hypot(mx, my)
        if st > 30: k = 30 / st; mx *= k; my *= k
        return mx, my

class KalmanCV:
    """Constant-velocity Kalman on the measured target offset (rx,ry), then the
    same nonlinear P + D + (gated) feedforward as OneEuro. Unlike the hand-tuned
    One Euro cutoffs, the Kalman gain is DERIVED from the measured noise: R = the
    real detector variance (px^2), Q = how much the target can accelerate. This is
    the optimal linear estimator for the noise we actually measured - the test of
    whether hand-tuning left anything on the table."""
    def __init__(self, p):
        self.p = p
        self.R = p.get("R", 15.0)          # measurement variance px^2 (~3.9px sigma)
        self.q = p.get("Q", 0.5)           # process (accel) noise
        self.gate_ff = p.get("gate_ff", True)
        self.vgate = p.get("vgate", 6.0)
        self.reset()
    def reset(self):
        self.has = False
        self.sx = [0.0, 0.0]; self.sy = [0.0, 0.0]         # [pos, vel] per axis
        self.Px = [[1e3, 0.0], [0.0, 1e3]]; self.Py = [[1e3, 0.0], [0.0, 1e3]]
        self.pex = self.pey = self.dex = self.dey = self.ax = self.ay = 0.0
    def set_applied(self, dx, dy): pass   # ego off (blows up under real noise)
    def _kf(self, P, s, meas, dt):
        s[0] += s[1] * dt                                  # predict pos
        q = self.q
        P00 = P[0][0] + dt*(P[1][0]+P[0][1]) + dt*dt*P[1][1] + q*dt*dt*dt/3.0
        P01 = P[0][1] + dt*P[1][1] + q*dt*dt/2.0
        P10 = P[1][0] + dt*P[1][1] + q*dt*dt/2.0
        P11 = P[1][1] + q*dt
        S = P00 + self.R                                   # innovation cov
        K0 = P00 / S; K1 = P10 / S
        r = meas - s[0]
        s[0] += K0 * r; s[1] += K1 * r
        P[0][0] = (1-K0)*P00; P[0][1] = (1-K0)*P01
        P[1][0] = P10 - K1*P00; P[1][1] = P11 - K1*P01
        return s[0], s[1]
    def step(self, rx, ry, dt=1.0):
        p = self.p; fresh = not self.has
        if fresh:
            self.sx = [rx, 0.0]; self.sy = [ry, 0.0]; self.has = True
            self.pex, self.pey = rx, ry; self.dex = self.dey = 0.0
            fx, vx, fy, vy = rx, 0.0, ry, 0.0
        else:
            fx, vx = self._kf(self.Px, self.sx, rx, dt)
            fy, vy = self._kf(self.Py, self.sy, ry, dt)
        ex, ey = fx, fy
        dex = clamp(ex - self.pex, -150, 150); dey = clamp(ey - self.pey, -150, 150)
        if fresh: self.dex = self.dey = 0.0
        else:
            self.dex = 0.6*dex + 0.4*self.dex; self.dey = 0.6*dey + 0.4*self.dey
        self.pex, self.pey = ex, ey
        spd = math.hypot(vx, vy); g = spd/(spd+self.vgate) if self.gate_ff else 1.0
        mx = nlp(ex, p["kp"], p["soft"]) + p["kd"]*self.dex + p["ff"]*vx*g
        my = nlp(ey, p["kp"]*1.1, p["soft"]*0.9) + p["kd"]*1.2*self.dey + p["ff"]*vy*g
        st = math.hypot(mx, my)
        if st > 30: k = 30/st; mx *= k; my *= k
        return mx, my

def project(az, el, yaw, pitch):
    d = math.atan2(math.sin(az - yaw), math.cos(az - yaw))
    return FOCAL * math.tan(d), -FOCAL * math.tan(el - pitch)

# ------------------------------------------------------------- one run
def run(make, regime, seed=0, frames=1400, lat_base=1):
    rng = random.Random(seed)
    def g(sd): return rng.gauss(0.0, sd)
    c = make()
    yaw = pitch = 0.0
    az, el, dist = (0.0 if regime == "step" else -0.30), 0.02, 12.0
    resx, resy = R(), R()
    drx = dry = lagj = 0.0
    hist = []
    errs = []; mvs = []; peak = 0.0; turnf = -999
    tv = 0.0045 if regime == "track" else 0.0
    STEPS = list(range(400, frames, 300))   # step regime: jump every 300 frames
    rises = []; pend = None                  # rise-time samples
    prev_az = az; view_blur = 0.0            # motion-blur state (view + world)
    for f in range(frames):
        # ---- target motion per regime ----
        if regime == "hold":
            pass
        elif regime == "track":
            az += tv
            if abs(az) > 0.55: tv = -tv; az = clamp(az, -0.55, 0.55)   # bounce, no teleport
        elif regime == "step":
            if f in STEPS:
                az = 0.16 if az <= 0 else -0.16                        # sudden 0.16 rad jump
                pend = f                                               # awaiting recovery
        else:  # reversal: strafe, sharp reverse every 240 frames
            if f % 240 == 0:
                tv = (0.0055 if (f // 240) % 2 else -0.0055); turnf = f
            az += tv; az = clamp(az, -0.7, 0.7)
        dist = clamp(dist + g(0.03), 8, 30)
        hist.append((az, el, yaw, pitch, dist))
        if len(hist) > 80: hist.pop(0)
        lagj = 0.6 * lagj + g(0.4); lag = int(round(clamp(lat_base + lagj, 0, 12)))
        s = hist[max(0, len(hist) - 1 - lag)]
        box = FOCAL * (HEADH / s[4]); nsc = clamp(18 / box, 0.5, 2.4)
        # MOTION BLUR: fast VIEW rotation (your own aiming) + target world motion
        # blur the capture -> noisier detection. This is the fact the calibration
        # exposed: noise is measured while the view moves, not on a static frame.
        # It penalises controllers that whip the view around (a self-defeating loop).
        world_px = abs(az - prev_az) * FOCAL
        # Real motion-blur factor measured ~1.0 (blur barely adds noise; the
        # detector floor dominates), so this amplification is now near-zero.
        nsc *= (1.0 + 0.005 * (view_blur + world_px)); prev_az = az
        drx = 0.9 * drx + g(0.32 * nsc); dry = 0.9 * dry + g(0.32 * nsc)
        # head/body dual detection (real head-selection rate P_HEAD)
        head_seen = rng.random() < P_HEAD
        hx, hy = project(s[0], s[1], s[2], s[3])
        if head_seen:
            detx, dety = hx, hy
        else:
            bx, by = project(s[0], s[1] - (0.62 * PERSON) / s[4], s[2], s[3])
            detx, dety = bx, by
        # White detector noise: X-heavy per the real calibration (WHITE_X>WHITE_Y).
        rx = detx + drx + g(WHITE_X * nsc); ry = dety + dry + g(WHITE_Y * nsc)
        # Fat-tailed detector outliers: the real aim-OFF noise is NOT gaussian
        # (kurtosis ~24, 3-4% of frames jump 15-80px). Inject a rare large single-
        # frame spike so the outlier-rejection gate has something real to reject.
        if rng.random() < P_OUTLIER:
            ang = rng.random() * 6.283
            mag = OUTLIER_MIN + rng.random() * (OUTLIER_MAX - OUTLIER_MIN)
            rx += mag * math.cos(ang); ry += mag * math.sin(ang)
        dtc = clamp(1 + 0.6 * g(0.5), 0.4, 2.2)
        # Dropout: on a miss the detector gives no fresh box -> feed the stale
        # previous measurement (controller must ride it out / coast).
        if f > 0 and rng.random() < P_DROP:
            rx, ry = prev_rx, prev_ry
        prev_rx, prev_ry = rx, ry
        mv = c.step(rx, ry, dtc)
        ax = emit_int(mv[0], resx); ay = emit_int(mv[1], resy)
        yaw += ax * SENS0; pitch -= ay * SENS0
        c.set_applied(ax * FOCAL * SENS0, ay * FOCAL * SENS0)
        view_blur = 0.6 * view_blur + 0.4 * math.hypot(ax, ay) * FOCAL * SENS0   # view-rotation blur
        nx, ny = project(az, el, yaw, pitch)
        e = math.hypot(nx, ny)
        # rise time: frames from a step jump until the crosshair recovers (<14px)
        if pend is not None:
            if e < 14.0: rises.append(f - pend); pend = None
            elif f - pend > 250: rises.append(250); pend = None
        if f > 300:
            errs.append(e); mvs.append(1 if (abs(ax) + abs(ay)) > 0 else 0)
            if regime == "reversal" and 0 <= f - turnf < 90:
                peak = max(peak, e)
    rms = math.sqrt(sum(x * x for x in errs) / len(errs))
    mv = 100.0 * sum(mvs) / len(mvs)
    rise = (sum(rises) / len(rises)) if rises else 250.0
    return rms, mv, peak, rise

def evalcfg(make, lat_base=1, seeds=4):
    hold = trk = rev = tw = rise = 0.0
    for s in range(seeds):
        rms, m, _, _ = run(make, "hold", s, lat_base=lat_base); hold += rms; tw += m
        rms, _, _, _ = run(make, "track", s, lat_base=lat_base); trk += rms
        _, _, p, _ = run(make, "reversal", s, lat_base=lat_base); rev += p
        _, _, _, ri = run(make, "step", s, lat_base=lat_base); rise += ri
    n = seeds
    return dict(hold=hold/n, track=trk/n, rev=rev/n, tw=tw/n, rise=rise/n)

# ------------------------------------------------------------- search space
def oe(**kw):
    base = dict(mincut=0.1, beta=0.02, dcut=0.5, kp=0.55, soft=11, kd=0.2, ff=0.9, filt=True, ego=False)
    base.update(kw); return lambda b=base: OneEuro(b)
def ab(**kw):
    base = dict(alpha=0.3, kp=0.55, soft=8, kd=0.2, ff=0.8, ego=True)
    base.update(kw); return lambda b=base: AlphaBeta(b)
def oex(**kw):
    base = dict(mincut=0.1, beta=0.02, dcut=0.5, kp=0.55, soft=11, kd=0.2, ff=0.9,
               filt=True, ego=False, gate_ff=False, predict=0.0, vgate=6.0,
               xcut_k=1.0, ykd_k=1.0)
    base.update(kw); return lambda b=base: OneEuroX(b)
def kf(**kw):
    base = dict(R=15.0, Q=0.5, kp=0.5, soft=11, kd=0.3, ff=0.6, gate_ff=True, vgate=6.0)
    base.update(kw); return lambda b=base: KalmanCV(b)

CANDS = []
# One Euro + PD + ff family (the shipped architecture) — sweep the levers
for mincut in (0.06, 0.12, 0.25):
    for beta in (0.015, 0.04):
        for ff in (0.0, 0.6, 1.0):
            for kp in (0.5, 0.65):
                for kd in (0.2, 0.32):
                    CANDS.append(("OE mc%.2f b%.3f ff%.1f kp%.2f kd%.2f" % (mincut, beta, ff, kp, kd),
                                  oe(mincut=mincut, beta=beta, ff=ff, kp=kp, kd=kd)))
# One Euro + ego variants
for mincut in (0.12, 0.25):
    for ff in (0.6, 1.0):
        CANDS.append(("OE+ego mc%.2f ff%.1f" % (mincut, ff), oe(mincut=mincut, ff=ff, ego=True)))
# filter off
CANDS.append(("filter-off ff0.9", oe(filt=False)))
CANDS.append(("filter-off ff0.0", oe(filt=False, ff=0.0)))
# --- NEW experimental controllers (OneEuroX): attack dead-time vs noise ---
# 1) confidence-gated feedforward: keep ff high but only when target truly moves
for ff in (0.6, 0.9):
    for vg in (4.0, 8.0):
        CANDS.append(("X gateFF ff%.1f vg%.0f" % (ff, vg),
                      oex(mincut=0.12, beta=0.04, kp=0.5, kd=0.32, ff=ff, gate_ff=True, vgate=vg)))
# 2) dead-time predictor (aim `predict` frames ahead, gated). ff off (predictor replaces it)
for pr in (1.5, 3.0):
    for vg in (4.0, 8.0):
        CANDS.append(("X predict h%.1f vg%.0f" % (pr, vg),
                      oex(mincut=0.12, beta=0.04, kp=0.5, kd=0.32, ff=0.0, predict=pr, vgate=vg)))
# 3) axis-asymmetric: filter X harder (X-heavy noise), plus gated ff
for xk in (0.5, 0.7):
    CANDS.append(("X axis xcut%.1f gateFF" % xk,
                  oex(mincut=0.12, beta=0.04, kp=0.5, kd=0.32, ff=0.6, gate_ff=True, xcut_k=xk)))
# 4) combined: axis-asym + gated predictor (the "maximize" candidate)
for pr in (1.5, 3.0):
    CANDS.append(("X combo h%.1f xcut0.6" % pr,
                  oex(mincut=0.12, beta=0.04, kp=0.5, kd=0.32, ff=0.0, predict=pr,
                      gate_ff=True, xcut_k=0.6, vgate=6.0)))
# 5) Kalman constant-velocity (R from real noise, Q swept) - optimal estimator test
for Rv in (8.0, 15.0, 25.0):
    for Qv in (0.3, 0.8, 2.0):
        for kdv in (0.2, 0.32):
            CANDS.append(("KF R%.0f Q%.1f kd%.2f" % (Rv, Qv, kdv),
                          kf(R=Rv, Q=Qv, kd=kdv, ff=0.6, gate_ff=True)))
# 6) JUMP-CLAMP test, on the current pure-reactive config (ff=0, predict=0).
#    Baseline = no clamp; then sweep the clamp threshold.
CANDS.append(("REACTIVE no-clamp (current)",
              oex(mincut=0.1, beta=0.02, kp=0.55, kd=0.2, ff=0.0, gate_ff=False)))
for jc in (10.0, 12.0, 15.0):
    CANDS.append(("REACTIVE + clamp%.0f" % jc,
                  oex(mincut=0.1, beta=0.02, kp=0.55, kd=0.2, ff=0.0, gate_ff=False,
                      jump_clamp=jc)))
# alpha-beta family
for al in (0.2, 0.3, 0.4):
    for ff in (0.4, 0.8):
        for ego in (True, False):
            CANDS.append(("AB a%.1f ff%.1f%s" % (al, ff, "+ego" if ego else ""), ab(alpha=al, ff=ff, ego=ego)))

# FEEL score (lower=better), calibrated so it reproduces the hardware verdict:
# crisp+sticky (One Euro+PD) beats floaty (alpha-beta heavy) and buzzy (filter-off).
#   rise  = response speed on a jump  -> penalises FLOATY (the axis RMS misses)
#   hold  = rest error (jitter)       -> penalises drift
#   tw    = rest twitch%              -> penalises BUZZ
#   rev   = reversal overshoot
#   track = de-weighted (rewards smoothing, which is the trap)
def score(m):
    return m["rise"]*2.4 + m["hold"]*0.9 + m["tw"]*0.05 + m["rev"]*0.06 + m["track"]*0.05

def sweep(lat_base, tag):
    print("== %s (latency base = %d frame%s, ~%d ms) ==" %
          (tag, lat_base, "" if lat_base == 1 else "s", round(lat_base * 6.94)))
    rows = []
    for name, mk in CANDS:
        m = evalcfg(mk, lat_base); rows.append((score(m), name, m))
    rows.sort(key=lambda x: x[0])
    hd = "%-30s | %5s | %5s | %5s | %6s | %5s | %4s" % ("combination", "score", "rise", "hold", "track", "rev", "tw%")
    print(hd); print("-" * len(hd))
    for sc, name, m in rows[:8]:
        print("%-30s | %5.1f | %5.1f | %5.2f | %6.2f | %5.1f | %3.0f" % (name, sc, m["rise"], m["hold"], m["track"], m["rev"], m["tw"]))
    print()
    return rows

ARCH = [
    ("One Euro+PD (your shipped feel)", oe(mincut=0.1, beta=0.02, kp=0.55, kd=0.2, ff=0.9)),
    ("alpha-beta a0.2 ego (floaty)",    ab(alpha=0.2, ff=0.4, ego=True)),
    ("filter-off ff0.9 (buzzy)",        oe(filt=False)),
]

if __name__ == "__main__":
    print("Searching %d combos. FEEL score (lower=better) is calibrated so it\n"
          "reproduces your hardware verdict: rise=crispness(anti-floaty),\n"
          "hold/tw=anti-buzz. It de-weights raw error (which rewards floaty).\n" % len(CANDS))
    print("== CALIBRATION check vs your known feel (latency 1 frame) ==")
    hd = "%-32s | %5s | %5s | %5s | %4s" % ("archetype", "score", "rise", "hold", "tw%")
    print(hd); print("-" * len(hd))
    for name, mk in ARCH:
        m = evalcfg(mk, 1)
        print("%-32s | %5.1f | %5.1f | %5.2f | %3.0f" % (name, score(m), m["rise"], m["hold"], m["tw"]))
    print("  (want: One Euro+PD LOWEST of these three)\n")
    sweep(1, "OPTIMISTIC latency (inference only, 5-7 ms)")
    rows3 = sweep(3, "REALISTIC 2-PC round trip (capture+net+infer+MAKCU+render)")

    # -------- Pareto frontier: FASTER (low rise) AND MORE ACCURATE (low error) --------
    # accuracy = mean of hold (rest RMS) and track (moving RMS); speed = rise.
    # A config is Pareto-optimal if nothing is both faster AND more accurate.
    print("== PARETO: faster (rise) AND more accurate (err) - realistic latency ==")
    pts = [(name, m["rise"], 0.5*(m["hold"] + m["track"])) for _, name, m in rows3]
    dominated = set()
    for i, (_, r1, a1) in enumerate(pts):
        for j, (_, r2, a2) in enumerate(pts):
            if j != i and r2 <= r1 and a2 <= a1 and (r2 < r1 or a2 < a1):
                dominated.add(i); break
    front = [pts[i] for i in range(len(pts)) if i not in dominated]
    front.sort(key=lambda x: x[1])
    hd = "%-30s | %6s | %8s" % ("Pareto-optimal (non-dominated)", "rise", "err")
    print(hd); print("-" * len(hd))
    for name, r, a in front:
        print("%-30s | %6.1f | %8.2f" % (name, r, a))
    # where does the shipped One Euro+PD sit?
    shipped = evalcfg(oe(mincut=0.1, beta=0.02, kp=0.55, kd=0.2, ff=0.9), 3)
    sr, sa = shipped["rise"], 0.5*(shipped["hold"] + shipped["track"])
    beats = [n for n, r, a in pts if r <= sr and a <= sa and (r < sr or a < sa)]
    print("\n  shipped One Euro+PD: rise=%.1f err=%.2f" % (sr, sa))
    print("  configs that DOMINATE shipped (faster AND more accurate): %d" % len(beats))
    for n in beats[:6]:
        print("    - %s" % n)
