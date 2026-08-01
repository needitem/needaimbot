#!/usr/bin/env python3
"""!! HISTORICAL - DOES NOT MODEL THE CURRENT CONTROLLER !!

This file describes the pipeline as it was BEFORE 2026-07-25: One Euro on the
frame-coord centre + a feedforward on the RAW (ego-polluted) velocity + coast
gap-glide. All three are gone from needaimbot/cuda/pd_controller.cuh:
  - coast was removed outright (no benefit under real noise, worse on reversals),
  - the raw-velocity feedforward was replaced by an EGO-CORRECTED lead built from
    the in-flight ring, plus a target-side dead-time extrapolation,
  - dead-time compensation (inflight_comp) did not exist here at all.
Tuning against this file will give WRONG answers. The current, measurement-
calibrated harness is bench/aim_opt.py (controller) and bench/aim_sim_select.py
(target selection). Kept only as a record of the earlier analysis.
"""

"""Comprehensive realistic 2-PC aim-pipeline sim (real-ish units).

Everything the tracking-scope viz models PLUS the pieces it was missing:
  - motion blur           detection noise up / confidence down with relative speed
  - confidence gating     low-confidence boxes dropped (head, being small, first)
  - occlusion             target hidden for stretches -> detection gaps
  - target switching      a second enemy briefly becomes the selected target
  - recoil                firing bursts kick the crosshair vertically (net of comp)
plus head/body dual detection + selection, correlated + white + outlier noise
(Y-heavy), multi-frame JITTERY latency, dt jitter, integer +-1 mouse quantization
with residual carry, and emit->screen scale error.

CALIBRATED TO TYPICAL VALUES (144 Hz, YOLOv12-nano mAP~0.5, ~20 ms 2-PC round
trip), NOT a specific rig. Read the RANKINGS and the ideal->realistic FLIP, not
the absolute pixels. Controllers are ported line-for-line from bench/aim_sim.py.
"""
import math
import random

MODEL = 320.0
SC = 160.0
FRAME_MS = 1000.0 / 144.0     # 144 Hz render cadence
HB = 35.0                     # head aim point sits this far above the body point

# ---------------------------------------------------------------- primitives
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def nlp(e, kp, s):
    ae = abs(e)
    return e * (max(kp, 0.0) * (ae / (ae + max(s, 1.0))))

def one_euro_alpha(cut):
    tau = 1.0 / (2.0 * math.pi * max(cut, 1e-4))
    return 1.0 / (1.0 + tau)

def trunc0(v):
    return math.floor(v) if v >= 0 else math.ceil(v)

class Res:
    __slots__ = ("v")
    def __init__(self): self.v = 0.0

def emit_int(move, res):
    c = res.v
    if move * c < 0.0:
        c = 0.0
    val = move + c
    e = trunc0(val)
    res.v = val - e
    return e

# ----------------------------------------------------------- controllers
class OneEuro:
    """Shipped One Euro + nonlinear P + D + velocity feedforward. filt=False
    removes only the low-pass (= 'filter off': PD+ff on raw). ego adds the
    crosshair's own applied delta back into the velocity."""
    def __init__(self, ego=False, filt=True, ff=0.9):
        self.ego, self.filt, self.ff = ego, filt, ff
        self.reset()
    def reset(self):
        self.has = False
        self.fx = self.fy = self.dfx = self.dfy = 0.0
        self.px = self.py = self.vx = self.vy = 0.0
        self.pex = self.pey = self.dex = self.dey = 0.0
        self.ax = self.ay = 0.0
    def set_applied(self, dx, dy):
        self.ax += dx; self.ay += dy
    def step(self, rx, ry, dt=1.0):
        fresh = not self.has
        tcx, tcy = rx, ry
        if self.filt:
            if not fresh:
                ad = one_euro_alpha(0.5)
                de = rx - self.fx; self.dfx = ad * de + (1 - ad) * self.dfx
                a = one_euro_alpha(0.1 + 0.02 * abs(self.dfx)); self.fx = a * rx + (1 - a) * self.fx
                de = ry - self.fy; self.dfy = ad * de + (1 - ad) * self.dfy
                a = one_euro_alpha(0.1 + 0.02 * abs(self.dfy)); self.fy = a * ry + (1 - a) * self.fy
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
        mx = nlp(ex, 0.55, 11) + 0.18 * self.dex + self.ff * self.vx
        my = nlp(ey, 0.6, 10) + 0.22 * self.dey + self.ff * self.vy
        st = math.hypot(mx, my)
        if st > 30: k = 30 / st; mx *= k; my *= k
        return mx, my

class AlphaBeta:
    """Estimate-then-control: gated alpha-beta estimator + ego + velocity
    feedforward + CV prediction (est += v*dt)."""
    def __init__(self, alpha=0.30, ego=True, tf=0.8, gate=50.0):
        self.alpha, self.ego, self.tf, self.gate = alpha, ego, tf, gate
        self.reset()
    def reset(self):
        self.has = False
        self.ex = self.ey = self.vx = self.vy = 0.0; self.gc = 0
        self.pex = self.pey = self.dex = self.dey = 0.0; self.ax = self.ay = 0.0
    def set_applied(self, dx, dy):
        self.ax += dx; self.ay += dy
    def step(self, rx, ry, dt=1.0):
        fresh = not self.has
        a = clamp(self.alpha, 0.01, 1); b = a * a / (2 - a)
        if fresh:
            self.ex, self.ey, self.vx, self.vy, self.gc = rx, ry, 0.0, 0.0, 0
        else:
            if self.ego: self.ex -= self.ax; self.ey -= self.ay
            px = self.ex + self.vx * dt; py = self.ey + self.vy * dt
            rrx, rry = rx - px, ry - py; rn = math.hypot(rrx, rry)
            if self.gate > 0 and rn > self.gate:
                self.gc += 1
                if self.gc >= 3:
                    px, py, self.vx, self.vy, self.gc = rx, ry, 0.0, 0.0, 0
            else:
                px += a * rrx; py += a * rry
                self.vx += (b / dt) * rrx; self.vy += (b / dt) * rry; self.gc = 0
            self.ex, self.ey = px, py
        self.ax = self.ay = 0.0; self.has = True
        ex, ey = self.ex, self.ey
        dex = clamp(ex - self.pex, -150, 150); dey = clamp(ey - self.pey, -150, 150)
        if fresh: self.dex = self.dey = 0.0
        else: self.dex = 0.6 * dex + 0.4 * self.dex; self.dey = 0.6 * dey + 0.4 * self.dey
        self.pex, self.pey = ex, ey
        mx = nlp(ex, 0.55, 8) + 0.18 * self.dex + self.tf * self.vx
        my = nlp(ey, 0.6, 7) + 0.22 * self.dey + self.tf * self.vy
        st = math.hypot(mx, my)
        if st > 30: k = 30 / st; mx *= k; my *= k
        return mx, my

# ------------------------------------------------------------- the pipeline
def run(make_ctrl, realistic=True, fast=False, frames=5000, seed=0):
    rng = random.Random(seed)
    ctrl = make_ctrl()
    dt_s = FRAME_MS / 1000.0

    hx, hy = SC - 70.0, SC
    C = [SC, SC]
    tvx, tvy = 150.0, 0.0                 # px/s
    holding, move_timer = False, 0.4
    drift = [0.0, 0.0]; lag_jit = 0.0; dt_jit = 0.0
    occluded, occ_timer = False, 2.0
    switching, switch_off = 0, [0.0, 0.0]
    firing, fire_timer = False, 1.0
    resx, resy = Res(), Res()
    hist = []; last_raw = None
    errs = []; mvs = []; ontgt = []

    for f in range(frames):
        # ---- target motion: strafe / hold + sharp reversals ----
        move_timer -= dt_s
        if move_timer <= 0:
            holding = not holding
            move_timer = 1.7 if holding else 1.9
            if not holding:
                if fast:
                    tvx = (1 if rng.random() < 0.5 else -1) * rng.uniform(180, 420)
                    tvy = rng.uniform(-80, 80)
                else:
                    tvx = (1 if rng.random() < 0.5 else -1) * rng.uniform(45, 150)
                    tvy = rng.uniform(-35, 35)
        if realistic and (not holding) and rng.random() < (0.05 if fast else 0.02):
            tvx = -tvx     # sharp counter-strafe (CV prediction overshoots these)
        if not holding:
            hx += tvx * dt_s; hy += tvy * dt_s
        if hx < 40: hx = 40; tvx = abs(tvx)
        if hx > MODEL - 40: hx = MODEL - 40; tvx = -abs(tvx)
        if hy < 50: hy = 50; tvy = abs(tvy)
        if hy > MODEL - 60: hy = MODEL - 60; tvy = -abs(tvy)

        # ---- recoil: firing bursts kick the crosshair up (net of comp) ----
        if realistic:
            fire_timer -= dt_s
            if fire_timer <= 0:
                firing = not firing
                fire_timer = 0.8 if firing else 1.4
            if firing:
                C[1] -= 0.2        # residual upward recoil (net of recoil-comp) the aim fights

        bx, by = hx, hy + HB
        hist.append((hx, hy, bx, by, C[0], C[1]))
        if len(hist) > 240: hist.pop(0)

        # ---- capture latency: multi-frame, jittery ----
        if realistic:
            lag_jit = 0.75 * lag_jit + rng.gauss(0, 1.1)
            lag = int(round(clamp(3 + lag_jit, 1, 10)))
        else:
            lag = 1
        snap = hist[max(0, len(hist) - 1 - lag)]

        # ---- motion blur: relative speed degrades detection ----
        blur = math.hypot(tvx * dt_s, tvy * dt_s) if realistic else 0.0
        noise_scale = 1.0 + 0.12 * blur

        # ---- correlated drift (both boxes breathe together) ----
        if realistic:
            drift[0] = 0.9 * drift[0] + rng.gauss(0, 0.5)
            drift[1] = 0.9 * drift[1] + rng.gauss(0, 0.6)

        # ---- occlusion (Markov) ----
        if realistic:
            occ_timer -= dt_s
            if occ_timer <= 0:
                occluded = rng.random() < 0.07
                occ_timer = rng.uniform(0.08, 0.28) if occluded else rng.uniform(2.5, 5.5)

        # ---- target switch (a second enemy) ----
        if realistic:
            if switching > 0:
                switching -= 1
            elif rng.random() < 0.002:
                switching = int(rng.uniform(3, 10))
                switch_off = [rng.uniform(-70, 70), rng.uniform(-50, 50)]

        # ---- confidence-gated head/body detection (blur lowers conf) ----
        head_conf = (0.9 / (1 + 0.15 * blur)) if realistic else 0.95
        body_conf = (0.97 / (1 + 0.06 * blur)) if realistic else 0.99
        head_seen = (rng.random() < head_conf) and not occluded
        body_seen = (rng.random() < body_conf) and not occluded

        det, on_head, dropped = None, False, False
        if head_seen:
            det, on_head = (snap[0], snap[1]), True
        elif body_seen:
            det, on_head = (snap[2], snap[3]), False
        else:
            dropped = True
        if det is not None and switching > 0:
            det = (det[0] + switch_off[0], det[1] + switch_off[1])

        # ---- measurement (frame coords) + noise / outliers ----
        if dropped:
            raw = last_raw if last_raw is not None else (0.0, 0.0)
        else:
            hw = (1.3 if on_head else 0.9) * noise_scale
            vw = (1.9 if on_head else 1.2) * noise_scale
            nx = rng.gauss(0, hw); ny = rng.gauss(0, vw)
            if realistic and rng.random() < 0.01:          # non-gaussian outlier tail
                nx += rng.gauss(0, 18); ny += rng.gauss(0, 18)
            dx = drift[0] if realistic else 0.0
            dy = drift[1] if realistic else 0.0
            detx, dety = det[0] + dx + nx, det[1] + dy + ny
            raw = (detx - snap[4], dety - snap[5])
            last_raw = raw

        # ---- dt jitter (variable frame interval) ----
        if realistic:
            dt_jit = 0.6 * dt_jit + rng.gauss(0, 0.4)
            dtc = clamp(1 + dt_jit, 0.35, 2.4)
        else:
            dtc = 1.0

        if dropped:
            # detection gap: HOLD (no chasing a stale measurement, which would
            # run the crosshair away). Real controllers coast/hold across gaps.
            ax = ay = 0
            ctrl.set_applied(0.0, 0.0)
        else:
            mx, my = ctrl.step(raw[0], raw[1], dtc)
            ax = emit_int(mx, resx); ay = emit_int(my, resy)   # integer mouse counts
            se = 0.08 if realistic else 0.0
            C[0] += ax * (1 + se); C[1] += ay * (1 + se)
            ctrl.set_applied(ax, ay)

        if f > 400:                                        # skip warm-up
            e = math.hypot(hx - C[0], hy - C[1])           # error vs the HEAD
            errs.append(e); mvs.append(1 if (abs(ax) + abs(ay)) > 0 else 0)
            ontgt.append(1 if e < 4.0 else 0)

    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    med = sorted(errs)[len(errs) // 2]                 # typical error (robust to gaps)
    return dict(rms=rms, med=med,
                mv=100.0 * sum(mvs) / len(mvs),
                on=100.0 * sum(ontgt) / len(ontgt))

def avg(make_ctrl, realistic, fast=False, n=6):
    acc = {}
    for s in range(n):
        m = run(make_ctrl, realistic=realistic, fast=fast, seed=s)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}

CTRLS = [
    ("filter off (PD+ff on raw)", lambda: OneEuro(ego=False, filt=False)),
    ("One Euro + PD  (shipped)",  lambda: OneEuro(ego=False, filt=True)),
    ("One Euro + ego",            lambda: OneEuro(ego=True,  filt=True)),
    ("Alpha-beta +ego +ff",       lambda: AlphaBeta(0.30, True, 0.8)),
]

def table(tag, realistic, fast=False):
    print("== %s ==" % tag)
    print("%-28s | %9s | %8s | %8s | %10s" %
          ("controller", "median px", "rms px", "move %", "on-head %"))
    print("-" * 76)
    for name, mk in CTRLS:
        m = avg(mk, realistic, fast=fast)
        print("%-28s | %9.1f | %8.1f | %7.0f%% | %9.0f%%" %
              (name, m["med"], m["rms"], m["mv"], m["on"]))
    print()

if __name__ == "__main__":
    print("Real-ish units: 320px model space, 144Hz, ms. Read RANKINGS, not")
    print("absolute px (calibrated to typical values, not a specific rig).")
    print("Mechanisms: head/body dual-detect + selection, blur, confidence")
    print("gating, occlusion, target switch, recoil, multi-frame jittery")
    print("latency, dt jitter, +-1 quantization, scale error, outlier noise.\n")
    table("IDEAL (base noise only)", realistic=False)
    table("REALISTIC - typical (holds + slow strafes)", realistic=True, fast=False)
    table("REALISTIC - duel (fast strafes + sharp reversals)", realistic=True, fast=True)
