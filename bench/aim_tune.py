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
        nsc *= (1.0 + 0.045 * (view_blur + world_px)); prev_az = az
        drx = 0.9 * drx + g(0.32 * nsc); dry = 0.9 * dry + g(0.32 * nsc)
        # head/body dual detection + dropout
        pHead = clamp(0.9 - 0.02 * (s[4] - 12), 0.55, 0.95)
        head_seen = rng.random() < pHead
        hx, hy = project(s[0], s[1], s[2], s[3])
        if head_seen:
            detx, dety = hx, hy
        else:
            bx, by = project(s[0], s[1] - (0.62 * PERSON) / s[4], s[2], s[3])
            detx, dety = bx, by
        rx = detx + drx + g(0.85 * nsc); ry = dety + dry + g(1.25 * nsc)
        dtc = clamp(1 + 0.6 * g(0.5), 0.4, 2.2)
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
    sweep(3, "REALISTIC 2-PC round trip (capture+net+infer+MAKCU+render)")
