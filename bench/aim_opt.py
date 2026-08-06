#!/usr/bin/env python3
"""Joint optimiser for the shipped controller: minimise ERROR subject to a HARD
no-ringing constraint.

Why joint: every previous pass swept one knob at a time, which cannot see
interactions (e.g. a higher kp may only be safe together with a fractional
dead-time comp and a tighter max_step).

Realism (all from this session's measurements, not assumptions):
  dead time   fractional AND jittery: 1.13 +/- 0.5 frames
              (= USB ~1ms + render wait 0..6.94ms + E2E 3.32ms @144Hz)
  noise       measured aim-ON aim-point sigma X 8.1 / Y 5.9 px
              (rig CSV, first-difference/sqrt2; aim-OFF floor is 3.5/5.0)
  quantisation integer mouse counts with residual carry, +-127 saturation
  gaps        1.5% dropout (hold, no coast - coast was removed)

Search variables: kp_x/y, soft_x/y, kd_x/y, max_step, dead-time comp weight
(FRACTIONAL - the shipped code only allows integers), One Euro min_cutoff/beta.
"""
import math
import random

from aim_sim_ring import (Cfg, one_euro_alpha, nonlinear_p, clamp_max_step,
                          emit_mouse_delta, K_INFLIGHT_MAX, SC)

V_NOISE_SCALE = True  # 속도 추정 노이즈 스케일 지원
# ---------------------------------------------------------------- measured rig
NOISE_X, NOISE_Y = 8.1, 5.9      # aim-ON aim-point sigma (measured)
DT_MEAN, DT_JIT = 1.13, 0.5      # dead time in frames: mean, +/- uniform jitter
P_DROP = 0.015                   # measured dropout under continuous aim
DRIFT, DRIFT_RHO = 1.2, 0.9      # correlated low-freq box breathing
P_OUTLIER, OUT_LO, OUT_HI = 0.035, 15.0, 60.0   # fat tail (kurtosis ~24)
# head<->body class switching, MEASURED from the rig CSVs (aim-point space):
# 0.6-2.4% of frames flip anchor, the offset PERSISTS 1-4 frames (unlike the
# single-frame outlier above, which reverts immediately) and is mostly vertical:
# |dY| mean 11-21px. This is ~27-36% of the Y variance, so tuning without it
# optimises against the wrong noise.
P_SWITCH, SW_LO, SW_HI, SW_XSD = 0.02, 11.0, 21.0, 3.0
# Box height varies with target distance; measured detector sigma scales steeply
# with it (h 17 -> ~0.9px, h 11 -> ~10.7px on the head box). BOX_H_REF is the size
# at which the nominal NOISE_X/Y hold; noise is scaled by (ref/h)^BOX_H_POW.
BOX_H_REF, BOX_H_POW, BOX_H_LO, BOX_H_HI = 68.0, 1.3, 40.0, 105.0
# The measured sigma (NOISE_X/Y) is a session AVERAGE that already contains this
# distance variation, so the per-frame scale must be NORMALISED to mean 1 -
# otherwise the variation is double-counted and the sim is simply noisier than
# reality. BOX_NSC_NORM is E[(ref/h)^pow] over the walk below (measured by
# simulation, see bench note); dividing by it keeps the mean noise on target.
BOX_NSC_NORM = 1.7365


class OptCtrl:
    """Shipped computeAimMovement(), with the ONE generalisation under test:
    the in-flight (dead-time) subtraction takes a FRACTIONAL frame count w.
      w = 1.0 reproduces the shipped integer deadtime_frames=1 exactly.
      w = 1.13 subtracts the whole last emit + 13% of the one before it.
    Everything else (One Euro, nonlinear P, D, max-step, integer emit with
    residual carry) is byte-for-byte the shipped math."""

    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self):
        self.has_track = 0
        self.filt_x = self.filt_y = 0.0
        self.dfilt_x = self.dfilt_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.derr_x = self.derr_y = 0.0
        self.res_x = self.res_y = 0.0
        self.inf_x = [0.0] * K_INFLIGHT_MAX
        self.inf_y = [0.0] * K_INFLIGHT_MAX
        self.head = 0
        # --- ego-corrected target velocity (for the lead/feedforward term) ---
        # raw = T - C + SC, so d_raw = dT - dC. Our own dC is known EXACTLY from
        # the in-flight ring, so dT (the target's TRUE screen velocity) = d_raw +
        # dC. This is the "lag-aligned ego-corrected velocity" that plain
        # screen-space velocity could never be - the reason the old feedforward
        # was ego-polluted and had to be removed. The ring did not exist then.
        self.prev_raw_x = self.prev_raw_y = 0.0
        self.vx = self.vy = 0.0

    def _push(self, dx, dy):
        self.inf_x[self.head] = dx
        self.inf_y[self.head] = dy
        self.head = (self.head + 1) % K_INFLIGHT_MAX

    def _inflight(self, w):
        sx = sy = 0.0
        w = max(0.0, min(float(w), float(K_INFLIGHT_MAX)))
        n = int(math.floor(w))
        for k in range(1, n + 1):
            i = (self.head - k + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
            sx += self.inf_x[i]; sy += self.inf_y[i]
        f = w - n
        if f > 0.0 and n + 1 <= K_INFLIGHT_MAX:
            i = (self.head - (n + 1) + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
            sx += f * self.inf_x[i]; sy += f * self.inf_y[i]
        return sx, sy

    def _ring_at(self, lag):
        """The emit `lag` frames back (1 = most recent). Fractional lag blends."""
        lag = max(1.0, min(float(lag), float(K_INFLIGHT_MAX)))
        n = int(math.floor(lag)); f = lag - n
        i = (self.head - n + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
        ax, ay = self.inf_x[i], self.inf_y[i]
        if f > 0.0 and n + 1 <= K_INFLIGHT_MAX:
            j = (self.head - (n+1) + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
            ax = (1-f)*ax + f*self.inf_x[j]
            ay = (1-f)*ay + f*self.inf_y[j]
        return ax, ay

    def step(self, rx, ry, cls_changed=False, box_h=BOX_H_REF):
        p = self.p
        fresh = (self.has_track == 0)
        tcx, tcy = rx, ry
        # SHIPPED: an anchor flip (head<->body) moves the measured centre by the
        # offset between two aim points.
        # For the VELOCITY estimate that is a pure artifact (the target did not
        # move) and the lead term would fling on it, so the frame is excluded -
        # that exclusion is where this feature's measured benefit comes from.
        # The DAMPING term deliberately still sees it: after a flip the aim point
        # really has moved to the other anchor, so the error step is a real
        # SETPOINT CHANGE and one D response helps cross it. Measured (3 blocks
        # x60, both gain profiles): suppressing it is 0.03-0.22% WORSE.
        # Keep this identical to pd_controller.cuh.
        if cls_changed and p.get("cls_reject", 1.0) != 0.0 and not fresh:
            self.prev_raw_x, self.prev_raw_y = rx, ry
        # --- ego-corrected target velocity ---
        # Two consecutive measurements are 1 frame apart in CAPTURE time, so the
        # crosshair motion between them is the emit that landed in that window -
        # which sits at ring lag ~(dead_time + 1). Adding it back to d_raw
        # recovers the target's true screen drift. ego_lag is tunable because the
        # earlier ego attempt failed by mis-aligning this ring lag.
        if (max(p.get("ff_x", p.get("ff", 0.0)), p.get("ff_y", p.get("ff", 0.0))) > 0.0
                or max(p.get("predict_x", p.get("predict", 0.0)),
                       p.get("predict_y", p.get("predict", 0.0))) > 0.0):
            if fresh:
                self.vx = self.vy = 0.0
            else:
                ax, ay = self._ring_at(p.get("ego_lag", 2.0))
                dvx = (rx - self.prev_raw_x) + ax
                dvy = (ry - self.prev_raw_y) + ay
                # 픽셀 정합으로 속도를 재면 박스 차분보다 노이즈가 작다(실측 X-32%/Y-23%).
                # vscale<1 로 그 개선을 모사: 참 속도 성분은 두고 추정 오차만 줄인다.
                vsx = p.get("v_noise_x", 1.0); vsy = p.get("v_noise_y", 1.0)
                if vsx != 1.0 or vsy != 1.0:
                    dvx = self.vx + (dvx - self.vx)*vsx
                    dvy = self.vy + (dvy - self.vy)*vsy
                dvx = max(-60.0, min(60.0, dvx)); dvy = max(-60.0, min(60.0, dvy))
                a = p.get("v_ema", 0.4)
                # REJECTED: a counter-strafe guard (zero the estimate when the fresh
                # sample opposes it) did NOT reduce the reversal peak (32.07 vs
                # 31.93) and cost 1% error by weakening the lead. The reversal
                # penalty comes from the GAINS, not the lead: it is present at
                # ff=0 too. Do not re-add.
                self.vx = (1-a)*self.vx + a*dvx
                self.vy = (1-a)*self.vy + a*dvy
            self.prev_raw_x, self.prev_raw_y = rx, ry
        if p["oneeuro"]:
            if self.has_track:
                ad = one_euro_alpha(p["dcut"])
                d = rx - self.filt_x
                self.dfilt_x = ad * d + (1 - ad) * self.dfilt_x
                a = one_euro_alpha(p["mincut"] + p["beta"] * abs(self.dfilt_x))
                self.filt_x = a * rx + (1 - a) * self.filt_x
                d = ry - self.filt_y
                self.dfilt_y = ad * d + (1 - ad) * self.dfilt_y
                a = one_euro_alpha(p["mincut"] + p["beta"] * abs(self.dfilt_y))
                self.filt_y = a * ry + (1 - a) * self.filt_y
            else:
                self.filt_x, self.filt_y = rx, ry
                self.dfilt_x = self.dfilt_y = 0.0
            tcx, tcy = self.filt_x, self.filt_y
        self.has_track = 1

        ex, ey = tcx - SC, tcy - SC
        if p["comp"] > 0.0 and p["w"] > 0.0:
            ix, iy = self._inflight(p["w"])
            ex -= p["comp"] * ix
            ey -= p["comp"] * iy
        # --- STRUCTURAL: symmetric dead-time compensation ---
        # The in-flight subtraction above removes OUR motion during the dead time.
        # But the TARGET also moved during that same dead time, and nothing
        # accounted for it - the compensation was half-done. Extrapolating the
        # target forward by predict*v completes it. Unlike a free ff gain on the
        # output, this has a PHYSICAL gain (the dead time) and passes through the
        # nonlinear P, so softness/saturation still apply consistently.
        prx = p.get("predict_x", p.get("predict", 0.0))
        pry = p.get("predict_y", p.get("predict", 0.0))
        pr = max(prx, pry)
        if pr > 0.0:
            g = 1.0
            vg = p.get("pred_vgate", 0.0)
            if vg > 0.0:
                sp = math.hypot(self.vx, self.vy)
                g *= sp / (sp + vg)
            e0 = p.get("pred_err_gate", 0.0)
            if e0 > 0.0:
                e2 = ex*ex + ey*ey
                g *= (e0*e0) / (e0*e0 + e2)
            ex += prx * g * self.vx
            ey += pry * g * self.vy

        dex = max(-150.0, min(150.0, ex - self.prev_err_x))
        dey = max(-150.0, min(150.0, ey - self.prev_err_y))
        if fresh:
            self.derr_x = self.derr_y = 0.0
        else:
            self.derr_x = 0.6 * dex + 0.4 * self.derr_x
            self.derr_y = 0.6 * dey + 0.4 * self.derr_y
        self.prev_err_x, self.prev_err_y = ex, ey

        # REJECTED: a plain LINEAR P (drop the softness term) blew up - step
        # overshoot 9.4px and 1.05 oscillations vs 2.5/0.03. The softness term is a
        # core stabiliser, not cosmetic. Do not re-add.
        mx = nonlinear_p(ex, p["kp_x"], p["soft_x"]) + p["kd_x"] * self.derr_x
        my = nonlinear_p(ey, p["kp_y"], p["soft_y"]) + p["kd_y"] * self.derr_y
        ffx = p.get("ff_x", p.get("ff", 0.0))
        ffy = p.get("ff_y", p.get("ff", 0.0))
        ff = max(ffx, ffy)
        if ff > 0.0:
            # Feedforward on the EGO-CORRECTED velocity: cancels the P
            # controller's steady-state ramp lag against a moving target.
            g = 1.0
            # (a) speed gate: g -> 0 at rest so detector jitter is not amplified.
            vg = p.get("vgate", 0.0)
            if vg > 0.0:
                sp = math.hypot(self.vx, self.vy)
                g *= sp / (sp + vg)
            # (b) ERROR gate: the lead must be OFF while acquiring. A position
            # jump (target switch / fresh lock) produces a huge one-frame
            # velocity estimate; feeding that forward is what overshoots. A real
            # moving target instead shows SMALL error with sustained velocity.
            # So fade the lead out as |error| grows: g_e = e0^2/(e0^2 + |e|^2).
            e0 = p.get("ff_err_gate", 0.0)
            if e0 > 0.0:
                e2 = ex*ex + ey*ey
                g *= (e0*e0) / (e0*e0 + e2)
            mx += ffx * g * self.vx
            my += ffy * g * self.vy
        # 세로 보정 세기. 클램프·emit 이전에 곱해야 max_step 이 실제 이동을 묶고
        # in-flight 링에도 실제로 나간 값이 들어간다. 커널과 동일한 순서.
        my *= p.get("y_scale", 1.0)
        mx, my = clamp_max_step(mx, my, p["max_step"])
        dx, self.res_x = emit_mouse_delta(mx, self.res_x)
        dy, self.res_y = emit_mouse_delta(my, self.res_y)
        self._push(float(dx), float(dy))
        return dx, dy


def base_params(**kw):
    p = dict(kp_x=0.75, kp_y=0.82, soft_x=9.0, soft_y=8.0, kd_x=0.05, kd_y=0.06,
             max_step=25.0, comp=1.0, w=1.0, oneeuro=True, mincut=0.1, beta=0.02,
             dcut=0.5,
             # ego-corrected feedforward (0 = shipped behaviour, no lead term)
             ff=0.0, ego_lag=2.0, v_ema=0.4, vgate=0.0, ff_err_gate=0.0,
             predict=0.0, pred_vgate=6.0, pred_err_gate=18.0, cls_reject=1.0,
             y_scale=1.0)
    p.update(kw); return p


def run(p, scenario, frames=None, seed=0, noise=True,
        dt_mean=DT_MEAN, dt_jit=DT_JIT, nx=NOISE_X, ny=NOISE_Y, vx=1.1,
        step_dist=60.0, ctrl_factory=None, switches=True, box_vary=True):
    """scenario 'track' uses vx px/frame (1.1 slow strafe .. 8 sprint). 'reach'
    measures frames to get within 4px of a step_dist-away target (acquisition
    speed). Targets BOUNCE at the frame edge - never teleport - so the RMS is not
    contaminated by a wrap discontinuity."""
    rng = random.Random(seed)
    if frames is None:
        frames = {"step": 150, "hold": 420, "track": 480, "reversal": 300,
                  "reach": 220}[scenario]
    c = (ctrl_factory(p) if ctrl_factory else OptCtrl(p)); c.reset()
    C = [SC, SC]
    if scenario == "step":
        T = [SC + step_dist*0.94, SC + step_dist*0.34]; v = [0.0, 0.0]
    elif scenario == "reach":
        T = [SC + step_dist*0.94, SC + step_dist*0.34]; v = [0.0, 0.0]
    elif scenario == "track":
        T = [SC - 60.0, SC]; v = [vx, vx*0.2]
    elif scenario == "reversal":
        T = [SC, SC]; v = [1.3, 0.0]
    else:
        T = [SC, SC]; v = [0.0, 0.0]
    reach = None
    d0 = math.hypot(T[0] - C[0], T[1] - C[1])
    au = ((T[0]-C[0])/d0, (T[1]-C[1])/d0) if d0 > 1e-9 else (1.0, 0.0)

    hist = []; errs = []; par = []; emits = []
    drift = [0.0, 0.0]
    box_h = BOX_H_REF          # AR(1) walk = the target closing / backing off
    sw_left = 0; sw_off = (0.0, 0.0)
    turn = None; peak = 0.0
    for f in range(frames):
        if scenario == "reversal" and f > 0 and f % 75 == 0:
            v[0] = -v[0]; turn = f
        T[0] += v[0]; T[1] += v[1]
        # BOUNCE inside the model frame (never wrap/teleport)
        if T[0] < 40.0:  T[0] = 40.0;  v[0] = abs(v[0])
        if T[0] > 280.0: T[0] = 280.0; v[0] = -abs(v[0])
        if T[1] < 40.0:  T[1] = 40.0;  v[1] = abs(v[1])
        if T[1] > 280.0: T[1] = 280.0; v[1] = -abs(v[1])
        hist.append((T[0], T[1], C[0], C[1]))

        lat = max(0.0, rng.uniform(dt_mean - dt_jit, dt_mean + dt_jit))
        idx = max(0.0, f - lat)
        i0 = int(idx); i1 = min(i0 + 1, len(hist) - 1); fr = idx - i0
        sT0 = hist[i0][0]*(1-fr) + hist[i1][0]*fr
        sT1 = hist[i0][1]*(1-fr) + hist[i1][1]*fr
        sC0 = hist[i0][2]*(1-fr) + hist[i1][2]*fr
        sC1 = hist[i0][3]*(1-fr) + hist[i1][3]*fr
        mx_, my_ = sT0 - sC0 + SC, sT1 - sC1 + SC

        gap = False
        cls_changed = False
        prev_sw = sw_left
        if noise and box_vary:
            box_h = min(BOX_H_HI, max(BOX_H_LO, 0.995*box_h + rng.gauss(0.0, 1.2)))
        nsc = ((BOX_H_REF / box_h) ** BOX_H_POW) / BOX_NSC_NORM if (noise and box_vary) else 1.0
        if noise:
            if rng.random() < P_DROP:
                gap = True
            else:
                s = math.sqrt(max(1e-9, 1 - DRIFT_RHO**2))
                drift[0] = DRIFT_RHO*drift[0] + rng.gauss(0, DRIFT*s)
                drift[1] = DRIFT_RHO*drift[1] + rng.gauss(0, DRIFT*s)
                mx_ += drift[0] + rng.gauss(0, nx*nsc)
                my_ += drift[1] + rng.gauss(0, ny*nsc)
                if rng.random() < P_OUTLIER:
                    a = rng.random()*6.283
                    m = OUT_LO + rng.random()*(OUT_HI-OUT_LO)
                    mx_ += m*math.cos(a); my_ += m*math.sin(a)
                # sustained head<->body anchor flip (measured; see P_SWITCH)
                if sw_left > 0:
                    mx_ += sw_off[0]; my_ += sw_off[1]; sw_left -= 1
                    if sw_left == 0: cls_changed = True   # snapping back is a change too
                elif switches and rng.random() < P_SWITCH:
                    sw_left = rng.randint(1, 4)
                    mag = rng.uniform(SW_LO, SW_HI)*(1 if rng.random() < 0.5 else -1)
                    sw_off = (rng.gauss(0.0, SW_XSD), mag)
                    mx_ += sw_off[0]; my_ += sw_off[1]
                    cls_changed = True

        if gap:
            dx = dy = 0
            c.res_x = c.res_y = 0.0        # shipped: hold on gap (no coast)
        else:
            try:
                dx, dy = c.step(mx_, my_, cls_changed, box_h)
            except TypeError:
                dx, dy = c.step(mx_, my_, cls_changed)
        C[0] += dx; C[1] += dy

        e = math.hypot(T[0]-C[0], T[1]-C[1])
        errs.append(e); emits.append((dx, dy))
        par.append((C[0]-(T[0]-d0*au[0]))*au[0] + (C[1]-(T[1]-d0*au[1]))*au[1])
        if reach is None and e < 4.0:
            reach = f
        if scenario == "reversal" and turn is not None and 0 <= f-turn < 50:
            peak = max(peak, e)

    out = {}
    if scenario == "reach":
        out["reach"] = reach if reach is not None else frames
        return out
    if scenario == "step":
        out["overshoot"] = max(0.0, max(par) - d0)
        rise = next((i for i, q in enumerate(par) if q >= d0 - 2.0), frames)
        out["rise"] = rise
        st = 0
        for i, e in enumerate(errs):
            if e > 2.0: st = i + 1
        out["settle"] = st
        # Count only excursions ABOVE the integer-quantisation floor: the mouse
        # emits whole counts, so a settled crosshair always dithers +-1px. A 0.5px
        # band would score that dither as "ringing"; real ringing is multi-px.
        OSC_BAND = 1.5
        osc = 0; sgn = 0
        for q in par[rise:]:
            s2 = 1 if (q-d0) > OSC_BAND else (-1 if (q-d0) < -OSC_BAND else 0)
            if s2 and sgn and s2 != sgn: osc += 1
            if s2: sgn = s2
        out["osc"] = osc
    elif scenario == "reversal":
        out["peak"] = peak
        w = errs[100:]; out["rms"] = math.sqrt(sum(x*x for x in w)/len(w))
    else:
        w = errs[100:]
        out["rms"] = math.sqrt(sum(x*x for x in w)/len(w))
        out["twitch"] = 100.0*sum(1 for d in emits[100:] if d != (0, 0))/len(emits[100:])
    return out


def evaluate(p, seeds=6, **kw):
    """RINGING on a CLEAN step (intrinsic transient - under sigma=8px noise a 2px
    settle band is unreachable, so a noisy step measures noise, not ringing).
    ERROR under the full measured noise, at SLOW *and* FAST target speeds - the
    original hardware complaint was chasing fast targets, so a config tuned only
    on a slow strafe is worthless. ACQUISITION speed via 'reach' on a clean
    120px step, so a tame max_step cannot silently destroy re-acquisition."""
    acc = {}
    jobs = [("step", False, {}), ("hold", True, {}), ("reversal", True, {}),
            ("track", True, dict(vx=1.1)), ("track_fast", True, dict(vx=4.0)),
            ("track_sprint", True, dict(vx=8.0)),
            ("reach", False, dict(step_dist=120.0))]
    for name, noisy, extra in jobs:
        sc = "track" if name.startswith("track") else name
        for s in range(seeds):
            m = run(p, sc, seed=s, noise=noisy, **dict(extra, **kw))
            for k, v in m.items():
                acc.setdefault(name+"_"+k, []).append(v)
    return {k: sum(v)/len(v) for k, v in acc.items()}


def error_score(m):
    """Total error: rest jitter + tracking at three speeds + reversal. Fast-target
    tracking carries the most weight because that was the real complaint."""
    return (0.18*m["hold_rms"] + 0.18*m["track_rms"] + 0.24*m["track_fast_rms"]
            + 0.24*m["track_sprint_rms"] + 0.16*m["reversal_rms"])


# Objective: minimise ERROR (hold + track + reversal rms) under HARD no-ring
# constraints. Ringing is a constraint, not a weighted term, because the goal is
# "no ringing" - a config that trades 0.1px of error for visible ring must lose.
RING_OVERSHOOT_MAX = 1.5     # px past the target on a 60px step
RING_OSC_MAX = 0.35          # mean post-settle direction reversals
def objective(m, settle_budget):
    if m["step_overshoot"] > RING_OVERSHOOT_MAX: return None, "overshoot"
    if m["step_osc"] > RING_OSC_MAX: return None, "osc"
    if m["step_settle"] > settle_budget: return None, "settle"
    err = 0.40*m["hold_rms"] + 0.40*m["track_rms"] + 0.20*m["reversal_rms"]
    return err, None
