#!/usr/bin/env python3
"""Genuinely different ARCHITECTURES, not new knobs on the shipped one.

The shipped controller mixes coordinate frames: the detection is in FRAME coords
(it moves when we move), it is low-passed there, then our own motion is
subtracted (inflight_comp), then the target's motion is added back
(predict_frames), then a lead is added to the output (ff_gain). Four separate
patches to one underlying problem.

UNIFIED-FRAME idea: the in-flight ring tells us exactly how far the view has
moved, so the measurement can be lifted into an EGO-FREE absolute frame:

    M          = cumulative emitted motion (model px)
    M_capture  = M_now - inflightSum(D)          # our motion when this frame was captured
    abs_k      = raw_k + M_capture               # target position, ego removed (constant if target still)

Estimate (position, velocity) in THAT frame, extrapolate to now, and the error is

    error = abs_now - M_now - SC

which is algebraically exact (verified: static target -> error == T - C_now).
Dead-time compensation, ego correction and target prediction all fall out of the
one formulation instead of being three tuned patches.

Estimators tested in the unified frame:
  LS   least-squares line fit over the last N samples. Unlike a low-pass this is
       UNBIASED for a constant-velocity target - a low-pass lags a ramp by
       construction, which is exactly the dominant error term. Gives position and
       velocity in one shot.
  KF   constant-velocity Kalman. R from the MEASURED noise variance, Q = target
       accel. The optimal linear estimator for this noise. (Earlier alpha-beta /
       Kalman attempts failed because they ran on the ego-POLLUTED signal; in the
       unified frame that objection is gone.)

Controllers on top:
  P    the shipped nonlinear P (+D)
  DB   deadbeat: emit the whole computed error * trust (inverse model, no P law)
"""
import math

from aim_sim_ring import (one_euro_alpha, nonlinear_p, clamp_max_step,
                          emit_mouse_delta, K_INFLIGHT_MAX, SC)


class UnifiedCtrl:
    """Unified-frame estimator + controller.

    p keys:
      est      'ls' | 'kf'
      win      LS window length (samples)
      kf_R     measurement variance (px^2); kf_Q process/accel variance
      dead     dead time D in frames (for M_capture and the extrapolation)
      law      'p' (nonlinear P + D) | 'db' (deadbeat)
      kp_*/soft_*/kd_*   as shipped (law='p')
      trust    deadbeat fraction of the full correction (law='db')
      max_step, vlim
    """

    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self):
        self.has = False
        self.M_x = self.M_y = 0.0          # cumulative emitted motion (model px)
        self.inf_x = [0.0]*K_INFLIGHT_MAX
        self.inf_y = [0.0]*K_INFLIGHT_MAX
        self.head = 0
        self.buf = []                       # (abs_x, abs_y) newest last
        self.res_x = self.res_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.derr_x = self.derr_y = 0.0
        # Kalman state [pos, vel] per axis + covariance
        self.sx = [0.0, 0.0]; self.sy = [0.0, 0.0]
        self.Px = [[1e3, 0.0], [0.0, 1e3]]; self.Py = [[1e3, 0.0], [0.0, 1e3]]

    # ---- in-flight bookkeeping (same ring as the shipped kernel) ----
    def _push(self, dx, dy):
        self.inf_x[self.head] = dx; self.inf_y[self.head] = dy
        self.head = (self.head + 1) % K_INFLIGHT_MAX

    def _inflight(self, w):
        sx = sy = 0.0
        w = max(0.0, min(float(w), float(K_INFLIGHT_MAX)))
        n = int(math.floor(w))
        for k in range(1, n+1):
            i = (self.head - k + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
            sx += self.inf_x[i]; sy += self.inf_y[i]
        f = w - n
        if f > 0.0 and n+1 <= K_INFLIGHT_MAX:
            i = (self.head - (n+1) + K_INFLIGHT_MAX) % K_INFLIGHT_MAX
            sx += f*self.inf_x[i]; sy += f*self.inf_y[i]
        return sx, sy

    # ---- estimators ----
    @staticmethod
    def _ls(vals):
        """Least-squares line fit; returns (value at last sample, slope/sample).
        Unbiased for a ramp - a low-pass is not."""
        n = len(vals)
        if n == 1:
            return vals[0], 0.0
        # t = 0..n-1, centred for numerical stability
        tbar = (n - 1) / 2.0
        sxx = sum((t - tbar)**2 for t in range(n))
        ybar = sum(vals) / n
        sxy = sum((t - tbar)*(vals[t] - ybar) for t in range(n))
        b = (sxy / sxx) if sxx > 0 else 0.0
        a = ybar - b*tbar
        return a + b*(n - 1), b

    def _kf(self, P, s, meas, R, Q):
        s[0] += s[1]                                   # dt = 1 sample
        P00 = P[0][0] + P[1][0] + P[0][1] + P[1][1] + Q/3.0
        P01 = P[0][1] + P[1][1] + Q/2.0
        P10 = P[1][0] + P[1][1] + Q/2.0
        P11 = P[1][1] + Q
        S = P00 + R
        K0 = P00/S; K1 = P10/S
        r = meas - s[0]
        s[0] += K0*r; s[1] += K1*r
        P[0][0] = (1-K0)*P00; P[0][1] = (1-K0)*P01
        P[1][0] = P10 - K1*P00; P[1][1] = P11 - K1*P01
        return s[0], s[1]

    def step(self, rx, ry):
        p = self.p
        D = p.get("dead", 1.13)
        # lift the measurement into the ego-free absolute frame
        ix, iy = self._inflight(D)
        cap_x = self.M_x - ix
        cap_y = self.M_y - iy
        ax = rx + cap_x
        ay = ry + cap_y

        if not self.has:
            self.has = True
            self.buf = [(ax, ay)]
            self.sx = [ax, 0.0]; self.sy = [ay, 0.0]
            self.Px = [[1e3, 0.0], [0.0, 1e3]]; self.Py = [[1e3, 0.0], [0.0, 1e3]]
            est_x, vx = ax, 0.0
            est_y, vy = ay, 0.0
        elif p.get("est", "ls") == "kf":
            R = p.get("kf_R", 66.0)      # 8.1^2 measured
            Q = p.get("kf_Q", 0.5)
            est_x, vx = self._kf(self.Px, self.sx, ax, R, Q)
            est_y, vy = self._kf(self.Py, self.sy, ay, p.get("kf_R_y", 35.0), Q)
        else:
            win = int(p.get("win", 6))
            self.buf.append((ax, ay))
            if len(self.buf) > win: self.buf.pop(0)
            est_x, vx = self._ls([b[0] for b in self.buf])
            est_y, vy = self._ls([b[1] for b in self.buf])

        vlim = p.get("vlim", 60.0)
        vx = max(-vlim, min(vlim, vx)); vy = max(-vlim, min(vlim, vy))

        # JUMP DETECTION: an LS window averages over N samples, which is what makes
        # it quiet - but it also means a real target JUMP (fresh lock / target
        # switch) is dragged in slowly, hurting acquisition. If the newest sample
        # disagrees with the fit by far more than the noise, the window is stale:
        # drop it and restart from this sample.
        jt = p.get("jump", 0.0)
        if jt > 0.0 and len(self.buf) >= 3:
            if math.hypot(ax - est_x, ay - est_y) > jt:
                self.buf = [(ax, ay)]
                self.sx = [ax, 0.0]; self.sy = [ay, 0.0]
                est_x, vx, est_y, vy = ax, 0.0, ay, 0.0

        # extrapolate the estimate from capture time to NOW, then the error against
        # the crosshair's known current position. Exact: error == T - C_now.
        # The extrapolation beyond the dead time is GATED exactly as in the shipped
        # design: the LS slope carries ~1px/frame of noise, so an ungated lead pays
        # more in rest jitter than it saves in lag (measured).
        base = D
        extra = max(0.0, (p.get("pred", D) if p.get("pred") is not None else D) - D)
        if extra > 0.0:
            g = 1.0
            vg = p.get("pred_vgate", 0.0)
            if vg > 0.0:
                sp = math.hypot(vx, vy); g *= sp / (sp + vg)
            e0 = p.get("pred_err_gate", 0.0)
            if e0 > 0.0:
                rex = est_x + vx*base - self.M_x - SC
                rey = est_y + vy*base - self.M_y - SC
                e2 = rex*rex + rey*rey
                g *= (e0*e0) / (e0*e0 + e2)
            pr = base + extra*g
        else:
            pr = base
        ex = (est_x + vx*pr) - self.M_x - SC
        ey = (est_y + vy*pr) - self.M_y - SC

        if p.get("law", "p") == "db":
            # Deadbeat / inverse model: emit the whole correction (scaled by trust).
            # No proportional law - the error is already a complete correction
            # because the frame is exact.
            t = p.get("trust", 0.8)
            mx, my = t*ex, t*ey
        else:
            dex = max(-150.0, min(150.0, ex - self.prev_err_x))
            dey = max(-150.0, min(150.0, ey - self.prev_err_y))
            self.derr_x = 0.6*dex + 0.4*self.derr_x
            self.derr_y = 0.6*dey + 0.4*self.derr_y
            self.prev_err_x, self.prev_err_y = ex, ey
            mx = nonlinear_p(ex, p["kp_x"], p["soft_x"]) + p["kd_x"]*self.derr_x
            my = nonlinear_p(ey, p["kp_y"], p["soft_y"]) + p["kd_y"]*self.derr_y

        mx, my = clamp_max_step(mx, my, p["max_step"])
        dx, self.res_x = emit_mouse_delta(mx, self.res_x)
        dy, self.res_y = emit_mouse_delta(my, self.res_y)
        self._push(float(dx), float(dy))
        self.M_x += float(dx); self.M_y += float(dy)
        return dx, dy


def unified(**kw):
    p = dict(est="ls", win=6, dead=1.13, pred=None, law="p",
             kp_x=0.75, kp_y=0.82, soft_x=9.0, soft_y=8.0, kd_x=0.05, kd_y=0.06,
             max_step=25.0, trust=0.8, vlim=60.0, kf_R=66.0, kf_R_y=35.0, kf_Q=0.5,
             # unused by UnifiedCtrl but aim_opt.run touches p["comp"]/p["w"] only
             # inside OptCtrl, so these are harmless placeholders
             comp=0.0, w=0.0, oneeuro=False, mincut=0.1, beta=0.02, dcut=0.5, ff=0.0)
    p.update(kw); return p
