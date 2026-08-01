#!/usr/bin/env python3
"""검출 공백이 남기는 두 가지 오차 — 게인과 무관하므로 곡선을 옮길 수 있다.

공백 프레임(검출 없음)에서 커널은 조기 반환한다(simple_postprocess.cu). 그래서:

  ① prev_raw 가 갱신되지 않는다. 공백이 N프레임이면 다음 dvx = raw - prev_raw 는
     N프레임치 이동인데 코드는 1프레임으로 나눈다 -> 속도가 N배 과대추정되고,
     리드(ff_gain·predict_frames)가 그 속도를 그대로 쓴다.
  ② in-flight 링이 전진하지 않는다. 공백 동안 우리는 아무것도 안 내보내므로 0 이
     쌓여야 하는데 링이 멈춰 있어, 재획득 프레임에서 inflightSum 이 '이미 보이게 된
     옛 이동'을 다시 뺀다 -> 과보정(오차를 실제보다 작게 봐서 덜 나간다).

둘 다 노이즈 증폭이 아니라 '가진 정보를 버려서' 생기는 오차다. 적응 데드타임과 같은
성질이므로 게인을 안 건드리고 고칠 수 있다.

여기서 재는 것 (전부 현재 배포 = 적응 데드타임 ON 위에서):
  base      현재 그대로
  velfix    ①만: 경과 프레임 수로 속도 델타를 나눈다
  ringfix   ②만: 공백 프레임에 0 을 링에 push
  both      둘 다
결측률을 실측(1.5%)과 그보다 높은 조건에서 함께 본다 - 교전 중 공백은 더 잦다.
"""
import math
import random

import realjit
from realjit import sc
from ctrl_zoo_opt import base
from aim_opt import OptCtrl, SC
from aim_opt import (NOISE_X, NOISE_Y, DRIFT, DRIFT_RHO,
                     P_OUTLIER, OUT_LO, OUT_HI, P_SWITCH, SW_LO, SW_HI, SW_XSD)
from adaptive_deadtime import OBSERVABLE, SHIP

SHIP = dict(SHIP, kp_x=0.765, kp_y=0.698, soft_x=8.6, soft_y=6.57, kd_x=0.052,
            kd_y=0.037, max_step=19.56, predict=3.18, v_ema=0.235, vgate=14.79,
            ff_err_gate=22.25, mincut=0.084)


class GapCtrl(OptCtrl):
    """공백 처리 두 갈래를 옵션으로 켜고 끈다."""
    def reset(self):
        OptCtrl.reset(self)
        self.lag_meas = None
        self.gap = 0            # 직전 검출 이후 지나간 공백 프레임 수

    def set_lag(self, v):
        self.lag_meas = v

    def on_gap(self, ringfix):
        """검출이 없던 프레임. 우리는 아무것도 안 내보냈다."""
        self.gap += 1
        if ringfix:
            self._push(0.0, 0.0)      # 실제로 0 을 내보냈으니 링도 그렇게 알아야 한다

    def step(self, rx, ry, cls_changed=False, velfix=False, alphafix=False,
             predfix=False):
        p = self.p
        saved_w = p["w"]
        saved_mc, saved_pr = p["mincut"], p["predict"]
        n_gap = self.gap
        if alphafix and n_gap > 0 and self.has_track:
            # 공백 동안 필터도 멈춰 있었다. 재획득 프레임은 (n_gap+1) 프레임치를
            # 섞어야 하므로 alpha_eff = 1-(1-a)^(n+1). One Euro 의 alpha 는 cutoff 의
            # 함수이므로 등가 cutoff 로 되돌려 넣는다: a = 2*pi*fc/(2*pi*fc+1).
            a1 = 2*math.pi*p["mincut"]/(2*math.pi*p["mincut"]+1.0)
            ae = 1.0 - (1.0-a1)**(n_gap+1)
            ae = min(ae, 0.98)
            p["mincut"] = ae/(2*math.pi*(1.0-ae))
        if predfix and n_gap > 0:
            # 측정이 데드타임뿐 아니라 공백만큼도 더 낡았다. 표적 전진도 그만큼.
            p["predict"] = min(6.0, p["predict"] + n_gap)
        if self.lag_meas is not None:
            p["w"] = max(0.0, min(4.0, self.lag_meas))
        # 공백 뒤 첫 프레임: 델타가 (gap+1) 프레임치다. 속도는 프레임당이어야 하므로
        # prev_raw 를 그만큼 당겨 놓으면 기존 식이 그대로 프레임당 값을 낸다.
        if velfix and self.gap > 0 and self.has_track:
            n = self.gap + 1
            self.prev_raw_x = rx - (rx - self.prev_raw_x) / n
            self.prev_raw_y = ry - (ry - self.prev_raw_y) / n
        try:
            out = OptCtrl.step(self, rx, ry, cls_changed)
        finally:
            p["w"] = saved_w
            p["mincut"], p["predict"] = saved_mc, saved_pr
        self.gap = 0
        return out


def run(p, scen, seed, mode, p_drop, frames=None, vx=1.1, step_dist=60.0, noise=True):
    velfix = "vel" in mode
    ringfix = "ring" in mode or mode == "all"
    alphafix = "alpha" in mode or mode == "all"
    predfix = "pred" in mode or mode == "all"
    frames = frames or {"step": 150, "hold": 420, "track": 480,
                        "reversal": 300, "reach": 220}[scen]
    rng = random.Random(seed)
    c = GapCtrl(p); c.reset()
    C = [SC, SC]
    if scen in ("step", "reach"):
        T = [SC + step_dist*.94, SC + step_dist*.34]; v = [0, 0]
    elif scen == "track":
        T = [SC - 60., SC]; v = [vx, vx*.2]
    elif scen == "reversal":
        T = [SC, SC]; v = [1.3, 0]
    else:
        T = [SC, SC]; v = [0, 0]
    d0 = math.hypot(T[0]-C[0], T[1]-C[1])
    au = ((T[0]-C[0])/d0, (T[1]-C[1])/d0) if d0 > 1e-9 else (1., 0.)
    hist = []; errs = []; par = []; drift = [0., 0.]
    sw = 0; so = (0., 0.); turn = None; peak = 0.; reach = None
    W = p["w"]
    for f in range(frames):
        if scen == "reversal" and f > 0 and f % 75 == 0:
            v[0] = -v[0]; turn = f
        T[0] += v[0]; T[1] += v[1]
        if T[0] < 40: T[0] = 40; v[0] = abs(v[0])
        if T[0] > 280: T[0] = 280; v[0] = -abs(v[0])
        hist.append((T[0], T[1], C[0], C[1]))
        lat = max(0., realjit.dt_sample(rng))
        c.set_lag(W + (lat - 1.10) * OBSERVABLE)     # 배포된 적응 데드타임
        idx = max(0., f - lat); i0 = int(idx); i1 = min(i0+1, len(hist)-1); fr = idx - i0
        sT0 = hist[i0][0]*(1-fr) + hist[i1][0]*fr
        sT1 = hist[i0][1]*(1-fr) + hist[i1][1]*fr
        sC0 = hist[i0][2]*(1-fr) + hist[i1][2]*fr
        sC1 = hist[i0][3]*(1-fr) + hist[i1][3]*fr
        mx, my = sT0 - sC0 + SC, sT1 - sC1 + SC
        cls = False
        if noise:
            if rng.random() < p_drop:
                dx = dy = 0; c.res_x = c.res_y = 0.
                c.on_gap(ringfix)
            else:
                s = math.sqrt(max(1e-9, 1 - DRIFT_RHO**2))
                drift[0] = DRIFT_RHO*drift[0] + rng.gauss(0, DRIFT*s)
                drift[1] = DRIFT_RHO*drift[1] + rng.gauss(0, DRIFT*s)
                mx += drift[0] + rng.gauss(0, NOISE_X)
                my += drift[1] + rng.gauss(0, NOISE_Y)
                if rng.random() < P_OUTLIER:
                    a = rng.random()*6.283
                    mg = OUT_LO + rng.random()*(OUT_HI-OUT_LO)
                    mx += mg*math.cos(a); my += mg*math.sin(a)
                if sw > 0:
                    mx += so[0]; my += so[1]; sw -= 1
                    if sw == 0: cls = True
                elif rng.random() < P_SWITCH:
                    sw = rng.randint(1, 4)
                    mg = rng.uniform(SW_LO, SW_HI)*(1 if rng.random() < .5 else -1)
                    so = (rng.gauss(0, SW_XSD), mg)
                    mx += so[0]; my += so[1]; cls = True
                dx, dy = c.step(mx, my, cls, velfix=velfix, alphafix=alphafix, predfix=predfix)
        else:
            dx, dy = c.step(mx, my, cls, velfix=velfix, alphafix=alphafix, predfix=predfix)
        C[0] += dx; C[1] += dy
        e = math.hypot(T[0]-C[0], T[1]-C[1]); errs.append(e)
        par.append((C[0]-(T[0]-d0*au[0]))*au[0] + (C[1]-(T[1]-d0*au[1]))*au[1])
        if reach is None and e < 4.: reach = f
        if scen == "reversal" and turn is not None and 0 <= f-turn < 50:
            peak = max(peak, e)
    if scen == "step":
        ov = max(0., max(par) - d0)
        ri = next((i for i, q in enumerate(par) if q >= d0-2.), frames)
        osc = 0; sg = 0
        for q in par[ri:]:
            s2 = 1 if (q-d0) > 1.5 else (-1 if (q-d0) < -1.5 else 0)
            if s2 and sg and s2 != sg: osc += 1
            if s2: sg = s2
        return dict(overshoot=ov, osc=osc)
    if scen == "reach":
        return dict(reach=reach if reach is not None else frames)
    if scen == "reversal":
        w = errs[100:]
        return dict(peak=peak, rms=math.sqrt(sum(x*x for x in w)/len(w)))
    w = errs[100:]
    return dict(rms=math.sqrt(sum(x*x for x in w)/len(w)))


SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))


def metrics(p, mode, p_drop, blocks=(500, 700, 900), n=60):
    o = {}
    for lo in blocks:
        for nm, s_, kw in SCN:
            for s in range(lo, lo + n):
                m = run(p, s_, s, mode, p_drop, **kw)
                for k, v in m.items():
                    o.setdefault(nm + "_" + k, []).append(v)
    return {k: sum(v)/len(v) for k, v in o.items()}


def main():
    p = base(**SHIP)
    for pd_ in (0.015, 0.05, 0.10):
        print("\n  결측률 %.1f%% %s" % (100*pd_, "(실측 조준중)" if pd_ == 0.015 else ""))
        print("  %-22s %8s %6s %6s %6s %6s %6s"
              % ("공백 처리", "에러", "오버슛", "반전pk", "t4", "t8", "r40"))
        print("  " + "-" * 66)
        b = None
        for lbl, mode in (("현재 (버림)", "base"), ("링 전진", "ring"),
                          ("+ 필터 alpha 보정", "ring+alpha"),
                          ("+ 표적 전진 보정", "ring+pred"),
                          ("셋 다", "all")):
            m = metrics(p, mode, pd_)
            if b is None:
                b = m
            d = "" if m is b else "  %+.1f%%" % (100*(sc(m)-sc(b))/sc(b))
            print("  %-22s %8.3f %6.2f %6.2f %6.2f %6.2f %6.2f%s"
                  % (lbl, sc(m), m['step_overshoot'], m['rev_peak'],
                     m['t4_rms'], m['t8_rms'], m['r40_reach'], d))


if __name__ == "__main__":
    main()
