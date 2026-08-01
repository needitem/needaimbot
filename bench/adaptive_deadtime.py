#!/usr/bin/env python3
"""고정 상수 w 대신 '그 프레임의 실측 지연'으로 데드타임을 보정하면?

지금까지 확인된 법칙: 컨트롤러 안의 모든 축(soft_x·soft_y·kp_y·max_step·w)에서
근거리 속도와 오버슈트는 같은 양이다. 게인을 만지는 한 곡선 위를 미끄러질 뿐이다.

그런데 오버슈트의 원인은 둘로 갈린다 - ① 노이즈 증폭 ② **데드타임 지터**.
`inflight_comp` 는 우리가 내보낸 이동을 빼서 이중보정을 막는데, 몇 프레임치를 뺄지가
**고정 상수 w** 다. 실제 데드타임은 프레임마다 흔들리므로(실측 E2E p50 2.7 / p95 4.8ms)
매 프레임 틀린 양을 빼고 있고, w=1.25 는 그 분포에 대한 타협값일 뿐이다.

②는 게인과 무관한 오차다. 없앨 수 있다면 곡선 위를 미끄러지는 게 아니라 곡선이 옮겨간다.
그리고 앱은 프레임마다 실제 지연(E2E)을 이미 재고 있다 - 컨트롤러에 안 줄 뿐이다.

여기서 재는 것:
  A) 완전 지식 (그 프레임의 진짜 지연을 그대로 사용) = 이론 천장
  B) 부분 지식 (E2E 부분만 관측 가능, USB/렌더대기는 미관측) = 실제로 구현 가능한 것
  C) 고정 w (현재)
"""
import math
import random

import realjit
from realjit import sc
from ctrl_zoo_opt import base
from aim_opt import OptCtrl, SC
from aim_opt import (NOISE_X, NOISE_Y, P_DROP, DRIFT, DRIFT_RHO,
                     P_OUTLIER, OUT_LO, OUT_HI, P_SWITCH, SW_LO, SW_HI, SW_XSD)

SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}

# 관측 가능한 비율: 데드타임 7.07ms 중 E2E 2.6ms 만 프레임별로 측정된다.
# 렌더 대기(3.47)와 USB(1.0)는 프레임별로 알 수 없다.
OBSERVABLE = 2.6 / 7.07


class AdaptCtrl(OptCtrl):
    """그 프레임의 (부분) 실측 지연으로 w 를 대체한다. set_lag() 로 주입."""
    def reset(self):
        OptCtrl.reset(self); self.lag_meas = None

    def set_lag(self, lag_frames):
        self.lag_meas = lag_frames

    def step(self, rx, ry, cls_changed=False, box_h=None):
        if self.lag_meas is None:
            return OptCtrl.step(self, rx, ry, cls_changed)
        saved = self.p["w"]
        self.p["w"] = max(0.0, min(4.0, self.lag_meas))
        try:
            return OptCtrl.step(self, rx, ry, cls_changed)
        finally:
            self.p["w"] = saved


def run(p, scen, seed, mode, frames=None, vx=1.1, step_dist=60.0, noise=True):
    """realjit.run_real 의 사본 + 프레임별 지연을 컨트롤러에 주입하는 갈래.
    (원본을 건드리지 않으려고 복제했다 - 원본은 다른 실험들이 계속 쓴다.)"""
    frames = frames or {"step": 150, "hold": 420, "track": 480,
                        "reversal": 300, "reach": 220}[scen]
    rng = random.Random(seed)
    c = AdaptCtrl(p); c.reset()
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
        # --- 이번 실험의 핵심: 컨트롤러가 무엇을 안다고 볼 것인가 ---
        if mode == "fixed":
            c.set_lag(None)                        # 현재: 고정 상수 w
        elif mode == "full":
            c.set_lag(lat + (W - 1.10))            # 천장: 진짜 지연을 그대로
        elif mode == "partial":
            # 실제 구현 가능한 것: 지연 변동 중 관측 가능한 몫만 반영
            c.set_lag(W + (lat - 1.10) * OBSERVABLE)
        elif mode.startswith("noisy"):
            # 실기에서는 편차를 '잡음 섞인 상태로' 관측한다(클럭 오프셋 추정 오차 등).
            # 잡음이 신호를 넘어서면 보정이 오히려 해가 되므로 내성을 확인해야 한다.
            frac = float(mode.split(":")[1])
            true_dev = (lat - 1.10) * OBSERVABLE
            meas = true_dev + rng.gauss(0.0, abs(frac) * 0.37 * OBSERVABLE)
            c.set_lag(W + max(-0.75, min(0.75, meas)))
        idx = max(0., f - lat); i0 = int(idx); i1 = min(i0+1, len(hist)-1); fr = idx - i0
        sT0 = hist[i0][0]*(1-fr) + hist[i1][0]*fr
        sT1 = hist[i0][1]*(1-fr) + hist[i1][1]*fr
        sC0 = hist[i0][2]*(1-fr) + hist[i1][2]*fr
        sC1 = hist[i0][3]*(1-fr) + hist[i1][3]*fr
        mx, my = sT0 - sC0 + SC, sT1 - sC1 + SC
        cls = False
        if noise:
            if rng.random() < P_DROP:
                dx = dy = 0; c.res_x = c.res_y = 0.
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
                dx, dy = c.step(mx, my, cls)
        else:
            dx, dy = c.step(mx, my, cls)
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


def metrics(p, mode, blocks=(500, 700, 900), n=60):
    o = {}
    for lo in blocks:
        for nm, s_, kw in SCN:
            for s in range(lo, lo + n):
                m = run(p, s_, s, mode, **kw)
                for k, v in m.items():
                    o.setdefault(nm + "_" + k, []).append(v)
    return {k: sum(v)/len(v) for k, v in o.items()}


def main():
    p = base(**SHIP)
    print("  데드타임 지터를 프레임별 실측으로 보정하면? (세 블록 x60)\n")
    print("  %-30s %8s %6s %6s %6s %6s"
          % ("보정 방식", "에러", "오버슛", "반전pk", "r40", "r120"))
    print("  " + "-" * 66)
    b = None
    for lbl, mode in (("고정 상수 w=1.25 (현재)", "fixed"),
                      ("부분 실측 (E2E 몫 %.0f%%)" % (100*OBSERVABLE), "partial"),
                      ("완전 실측 (이론 천장)", "full")):
        m = metrics(p, mode)
        if b is None:
            b = m
        d = "" if m is b else "  에러%+.1f%% 오버슛%+.0f%% r40%+.1f%%" % (
            100*(sc(m)-sc(b))/sc(b),
            100*(m['step_overshoot']-b['step_overshoot'])/b['step_overshoot'],
            100*(m['r40_reach']-b['r40_reach'])/b['r40_reach'])
        print("  %-30s %8.3f %6.2f %6.2f %6.2f %6.2f%s"
              % (lbl, sc(m), m['step_overshoot'], m['rev_peak'],
                 m['r40_reach'], m['r120_reach'], d))


if __name__ == "__main__":
    main()
