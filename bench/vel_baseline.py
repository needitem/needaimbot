#!/usr/bin/env python3
"""속도 추정기를 바꾸면 t8/t4 가 줄어드는가 — 게인은 그대로.

에러의 최대 항은 t8(빠른 표적 추적)이고, 그 지배 성분은 리드가 쓰는 속도 추정의
노이즈다. 현재는 '연속 프레임 차분의 EMA':

    d_k = x_k - x_{k-1} + ego_k          (프레임당 차분, 노이즈 sigma*sqrt(2))
    v   = (1-a)*v + a*d_k                 (a = ff_v_ema = 0.235)

sigma 8.1px 에서 d 의 노이즈는 11.5px/f, EMA 후에도 ~4.2px/f 다. 참 속도가 8px/f 인
t8 에서 **추정의 절반이 잡음**이고 ff=1.7 이 그걸 증폭한다.

같은 '기억 길이'에서 노이즈가 더 작은 추정기가 있다. N프레임 기준선 차분:

    v = (x_k - x_{k-N} + ego_over_N) / N   (노이즈 sigma*sqrt(2)/N)

연속 차분은 이웃끼리 끝점을 공유해 평균이 잘 안 내려가지만, 끝점 하나짜리 차분은
N 에 반비례해 바로 내려간다. 대신 그 창의 '중간 시점' 속도라 N/2 프레임 지연이 붙는다
- EMA 도 이미 ~1/a = 4.3 프레임의 지연을 갖고 있으므로 공짜로 바꿔치기가 가능하다.

주의: 최소자승 궤적적합은 이미 기각됐다(에러 +8.2%). 그건 **위치 추정까지** 창으로
대체해 반전에 약했다. 여기서는 위치는 One Euro 그대로 두고 **속도 추정만** 바꾼다.
"""
import math
import random

import realjit
from realjit import sc
from ctrl_zoo_opt import base
from aim_opt import OptCtrl, SC
from aim_opt import (NOISE_X, NOISE_Y, P_DROP, DRIFT, DRIFT_RHO,
                     P_OUTLIER, OUT_LO, OUT_HI, P_SWITCH, SW_LO, SW_HI, SW_XSD)
from adaptive_deadtime import OBSERVABLE
from gap_fix import SHIP

KMAX = 8


class BaseVelCtrl(OptCtrl):
    """속도만 N프레임 기준선 차분으로. N=1 이면 현재 동작과 같다.

    핵심은 ego 되돌리기의 '창'을 raw 차분과 맞추는 것이다. raw_k = T_k - C_k + SC 이므로
        raw_k - raw_{k-N} = (T_k - T_{k-N}) - (C_k - C_{k-N})
    이고 C 의 변화는 그 구간에 적용된 emit 의 합이다. 게다가 측정은 데드타임만큼 낡았으므로
    창 자체를 ego_lag 만큼 뒤로 밀어야 한다. 링은 4프레임뿐이라 N>4 를 보려면 emit 이력을
    따로 들어야 한다 - 첫 시도가 1프레임치 ego 만 더해서 발산했다(오버슈트 21~25).
    """
    def reset(self):
        OptCtrl.reset(self)
        self.lag_meas = None
        self.hist_raw = []
        self.hist_emit = []

    def set_lag(self, v):
        self.lag_meas = v

    def note_emit(self, dx, dy):
        self.hist_emit.append((dx, dy))
        if len(self.hist_emit) > 64:
            self.hist_emit.pop(0)

    def step(self, rx, ry, cls_changed=False, nbase=1):
        p = self.p
        saved_w = p["w"]
        if self.lag_meas is not None:
            p["w"] = max(0.0, min(4.0, self.lag_meas))
        lag = int(round(p.get("ego_lag", 2.0)))
        need = nbase + lag
        if nbase > 1 and self.has_track and len(self.hist_raw) >= nbase \
                and len(self.hist_emit) >= need:
            ox, oy = self.hist_raw[-nbase]
            hi = len(self.hist_emit) - lag
            win = self.hist_emit[hi-nbase:hi]
            ex = sum(e[0] for e in win); ey = sum(e[1] for e in win)
            dvx = ((rx - ox) + ex) / nbase
            dvy = ((ry - oy) + ey) / nbase
            ax, ay = self._ring_at(p.get("ego_lag", 2.0))
            # OptCtrl 은 dv = (rx - prev_raw) + ax 를 쓴다. 원하는 dv 가 나오도록 역산.
            self.prev_raw_x = rx + ax - dvx
            self.prev_raw_y = ry + ay - dvy
        try:
            out = OptCtrl.step(self, rx, ry, cls_changed)
        finally:
            p["w"] = saved_w
        self.hist_raw.append((rx, ry))
        if len(self.hist_raw) > 64:
            self.hist_raw.pop(0)
        return out


def run(p, scen, seed, nbase, frames=None, vx=1.1, step_dist=60.0, noise=True):
    frames = frames or {"step": 150, "hold": 420, "track": 480,
                        "reversal": 300, "reach": 220}[scen]
    rng = random.Random(seed)
    c = BaseVelCtrl(p); c.reset()
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
        c.set_lag(W + (lat - 1.10) * OBSERVABLE)
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
                c._push(0.0, 0.0)          # 링 전진 (이번 세션에서 확인된 수정)
                c.note_emit(0.0, 0.0)
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
                dx, dy = c.step(mx, my, cls, nbase=nbase)
                c.note_emit(dx, dy)
        else:
            dx, dy = c.step(mx, my, cls, nbase=nbase)
            c.note_emit(dx, dy)
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


def metrics(p, nbase, blocks=(500, 700, 900), n=60):  # noqa: E302
    o = {}
    for lo in blocks:
        for nm, s_, kw in SCN:
            for s in range(lo, lo + n):
                m = run(p, s_, s, nbase, **kw)
                for k, v in m.items():
                    o.setdefault(nm + "_" + k, []).append(v)
    return {k: sum(v)/len(v) for k, v in o.items()}


def main():
    print("  속도 추정 기준선 N (N=1 = 현재). 게인은 전부 그대로.\n")
    print("  %-26s %8s %6s %6s %6s %6s %6s %6s"
          % ("추정기", "에러", "오버슛", "반전pk", "hold", "t4", "t8", "r40"))
    print("  " + "-" * 74)
    # 기준선 차분은 그 자체가 창 평균이므로 EMA 를 '대체'해야 한다. 위에 얹으면
    # 지연이 합산돼(N/2 + 1/a) 당연히 나빠진다 - 첫 시도의 실수였다.
    b = None
    cases = [("N=1 + EMA 0.235 (현재)", 1, 0.235)]
    for nb in (3, 4, 6, 8):
        cases.append(("N=%d, EMA 없음" % nb, nb, 1.0))
    for nb in (4, 6):
        cases.append(("N=%d + EMA 0.5" % nb, nb, 0.5))
    for lbl, nb, ve in cases:
        p = base(**dict(SHIP, v_ema=ve))
        m = metrics(p, nb)
        if b is None:
            b = m
        d = "" if m is b else "  %+.1f%%" % (100*(sc(m)-sc(b))/sc(b))
        print("  %-26s %8.3f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f%s"
              % (lbl, sc(m), m['step_overshoot'], m['rev_peak'], m['hold_rms'],
                 m['t4_rms'], m['t8_rms'], m['r40_reach'], d))


if __name__ == "__main__":
    main()
