#!/usr/bin/env python3
"""설정만으로 얻은 -4.6% 후보를 굳히고 독립 seed 로 확정 검증.

두 가지를 더 본다:
  1) reach 회귀 제거 - 앞 결과는 획득시간이 6.5 -> 7.5 프레임 늘었다. 사용자의 목표는
     '링잉 없이 더 빠르게' 이므로 획득이 느려지는 것은 대가로 받아들이기 어렵다.
     holdout 에서 잰 reach 를 제약으로 걸고 다시 정련한다.
  2) 속도추정 입력 - KF/EMA 실험은 '필터된 위치'를 리드항 속도추정에도 먹였다.
     One Euro 경로는 원시 위치로 속도를 추정한다. 이 차이가 KF 의 -2.5% 와
     같은 설정의 -0.5% 를 갈랐다. 이건 설정이 아니라 코드 한 줄이므로 별도로 잰다.
확정 평가는 탐색·검증 어디에도 쓰지 않은 seed 500-559 로 한다.
"""
import json
import math
import random

from realjit import ev, sc, run_real
from holdout import ev_holdout
from ctrl_zoo_opt import base, COMMON_BOX, refine
from aim_opt import OptCtrl

random.seed(31337)

CAND = {"kp_x": 0.7797, "kp_y": 0.611, "soft_x": 9.0161, "soft_y": 15.9729,
        "kd_x": 0.1635, "kd_y": 0.0173, "max_step": 29.8496, "w": 1.523,
        "ff": 1.2777, "predict": 1.1475, "v_ema": 0.2476, "vgate": 2.5634,
        "ff_err_gate": 23.8051, "mincut": 0.2238, "beta": 0.0006}


class FiltVelCtrl(OptCtrl):
    """리드항 속도추정 입력을 원시 위치 -> 평활 위치로.

    OptCtrl 은 dv = rx - prev_raw_x 로 속도를 잰다. 같은 alpha 의 EMA g 를 따로 돌리고
    prev_raw_x 를 g_prev - (rx - g_now) 로 넣으면 그 차분이 정확히 g_now - g_prev 가
    된다 - 원본 코드를 건드리지 않고 등가 동작을 만드는 방법."""
    def reset(self):
        OptCtrl.reset(self); self.gx = self.gy = None

    def step(self, rx, ry, cls_changed=False, box_h=None):
        from aim_sim_ring import one_euro_alpha
        a = one_euro_alpha(self.p["mincut"])
        if self.gx is None or cls_changed:
            self.gx, self.gy = rx, ry
        gpx, gpy = self.gx, self.gy
        self.gx += a*(rx - self.gx); self.gy += a*(ry - self.gy)
        self.prev_raw_x = gpx - (rx - self.gx)
        self.prev_raw_y = gpy - (ry - self.gy)
        return OptCtrl.step(self, rx, ry, cls_changed)


def ev_final(p, ctrl=None, lo=500, n=60):
    o = {}
    for nm, scn, kw in (("step", "step", dict(noise=False)),
                        ("hold", "hold", {}), ("rev", "reversal", {}),
                        ("t1", "track", dict(vx=1.1)), ("t4", "track", dict(vx=4.0)),
                        ("t8", "track", dict(vx=8.0)),
                        ("reach", "reach", dict(noise=False, step_dist=120.))):
        acc = {}
        for s in range(lo, lo + n):
            m = run_real(p, scn, s, ctrl=ctrl, **kw)
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            o[nm + "_" + k] = sum(v) / len(v)
    return o


def show(lbl, p, ctrl=None, ref=None):
    m = ev_final(p, ctrl=ctrl)
    e = sc(m)
    d = "" if ref is None else "  %+6.1f%%" % (100*(e-ref)/ref)
    print("  %-36s %8.3f  %5.2f %6.2f %5.2f %6.2f %5.1f%s"
          % (lbl, e, m['step_overshoot'], m['rev_peak'], m['hold_rms'],
             m['t8_rms'], m['reach_reach'], d))
    return e, m


def main():
    cur = base()
    # 제약은 holdout(200-239)에서 잰 현재값 기준 - 정련 점수와 제약을 같은 데이터에서
    # 재지 않기 위해 정련은 seed 0-9, 제약 판정은 그대로 seed 0-9 를 쓰되 reach 만
    # 현재값 이하로 죈다.
    m0 = ev(cur, seeds=10)
    ok = lambda m: (m['step_overshoot'] <= m0['step_overshoot'] + 0.05
                    and m['step_osc'] <= 0.35
                    and m['rev_peak'] <= m0['rev_peak'] * 1.03
                    and m['reach_reach'] <= m0['reach_reach'])   # 회귀 금지
    box = dict(COMMON_BOX, beta=(0.0, 0.12))

    print("  reach 제약을 죄고 재정련 중...")
    p2, e2 = refine(base(**CAND), sc(ev(base(**CAND), seeds=10)), box, None, ok,
                    iters=200, seeds=10)
    print("  속도추정 입력 변형 정련 중...")
    p3, e3 = refine(base(**CAND), sc(ev(base(**CAND), seeds=10, ctrl=FiltVelCtrl)),
                    box, FiltVelCtrl, ok, iters=200, seeds=10)

    print("\n  확정 평가: seed 500-559 (탐색·검증 어디에도 미사용)")
    print("  %-36s %8s  %5s %6s %5s %6s %5s"
          % ("구성", "에러", "오버슛", "반전pk", "hold", "t8", "reach"))
    print("  " + "-" * 84)
    ref, _ = show("현재 배포 설정", cur)
    show("설정만 (-4.6% 후보)", base(**CAND), None, ref)
    e_a, m_a = show("설정만 + reach 제약 재정련", p2, None, ref)
    e_b, m_b = show("+ 속도추정에 필터값 (코드 1줄)", p3, FiltVelCtrl, ref)

    print("\n  reach 제약 재정련 결과:")
    print("  " + json.dumps({k: round(p2[k], 4) for k in box if k in p2}))
    if e_b < e_a * 0.995:
        print("\n  속도추정 변형 결과:")
        print("  " + json.dumps({k: round(p3[k], 4) for k in box if k in p3}))


if __name__ == "__main__":
    main()
