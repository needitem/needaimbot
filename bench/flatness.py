#!/usr/bin/env python3
"""파라미터 최적점이 정말 '최적점'인가, 아니면 표면이 평평한가.

sigma_law.py 가 낸 다섯 최적점은 서로 완전히 다르고(kp_x 0.688~1.200, soft_x 2.0~16.4)
여러 값이 탐색 상자 경계에 붙어 있었다. R² 도 0.02~0.54 로 낮다. 두 가지 해석이 있다:
  (a) sigma 에 대한 법칙이 없다
  (b) 애초에 목적함수가 평평해서 '최적점'이 정해지지 않는다
가르는 방법: 다섯 최적점을 **같은 sigma** 에서 평가한다. 점수가 비슷하면 (b) 다.

(b) 라면 이번 세션 내내 겪은 탐색 불일치(같은 제약에서 -1.3% 와 -3.0%)도 같은 원인이고,
sigma 를 온라인 추정해 파라미터를 따라가게 만들 이유도 사라진다.
"""
import math
import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base

FPS, DEAD_MS, SIG_DT, BASE = 201.0, 11.60, 0.295, 143.0
T = 1000.0/FPS
realjit.dt_sample = (lambda m: (lambda rng: m*math.exp(rng.gauss(0., SIG_DT))))(DEAD_MS/T)

SHIP = {"kp_x":0.445,"kp_y":0.406,"soft_x":8.6,"soft_y":6.57,"kd_x":0.052,"kd_y":0.037,
        "max_step":11.37,"w":2.85,"ff":2.924,"beta":0.02,"predict":5.47,"v_ema":0.137,
        "vgate":8.60,"ff_err_gate":22.25,"mincut":0.049,"ego_lag":2.85}
OPT = {0.4: dict(kp_x=0.952,kp_y=0.746,soft_x=4.30,soft_y=26.31,mincut=0.2881,vgate=6.22,ff_err_gate=20.41),
       0.7: dict(kp_x=0.760,kp_y=0.211,soft_x=2.00,soft_y=2.00,mincut=0.0100,vgate=9.88,ff_err_gate=48.88),
       1.0: dict(kp_x=1.200,kp_y=0.212,soft_x=16.43,soft_y=2.00,mincut=0.0100,vgate=11.40,ff_err_gate=67.14),
       1.5: dict(kp_x=0.743,kp_y=0.221,soft_x=7.50,soft_y=2.66,mincut=0.0100,vgate=10.98,ff_err_gate=18.13),
       2.5: dict(kp_x=0.688,kp_y=0.331,soft_x=9.45,soft_y=14.50,mincut=0.0100,vgate=1.55,ff_err_gate=4.00)}


def ev(p, n=40, lo=500):
    k = FPS/BASE; vs = BASE/FPS; o = {}
    spec = (("step","step",dict(noise=False,frames=int(150*k))),("hold","hold",dict(frames=int(420*k))),
            ("rev","reversal",dict(frames=int(300*k))),("t1","track",dict(vx=1.1*vs,frames=int(480*k))),
            ("t4","track",dict(vx=4.0*vs,frames=int(480*k))),("t8","track",dict(vx=8.0*vs,frames=int(480*k))),
            ("r40","reach",dict(noise=False,step_dist=40.,frames=int(220*k))))
    for nm, s_, kw in spec:
        acc = {}
        for s in range(lo, lo+n):
            m = run_real(p, s_, s, **kw)
            for kk, v in m.items(): acc.setdefault(kk, []).append(v)
        for kk, v in acc.items(): o[nm+"_"+kk] = sum(v)/len(v)
    o["r40_ms"] = o["r40_reach"]*T
    return o


def main():
    print("  다섯 최적점 + 현재 배포값을, 전부 σ배율 1.0 에서 평가\n")
    print("  %-20s %8s %7s %8s %8s"%("설정","에러","오버슛","반전pk","획득40"))
    print("  "+"-"*56)
    res = []
    m = ev(base(**SHIP)); res.append(sc(m))
    print("  %-20s %8.3f %7.2f %8.2f %7.1fms"
          % ("현재 배포", sc(m), m['step_overshoot'], m['rev_peak'], m['r40_ms']))
    for mul, o in OPT.items():
        m = ev(base(**dict(SHIP, **o))); res.append(sc(m))
        print("  %-20s %8.3f %7.2f %8.2f %7.1fms"
              % ("σ%.1f 최적점" % mul, sc(m), m['step_overshoot'], m['rev_peak'], m['r40_ms']))
    print("\n  에러 범위 %.3f ~ %.3f  (폭 %.1f%%)"
          % (min(res), max(res), 100*(max(res)-min(res))/min(res)))


if __name__ == "__main__":
    main()
