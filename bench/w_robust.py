#!/usr/bin/env python3
"""데드타임이 변할 때, 어떤 w 가 가장 덜 흔들리나.

폐루프로는 emit->visible 을 못 잰다(deadtime_closedloop.py: 143fps 진실 1.66 인데
분할별로 1.45~4.85 가 나온다). 주입 측정은 신뢰할 수 있지만 플레이 중엔 못 한다.
그러면 실제 데드타임은 플레이마다 얼마간 모를 수밖에 없다.

그래서 '맞히기'가 아니라 '견디기'로 간다: 플랜트를 넓게 흔들면서 각 w 의 최악을 본다.
교차표에서 이상한 걸 봤다 - w 를 플랜트에 **정확히** 맞췄을 때보다 과대로 잡았을 때
오버슈트가 작았다. 사실이면 w 는 '맞히는 값'이 아니라 감쇠 다이얼이고, 모르는 쪽으로
안전하게 치우칠 수 있다는 뜻이다.
"""
import math
import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base

FPS, BASE, SIG_DT = 231.0, 143.0, 0.295
T, TB = 1000.0/FPS, 1000.0/BASE
R = T/TB

S143 = {"kp_x":0.765,"kp_y":0.698,"soft_x":8.6,"soft_y":6.57,"kd_x":0.052,"kd_y":0.037,
        "max_step":19.56,"ff":1.7,"beta":0.02,"predict":3.18,"v_ema":0.235,
        "vgate":14.79,"ff_err_gate":22.25,"mincut":0.084}
P = dict(S143)
for k in ("max_step","vgate","kp_x","kp_y","v_ema","mincut"): P[k] *= R
for k in ("predict","ff"): P[k] /= R


def ev(w, plant, n=32, lo=500):
    realjit.dt_sample = (lambda m: (lambda rng: m*math.exp(rng.gauss(0., SIG_DT))))(plant)
    p = base(**dict(P, w=w, ego_lag=w))
    k, vs = FPS/BASE, BASE/FPS; o = {}
    for nm, s_, kw in (("step","step",dict(noise=False,frames=int(150*k))),
                       ("hold","hold",dict(frames=int(420*k))),("rev","reversal",dict(frames=int(300*k))),
                       ("t1","track",dict(vx=1.1*vs,frames=int(480*k))),("t4","track",dict(vx=4.0*vs,frames=int(480*k))),
                       ("t8","track",dict(vx=8.0*vs,frames=int(480*k))),
                       ("r40","reach",dict(noise=False,step_dist=40.,frames=int(220*k)))):
        acc = {}
        for s in range(lo, lo+n):
            for kk, v in run_real(p, s_, s, **kw).items(): acc.setdefault(kk, []).append(v)
        for kk, v in acc.items(): o[nm+"_"+kk] = sum(v)/len(v)
    return sc(o), o["step_overshoot"], o["r40_reach"]*T


def main():
    PLANTS = [1.85, 2.37, 2.90, 3.40]      # 8.0~14.7ms @231fps - 있을 법한 범위
    WS = [2.10, 2.37, 2.60, 2.85, 3.10]
    print("  231fps. 행=컨트롤러 w, 열=실제 플랜트 데드타임(프레임)\n")
    print("  오버슛")
    print("  %6s" % "w", "".join("%9.2f" % p for p in PLANTS), "   최악")
    rows = {}
    for w in WS:
        r = [ev(w, p) for p in PLANTS]
        rows[w] = r
        ov = [x[1] for x in r]
        print("  %6.2f" % w, "".join("%9.2f" % v for v in ov), "  %6.2f" % max(ov))
    print("\n  획득 40px (ms)")
    print("  %6s" % "w", "".join("%9.2f" % p for p in PLANTS), "   최악")
    for w in WS:
        rc = [x[2] for x in rows[w]]
        print("  %6.2f" % w, "".join("%9.1f" % v for v in rc), "  %6.1f" % max(rc))
    print("\n  총에러")
    for w in WS:
        er = [x[0] for x in rows[w]]
        print("  %6.2f" % w, "".join("%9.3f" % v for v in er), "  최악 %.3f  폭 %.1f%%"
              % (max(er), 100*(max(er)-min(er))/min(er)))


if __name__ == "__main__":
    main()
