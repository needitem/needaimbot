#!/usr/bin/env python3
"""206 fps 실측 레이트로 환산을 정정한다.

배포 중인 값은 246 fps 가정으로 환산돼 있는데, 실측은 206 fps 다(평균 프레임간격
4.85ms, perf R 중앙 208, 벽시계 201 - 전부 일치). 중앙 간격 4.38ms 가 229 를 가리켜
그걸 썼던 게 화근이다. 중앙값은 긴 꼬리를 무시하므로 평균 px/초를 결정하지 못한다.

두 군데가 틀렸다:
  1) 환산비 r = T_new/T_base 가 0.581 이어야 할 게 아니라 0.694
  2) w: 데드타임 11.60ms 는 143fps 에서 쟀는데 246fps 의 T 로 나눴다

11.60ms 중 캡처 샘플링분 T/2=3.50ms 는 레이트를 따라 줄고, 나머지 8.10ms
(USB + 게임 렌더 파이프라인)는 남는다. 206fps 에서는 8.10+2.43=10.53ms -> 2.17 프레임.
게임 렌더분도 fps 를 따라 줄 수 있으므로 2.17 은 상한이다.

플랜트의 진짜 데드타임을 모르므로 교차표로 본다: 플랜트 2.17/2.39 x 컨트롤러 w.
"""
import math
import realjit
from realjit import run_real, sc

FPS, BASE, SIG_DT = 206.0, 143.0, 0.295
T, TB = 1000.0/FPS, 1000.0/BASE
R = T/TB

# 143fps 에서 튜닝이 끝났던 값 (환산의 출발점)
S143 = {"kp_x":0.765,"kp_y":0.698,"soft_x":8.6,"soft_y":6.57,"kd_x":0.052,"kd_y":0.037,
        "max_step":19.56,"ff":1.7,"beta":0.02,"predict":3.18,"v_ema":0.235,
        "vgate":14.79,"ff_err_gate":22.25,"mincut":0.084}
# 246 가정으로 환산돼 배포 중인 값
SHIP = {"kp_x":0.445,"kp_y":0.406,"soft_x":8.6,"soft_y":6.57,"kd_x":0.052,"kd_y":0.037,
        "max_step":11.37,"ff":2.924,"beta":0.02,"predict":5.47,"v_ema":0.137,
        "vgate":8.60,"ff_err_gate":22.25,"mincut":0.049}


def rescale(r):
    """물리 환산. px/프레임 xr, 프레임수 /r, 프레임당계수 xr, 속도곱 /r, px 불변."""
    p = dict(S143)
    for k in ("max_step", "vgate", "kp_x", "kp_y", "v_ema", "mincut"): p[k] *= r
    for k in ("predict", "ff"): p[k] /= r
    return p


def ev(p, plant_w, n=40, lo=500):
    from ctrl_zoo_opt import base
    realjit.dt_sample = (lambda m: (lambda rng: m*math.exp(rng.gauss(0., SIG_DT))))(plant_w)
    k, vs = FPS/BASE, BASE/FPS; o = {}
    for nm, s_, kw in (("step","step",dict(noise=False,frames=int(150*k))),
                       ("hold","hold",dict(frames=int(420*k))),("rev","reversal",dict(frames=int(300*k))),
                       ("t1","track",dict(vx=1.1*vs,frames=int(480*k))),("t4","track",dict(vx=4.0*vs,frames=int(480*k))),
                       ("t8","track",dict(vx=8.0*vs,frames=int(480*k))),
                       ("r40","reach",dict(noise=False,step_dist=40.,frames=int(220*k)))):
        acc = {}
        for s in range(lo, lo+n):
            for kk, v in run_real(base(**p), s_, s, **kw).items(): acc.setdefault(kk, []).append(v)
        for kk, v in acc.items(): o[nm+"_"+kk] = sum(v)/len(v)
    o["r40_ms"] = o["r40_reach"]*T
    return o


def main():
    from ctrl_zoo_opt import base
    print("  환산비 r = %.4f  (배포본은 0.581 로 환산됨)\n" % R)
    fix = rescale(R)
    print("  %-12s %10s %10s %8s" % ("파라미터", "배포(246)", "정정(206)", "차이"))
    print("  " + "-"*44)
    for k in ("kp_x","kp_y","max_step","vgate","predict","v_ema","mincut","ff"):
        print("  %-12s %10.3f %10.3f %+7.1f%%"
              % (k, SHIP[k], fix[k], 100*(fix[k]-SHIP[k])/SHIP[k]))

    print("\n  플랜트 데드타임을 모르므로 교차표 (행=진짜 플랜트, 열=컨트롤러 w)\n")
    for plant in (2.17, 2.39):
        print("  ── 플랜트 %.2f 프레임 (%.1f ms) ──" % (plant, plant*T))
        print("    %-26s %8s %7s %8s %8s" % ("설정", "에러", "오버슛", "반전pk", "획득40"))
        for lbl, p, w in ([("배포(246환산) w=2.85", SHIP, 2.85)]
                          + [("정정(206환산) w=%.2f" % w, fix, w) for w in (2.85, 2.39, 2.17, 1.95)]):
            m = ev(dict(p, w=w, ego_lag=w), plant)
            print("    %-26s %8.3f %7.2f %8.2f %7.1fms"
                  % (lbl, sc(m), m['step_overshoot'], m['rev_peak'], m['r40_ms']))
        print()


if __name__ == "__main__":
    main()
