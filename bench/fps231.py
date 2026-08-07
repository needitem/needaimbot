#!/usr/bin/env python3
"""실전 플레이에서 잰 레이트로 환산을 다시 정정한다.

오늘 두 번 틀렸다. 처음엔 중앙 간격(229)을 써서 246 을 가정했고, 그 다음엔 정지 표적
수집의 평균(206)으로 '정정'했다. 그런데 **정지 표적 수집이 대표성이 없었다** - 지터가
플레이의 두 배(간격 표준편차 1.61 vs 0.74)고 레이트가 낮다.

실전 401초(클럭 고정, needaimbot.sh):
    전체 평균 4.22ms -> 237 fps,  중앙 4.04 -> 247,  조준 중 4.33 -> 231
    30초 구간별 226~246 으로 안정, 교전이 238->231 로 거의 안 떨어뜨린다.

컨트롤러는 조준 중에만 루프를 닫으므로 **231** 을 쓴다(237 과는 2.6% 차이라 무의미).
따라서 정정 전(246, r=0.581)이 오늘의 '정정'(206, r=0.694)보다 오히려 가까웠다.

플랜트 데드타임: 11.60ms 는 143fps 실측이고 캡처 샘플링분 T/2=3.50 을 빼면 8.10ms 가
레이트 무관분. 231fps(T=4.33)에서 8.10+2.16=10.26ms -> 2.37 프레임.
"""
import math
import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base

FPS, BASE, SIG_DT = 231.0, 143.0, 0.295
T, TB = 1000.0/FPS, 1000.0/BASE
PLANT_W = (11.60 - 0.5*TB + 0.5*T)/T

S143 = {"kp_x":0.765,"kp_y":0.698,"soft_x":8.6,"soft_y":6.57,"kd_x":0.052,"kd_y":0.037,
        "max_step":19.56,"ff":1.7,"beta":0.02,"predict":3.18,"v_ema":0.235,
        "vgate":14.79,"ff_err_gate":22.25,"mincut":0.084}
MUL = ("max_step","vgate","kp_x","kp_y","v_ema","mincut")
DIV = ("predict","ff")


def rescale(fps):
    r = (1000.0/fps)/TB
    p = dict(S143)
    for k in MUL: p[k] *= r
    for k in DIV: p[k] /= r
    return p


def ev(p, plant, n=40, lo=500):
    realjit.dt_sample = (lambda m: (lambda rng: m*math.exp(rng.gauss(0., SIG_DT))))(plant)
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
    print("  실제 플랜트: %.0f fps (T=%.2fms), 데드타임 %.2f 프레임 (%.1fms)\n"
          % (FPS, T, PLANT_W, PLANT_W*T))
    cands = [("배포 중 (206 환산) w=2.85", rescale(206.0), 2.85),
             ("정정 전 (246 환산) w=2.85", rescale(246.0), 2.85),
             ("231 환산       w=2.85", rescale(FPS), 2.85),
             ("231 환산       w=2.37", rescale(FPS), PLANT_W)]
    print("  %-26s %8s %7s %8s %8s" % ("설정", "에러", "오버슛", "반전pk", "획득40"))
    print("  " + "-"*62)
    for lbl, p, w in cands:
        m = ev(dict(p, w=w, ego_lag=w), PLANT_W)
        print("  %-26s %8.3f %7.2f %8.2f %7.1fms"
              % (lbl, sc(m), m['step_overshoot'], m['rev_peak'], m['r40_ms']))
    print("\n  참고 환산비: 206 -> %.3f | 231 -> %.3f | 246 -> %.3f"
          % ((1000/206)/TB, (1000/FPS)/TB, (1000/246)/TB))
    q = rescale(FPS)
    print("  231 환산값: kp %.3f/%.3f  max_step %.2f  vgate %.2f  predict %.2f  ff %.3f"
          % (q["kp_x"], q["kp_y"], q["max_step"], q["vgate"], q["predict"], q["ff"]))


if __name__ == "__main__":
    main()
