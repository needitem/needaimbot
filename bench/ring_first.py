#!/usr/bin/env python3
"""링잉을 목적함수에 넣고 최적화 (지금까지는 제약으로만 걸었다).

플레이 체감 판정: sim 이 -3% 라고 한 변경을 사용자가 "훨씬 좋다"고 했다. 그 변경에서
가장 크게 움직인 지표는 RMS 가 아니라 반전 최대이탈(-5.1%)과 정지유지(-4.3%)였다.
=> 지각되는 양은 RMS 가 아니라 링잉이다. 그런데 지금까지의 점수 sc() 는 오버슈트와
반전 최대이탈을 아예 세지 않고 '나빠지지만 마라'는 제약으로만 썼다. 목적을 바꾼다.

비용 배분도 체감에 맞춘다: 사용자 조건은 "너무 느려지지만 않으면". 파레토에서 링잉을
줄일 때 근거리 획득은 거의 안 상하고 원거리 획득만 크게 상하는 것이 관측됐으므로,
근거리(40px)는 죄고 원거리(120px)는 풀어준다. 에러는 현재보다 나빠지지 않게 못박는다.
"""
import json
import random

from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX, common

random.seed(8888)

BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))

# 방금 stable 로 승격된 설정 = 새 기준선
SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}


def metrics(p, lo=0, n=16):
    o = {}
    for nm, s_, kw in SCN:
        acc = {}
        for s in range(lo, lo + n):
            m = run_real(p, s_, s, **kw)
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            o[nm + "_" + k] = sum(v)/len(v)
    return o


def main():
    b = metrics(base(**SHIP))
    B = (b['rev_peak'], b['step_overshoot'], sc(b))

    def J(m):
        """링잉 우선 목적함수. 반전 최대이탈 45% + 스텝 오버슈트 35% + 에러 20%."""
        return (0.45*m['rev_peak']/B[0] + 0.35*m['step_overshoot']/B[1]
                + 0.20*sc(m)/B[2])

    def ok(m):
        return (sc(m) <= B[2]                       # 에러 회귀 금지
                and m['step_osc'] <= 0.35           # 지속 진동 금지
                and m['r40_reach'] <= b['r40_reach']*1.05   # 근거리 스냅 사수
                and m['r120_reach'] <= b['r120_reach']*1.35) # 원거리는 풀어줌

    def refine(p0, iters=400):
        p, e = dict(p0), J(metrics(p0))
        step, since = 0.25, 0
        for _ in range(iters):
            q = dict(p)
            for k, (lo, hi) in BOX.items():
                if random.random() < 0.35:
                    q[k] = min(hi, max(lo, q[k] + random.gauss(0, step*(hi-lo))))
            m = metrics(q)
            if ok(m) and J(m) < e:
                p, e, since = q, J(m), 0
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008:
                        break
        return p, e

    starts = [("승격설정", base(**SHIP))]
    tried = 0
    while len(starts) < 4 and tried < 600:
        tried += 1
        q = base(**common(), beta=round(random.uniform(0, .12), 4))
        if ok(metrics(q, n=8)):
            starts.append(("랜덤%d" % (len(starts)), q))
    print("  출발점 %d개 (랜덤 %d회 시도)" % (len(starts), tried))

    best = None
    for tag, p0 in starts:
        p1, j1 = refine(p0)
        m = metrics(p1)
        print("  %-10s -> J=%.4f  반전pk %.2f  오버슛 %.2f  에러 %.3f%s"
              % (tag, j1, m['rev_peak'], m['step_overshoot'], sc(m),
                 "" if ok(m) else " [이탈]"))
        if ok(m) and (best is None or j1 < best[0]):
            best = (j1, p1)

    if not best:
        print("\n  통과 후보 없음")
        return
    print("\n  확정 평가 (독립 seed 두 블록, 모든 행 같은 블록)")
    print("  %-14s %8s %7s %7s %7s  %5s %6s"
          % ("구성", "에러", "반전pk", "오버슛", "hold", "r40", "r120"))
    print("  " + "-" * 68)
    for lo in (500, 700):
        print("  --- seed %d-%d ---" % (lo, lo+59))
        for lbl, p in (("승격설정(기준)", base(**SHIP)), ("링잉우선", best[1])):
            m = metrics(p, lo=lo, n=60)
            print("  %-14s %8.3f %7.2f %7.2f %7.2f  %5.2f %6.2f"
                  % (lbl, sc(m), m['rev_peak'], m['step_overshoot'],
                     m['hold_rms'], m['r40_reach'], m['r120_reach']))
    print("\n  링잉우선 파라미터:")
    print("  " + json.dumps({k: round(best[1][k], 4) for k in BOX if k in best[1]}))


if __name__ == "__main__":
    main()
