#!/usr/bin/env python3
"""-4.6% 후보의 유일한 대가인 reach 회귀를 값매김한다.

확정 평가에서 획득시간이 6.2 -> 7.2 프레임 늘었다. 다만 그 수치는 120px 스텝 하나에서
잰 것이라 '먼 표적만 느려진 것'인지 '전 거리에서 느려진 것'인지 알 수 없다. 실제
교전에서 흔한 것은 짧은 재조준이므로 거리별로 나눠 본다. 동시에 reach 를 현재값 이하로
못박은 탐색을 여러 출발점에서 돌려, 대가 없이 이득만 남는 지점이 있는지 확인한다.
"""
import json
import random

from realjit import ev, sc, run_real
from ctrl_zoo_opt import base, COMMON_BOX, refine, common

random.seed(20260730)

CAND = {"kp_x": 0.7797, "kp_y": 0.611, "soft_x": 9.0161, "soft_y": 15.9729,
        "kd_x": 0.1635, "kd_y": 0.0173, "max_step": 29.8496, "w": 1.523,
        "ff": 1.2777, "predict": 1.1475, "v_ema": 0.2476, "vgate": 2.5634,
        "ff_err_gate": 23.8051, "mincut": 0.2238, "beta": 0.0006}


def reach_at(p, dist, lo=500, n=60):
    v = [run_real(p, "reach", s, noise=False, step_dist=dist)["reach"]
         for s in range(lo, lo + n)]
    return sum(v)/len(v)


def main():
    cur = base()
    cand = base(**CAND)
    print("  [1] 거리별 획득시간 (프레임, 노이즈 없음, seed 500-559)")
    print("  %-10s %10s %10s %8s" % ("스텝거리", "현재", "후보", "차이"))
    print("  " + "-" * 42)
    for d in (20., 40., 60., 90., 120., 180.):
        a, b = reach_at(cur, d), reach_at(cand, d)
        print("  %8.0fpx %10.2f %10.2f %+8.2f" % (d, a, b, b-a))

    print("\n  [2] reach 를 현재값 이하로 못박은 탐색 (대가 없는 지점이 있는가)")
    m0 = ev(cur, seeds=10)
    ok = lambda m: (m['step_overshoot'] <= m0['step_overshoot'] + 0.05
                    and m['step_osc'] <= 0.35
                    and m['rev_peak'] <= m0['rev_peak'] * 1.03
                    and m['reach_reach'] <= m0['reach_reach'])
    box = dict(COMMON_BOX, beta=(0.0, 0.12))
    starts = [("현재설정", cur)]
    tried = 0
    while len(starts) < 4 and tried < 400:      # 제약을 만족하는 랜덤 출발점 확보
        tried += 1
        q = base(**common(), beta=round(random.uniform(0, .12), 4))
        if ok(ev(q, seeds=6)):
            starts.append(("랜덤%d" % (len(starts)), q))
    e_cur = sc(ev(cur, seeds=40))
    best = None
    print("  %-14s %10s %8s" % ("출발점", "에러", "vs현재"))
    print("  " + "-" * 36)
    for tag, p0 in starts:
        p1, _ = refine(p0, sc(ev(p0, seeds=10)), box, None, ok, iters=250, seeds=10)
        m = ev(p1, seeds=40)
        if not ok(m):
            print("  %-14s %10s" % (tag, "제약 이탈"))
            continue
        e = sc(m)
        print("  %-14s %10.3f %+7.1f%%" % (tag, e, 100*(e-e_cur)/e_cur))
        if best is None or e < best[0]:
            best = (e, p1)
    if best:
        print("\n  reach 무회귀 최선: %+.1f%%" % (100*(best[0]-e_cur)/e_cur))
        print("  " + json.dumps({k: round(best[1][k], 4) for k in box if k in best[1]}))


if __name__ == "__main__":
    main()
