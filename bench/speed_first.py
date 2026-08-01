#!/usr/bin/env python3
"""링잉을 현재 이하로 못박고 '획득 속도'를 목적함수로 최적화.

사용자 판정: 지금 설정이 링잉은 좋은데 속도가 답답하다. 그래서 목적을 뒤집는다.
지금까지는 전부 에러(RMS)를 목적으로 놓고 획득을 제약으로만 썼다 - 즉 한 번도
'빠르게'를 직접 최적화한 적이 없다.

주의(메모리 결론): 이 시스템에서 빠른 획득과 오버슈트는 같은 메커니즘이라 max_step
같은 단일 노브로는 트레이드밖에 안 된다. 같은 링잉에서 더 빠르려면 감쇠(kd),
데드타임 보정(w), 리드(ff/predict/게이트)가 같이 움직여야 한다. 그 지점이 실제로
있는지가 이 스크립트의 질문이다.

두 단계로 본다:
  A) 링잉·에러 모두 무회귀 -> 순수 공짜 속도가 있는가
  B) 링잉 무회귀, 에러는 +2% 까지 허용 -> 에러를 조금 팔면 얼마나 빨라지는가
"""
import json
import random

from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX, common

random.seed(24680)

BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r80", "reach", dict(noise=False, step_dist=80.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))

# 현재 stable = 기준선
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
    B = (b['r40_reach'], b['r80_reach'], b['r120_reach'])

    def speed(m):
        """획득 시간. 근거리에 더 큰 가중 - 실전 재조준은 대부분 근거리."""
        return (0.40*m['r40_reach']/B[0] + 0.35*m['r80_reach']/B[1]
                + 0.25*m['r120_reach']/B[2])

    def mk_ok(err_allow):
        def ok(m):
            return (m['step_overshoot'] <= b['step_overshoot']   # 링잉 무회귀
                    and m['step_osc'] <= max(0.05, b['step_osc'])
                    and m['rev_peak'] <= b['rev_peak']
                    and sc(m) <= sc(b)*err_allow)
        return ok

    def refine(p0, ok, iters=400):
        p, e = dict(p0), speed(metrics(p0))
        step, since = 0.25, 0
        for _ in range(iters):
            q = dict(p)
            for k, (lo, hi) in BOX.items():
                if random.random() < 0.35:
                    q[k] = min(hi, max(lo, q[k] + random.gauss(0, step*(hi-lo))))
            m = metrics(q)
            if ok(m) and speed(m) < e:
                p, e, since = q, speed(m), 0
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008:
                        break
        return p, e

    print("  기준(seed 0-15): 오버슛 %.2f  반전pk %.2f  에러 %.3f  r40/80/120 %.2f/%.2f/%.2f\n"
          % (b['step_overshoot'], b['rev_peak'], sc(b), *B))

    results = {}
    for tier, allow in (("A 에러도 무회귀", 1.00), ("B 에러 +2% 허용", 1.02)):
        ok = mk_ok(allow)
        starts = [("현재설정", base(**SHIP))]
        tried = 0
        while len(starts) < 4 and tried < 700:
            tried += 1
            q = base(**common(), beta=round(random.uniform(0, .12), 4))
            if ok(metrics(q, n=8)):
                starts.append(("랜덤%d" % (len(starts)), q))
        best = None
        for tag, p0 in starts:
            p1, s1 = refine(p0, ok)
            m = metrics(p1)
            if ok(m) and (best is None or s1 < best[0]):
                best = (s1, p1)
        if best:
            results[tier] = best[1]
            m = metrics(best[1])
            print("  [%s] 출발점 %d개 -> 속도지수 %.4f (기준 1.0), r40 %.2f r120 %.2f"
                  % (tier, len(starts), best[0], m['r40_reach'], m['r120_reach']))
        else:
            print("  [%s] 통과 후보 없음 (출발점 %d개)" % (tier, len(starts)))

    if not results:
        print("\n  링잉 무회귀 상태로 빨라지는 지점이 없음 = 트레이드가 전부")
        return
    print("\n  확정 평가 (독립 seed 두 블록, 모든 행 같은 블록)")
    print("  %-16s %8s %7s %7s  %5s %5s %6s"
          % ("구성", "에러", "오버슛", "반전pk", "r40", "r80", "r120"))
    print("  " + "-" * 66)
    for lo in (500, 700):
        print("  --- seed %d-%d ---" % (lo, lo+59))
        for lbl, p in [("현재 stable", base(**SHIP))] + [(k, v) for k, v in results.items()]:
            m = metrics(p, lo=lo, n=60)
            print("  %-16s %8.3f %7.2f %7.2f  %5.2f %5.2f %6.2f"
                  % (lbl, sc(m), m['step_overshoot'], m['rev_peak'],
                     m['r40_reach'], m['r80_reach'], m['r120_reach']))
    for k, v in results.items():
        print("\n  %s: %s" % (k, json.dumps({j: round(v[j], 4) for j in BOX if j in v})))


if __name__ == "__main__":
    main()
