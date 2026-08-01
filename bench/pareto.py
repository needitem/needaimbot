#!/usr/bin/env python3
"""매끄러움 <-> 스냅 파레토 곡선.

지금까지 나온 것:
  스냅 사수(r40 +2% 이내)  -> 에러 -1.3%
  스냅 무시               -> 에러 -4.6%, 단 r40 +46%
둘 중 어느 쪽이 낫다고 sim 이 말해줄 수 없다(추적정밀도와 재조준속도는 다른 축이고,
어느 쪽이 실전에서 중요한지는 플레이가 정한다). 그래서 근거리 획득 허용치를 단계적으로
풀면서 각 단계의 최선 에러를 구한다 = 선택지를 수치로 만든다.
"""
import json
import random

from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX

random.seed(2024)

CAND = {"kp_x": 0.7797, "kp_y": 0.611, "soft_x": 9.0161, "soft_y": 15.9729,
        "kd_x": 0.1635, "kd_y": 0.0173, "max_step": 29.8496, "w": 1.523,
        "ff": 1.2777, "predict": 1.1475, "v_ema": 0.2476, "vgate": 2.5634,
        "ff_err_gate": 23.8051, "mincut": 0.2238, "beta": 0.0006}
BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))


def metrics(p, lo=0, n=12):
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
    cur = base()
    b = metrics(cur)
    rows = []
    for allow in (1.02, 1.10, 1.20, 1.35, 1.50):
        def ok(m, a=allow):
            return (m['step_overshoot'] <= b['step_overshoot'] + 0.05
                    and m['step_osc'] <= 0.35
                    and m['rev_peak'] <= b['rev_peak']*1.03
                    and m['r40_reach'] <= b['r40_reach']*a
                    and m['r120_reach'] <= b['r120_reach']*a)
        best = None
        for p0 in (cur, base(**CAND)):
            if not ok(metrics(p0)):
                p0 = cur
            p, e = dict(p0), sc(metrics(p0))
            step, since = 0.30, 0
            for _ in range(220):
                q = dict(p)
                for k, (lo, hi) in BOX.items():
                    if random.random() < 0.35:
                        q[k] = min(hi, max(lo, q[k] + random.gauss(0, step*(hi-lo))))
                m = metrics(q)
                if ok(m) and sc(m) < e:
                    p, e, since = q, sc(m), 0
                else:
                    since += 1
                    if since >= 12:
                        step *= 0.6; since = 0
                        if step < 0.01:
                            break
            if best is None or e < best[0]:
                best = (e, p)
        rows.append((allow, best[1]))
        print("  허용 %+.0f%% 탐색 완료" % (100*(allow-1)))

    print("\n  확정 평가 seed 500-559 (탐색 미사용)")
    print("  %-22s %8s %7s  %5s %6s %5s %6s"
          % ("구성", "에러", "vs현재", "오버슛", "반전pk", "r40", "r120"))
    print("  " + "-" * 72)
    m = metrics(cur, lo=500, n=60); ref = sc(m)
    print("  %-22s %8.3f %7s  %5.2f %6.2f %5.2f %6.2f"
          % ("현재 배포", ref, "-", m['step_overshoot'], m['rev_peak'],
             m['r40_reach'], m['r120_reach']))
    out = {}
    for allow, p in rows:
        m = metrics(p, lo=500, n=60); e = sc(m)
        tag = "스냅허용 %+.0f%%" % (100*(allow-1))
        print("  %-22s %8.3f %+6.1f%%  %5.2f %6.2f %5.2f %6.2f"
              % (tag, e, 100*(e-ref)/ref, m['step_overshoot'], m['rev_peak'],
                 m['r40_reach'], m['r120_reach']))
        out[tag] = {k: round(p[k], 4) for k in BOX if k in p}
    m = metrics(base(**CAND), lo=500, n=60)
    print("  %-22s %8.3f %+6.1f%%  %5.2f %6.2f %5.2f %6.2f"
          % ("스냅 무시(참고)", sc(m), 100*(sc(m)-ref)/ref, m['step_overshoot'],
             m['rev_peak'], m['r40_reach'], m['r120_reach']))
    print()
    for k, v in out.items():
        print("  %s: %s" % (k, json.dumps(v)))


if __name__ == "__main__":
    main()
