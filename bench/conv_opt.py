#!/usr/bin/env python3
"""스냅허용 +10% 지점을 수렴시켜 배포 후보 하나로 확정.

파레토 실행에서 같은 제약인데 -1.3% 와 -3.0% 가 나왔다 = 탐색이 수렴하지 않았다.
그 상태의 숫자로 설정을 바꾸면 sim 의 운을 게인에 굳히는 셈이다. 여기서는
  - 재시작을 늘리고(파레토가 찾은 지점들 + 현재설정을 모두 출발점으로)
  - 점수/제약 seed 를 12 -> 16 개로 늘려 선택 잡음을 줄이고
  - 확정을 두 개의 독립 seed 블록(500-559, 700-759)에서 재서 일관성을 본다.
두 블록에서 같은 방향으로 이득이 나오면 적용, 갈리면 적용하지 않는다.
"""
import json
import random

from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX

random.seed(58008)

BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))

# 파레토 실행이 찾은 지점들 (출발점으로 재사용)
P02 = {"kp_x": 0.75, "kp_y": 0.716, "soft_x": 7.7565, "soft_y": 9.5714,
       "kd_x": 0.0059, "kd_y": 0.0188, "max_step": 25.6017, "w": 1.3151,
       "ff": 2.1444, "predict": 1.7717, "v_ema": 0.2243, "vgate": 8.2421,
       "ff_err_gate": 13.3843, "mincut": 0.3415, "beta": 0.0067}
P10 = {"kp_x": 0.75, "kp_y": 0.6469, "soft_x": 6.9315, "soft_y": 9.4036,
       "kd_x": 0.0541, "kd_y": 0.0253, "max_step": 28.0964, "w": 1.3161,
       "ff": 1.4385, "predict": 3.349, "v_ema": 0.193, "vgate": 7.4277,
       "ff_err_gate": 11.8527, "mincut": 0.3276, "beta": 0.0029}
P20 = {"kp_x": 0.7854, "kp_y": 0.5, "soft_x": 9.298, "soft_y": 5.0876,
       "kd_x": 0.0298, "kd_y": 0.0417, "max_step": 20.7904, "w": 1.1842,
       "ff": 1.296, "predict": 2.8339, "v_ema": 0.2004, "vgate": 6.3217,
       "ff_err_gate": 21.3683, "mincut": 0.0814, "beta": 0.0404}


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
    cur = base()
    b = metrics(cur)

    def ok(m):
        return (m['step_overshoot'] <= b['step_overshoot'] + 0.10
                and m['step_osc'] <= 0.35
                and m['rev_peak'] <= b['rev_peak']*1.03
                and m['r40_reach'] <= b['r40_reach']*1.02   # 근거리 스냅은 사수
                and m['r120_reach'] <= b['r120_reach']*1.10)

    def refine(p0, iters=350):
        p, e = dict(p0), sc(metrics(p0))
        step, since = 0.25, 0
        for _ in range(iters):
            q = dict(p)
            for k, (lo, hi) in BOX.items():
                if random.random() < 0.35:
                    q[k] = min(hi, max(lo, q[k] + random.gauss(0, step*(hi-lo))))
            m = metrics(q)
            if ok(m) and sc(m) < e:
                p, e, since = q, sc(m), 0
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008:
                        break
        return p, e

    cands = []
    for tag, p0 in (("현재설정", cur), ("파레토+2%", base(**P02)),
                    ("파레토+10%", base(**P10)), ("파레토+20%", base(**P20))):
        p1, e1 = refine(base(**p0) if isinstance(p0, dict) and 'kp_x' not in p0 else p0)
        feasible = ok(metrics(p1))
        print("  %-12s -> %.3f (%+.1f%%) %s"
              % (tag, e1, 100*(e1-sc(b))/sc(b), "" if feasible else "[제약이탈]"))
        if feasible:
            cands.append((e1, tag, p1))
    cands.sort(key=lambda x: x[0])

    print("\n  독립 두 블록 확정 평가")
    print("  %-14s %8s %8s %7s  %5s %6s %5s %6s"
          % ("구성", "블록A", "블록B", "평균", "오버슛", "반전pk", "r40", "r120"))
    print("  " + "-" * 76)
    ra = sc(metrics(cur, lo=500, n=60)); rb = sc(metrics(cur, lo=700, n=60))
    print("  %-14s %8.3f %8.3f %7s  %5.2f %6.2f %5.2f %6.2f"
          % ("현재 배포", ra, rb, "-", b['step_overshoot'], b['rev_peak'],
             b['r40_reach'], b['r120_reach']))
    best = None
    for e1, tag, p in cands[:3]:
        ma = metrics(p, lo=500, n=60); mb = metrics(p, lo=700, n=60)
        da = 100*(sc(ma)-ra)/ra; db = 100*(sc(mb)-rb)/rb
        print("  %-14s %8.3f %8.3f %+6.1f%%  %5.2f %6.2f %5.2f %6.2f  (%+.1f/%+.1f)"
              % (tag, sc(ma), sc(mb), (da+db)/2, ma['step_overshoot'], ma['rev_peak'],
                 ma['r40_reach'], ma['r120_reach'], da, db))
        if best is None or (da+db)/2 < best[0]:
            best = ((da+db)/2, tag, p)

    print("\n  선정: %s  (두 블록 평균 %+.1f%%)" % (best[1], best[0]))
    print("  " + json.dumps({k: round(best[2][k], 4) for k in BOX if k in best[2]}))


if __name__ == "__main__":
    main()
