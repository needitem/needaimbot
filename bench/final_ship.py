#!/usr/bin/env python3
"""배포 후보 확정: 링잉·스냅 무회귀를 못박고 에러만 줄인다.

앞 실행(conv_opt)은 -4.2% 를 냈지만 오버슈트가 1.24 -> 1.62 (+31%) 였다. 제약을
'현재값 +0.10 절대' 로 느슨하게 준 탓이다. 사용자의 1순위는 링잉이므로 여기서는
오버슈트를 현재값 이하로 못박는다. 즉 어떤 지표도 나빠지지 않고 에러만 줄어드는
지점만 후보로 인정한다.

conv_opt 의 출력 버그도 고친다: 현재값 행은 in-sample 지표를, 후보 행은 holdout
지표를 찍어 비교가 어긋났다. 모든 행을 같은 블록에서 잰다.
"""
import json
import random

from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX

random.seed(13579)

BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))

STARTS = {
    "conv+10%": {"kp_x": 0.7395, "kp_y": 0.6469, "soft_x": 5.8099, "soft_y": 8.2498,
                 "kd_x": 0.0573, "kd_y": 0.0186, "max_step": 24.2544, "w": 1.3376,
                 "ff": 1.4657, "predict": 3.3402, "v_ema": 0.1923, "vgate": 7.2232,
                 "ff_err_gate": 11.5872, "mincut": 0.3276, "beta": 0.0029},
    "파레토+2%": {"kp_x": 0.75, "kp_y": 0.716, "soft_x": 7.7565, "soft_y": 9.5714,
                "kd_x": 0.0059, "kd_y": 0.0188, "max_step": 25.6017, "w": 1.3151,
                "ff": 2.1444, "predict": 1.7717, "v_ema": 0.2243, "vgate": 8.2421,
                "ff_err_gate": 13.3843, "mincut": 0.3415, "beta": 0.0067},
}


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


def show(lbl, m, ref=None):
    e = sc(m)
    d = "-" if ref is None else "%+.1f%%" % (100*(e-ref)/ref)
    print("  %-14s %8.3f %8s  %5.2f %5.2f %6.2f %5.2f %6.2f"
          % (lbl, e, d, m['step_overshoot'], m['step_osc'], m['rev_peak'],
             m['r40_reach'], m['r120_reach']))
    return e


def main():
    cur = base()
    b = metrics(cur)

    # 무회귀: 링잉·진동·반전피크·근거리스냅 모두 현재 이하. r120 만 +5% 허용.
    def ok(m):
        return (m['step_overshoot'] <= b['step_overshoot']
                and m['step_osc'] <= max(0.05, b['step_osc'])
                and m['rev_peak'] <= b['rev_peak']
                and m['r40_reach'] <= b['r40_reach']
                and m['r120_reach'] <= b['r120_reach']*1.05)

    def refine(p0, iters=400):
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

    print("  기준 오버슛 %.2f  osc %.2f  반전pk %.2f  r40 %.2f  r120 %.2f (seed 0-15)"
          % (b['step_overshoot'], b['step_osc'], b['rev_peak'],
             b['r40_reach'], b['r120_reach']))
    print("  제약: 오버슛/osc/반전pk/r40 무회귀, r120 +5% 까지\n")

    cands = []
    for tag, p0 in [("현재설정", cur)] + [(k, base(**v)) for k, v in STARTS.items()]:
        if not ok(metrics(p0)):
            print("  %-12s 출발점이 제약 이탈 -> 현재설정에서 출발" % tag)
            p0 = cur
        p1, e1 = refine(p0)
        f = ok(metrics(p1))
        print("  %-12s -> %.3f (%+.1f%%)%s"
              % (tag, e1, 100*(e1-sc(b))/sc(b), "" if f else " [이탈]"))
        if f:
            cands.append((e1, tag, p1))
    cands.sort(key=lambda x: x[0])

    print("\n  독립 두 블록 확정 (모든 행 같은 블록에서 측정)")
    print("  %-14s %8s %8s  %5s %5s %6s %5s %6s"
          % ("구성", "에러", "vs현재", "오버슛", "osc", "반전pk", "r40", "r120"))
    print("  " + "-" * 74)
    for lo in (500, 700):
        print("  --- seed %d-%d ---" % (lo, lo+59))
        ref = show("현재 배포", metrics(cur, lo=lo, n=60))
        for _, tag, p in cands[:2]:
            show(tag, metrics(p, lo=lo, n=60), ref)
    if cands:
        print("\n  최선 후보 (%s):" % cands[0][1])
        print("  " + json.dumps({k: round(cands[0][2][k], 4) for k in BOX
                                 if k in cands[0][2]}))


if __name__ == "__main__":
    main()
