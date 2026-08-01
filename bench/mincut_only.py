#!/usr/bin/env python3
"""무회귀 최적점의 이득 중 mincut 하나가 얼마나 차지하는가.

무회귀 탐색이 찾은 -3.0% 설정은 8개 값이 움직였는데, 그중 beta 는 현재값(0.02)으로
돌아왔고 가장 크게 움직인 것은 mincut 0.3 -> 0.225 였다. 값 하나만 바꿔 대부분을
얻을 수 있으면 그게 실전에 넣기 좋다 - 되돌리기 쉽고, 무엇이 효과를 냈는지도 분명하다.

전체 최적점과 mincut 단독 변경을 같은 블록에서 나란히 잰다.
"""
from realjit import run_real, sc
from ctrl_zoo_opt import base

FULL = {"kp_x": 0.75, "kp_y": 0.7441, "soft_x": 9.3297, "soft_y": 11.0253,
        "kd_x": 0.0407, "kd_y": 0.054, "max_step": 23.4887, "w": 1.2397,
        "ff": 1.7374, "predict": 2.936, "v_ema": 0.2244, "vgate": 11.1355,
        "ff_err_gate": 19.6875, "mincut": 0.2252, "beta": 0.0216}
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))


def metrics(p, lo, n=60):
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
    rows = [("현재 배포 (mincut 0.30)", base())]
    for mc in (0.28, 0.25, 0.225, 0.20, 0.17):
        rows.append(("mincut %.3f 단독" % mc, base(mincut=mc)))
    rows.append(("무회귀 최적 (8개값)", base(**FULL)))

    for lo in (500, 700):
        print("  --- seed %d-%d ---" % (lo, lo+59))
        print("  %-24s %8s %8s  %5s %6s %5s %6s"
              % ("구성", "에러", "vs현재", "오버슛", "반전pk", "r40", "r120"))
        print("  " + "-" * 70)
        ref = None
        for lbl, p in rows:
            m = metrics(p, lo)
            e = sc(m)
            if ref is None:
                ref = e
            print("  %-24s %8.3f %8s  %5.2f %6.2f %5.2f %6.2f"
                  % (lbl, e, "-" if e == ref else "%+.1f%%" % (100*(e-ref)/ref),
                     m['step_overshoot'], m['rev_peak'], m['r40_reach'],
                     m['r120_reach']))
        print()


if __name__ == "__main__":
    main()
