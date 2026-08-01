#!/usr/bin/env python3
"""실제로 배포할 (반올림된) 값으로 이득이 남는지 확인.

탐색 최적점에는 sim 해상도 이하의 변화가 섞여 있다 (beta 0.02 -> 0.0216, 데드타임
1.25 -> 1.2397, ff 1.7 -> 1.7374). 그런 자리를 설정에 박으면 diff 만 지저분해지고
의미는 없다. 그래서 그 셋은 현재값으로 되돌리고, 의미 있게 움직인 값만 남긴 뒤
'실제로 넣을 설정' 그 자체를 다시 잰다. 최적점이 아니라 배포본을 검증해야 한다.
"""
from realjit import run_real, sc
from ctrl_zoo_opt import base

FULL = {"kp_x": 0.75, "kp_y": 0.7441, "soft_x": 9.3297, "soft_y": 11.0253,
        "kd_x": 0.0407, "kd_y": 0.054, "max_step": 23.4887, "w": 1.2397,
        "ff": 1.7374, "predict": 2.936, "v_ema": 0.2244, "vgate": 11.1355,
        "ff_err_gate": 19.6875, "mincut": 0.2252, "beta": 0.0216}

# 배포본: sim 해상도 이하 변화(beta/w/ff)는 현재값 유지, 나머지는 설정에 적기 좋게 반올림
SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5,
        "w": 1.25, "ff": 1.7, "beta": 0.02,          # <- 현재값 유지
        "predict": 2.94, "v_ema": 0.224, "vgate": 11.14,
        "ff_err_gate": 19.69, "mincut": 0.225}

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
    rows = (("현재 배포", base()), ("탐색 최적점", base(**FULL)),
            ("배포본(반올림)", base(**SHIP)))
    for lo in (500, 700, 900):
        print("  --- seed %d-%d ---" % (lo, lo+59))
        print("  %-18s %8s %8s  %5s %5s %6s %5s %6s %6s"
              % ("구성", "에러", "vs현재", "오버슛", "osc", "반전pk", "r40", "r120", "hold"))
        print("  " + "-" * 76)
        ref = None
        for lbl, p in rows:
            m = metrics(p, lo); e = sc(m)
            if ref is None:
                ref = e
            print("  %-18s %8.3f %8s  %5.2f %5.2f %6.2f %5.2f %6.2f %6.2f"
                  % (lbl, e, "-" if e == ref else "%+.1f%%" % (100*(e-ref)/ref),
                     m['step_overshoot'], m['step_osc'], m['rev_peak'],
                     m['r40_reach'], m['r120_reach'], m['hold_rms']))
        print()

    print("  160 프리셋 값 (softness/vgate/err_gate 만 x2):")
    x2 = {"soft_x", "soft_y", "vgate", "ff_err_gate"}
    for k, v in SHIP.items():
        print("    %-14s %s" % (k, round(v*2, 3) if k in x2 else v))


if __name__ == "__main__":
    main()
