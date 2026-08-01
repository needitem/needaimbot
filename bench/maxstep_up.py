#!/usr/bin/env python3
"""max_step 다이얼의 '위쪽'을 측정 (아래쪽만 재봤다).

maxstep_dial.py 는 23.5 에서 아래로만(20/18/15/13) 쟀다. 사용자가 이제 '더 빠르게'를
원하므로 위쪽 값이 필요하다. 이 다이얼은 트레이드이므로 위로 올리면 링잉이 나빠질 텐데,
얼마나 나빠지고 얼마나 빨라지는지를 알아야 speed_first 의 탐색 결과와 비교가 된다
(탐색이 찾은 지점이 '그냥 max_step 올린 것'과 다르지 않으면 의미가 없다).

참고: 이전 배포값이 25 였고 그때 사용자는 링잉을 문제로 느끼지 않았다. 즉 25 정도는
허용 범위일 가능성이 있다.
"""
from realjit import run_real, sc
from ctrl_zoo_opt import base

SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r80", "reach", dict(noise=False, step_dist=80.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))


def metrics(p, blocks=(500, 700), n=60):
    out = {}
    for lo in blocks:
        for nm, s_, kw in SCN:
            for s in range(lo, lo + n):
                m = run_real(p, s_, s, **kw)
                for k, v in m.items():
                    out.setdefault(nm + "_" + k, []).append(v)
    return {k: sum(v)/len(v) for k, v in out.items()}


def main():
    print("  두 블록(500-559, 700-759) 합산 평균. 화살표 = 기준 대비\n")
    print("  %-22s %8s %7s %7s   %5s %5s %6s"
          % ("max_step", "에러", "오버슛", "반전pk", "r40", "r80", "r120"))
    print("  " + "-" * 68)
    b = metrics(base(**SHIP))
    print("  %-22s %8.3f %7.2f %7.2f   %5.2f %5.2f %6.2f"
          % ("23.5 (현재 stable)", sc(b), b['step_overshoot'], b['rev_peak'],
             b['r40_reach'], b['r80_reach'], b['r120_reach']))
    for ms in (25.0, 28.0, 32.0, 38.0):
        m = metrics(base(**dict(SHIP, max_step=ms)))
        print("  %-22s %8.3f %7.2f %7.2f   %5.2f %5.2f %6.2f   링잉%+5.1f%% 속도%+5.1f%%"
              % ("%.1f" % ms, sc(m), m['step_overshoot'], m['rev_peak'],
                 m['r40_reach'], m['r80_reach'], m['r120_reach'],
                 100*(m['rev_peak']-b['rev_peak'])/b['rev_peak'],
                 100*(m['r80_reach']-b['r80_reach'])/b['r80_reach']))
    print("\n  참고: 이전 배포값 25.0 은 kp_y/softness_y 가 달랐으므로 위 25.0 행과 다름")


if __name__ == "__main__":
    main()
