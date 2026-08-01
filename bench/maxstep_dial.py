#!/usr/bin/env python3
"""링잉 감소가 감쇠 개선인가, 그냥 걸음을 묶은 것인가.

링잉 우선 최적화가 max_step 을 탐색 하한(15)까지 밀어붙였다. 걸음 상한을 줄이면
오버슈트는 기계적으로 줄어든다 - 넘어갈 수 있는 거리 자체가 작아지므로. 그건 감쇠가
좋아진 게 아니라 느려진 것이고, 목적함수가 링잉에 45% 를 주면 그 방향으로 무한히
끌려간다(퇴화 해). 그렇다면 결론은 "새 튜닝"이 아니라 "max_step 이 곧 링잉/속도
다이얼"이다 - 그게 사실이면 사용자가 체감으로 직접 돌릴 수 있는 노브 하나가 된다.

승격설정에서 max_step 만 바꿔가며 재고, 링잉우선 전체 설정과 나란히 놓는다.
같은 링잉을 max_step 단독으로 얻을 수 있으면 나머지 14개 값은 무의미하다.
"""
from realjit import run_real, sc
from ctrl_zoo_opt import base

SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}
RING = {"kp_x": 0.9924, "kp_y": 1.0848, "soft_x": 11.8108, "soft_y": 11.3429,
        "kd_x": 0.124, "kd_y": 0.1012, "max_step": 15, "w": 1.429,
        "ff": 1.7338, "predict": 1.7191, "v_ema": 0.2596, "vgate": 11.0581,
        "ff_err_gate": 17.5089, "mincut": 0.3001, "beta": 0.0}
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
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


def row(lbl, p, ref=None):
    m = metrics(p)
    e = sc(m)
    d = "" if ref is None else " %+6.1f%%" % (100*(m['rev_peak']-ref)/ref)
    print("  %-26s %8.3f %7.2f%s %7.2f %7.2f  %5.2f %6.2f"
          % (lbl, e, m['rev_peak'], d, m['step_overshoot'], m['hold_rms'],
             m['r40_reach'], m['r120_reach']))
    return m


def main():
    print("  두 블록(500-559, 700-759) 합산 평균\n")
    print("  %-26s %8s %7s %8s %7s %7s  %5s %6s"
          % ("구성", "에러", "반전pk", "vs기준", "오버슛", "hold", "r40", "r120"))
    print("  " + "-" * 80)
    b = row("승격설정 (max_step 23.5)", base(**SHIP))
    ref = b['rev_peak']
    for ms in (20.0, 18.0, 15.0, 13.0):
        row("  └ max_step %.1f 만 변경" % ms, base(**dict(SHIP, max_step=ms)), ref)
    print()
    row("링잉우선 (14개값+ms 15)", base(**RING), ref)
    row("  └ 그 설정에서 ms 23.5 로", base(**dict(RING, max_step=23.5)), ref)


if __name__ == "__main__":
    main()
