#!/usr/bin/env python3
"""속도 최적화 A안 검증 후 배포값 확정.

두 가지가 걸린다:
  1) soft_x 가 탐색 하한 4 에 붙었다(9.33->4). 경계에 붙은 값은 '최적'이 아니라
     '상자가 막은 지점'이라 상자를 넓히면 더 갈 수도, 사실은 그 근처가 평평해서
     아무 값이나 된 것일 수도 있다. 주변을 훑어 확인한다.
  2) 스텝 오버슈트가 holdout 에서 +3% 나빠졌다. 제약은 seed 0-15 에서 걸었는데
     확정은 다른 블록에서 재니 제약 추정 잡음이다. 사용자 조건이 "링잉은 그대로
     두거나 좋아지고" 이므로 이게 실제 회귀인지 봐야 한다.
확정은 탐색·앞선 검증 어디에도 안 쓴 seed 900-959 를 추가해 세 블록으로 한다.
"""
import json

from realjit import run_real, sc
from ctrl_zoo_opt import base

SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}
A = {"kp_x": 0.541, "kp_y": 0.701, "soft_x": 4.0, "soft_y": 12.9809,
     "kd_x": 0.0543, "kd_y": 0.0261, "max_step": 22.9269, "w": 1.1649,
     "ff": 1.9877, "predict": 1.8612, "v_ema": 0.2698, "vgate": 10.0404,
     "ff_err_gate": 17.2054, "mincut": 0.1223, "beta": 0.0187}
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r80", "reach", dict(noise=False, step_dist=80.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))


def metrics(p, blocks=(500, 700, 900), n=60):
    out = {}
    for lo in blocks:
        for nm, s_, kw in SCN:
            for s in range(lo, lo + n):
                m = run_real(p, s_, s, **kw)
                for k, v in m.items():
                    out.setdefault(nm + "_" + k, []).append(v)
    return {k: sum(v)/len(v) for k, v in out.items()}


def row(lbl, p, b=None):
    m = metrics(p)
    f = lambda k: "" if b is None else " (%+.1f%%)" % (100*(m[k]-b[k])/b[k])
    print("  %-22s %8.3f %6.2f %6.2f  %5.2f %5.2f %6.2f"
          % (lbl, sc(m), m['step_overshoot'], m['rev_peak'],
             m['r40_reach'], m['r80_reach'], m['r120_reach']))
    return m


def main():
    print("  세 블록(500/700/900 x60) 합산\n")
    print("  %-22s %8s %6s %6s  %5s %5s %6s"
          % ("구성", "에러", "오버슛", "반전pk", "r40", "r80", "r120"))
    print("  " + "-" * 64)
    b = row("현재 stable", base(**SHIP))
    a = row("A안 (탐색 그대로)", base(**A), b)

    print("\n  soft_x 민감도 (A안에서 이 값만 변경) — 경계 아티팩트인지 확인")
    print("  " + "-" * 64)
    for sx in (4.0, 5.5, 7.0, 9.33):
        row("  soft_x %.2f" % sx, base(**dict(A, soft_x=sx)), b)

    print("\n  변화 요약 (A안 vs 현재)")
    for k, lbl in (("step_overshoot", "스텝 오버슈트"), ("rev_peak", "반전 최대이탈"),
                   ("hold_rms", "정지 유지"), ("t8_rms", "전력질주 추적"),
                   ("r40_reach", "획득 40px"), ("r80_reach", "획득 80px"),
                   ("r120_reach", "획득 120px")):
        d = 100*(a[k]-b[k])/b[k]
        print("    %-14s %8.2f -> %8.2f  %+6.1f%%  %s"
              % (lbl, b[k], a[k], d, "나빠짐" if d > 1.0 else ("좋아짐" if d < -1.0 else "-")))
    print("    %-14s %8.3f -> %8.3f  %+6.1f%%" % ("에러(가중)", sc(b), sc(a),
                                                  100*(sc(a)-sc(b))/sc(b)))


if __name__ == "__main__":
    main()
