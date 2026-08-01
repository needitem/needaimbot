#!/usr/bin/env python3
"""이득을 코드 변경 없이 설정만으로 얻을 수 있는지 확인하고, beta 를 넣어 재최적화.

앞의 발견: 이긴 것은 칼만이 아니라 alpha 고정 EMA 였다. One Euro 는 beta=0 이면
cutoff 가 상수가 되어 정확히 고정 EMA 다 (alpha = 2*pi*fc / (2*pi*fc + 1)). 즉
필요한 것은 CUDA 포팅이 아니라 oneeuro_beta 0.02 -> 0 과 min_cutoff 재설정뿐일 수 있다.

동시에 이번 세션의 모든 탐색이 beta 를 0.02 에 고정해 뒀다는 점을 바로잡는다. beta 는
속도 적응 계수인데, 여기서 '속도'는 sigma 8px 노이즈의 차분이라 신호가 아니라 잡음으로
구동된다. 그래서 0 이 최적일 수 있다 - 확인해 본 적이 없을 뿐이다.
"""
import json
import math
import random

from realjit import ev, sc
from holdout import ev_holdout
from ctrl_zoo_opt import base, COMMON_BOX, refine

random.seed(909)

KFCA_GAINS = {"kp_x": 0.71, "kp_y": 0.764, "soft_x": 10.993, "soft_y": 14.79,
              "kd_x": 0.1288, "kd_y": 0.0367, "max_step": 24.256, "w": 1.591,
              "ff": 1.6282, "predict": 2.6993, "v_ema": 0.2165, "vgate": 5.7821,
              "ff_err_gate": 27.14}


def fc_for_alpha(a):
    return a / (2.0 * math.pi * (1.0 - a))


def line(lbl, p, ref=None):
    m = ev_holdout(p)
    e = sc(m)
    d = "" if ref is None else "  %+6.1f%%" % (100*(e-ref)/ref)
    print("  %-40s %8.3f  %5.2f %6.2f %5.1f%s"
          % (lbl, e, m['step_overshoot'], m['rev_peak'], m['reach_reach'], d))
    return e, m


def main():
    cur = base()
    print("  %-40s %8s  %5s %6s %5s"
          % ("구성 (holdout seed 200-239)", "에러", "오버슛", "반전pk", "reach"))
    print("  " + "-" * 74)
    ref, m_cur = line("현재 배포 (mincut 0.3, beta 0.02)", cur)

    fc = fc_for_alpha(0.5802)
    print("\n  [1] EMA 등가 설정으로 재현되는가  (alpha 0.5802 -> min_cutoff %.4f)" % fc)
    print("  " + "-" * 74)
    line("KFCA게인 + beta=0, mincut %.4f" % fc,
         base(**KFCA_GAINS, mincut=fc, beta=0.0), ref)
    line("KFCA게인 + beta=0.02, mincut %.4f" % fc,
         base(**KFCA_GAINS, mincut=fc, beta=0.02), ref)
    line("현재게인 + beta=0, mincut %.4f" % fc, base(mincut=fc, beta=0.0), ref)

    print("\n  [2] beta 를 탐색에 넣고 재최적화 (설정만 바꾸는 범위)")
    print("  " + "-" * 74)
    box = dict(COMMON_BOX, beta=(0.0, 0.12))
    ok = lambda m: (m['step_overshoot'] <= m_cur['step_overshoot'] + 0.05
                    and m['step_osc'] <= 0.35
                    and m['rev_peak'] <= m_cur['rev_peak'] * 1.03
                    and m['reach_reach'] <= m_cur['reach_reach'] * 1.10)
    starts = [("현재설정 출발", base()),
              ("EMA등가 출발", base(**KFCA_GAINS, mincut=fc, beta=0.0))]
    best = None
    for tag, p0 in starts:
        e0 = sc(ev(p0, seeds=10))
        p1, e1 = refine(p0, e0, box, None, ok, iters=200, seeds=10)
        e, m = line("ES 정련 (%s)" % tag, p1, ref)
        if best is None or e < best[0]:
            best = (e, p1, m)

    print("\n  최종: %+.1f%%  (에러 %.3f)" % (100*(best[0]-ref)/ref, best[0]))
    keys = list(COMMON_BOX) + ["beta"]
    print("  " + json.dumps({k: round(best[1][k], 4) for k in keys if k in best[1]}))


if __name__ == "__main__":
    main()
