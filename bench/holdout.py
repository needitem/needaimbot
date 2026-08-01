#!/usr/bin/env python3
"""탐색에 쓰지 않은 seed 로 재평가 (과적합 검증).

ctrl_zoo_opt 의 정련은 seed 0..9 로 점수를 매긴다. 같은 seed 로 확정 평가까지 하면
탐색이 그 seed 들의 노이즈 실현값에 맞춰진 것을 실력으로 착각할 수 있다. 여기서는
seed 200.. 의 완전히 새로운 노이즈 실현으로 다시 잰다.
"""
import math
import random

from realjit import run_real, sc


def ev_holdout(p, ctrl=None, lo=200, n=40):
    o = {}
    for nm, scn, kw in (("step", "step", dict(noise=False)),
                        ("hold", "hold", {}),
                        ("rev", "reversal", {}),
                        ("t1", "track", dict(vx=1.1)),
                        ("t4", "track", dict(vx=4.0)),
                        ("t8", "track", dict(vx=8.0)),
                        ("reach", "reach", dict(noise=False, step_dist=120.))):
        acc = {}
        for s in range(lo, lo + n):
            m = run_real(p, scn, s, ctrl=ctrl, **kw)
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            o[nm + "_" + k] = sum(v) / len(v)
    return o
