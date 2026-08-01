#!/usr/bin/env python3
"""트레이드 없는 개선이 가능한가 = 파레토 곡선 자체를 옮길 수 있는가.

컨트롤러 안에서는 근거리 속도와 오버슈트가 같은 양이라 맞교환밖에 없다(이번 세션에서
soft_x/soft_y/kp_y/max_step 네 축 모두에서 확인). 그건 튜닝 실패가 아니라 노이즈 하
루프 게인의 성질이다: 게인을 올리면 반응과 노이즈 증폭이 같이 커진다.

그러나 곡선의 '위치'는 컨트롤러 밖의 두 값이 정한다 - 검출기 σ 와 데드타임. 이 둘이
줄면 같은 링잉에서 더 빠른 지점이 생긴다. 여기서 각 조건마다 '링잉을 현재 이하로
못박고 속도만 최적화'해서, 실제로 두 지표가 동시에 좋아지는지 확인한다.

조건은 전부 실현 경로가 있는 것만:
  데드타임 -30%  게임 fps 상향(렌더 대기가 데드타임 최대 성분) + INT8 엔진
  σ -15%        유휴 GPU 활용(추론 2회 평균/TTA). 모델 교체 아님.
"""
import json
import math
import random

import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX, common

random.seed(11235)

BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SCN = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
       ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
       ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
       ("r40", "reach", dict(noise=False, step_dist=40.)),
       ("r80", "reach", dict(noise=False, step_dist=80.)),
       ("r120", "reach", dict(noise=False, step_dist=120.)))
SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}

NX0, NY0 = realjit.NOISE_X, realjit.NOISE_Y
DT0 = realjit.dt_sample


def set_cond(dt_scale, n_scale):
    realjit.dt_sample = (lambda s: (lambda rng: 1.10*s*math.exp(rng.gauss(0., 0.335))))(dt_scale)
    realjit.NOISE_X, realjit.NOISE_Y = NX0*n_scale, NY0*n_scale


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
    # 기준 링잉 = 현재 조건·현재 설정
    set_cond(1.0, 1.0)
    b0 = metrics(base(**SHIP))
    RING_OVS, RING_REV = b0['step_overshoot'], b0['rev_peak']
    print("  링잉 상한(현재 설정·현재 조건): 오버슛 %.2f  반전pk %.2f"
          % (RING_OVS, RING_REV))
    print("  기준 속도: r40 %.2f  r80 %.2f  r120 %.2f  에러 %.3f\n"
          % (b0['r40_reach'], b0['r80_reach'], b0['r120_reach'], sc(b0)))

    def speed(m):
        return 0.45*m['r40_reach'] + 0.35*m['r80_reach'] + 0.20*m['r120_reach']

    def ok(m):
        return (m['step_overshoot'] <= RING_OVS and m['rev_peak'] <= RING_REV
                and m['step_osc'] <= 0.35 and sc(m) <= sc(b0))

    def refine(p0, iters=300):
        p, e = dict(p0), speed(metrics(p0))
        step, since = 0.25, 0
        for _ in range(iters):
            q = dict(p)
            for k, (lo, hi) in BOX.items():
                if random.random() < 0.35:
                    q[k] = min(hi, max(lo, q[k] + random.gauss(0, step*(hi-lo))))
            m = metrics(q)
            if ok(m) and speed(m) < e:
                p, e, since = q, speed(m), 0
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008:
                        break
        return p

    conds = (("현재 조건 (대조군)", 1.0, 1.00),
             ("데드타임 -30%", 0.70, 1.00),
             ("σ -15%", 1.00, 0.85),
             ("둘 다", 0.70, 0.85))
    print("  각 조건에서 '링잉·에러 무회귀' 하에 속도만 최적화한 결과")
    print("  %-20s %6s %6s  %5s %5s %6s %8s   %s"
          % ("조건", "오버슛", "반전pk", "r40", "r80", "r120", "에러", "판정"))
    print("  " + "-" * 84)
    out = {}
    for lbl, ds, ns in conds:
        set_cond(ds, ns)
        starts = [base(**SHIP)]
        tried = 0
        while len(starts) < 3 and tried < 400:
            tried += 1
            q = base(**common(), beta=round(random.uniform(0, .12), 4))
            if ok(metrics(q, n=8)):
                starts.append(q)
        best = None
        for p0 in starts:
            p1 = refine(p0)
            m = metrics(p1)
            if ok(m) and (best is None or speed(m) < best[0]):
                best = (speed(m), p1, m)
        if best is None:
            print("  %-20s %s" % (lbl, "통과 후보 없음"))
            continue
        _, p, m = best
        faster = (m['r40_reach'] < b0['r40_reach'] and m['r80_reach'] < b0['r80_reach'])
        verdict = ("동시개선 <==" if faster else "속도 못 얻음")
        print("  %-20s %6.2f %6.2f  %5.2f %5.2f %6.2f %8.3f   %s"
              % (lbl, m['step_overshoot'], m['rev_peak'], m['r40_reach'],
                 m['r80_reach'], m['r120_reach'], sc(m), verdict))
        out[lbl] = (p, m)

    print("\n  r40 개선폭 (링잉·에러를 하나도 안 내주고 얻은 속도)")
    for lbl, (p, m) in out.items():
        print("    %-20s %+6.1f%%" % (lbl, 100*(m['r40_reach']-b0['r40_reach'])/b0['r40_reach']))
    realjit.dt_sample = DT0
    realjit.NOISE_X, realjit.NOISE_Y = NX0, NY0
    for lbl, (p, m) in out.items():
        print("\n  %s: %s" % (lbl, json.dumps({k: round(p[k], 4) for k in BOX if k in p})))


if __name__ == "__main__":
    main()
