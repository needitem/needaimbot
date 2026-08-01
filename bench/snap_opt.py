#!/usr/bin/env python3
"""스냅(근거리 획득)을 잃지 않으면서 에러를 줄이는 지점을 찾는다.

앞 결과: beta 를 풀면 에러 -4.6% 가 나오지만 20-60px 재조준이 45% 느려진다. 실전
재조준은 대부분 근거리이므로 받아들일 수 없는 대가다. 그래서 reach 를 '먼 거리 하나'가
아니라 '근거리 위주'로 재고 제약에 넣는다.

앞 스크립트의 제약 판정 결함도 고친다: 제약을 seed 10개로 재고 seed 40개로 재판정하면
현재 설정조차 탈락한다(스텝 시나리오도 데드타임 seed 에 따라 변한다). 여기서는 제약과
점수를 같은 seed 집합에서 재고, 확정만 미사용 seed 로 한다.
"""
import json
import random

from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX, common
from aim_opt import OptCtrl

random.seed(777)

CAND = {"kp_x": 0.7797, "kp_y": 0.611, "soft_x": 9.0161, "soft_y": 15.9729,
        "kd_x": 0.1635, "kd_y": 0.0173, "max_step": 29.8496, "w": 1.523,
        "ff": 1.2777, "predict": 1.1475, "v_ema": 0.2476, "vgate": 2.5634,
        "ff_err_gate": 23.8051, "mincut": 0.2238, "beta": 0.0006}
BOX = dict(COMMON_BOX, beta=(0.0, 0.12))


def metrics(p, ctrl=None, lo=0, n=12):
    o = {}
    scn = (("step", "step", dict(noise=False)), ("hold", "hold", {}),
           ("rev", "reversal", {}), ("t1", "track", dict(vx=1.1)),
           ("t4", "track", dict(vx=4.0)), ("t8", "track", dict(vx=8.0)),
           ("r40", "reach", dict(noise=False, step_dist=40.)),
           ("r120", "reach", dict(noise=False, step_dist=120.)))
    for nm, s_, kw in scn:
        acc = {}
        for s in range(lo, lo + n):
            m = run_real(p, s_, s, ctrl=ctrl, **kw)
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            o[nm + "_" + k] = sum(v) / len(v)
    return o


def main():
    cur = base()
    b = metrics(cur)
    print("  기준(seed 0-11): 에러 %.3f  오버슛 %.2f  반전pk %.2f  r40 %.2f  r120 %.2f"
          % (sc(b), b['step_overshoot'], b['rev_peak'], b['r40_reach'], b['r120_reach']))

    def ok(m):
        return (m['step_overshoot'] <= b['step_overshoot'] + 0.05
                and m['step_osc'] <= 0.35
                and m['rev_peak'] <= b['rev_peak']*1.03
                and m['r40_reach'] <= b['r40_reach']*1.02      # 스냅 사수
                and m['r120_reach'] <= b['r120_reach']*1.05)

    def refine2(p0, iters=300):
        p, e = dict(p0), sc(metrics(p0))
        step, since = 0.30, 0
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
                if since >= 12:
                    step *= 0.6; since = 0
                    if step < 0.01:
                        break
        return p, e

    starts = [("현재설정", cur), ("beta만 0", base(beta=0.0))]
    tried = 0
    while len(starts) < 5 and tried < 500:
        tried += 1
        q = base(**common(), beta=round(random.uniform(0, .12), 4))
        if ok(metrics(q, n=6)):
            starts.append(("랜덤%d" % (len(starts)-1), q))
    print("  출발점 %d개 (랜덤 후보 %d회 시도)\n" % (len(starts), tried))

    best = None
    for tag, p0 in starts:
        p1, e1 = refine2(p0)
        print("  %-10s -> %.3f (%+.1f%%)" % (tag, e1, 100*(e1-sc(b))/sc(b)))
        if best is None or e1 < best[0]:
            best = (e1, p1)

    print("\n  확정 평가 seed 500-559")
    print("  %-26s %8s  %5s %6s %5s %6s %6s"
          % ("구성", "에러", "오버슛", "반전pk", "r40", "r120", "hold"))
    print("  " + "-" * 74)
    ref = None
    for lbl, p in (("현재 배포", cur), ("스냅 무시 후보(-4.6%)", base(**CAND)),
                   ("스냅 사수 최적", best[1])):
        m = metrics(p, lo=500, n=60)
        e = sc(m)
        if ref is None:
            ref = e
        print("  %-26s %8.3f  %5.2f %6.2f %5.2f %6.2f %6.2f  %+5.1f%%"
              % (lbl, e, m['step_overshoot'], m['rev_peak'], m['r40_reach'],
                 m['r120_reach'], m['hold_rms'], 100*(e-ref)/ref))
    print("\n  스냅 사수 최적 파라미터:")
    print("  " + json.dumps({k: round(best[1][k], 4) for k in BOX if k in best[1]}))


if __name__ == "__main__":
    main()
