#!/usr/bin/env python3
"""thumb(사이드버튼) 프로필 최적화 — 이 프로필은 한 번도 튜닝된 적이 없다.

프로필별로 갈리는 값은 6개뿐이다(kp_x/y, softness_x/y, kd_x/y). 나머지(데드타임 보정,
리드, One Euro)는 공유다. 그런데 우클릭 프로필은 이번 세션까지 여러 번 재최적화된 반면
thumb 은 초기값 그대로다:

    main   kp 0.765/0.698  softness 8.6/6.57  kd 0.052/0.037
    thumb  kp 0.6  /0.62   softness 11.0/10.0 kd 0.18 /0.22

kd 가 메인의 4~5배다. "데드타임 하에서 D항은 감쇠가 아니라 링잉을 키운다"는 게 이
프로젝트의 확립된 결론(실측 kd 0.18->0.10 에서 오버슈트 0.5->0.3회)인데 thumb 만 그
이전 값에 남아 있다. 사용자가 "근거리에서 thumb 쓸 때 잘 안 된다"고 한 것과 맞는다.

배포된 공유 설정(적응 데드타임 포함) 위에서 6개 값만 탐색한다.
"""
import json
import random

from realjit import sc
from ctrl_zoo_opt import base
from gap_fix import SHIP, metrics

random.seed(1207)

# 프로필별 6개 값만 (나머지는 SHIP 공유값 그대로)
BOX = {"kp_x": (0.35, 1.15), "kp_y": (0.35, 1.15),
       "soft_x": (4.0, 20.0), "soft_y": (4.0, 20.0),
       "kd_x": (0.0, 0.30), "kd_y": (0.0, 0.30)}
THUMB = {"kp_x": 0.6, "kp_y": 0.62, "soft_x": 11.0, "soft_y": 10.0,
         "kd_x": 0.18, "kd_y": 0.22}


def M(d, **kw):
    return metrics(base(**dict(SHIP, **dict(d, **kw))), "ring", 0.015, **{})


def main():
    cur = metrics(base(**dict(SHIP, **THUMB)), "ring", 0.015, blocks=(0,), n=16)
    ship = metrics(base(**SHIP), "ring", 0.015, blocks=(0,), n=16)
    print("  우클릭 프로필(참고): 에러 %.3f 오버슛 %.2f r40 %.2f"
          % (sc(ship), ship['step_overshoot'], ship['r40_reach']))
    print("  thumb 현재       : 에러 %.3f 오버슛 %.2f r40 %.2f\n"
          % (sc(cur), cur['step_overshoot'], cur['r40_reach']))

    def ok(m):
        # thumb 은 근거리 전용이므로 링잉을 우클릭 프로필 수준으로 죈다.
        return (m['step_overshoot'] <= ship['step_overshoot']
                and m['step_osc'] <= 0.35
                and m['rev_peak'] <= cur['rev_peak']
                and m['r40_reach'] <= cur['r40_reach'])

    def ev(d, n=16, lo=0):
        return metrics(base(**dict(SHIP, **d)), "ring", 0.015, blocks=(lo,), n=n)

    best = None
    for tag, start in (("현재값", dict(THUMB)),
                       ("우클릭값", {k: SHIP[k] for k in BOX})):
        p, e = dict(start), sc(ev(start))
        if ok(ev(p)):
            best = best or (e, dict(p))
        step, since = 0.25, 0
        for _ in range(260):
            q = dict(p)
            for k, (lo_, hi) in BOX.items():
                if random.random() < 0.45:
                    q[k] = min(hi, max(lo_, q[k] + random.gauss(0, step*(hi-lo_))))
            m = ev(q)
            if ok(m) and sc(m) < e:
                p, e, since = q, sc(m), 0
                if best is None or e < best[0]:
                    best = (e, dict(q))
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008:
                        break
        print("  %-10s 출발 -> %.3f" % (tag, e))

    if not best:
        print("\n  통과 후보 없음")
        return
    print("\n  확정 (세 블록 x60)")
    print("  %-22s %8s %6s %6s %6s %6s %6s"
          % ("thumb 프로필", "에러", "오버슛", "반전pk", "hold", "t4", "r40"))
    print("  " + "-" * 62)
    ref = None
    for lbl, d in (("현재 (미튜닝)", THUMB), ("최적화", best[1])):
        m = metrics(base(**dict(SHIP, **d)), "ring", 0.015)
        if ref is None:
            ref = m
        dd = "" if m is ref else "  %+.1f%%" % (100*(sc(m)-sc(ref))/sc(ref))
        print("  %-22s %8.3f %6.2f %6.2f %6.2f %6.2f %6.2f%s"
              % (lbl, sc(m), m['step_overshoot'], m['rev_peak'], m['hold_rms'],
                 m['t4_rms'], m['r40_reach'], dd))
    print("\n  최적 thumb 게인 (320 공간): " +
          json.dumps({k: round(best[1][k], 4) for k in BOX}))


if __name__ == "__main__":
    main()
