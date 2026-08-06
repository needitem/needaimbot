#!/usr/bin/env python3
"""2026-08-03 리그 실측 조건에서 재튜닝 (240Hz 모니터 교체 후).

여태 sim 은 데드타임을 1.10 프레임으로 모델링했다. 정지 표적 + 스텝 주입으로 처음
실측해 보니 **2.85 프레임(11.60ms)** 이었다 - 부품 합산(USB+렌더대기+E2E=6.42ms)이
놓친 ~5ms 는 게임 자체의 렌더 파이프라인 지연으로 보인다. 부품을 더해서는 알 수 없는
값이고, 7개 분할에서 2.79~2.99 (표준편차 0.07) 로 재현된다.

즉 지금까지의 모든 튜닝은 **실제보다 2.6배 빠른 플랜트**를 가정하고 이뤄졌다.
여기서 실측 조건으로 갈아끼우고 다시 찾는다.

  프레임 주기   4.07 ms (246 fps)      실측
  데드타임      2.85 프레임            실측 (스텝 응답)
  지터 sigma    0.295 (로그정규)       실측 E2E p50/p95 에서
  표적 속도     화면 속도 고정으로 환산   (143Hz 기준 대비 x143/246)
"""
import json, math, random, sys
import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX

random.seed(20260803)

FPS, DEAD_FR, SIG = 246.0, 2.85, 0.295
BASE_FPS = 143.0
BOX = dict(COMMON_BOX, beta=(0.0, 0.12), w=(0.0, 4.0), predict=(0.0, 6.0))
SHIP = {"kp_x": 0.765, "kp_y": 0.698, "soft_x": 8.6, "soft_y": 6.57,
        "kd_x": 0.052, "kd_y": 0.037, "max_step": 19.56, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 3.18, "v_ema": 0.235,
        "vgate": 14.79, "ff_err_gate": 22.25, "mincut": 0.084}

realjit.dt_sample = lambda rng: DEAD_FR * math.exp(rng.gauss(0.0, SIG))


def ev(p, n=24, lo=500):
    T = 1000.0 / FPS
    k = FPS / BASE_FPS
    vs = BASE_FPS / FPS
    spec = (("step", "step", dict(noise=False, frames=int(150*k))),
            ("hold", "hold", dict(frames=int(420*k))),
            ("rev", "reversal", dict(frames=int(300*k))),
            ("t1", "track", dict(vx=1.1*vs, frames=int(480*k))),
            ("t4", "track", dict(vx=4.0*vs, frames=int(480*k))),
            ("t8", "track", dict(vx=8.0*vs, frames=int(480*k))),
            ("r40", "reach", dict(noise=False, step_dist=40., frames=int(220*k))),
            ("r120", "reach", dict(noise=False, step_dist=120., frames=int(220*k))))
    o = {}
    for nm, s_, kw in spec:
        acc = {}
        for s in range(lo, lo + n):
            m = run_real(p, s_, s, **kw)
            for kk, v in m.items():
                acc.setdefault(kk, []).append(v)
        for kk, v in acc.items():
            o[nm + "_" + kk] = sum(v)/len(v)
    for nm in ("r40", "r120"):
        o[nm + "_ms"] = o[nm + "_reach"] * T
    return o


def rescale():
    """물리 환산 + w 는 실측 데드타임으로 직접."""
    r = (1000.0/FPS) / (1000.0/BASE_FPS)
    q = dict(base(**SHIP))
    for k in ("max_step", "vgate", "kp_x", "kp_y", "v_ema", "mincut"):
        q[k] *= r
    q["predict"] = min(6.0, q["predict"] / r)
    q["ff"] = q["ff"] / r
    q["w"] = min(4.0, DEAD_FR)
    return q


def main():
    cur = ev(base(**SHIP), n=40)
    T = 1000.0/FPS
    print("  실측 조건: %.0f fps (프레임 %.2fms), 데드타임 %.2f 프레임 (%.1fms), 지터 s=%.3f\n"
          % (FPS, T, DEAD_FR, DEAD_FR*T, SIG))
    print("  %-30s %8s %6s %6s   %8s %8s"
          % ("구성", "에러", "오버슛", "반전pk", "획득40", "획득120"))
    print("  " + "-" * 72)
    print("  %-30s %8.3f %6.2f %6.2f   %7.1fms %7.1fms"
          % ("현재 배포 (143Hz 기준값)", sc(cur), cur['step_overshoot'],
             cur['rev_peak'], cur['r40_ms'], cur['r120_ms']))

    rs = rescale()
    m = ev(rs, n=40)
    print("  %-30s %8.3f %6.2f %6.2f   %7.1fms %7.1fms"
          % ("환산 + w=실측 2.85", sc(m), m['step_overshoot'],
             m['rev_peak'], m['r40_ms'], m['r120_ms']))

    # w 만 실측값으로 (다른 건 그대로) - 어느 쪽이 이득의 출처인지 가른다
    wonly = dict(base(**SHIP)); wonly["w"] = DEAD_FR
    mw = ev(wonly, n=40)
    print("  %-30s %8.3f %6.2f %6.2f   %7.1fms %7.1fms"
          % ("w 만 2.85 로 (게인 그대로)", sc(mw), mw['step_overshoot'],
             mw['rev_peak'], mw['r40_ms'], mw['r120_ms']))

    def ok(x):
        return (x['step_overshoot'] <= cur['step_overshoot']
                and x['step_osc'] <= max(0.05, cur['step_osc'])
                and x['rev_peak'] <= cur['rev_peak']
                and sc(x) <= sc(cur))

    def score(x):
        return x['r40_ms']

    best = None
    for tag, seed0 in (("환산", rs), ("w만", wonly), ("현재", base(**SHIP))):
        p = dict(seed0)
        m0 = ev(p, n=12, lo=0)
        e = score(m0)
        if ok(m0) and (best is None or e < best[0]):
            best = (e, dict(p))
        step, since = 0.25, 0
        for _ in range(300):
            q = dict(p)
            for k, (lo_, hi) in BOX.items():
                if random.random() < 0.35:
                    q[k] = min(hi, max(lo_, q[k] + random.gauss(0, step*(hi-lo_))))
            mm = ev(q, n=12, lo=0)
            if ok(mm) and (best is None or score(mm) < best[0]):
                best = (score(mm), dict(q)); p, e, since = q, score(mm), 0
            elif ok(mm) and score(mm) < e:
                p, e, since = q, score(mm), 0
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008: break

    if best is None:
        print("\n  제약 통과 지점 없음")
        return
    mb = ev(best[1], n=40)
    print("  %-30s %8.3f %6.2f %6.2f   %7.1fms %7.1fms   획득40 %+.0f%%"
          % ("실측조건 재최적화", sc(mb), mb['step_overshoot'], mb['rev_peak'],
             mb['r40_ms'], mb['r120_ms'], 100*(mb['r40_ms']-cur['r40_ms'])/cur['r40_ms']))
    print("\n  값 (320 공간): " + json.dumps({k: round(best[1][k], 4) for k in BOX if k in best[1]}))
    x2 = {"soft_x", "soft_y", "vgate", "ff_err_gate"}
    print("\n  160 프리셋:")
    for k in BOX:
        if k in best[1]:
            v = best[1][k]
            print("    %-14s %s" % (k, round(v*2, 3) if k in x2 else round(v, 4)))


if __name__ == "__main__":
    main()
