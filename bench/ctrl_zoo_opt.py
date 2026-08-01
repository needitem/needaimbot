#!/usr/bin/env python3
"""각 대안 기법을 '제대로 튜닝해서' 현재 구성과 비교.

앞선 1차 비교는 불공정했다: 현재 구성은 이번 세션 내내 최적화된 값인데 대안들은
손으로 찍은 파라미터 2~3개만 시험했다. 여기서는 기법마다 고유 파라미터 + 기본
게인(kp/softness/kd/max_step/데드타임/리드)을 함께 무작위 탐색하고, 상위 후보를
많은 seed 로 재검증한다. 기법이 지는 것인지, 튜닝이 모자랐던 것인지 가른다.
"""
import random
import sys

from realjit import ev, sc
from aim_opt import base_params, OptCtrl
from ctrl_zoo import PIDCtrl, HoltCtrl, KFCACtrl, SchedCtrl

random.seed(6060)


def base(**kw):
    d = dict(ego_lag=2.25, v_ema=0.2, vgate=9., ff_err_gate=18., pred_vgate=9.,
             pred_err_gate=18., mincut=0.3, dcut=1.0, w=1.25, ff=1.7, predict=3.2)
    d.update(kw)
    return base_params(**d)


def common():
    """기법과 무관한 공통 게인 탐색 범위."""
    return dict(
        kp_x=round(random.uniform(.55, 1.15), 3), kp_y=round(random.uniform(.55, 1.15), 3),
        soft_x=round(random.uniform(5, 16), 2), soft_y=round(random.uniform(5, 16), 2),
        kd_x=round(random.uniform(0, .16), 3), kd_y=round(random.uniform(0, .16), 3),
        max_step=round(random.uniform(18, 38), 1),
        w=round(random.uniform(.9, 1.7), 3),
        ff=round(random.uniform(0, 2.4), 3),
        predict=round(random.uniform(0, 4.2), 3),
        v_ema=round(random.uniform(.12, .45), 3),
        vgate=round(random.uniform(4, 14), 2),
        ff_err_gate=round(random.uniform(10, 30), 2),
        mincut=round(random.uniform(.1, .6), 3),
    )


# 공통 게인의 (하한, 상한) — 국소 정련이 벗어나지 못하는 상자
COMMON_BOX = dict(kp_x=(.5, 1.3), kp_y=(.5, 1.3), soft_x=(4, 20), soft_y=(4, 20),
                  kd_x=(0, .2), kd_y=(0, .2), max_step=(15, 45), w=(.8, 2.0),
                  ff=(0, 3.0), predict=(0, 5.0), v_ema=(.08, .6), vgate=(2, 20),
                  ff_err_gate=(6, 40), mincut=(.05, .8))

TECHS = {
    "PID": (PIDCtrl, dict(ki=(.001, .08), i_band=(5, 50), i_clamp=(3, 40))),
    "HOLT": (HoltCtrl, dict(holt_a=(.15, .9), holt_b=(.005, .45))),
    "KF-CA": (KFCACtrl, dict(kf_R=(10, 300), kf_Q=(.02, 8))),
    "SCHED": (SchedCtrl, dict(sched_gain=(.02, 1.5), sched_vref=(1.0, 14))),
    "BASE": (None, {}),      # 현재 구조도 같은 예산으로 재탐색 (공정 비교)
}


def sample_box(box):
    return {k: round(random.uniform(lo, hi), 4) for k, (lo, hi) in box.items()}


def refine(p0, e0, box, ctrl, ok, iters, seeds):
    """(1+1) ES: 상자 안에서 스텝을 줄여가며 국소 최적화. 랜덤 스크리닝만으로는
    17차원에서 봉우리 근처까지만 가고 정상에는 못 올라가므로 이 단계가 필요하다."""
    p, e = dict(p0), e0
    step = 0.30
    since = 0
    for _ in range(iters):
        q = dict(p)
        for k, (lo, hi) in box.items():
            if random.random() < 0.35:           # 매번 일부 축만 흔든다
                q[k] = min(hi, max(lo, q[k] + random.gauss(0, step*(hi-lo))))
        m = ev(q, seeds=seeds, ctrl=ctrl)
        if ok(m) and sc(m) < e:
            p, e, since = q, sc(m), 0
        else:
            since += 1
            if since >= 12:                      # 정체 시 스텝 축소
                step *= 0.6; since = 0
                if step < 0.01:
                    break
    return p, e


def main():
    n_screen = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    cur = ev(base())
    RING = cur['step_overshoot'] + 0.05
    REV = cur['rev_peak'] * 1.03
    RCH = cur['reach_reach'] * 1.10
    print("기준(현재 구성): 오버슈트 %.2f  반전pk %.2f  reach %.1f  에러 %.3f"
          % (cur['step_overshoot'], cur['rev_peak'], cur['reach_reach'], sc(cur)))
    print("제약: 오버슈트<=%.2f, 반전pk<=%.2f, reach<=%.1f | 기법당 %d 샘플\n"
          % (RING, REV, RCH, n_screen))

    def ok(m):
        return (m['step_overshoot'] <= RING and m['step_osc'] <= 0.35
                and m['rev_peak'] <= REV and m['reach_reach'] <= RCH)

    import json
    results = {}
    for name, (ctrl, tbox) in TECHS.items():
        box = dict(COMMON_BOX, **tbox)
        pool = []
        for _ in range(n_screen):
            p = base(**common(), **sample_box(tbox))
            m = ev(p, seeds=6, ctrl=ctrl)
            if ok(m):
                pool.append((sc(m), p))
        pool.sort(key=lambda x: x[0])
        if not pool:
            print("  %-6s 스크리닝 통과 0/%d" % (name, n_screen))
            continue
        # 상위 3개 시드에서 각각 국소 정련 -> 그 중 최고를 많은 seed 로 확정
        cands = []
        for _, p0 in pool[:3]:
            p1, e1 = refine(p0, sc(ev(p0, seeds=10, ctrl=ctrl)), box, ctrl, ok,
                            iters=70, seeds=10)
            cands.append((e1, p1))
        cands.sort(key=lambda x: x[0])
        best = None
        for _, p in cands:
            m = ev(p, seeds=40, ctrl=ctrl)
            if ok(m) and (best is None or sc(m) < best[0]):
                best = (sc(m), p, m)
        if best is None:
            print("  %-6s 정련 후 제약 통과 실패 (스크리닝 %d/%d)"
                  % (name, len(pool), n_screen))
            continue
        e, p, m = best
        results[name] = (e, p, m, ctrl, box)
        print("  %-6s 스크리닝 %3d/%d  ->  정련후 에러 %.3f (오버슈트 %.2f, 반전pk %.2f)"
              % (name, len(pool), n_screen, e, m['step_overshoot'], m['rev_peak']))

    # 현재 구조를 같은 예산으로 재탐색한 BASE 가 기준선. 대안은 BASE 를 이겨야 의미가 있다.
    ref = results.get("BASE", (sc(cur),))[0]
    print("\n  %-6s %8s %8s %8s %8s %9s  %8s %8s"
          % ("기법", "오버슈트", "반전pk", "hold", "t8", "에러", "vs현재", "vs BASE"))
    print("  " + "-" * 78)
    for name, (e, p, m, ctrl, box) in sorted(results.items(), key=lambda x: x[1][0]):
        print("  %-6s %8.2f %8.2f %8.2f %8.2f %9.3f  %+7.1f%% %+7.1f%%%s"
              % (name, m['step_overshoot'], m['rev_peak'], m['hold_rms'], m['t8_rms'],
                 e, 100*(e-sc(cur))/sc(cur), 100*(e-ref)/ref,
                 "  <== BASE 초과" if e < ref*0.995 else ""))
    keys = list(COMMON_BOX)
    for name, (e, p, m, ctrl, box) in sorted(results.items(), key=lambda x: x[1][0]):
        ks = keys + [k for k in box if k not in COMMON_BOX]
        print("\n  %s: %s" % (name, json.dumps({k: p[k] for k in ks if k in p})))


if __name__ == "__main__":
    main()
