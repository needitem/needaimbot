#!/usr/bin/env python3
"""이번 재조정 11개 값 중 '무엇이' 속도를 깎았는지 항목별 귀속.

사용자는 재조정 후 링잉은 좋아졌지만 속도가 답답하다고 했다. 재조정은 획득을
40px +4.3% / 120px +6.5% 늘렸는데, 11개가 동시에 움직였으니 범인이 누군지 모른다.
범인이 하나뿐이고 그것만 되돌려도 링잉 이득이 남으면 그게 최선의 수정이다.

각 값을 하나씩 이전 배포값으로 되돌려(나머지는 새 값 유지) 속도와 링잉을 함께 본다.
'속도 회복은 크고 링잉 손실은 작은' 항목이 있으면 그것만 되돌린다.
"""
from realjit import run_real, sc
from ctrl_zoo_opt import base

# 이전 배포값(= sim 기본값)과 현재 stable 값
OLD = {"kp_y": 0.82, "soft_x": 9.0, "soft_y": 8.0, "kd_x": 0.05, "kd_y": 0.06,
       "max_step": 25.0, "mincut": 0.3, "v_ema": 0.2, "vgate": 9.0,
       "ff_err_gate": 18.0, "predict": 3.2}
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


def spd(m):
    return 0.40*m['r40_reach'] + 0.35*m['r80_reach'] + 0.25*m['r120_reach']


def main():
    cur = metrics(base(**SHIP))
    old = metrics(base())
    s0, r0 = spd(cur), cur['rev_peak']
    print("  현재 stable : 속도지수 %.3f  반전pk %.2f  오버슛 %.2f  에러 %.3f"
          % (s0, r0, cur['step_overshoot'], sc(cur)))
    print("  이전 배포   : 속도지수 %.3f  반전pk %.2f  오버슛 %.2f  에러 %.3f  (속도 %+.1f%%)"
          % (spd(old), old['rev_peak'], old['step_overshoot'], sc(old),
             100*(spd(old)-s0)/s0))
    print("\n  각 값을 하나씩 이전값으로 되돌렸을 때 (나머지는 새 값 유지)")
    print("  %-22s %8s %8s %8s %8s"
          % ("되돌린 항목", "속도", "링잉", "오버슛", "에러"))
    print("  " + "-" * 60)
    rows = []
    for k, v in OLD.items():
        m = metrics(base(**dict(SHIP, **{k: v})))
        ds = 100*(spd(m)-s0)/s0
        dr = 100*(m['rev_peak']-r0)/r0
        rows.append((ds, k, v, ds, dr, m))
    rows.sort()
    for _, k, v, ds, dr, m in rows:
        flag = ""
        if ds < -1.0 and dr < 1.0:
            flag = "  <== 속도↑ 링잉 유지"
        elif ds < -1.0:
            flag = "  (속도↑ 링잉↓)"
        print("  %-22s %+7.1f%% %+7.1f%% %8.2f %8.3f%s"
              % ("%s %s→%s" % (k, SHIP[k], v), ds, dr,
                 m['step_overshoot'], sc(m), flag))
    print("\n  속도 = 0.40*r40 + 0.35*r80 + 0.25*r120 (낮을수록 빠름, 음수 = 빨라짐)")


if __name__ == "__main__":
    main()
