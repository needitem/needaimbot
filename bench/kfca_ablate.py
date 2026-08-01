#!/usr/bin/env python3
"""KF-CA 의 -2.1% 가 진짜인지 가른다.

두 가지 교란요인이 있다:
  1) 과적합 - 정련은 seed 0..9 로 점수를 매겼다. 새 seed 에서도 유지되는가?
  2) 귀속오류 - 탐색이 추정기와 게인을 '동시에' 움직였다. 같은 게인을 현재 추정기
     (One Euro)에 물리면 이득이 사라지는가? 사라지면 칼만이 아니라 게인이 한 일이다.
그리고 공정한 대조군: 현재 구조(BASE)도 현재 설정에서 출발해 같은 ES 예산으로 정련한다.
랜덤 스크리닝만으로는 17차원에서 현재 설정만큼 좋은 점을 못 찾으므로, 이전 실행의
"BASE 통과 실패"는 BASE 가 약하다는 뜻이 아니라 예산이 얕다는 뜻이었다.
"""
import json
import random

from realjit import ev, sc
from holdout import ev_holdout
from ctrl_zoo_opt import base, COMMON_BOX, refine
from ctrl_zoo import KFCACtrl, HoltCtrl

random.seed(4242)

KFCA = {"kp_x": 0.71, "kp_y": 0.764, "soft_x": 10.993, "soft_y": 14.79,
        "kd_x": 0.1288, "kd_y": 0.0367, "max_step": 24.256, "w": 1.591,
        "ff": 1.6282, "predict": 2.6993, "v_ema": 0.2165, "vgate": 5.7821,
        "ff_err_gate": 27.14, "mincut": 0.106, "kf_R": 19.032, "kf_Q": 5.0879}
HOLT = {"kp_x": 0.857, "kp_y": 0.6553, "soft_x": 14.135, "soft_y": 7.519,
        "kd_x": 0.112, "kd_y": 0.108, "max_step": 28.17, "w": 1.4563,
        "ff": 1.8181, "predict": 1.7544, "v_ema": 0.1727, "vgate": 6.7286,
        "ff_err_gate": 13.79, "mincut": 0.443, "holt_a": 0.5745, "holt_b": 0.0559}


def row(lbl, p, ctrl):
    a, b = ev(p, seeds=40, ctrl=ctrl), ev_holdout(p, ctrl=ctrl)
    print("  %-34s %8.3f %8.3f   %6.2f %6.2f %6.1f"
          % (lbl, sc(a), sc(b), b['step_overshoot'], b['rev_peak'], b['reach_reach']))
    return sc(b), b


def main():
    cur = base()
    print("  %-34s %8s %8s   %6s %6s %6s"
          % ("구성", "in-sample", "holdout", "오버슛", "반전pk", "reach"))
    print("  " + "-" * 76)
    e_cur, m_cur = row("현재 배포 설정", cur, None)
    e_kf, _ = row("KF-CA (최적화)", base(**KFCA), KFCACtrl)
    e_kfoe, _ = row("  └ 같은 게인 + One Euro (귀속검증)", base(**KFCA), None)
    e_ho, _ = row("HOLT (최적화)", base(**HOLT), HoltCtrl)

    # 공정 대조군: 현재 설정에서 출발한 BASE 국소 정련
    ok = lambda m: (m['step_overshoot'] <= m_cur['step_overshoot'] + 0.05
                    and m['step_osc'] <= 0.35
                    and m['rev_peak'] <= m_cur['rev_peak'] * 1.03
                    and m['reach_reach'] <= m_cur['reach_reach'] * 1.10)
    p1, _ = refine(cur, sc(ev(cur, seeds=10)), COMMON_BOX, None, ok,
                   iters=140, seeds=10)
    e_base, _ = row("BASE (현재설정 출발 ES 정련)", p1, None)

    print("\n  holdout 기준 현재 대비:")
    for lbl, e in (("KF-CA", e_kf), ("KF-CA 게인만", e_kfoe),
                   ("HOLT", e_ho), ("BASE 정련", e_base)):
        print("    %-14s %+6.1f%%" % (lbl, 100*(e-e_cur)/e_cur))
    print("\n  BASE 정련 결과: " + json.dumps({k: round(v, 4) for k, v in p1.items()
                                          if k in COMMON_BOX}))


if __name__ == "__main__":
    main()
