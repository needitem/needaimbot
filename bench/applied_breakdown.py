#!/usr/bin/env python3
"""실제로 적용한 설정이 '어느 상황에서' 좋아졌는지 시나리오별 분해.

지금까지의 보고는 가중합 하나(에러 -3.0%)였는데, 그것만으로는 무엇이 나아졌는지
알 수 없다. 정지 표적 유지가 좋아진 것과 전력질주 추적이 좋아진 것은 체감이 전혀
다르다. 적용 전/후를 시나리오별로 나란히 놓는다. 세 개의 독립 seed 블록 평균.
"""
from realjit import run_real, sc
from ctrl_zoo_opt import base

BEFORE = {}                                   # 적용 전 = sim 기본값(= 이전 배포 설정)
AFTER = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
         "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
         "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
         "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}

SCN = (("정지 표적 유지", "hold", {}, "hold_rms"),
       ("느린 스트레이프 1.1px/f", "track", dict(vx=1.1), "t1_rms"),
       ("보통 스트레이프 4px/f", "track", dict(vx=4.0), "t4_rms"),
       ("전력질주 8px/f", "track", dict(vx=8.0), "t8_rms"),
       ("좌우 반전 평균오차", "reversal", {}, "rev_rms"),
       ("좌우 반전 최대이탈", "reversal", {}, "rev_peak"),
       ("스텝 오버슈트", "step", dict(noise=False), "step_overshoot"),
       ("획득 40px", "reach", dict(noise=False, step_dist=40.), "r_reach"),
       ("획득 120px", "reach", dict(noise=False, step_dist=120.), "r_reach"))


def one(p, scn, kw, key, blocks=(500, 700, 900), n=60):
    vals = []
    for lo in blocks:
        acc = []
        for s in range(lo, lo+n):
            m = run_real(p, scn, s, **kw)
            acc.append(m[key.split("_", 1)[1] if key.startswith("r_") else
                         key.split("_", 1)[1]])
        vals.append(sum(acc)/len(acc))
    return sum(vals)/len(vals)


def main():
    a, b = base(**BEFORE), base(**AFTER)
    print("  %-24s %9s %9s %9s" % ("상황", "적용 전", "적용 후", "변화"))
    print("  " + "-" * 56)
    for lbl, scn, kw, key in SCN:
        va, vb = one(a, scn, kw, key), one(b, scn, kw, key)
        d = 100*(vb-va)/va if va else 0.0
        mark = "  좋아짐" if d < -1.5 else ("  나빠짐" if d > 1.5 else "  -")
        print("  %-24s %9.2f %9.2f %+8.1f%%%s" % (lbl, va, vb, d, mark))


if __name__ == "__main__":
    main()
