#!/usr/bin/env python3
"""-4.6% 후보가 160 크롭의 낮은 노이즈에서도 유효한가.

sim 의 NOISE_X/Y 8.1/5.9 는 320 캡처 시절 실측치다. 그래서 sim 기본 softness 9/8 이
320 프리셋과 1:1 로 맞고, 지금 쓰는 160 프리셋(18/16)은 그 2배다. 그런데 160 크롭은
검출 sigma 를 X -34% / Y -23% 줄였다 - 즉 지금 사용자가 돌리는 조건은 sim 보다 조용하다.

beta=0 이 이긴 이유가 '노이즈가 One Euro 의 속도적응을 헛돌린다' 였으므로, 노이즈가
줄면 이득도 줄 수 있다. 노이즈를 실측 비율대로 낮춰 이득이 남는지 확인한다. 남으면
두 프리셋 모두에 적용해도 되고, 사라지면 320 프리셋에만 적용해야 한다.
"""
import aim_opt
from realjit import run_real, sc
from ctrl_zoo_opt import base

CAND = {"kp_x": 0.7797, "kp_y": 0.611, "soft_x": 9.0161, "soft_y": 15.9729,
        "kd_x": 0.1635, "kd_y": 0.0173, "max_step": 29.8496, "w": 1.523,
        "ff": 1.2777, "predict": 1.1475, "v_ema": 0.2476, "vgate": 2.5634,
        "ff_err_gate": 23.8051, "mincut": 0.2238, "beta": 0.0006}


def ev_at(p, lo=500, n=60):
    o = {}
    for nm, scn, kw in (("step", "step", dict(noise=False)),
                        ("hold", "hold", {}), ("rev", "reversal", {}),
                        ("t1", "track", dict(vx=1.1)), ("t4", "track", dict(vx=4.0)),
                        ("t8", "track", dict(vx=8.0))):
        acc = {}
        for s in range(lo, lo + n):
            m = run_real(p, scn, s, **kw)
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            o[nm + "_" + k] = sum(v) / len(v)
    return o


def main():
    import realjit
    nx0, ny0 = aim_opt.NOISE_X, aim_opt.NOISE_Y
    print("  %-30s %9s %9s %8s %7s"
          % ("노이즈 조건", "현재", "후보", "이득", "오버슛"))
    print("  " + "-" * 68)
    for tag, fx, fy in (("320 실측 (sigma 8.1/5.9)", 1.00, 1.00),
                        ("160 실측 (X-34% Y-23%)", 0.66, 0.77),
                        ("검출기 개선 가정 (절반)", 0.50, 0.50),
                        ("노이즈 악화 (1.3배)", 1.30, 1.30)):
        aim_opt.NOISE_X, aim_opt.NOISE_Y = nx0*fx, ny0*fy
        realjit.NOISE_X, realjit.NOISE_Y = aim_opt.NOISE_X, aim_opt.NOISE_Y
        a, b = ev_at(base()), ev_at(base(**CAND))
        print("  %-30s %9.3f %9.3f %+7.1f%% %7.2f"
              % (tag, sc(a), sc(b), 100*(sc(b)-sc(a))/sc(a), b['step_overshoot']))
    aim_opt.NOISE_X, aim_opt.NOISE_Y = nx0, ny0
    realjit.NOISE_X, realjit.NOISE_Y = nx0, ny0


if __name__ == "__main__":
    main()
