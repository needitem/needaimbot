#!/usr/bin/env python3
"""캡처 프레임레이트가 바뀌었을 때 컨트롤러를 재튜닝한다.

컨트롤러 파라미터는 대부분 '프레임' 단위다. 캡처 레이트가 바뀌면 같은 숫자가 다른
물리량을 뜻하게 되므로 그대로 두면 안 된다 - 143Hz 설정을 240Hz 에 그대로 꽂았을 때
스텝 오버슈트가 0.94 -> 4.55 로 터진 적이 있다.

두 단계로 한다:
  1) 물리 환산으로 시드를 만든다 (r = 새 프레임주기 / 기준 프레임주기)
       px/프레임인 것 (max_step, vgate)          x r    같은 px/초
       프레임 수인 것 (predict)                  / r    같은 ms 선행
       프레임당 계수인 것 (kp, v_ema, mincut)      x r    같은 Hz 대역
       속도(px/프레임)에 곱하는 것 (ff)            / r    같은 px 리드
       px 인 것 (softness, err_gate, kd)         불변
       w 는 그 조건의 실제 데드타임(프레임)으로 직접   (USB + T/2 + E2E) / T
  2) 그 시드에서 ES 로 정련한다. 환산만으로는 최적이 아니다.

sim 자체도 함께 재파라미터화한다 - 프레임 주기, 표적 속도(같은 화면 속도 유지),
시나리오 길이(같은 벽시계 시간 유지), 데드타임 분포. 비교는 프레임이 아니라 **ms** 로
한다. 프레임 수는 레이트가 바뀌면 뜻이 달라지기 때문이다.

  사용법:  python3 bench/fps_retune.py [새_fps] [E2E_ms]
"""
import json
import math
import random
import sys

import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX

random.seed(2390)

BASE_FPS = 143.0
USB_MS = 1.0
BOX = dict(COMMON_BOX, beta=(0.0, 0.12))

# 현재 배포 중인 값 (320 공간). 160 프리셋은 softness/vgate/err_gate 만 x2.
SHIP = {"kp_x": 0.765, "kp_y": 0.698, "soft_x": 8.6, "soft_y": 6.57,
        "kd_x": 0.052, "kd_y": 0.037, "max_step": 19.56, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 3.18, "v_ema": 0.235,
        "vgate": 14.79, "ff_err_gate": 22.25, "mincut": 0.084}

DT0 = realjit.dt_sample


def configure(fps, e2e_ms):
    T = 1000.0 / fps
    dead_ms = USB_MS + 0.5 * T + e2e_ms
    med = dead_ms / T
    realjit.dt_sample = (lambda m: (lambda rng: m * math.exp(rng.gauss(0., 0.335))))(med)
    return T, dead_ms, med


def rescale(fps, e2e_ms):
    """물리량을 보존하는 환산. 정련의 출발점일 뿐 최적값은 아니다.

    BOX 로 자르지 않는다 - 그건 '탐색 범위'지 물리 한계가 아니다. 자르면 환산이
    환산이 아니게 된다(239Hz 에서 kp/max_step/predict 네 값이 경계에 걸렸다).
    커널이 실제로 요구하는 한계만 지킨다: w 와 predict 는 in-flight 링 깊이(4)와
    커널 clamp(predict<=6) 안에 있어야 한다."""
    T = 1000.0 / fps
    r = T / (1000.0 / BASE_FPS)
    q = dict(base(**SHIP))
    for k in ("max_step", "vgate", "kp_x", "kp_y", "v_ema", "mincut"):
        q[k] = q[k] * r
    q["predict"] = min(6.0, q["predict"] / r)          # 커널 clamp
    q["ff"] = q["ff"] / r
    q["w"] = min(4.0, (USB_MS + 0.5 * T + e2e_ms) / T)  # 링 깊이
    return q


def ev(p, fps, n=24, lo=500):
    """모든 시나리오를 그 fps 로. 시간 지표는 ms 로 환산해 돌려준다."""
    T = 1000.0 / fps
    k = fps / BASE_FPS          # 프레임 수 배율 (벽시계 고정)
    vs = BASE_FPS / fps         # 속도 배율 (화면 속도 고정)
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
            o[nm + "_" + kk] = sum(v) / len(v)
    for nm in ("r40", "r120"):
        o[nm + "_ms"] = o[nm + "_reach"] * T
    return o


def main():
    fps = float(sys.argv[1]) if len(sys.argv) > 1 else 239.0
    e2e = float(sys.argv[2]) if len(sys.argv) > 2 else 2.6

    configure(BASE_FPS, e2e)
    b = ev(base(**SHIP), BASE_FPS)
    T0 = 1000.0 / BASE_FPS
    print("  기준 %.0ffps: 프레임 %.2fms, 오버슛 %.2f, 반전pk %.2f, 에러 %.3f"
          % (BASE_FPS, T0, b['step_overshoot'], b['rev_peak'], sc(b)))
    print("  기준 획득: 40px %.1fms / 120px %.1fms\n" % (b['r40_ms'], b['r120_ms']))

    T, dms, med = configure(fps, e2e)
    print("  목표 %.0ffps: 프레임 %.2fms, 데드타임 %.2fms (%.2f 프레임)"
          % (fps, T, dms, med))

    def ok(m):
        return (m['step_overshoot'] <= b['step_overshoot']
                and m['step_osc'] <= max(0.05, b['step_osc'])
                and m['rev_peak'] <= b['rev_peak']
                and sc(m) <= sc(b))

    # 사고 싶은 것만 목적에 넣는다. 앞선 실행은 r120 에 0.35 를 줬다가 탐색이
    # 근거리·에러·오버슈트를 팔아서 원거리를 샀고, 환산본보다 나쁜 결과가 나왔다.
    def score(m):
        return m['r40_ms']

    seed = rescale(fps, e2e)
    ms = ev(seed, fps, n=12, lo=0)
    print("  환산 시드: 오버슛 %.2f 반전pk %.2f 에러 %.3f %s"
          % (ms['step_overshoot'], ms['rev_peak'], sc(ms),
             "(제약 통과)" if ok(ms) else "(제약 이탈 - 정련 필요)"))

    p = dict(seed)
    best = (score(ms), dict(p)) if ok(ms) else None
    e = score(ms)
    step, since = 0.25, 0
    for _ in range(320):
        q = dict(p)
        for k, (lo_, hi) in BOX.items():
            if random.random() < 0.35:
                q[k] = min(hi, max(lo_, q[k] + random.gauss(0, step*(hi-lo_))))
        m = ev(q, fps, n=12, lo=0)
        if ok(m) and (best is None or score(m) < best[0]):
            best = (score(m), dict(q)); p, e, since = q, score(m), 0
        elif ok(m) and score(m) < e:
            p, e, since = q, score(m), 0
        else:
            since += 1
            if since >= 14:
                step *= 0.65; since = 0
                if step < 0.008:
                    break

    if best is None:
        print("\n  제약을 만족하는 지점을 못 찾음 - 범위를 넓히거나 제약을 재검토할 것")
        realjit.dt_sample = DT0
        return

    print("\n  확정 (seed 500-523, 탐색 미사용)")
    print("  %-20s %8s %6s %6s   %9s %9s"
          % ("구성", "에러", "오버슛", "반전pk", "획득40", "획득120"))
    print("  " + "-" * 64)
    configure(BASE_FPS, e2e)
    m0 = ev(base(**SHIP), BASE_FPS)
    print("  %-20s %8.3f %6.2f %6.2f   %7.1fms %7.1fms"
          % ("%.0ffps 현재" % BASE_FPS, sc(m0), m0['step_overshoot'],
             m0['rev_peak'], m0['r40_ms'], m0['r120_ms']))
    configure(fps, e2e)
    for lbl, q in (("%.0ffps 환산만" % fps, seed), ("%.0ffps 정련" % fps, best[1])):
        m = ev(q, fps)
        print("  %-20s %8.3f %6.2f %6.2f   %7.1fms %7.1fms   획득40 %+.0f%%"
              % (lbl, sc(m), m['step_overshoot'], m['rev_peak'],
                 m['r40_ms'], m['r120_ms'],
                 100*(m['r40_ms']-m0['r40_ms'])/m0['r40_ms']))
    realjit.dt_sample = DT0

    print("\n  %.0ffps 용 값 (320 공간):" % fps)
    print("  " + json.dumps({k: round(best[1][k], 4) for k in BOX if k in best[1]}))
    x2 = {"soft_x", "soft_y", "vgate", "ff_err_gate"}
    print("\n  160 프리셋 (softness/vgate/err_gate x2):")
    for k in BOX:
        if k in best[1]:
            v = best[1][k]
            print("    %-14s %s" % (k, round(v*2, 3) if k in x2 else round(v, 4)))


if __name__ == "__main__":
    main()
