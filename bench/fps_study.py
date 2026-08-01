#!/usr/bin/env python3
"""게임 fps 와 엔진 정밀도가 실제로 얼마나 사주는가 (프레임이 아니라 ms 로).

앞 실행에서 '데드타임 -30%'를 프레임 단위로 스케일했는데 그건 fps 상향과 다르다.
fps 를 올리면 프레임 주기 T 가 같이 짧아지므로:
  렌더 대기 = T/2  -> 프레임 단위로는 항상 0.5, 안 변함
  USB·E2E   = ms 고정 -> 프레임 단위로는 오히려 커짐
따라서 프레임 단위 데드타임은 fps 를 올릴수록 '늘어난다'. 그런데 프레임이 짧아지니
컨트롤러가 더 자주 돌고, 표적은 프레임당 덜 움직인다. 어느 쪽이 이기는지는 재봐야 안다.

그래서 fps 마다 sim 을 통째로 재파라미터화한다:
  T = 1000/fps                       프레임 주기(ms)
  데드타임(ms) = USB 1.0 + T/2 + E2E
  데드타임(프레임) = 그것 / T
  표적 속도(px/프레임) = 기준속도 * 144/fps      같은 화면 속도를 유지
  시나리오 길이(프레임) = 기준 * fps/144         같은 벽시계 시간을 유지
  획득 시간은 프레임이 아니라 ms 로 환산해 비교    <- 이게 핵심
검출기 σ 는 검출당 값이라 fps 와 무관하게 그대로 둔다(프레임당 노이즈 불변).
"""
import json
import math
import random

import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base, COMMON_BOX

random.seed(4747)

BOX = dict(COMMON_BOX, beta=(0.0, 0.12))
SHIP = {"kp_x": 0.75, "kp_y": 0.744, "soft_x": 9.33, "soft_y": 11.0,
        "kd_x": 0.041, "kd_y": 0.054, "max_step": 23.5, "w": 1.25,
        "ff": 1.7, "beta": 0.02, "predict": 2.94, "v_ema": 0.224,
        "vgate": 11.14, "ff_err_gate": 19.69, "mincut": 0.225}
USB_MS, BASE_FPS = 1.0, 144.0
DT0 = realjit.dt_sample


def configure(fps, e2e_ms):
    """그 fps/엔진에서의 데드타임 분포를 프레임 단위로 심는다."""
    T = 1000.0 / fps
    dead_ms = USB_MS + 0.5 * T + e2e_ms
    med = dead_ms / T
    realjit.dt_sample = (lambda m: (lambda rng: m * math.exp(rng.gauss(0., 0.335))))(med)
    return T, dead_ms, med


def ev(p, fps, n=40, lo=500):
    """모든 시나리오를 그 fps 로 돌리고, 시간 지표는 ms 로 환산해 돌려준다."""
    T = 1000.0 / fps
    k = fps / BASE_FPS                      # 프레임 수 배율(벽시계 고정)
    vs = BASE_FPS / fps                     # 속도 배율(화면 속도 고정)
    spec = (("step", "step", dict(noise=False, frames=int(150*k))),
            ("hold", "hold", dict(frames=int(420*k))),
            ("rev", "reversal", dict(frames=int(300*k))),
            ("t1", "track", dict(vx=1.1*vs, frames=int(480*k))),
            ("t4", "track", dict(vx=4.0*vs, frames=int(480*k))),
            ("t8", "track", dict(vx=8.0*vs, frames=int(480*k))),
            ("r40", "reach", dict(noise=False, step_dist=40., frames=int(220*k))),
            ("r80", "reach", dict(noise=False, step_dist=80., frames=int(220*k))),
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
    for nm in ("r40", "r80", "r120"):
        o[nm + "_ms"] = o[nm + "_reach"] * T      # 프레임 -> ms
    return o


def main():
    # 기준: 현재 조건(144fps, FP16 E2E 2.6ms)에서의 현재 설정
    T0, d0, m0 = configure(BASE_FPS, 2.6)
    b = ev(base(**SHIP), BASE_FPS)
    print("  기준 144fps/FP16: 프레임 %.2fms, 데드타임 %.2fms(%.2f프레임)" % (T0, d0, m0))
    print("  현재 설정: 오버슛 %.2f 반전pk %.2f 에러 %.3f | 획득 %.1f/%.1f/%.1f ms\n"
          % (b['step_overshoot'], b['rev_peak'], sc(b),
             b['r40_ms'], b['r80_ms'], b['r120_ms']))

    def ok(m):
        return (m['step_overshoot'] <= b['step_overshoot']
                and m['rev_peak'] <= b['rev_peak']
                and m['step_osc'] <= 0.35 and sc(m) <= sc(b))

    def rescale(fps, e2e):
        """fps 가 바뀌면 '프레임 단위' 파라미터는 전부 재해석돼야 한다. 같은 물리량을
        유지하는 환산을 출발점으로 준다 (r = 새 프레임주기 / 기준 프레임주기).
          px/프레임 인 것 (max_step, vgate)         -> xr   같은 px/초
          프레임 수인 것 (predict)                 -> /r   같은 ms 선행
          프레임당 계수인 것 (kp, v_ema, mincut)     -> xr   같은 Hz 대역
          px 인 것 (softness, err_gate)           -> 불변
          w 는 그 조건의 실제 데드타임(프레임)으로 직접 설정
        """
        T = 1000.0/fps
        r = T / (1000.0/BASE_FPS)
        q = dict(base(**SHIP))
        for k in ("max_step", "vgate"):
            q[k] = min(BOX[k][1], max(BOX[k][0], q[k]*r))
        q["predict"] = min(BOX["predict"][1], q["predict"]/r)
        for k in ("kp_x", "kp_y", "v_ema", "mincut"):
            q[k] = min(BOX[k][1], max(BOX[k][0], q[k]*r))
        q["ff"] = min(BOX["ff"][1], max(BOX["ff"][0], q["ff"]/r))
        q["w"] = min(BOX["w"][1], max(BOX["w"][0], (USB_MS + 0.5*T + e2e)/T))
        return q

    def refine(fps, e2e, iters=220):
        """물리 환산 출발점에서 ES. 통과 후보를 못 찾으면 None 을 돌려준다
        (출발점을 최적화 결과로 오인하지 않기 위해 - 앞 실행의 실수)."""
        def score(m): return 0.45*m['r40_ms'] + 0.35*m['r80_ms'] + 0.20*m['r120_ms']
        p0 = rescale(fps, e2e)
        best = None
        p = dict(p0); e = score(ev(p, fps, n=12, lo=0))
        if ok(ev(p, fps, n=12, lo=0)):
            best = (e, dict(p))
        step, since = 0.25, 0
        for _ in range(iters):
            q = dict(p)
            for kk, (lo_, hi) in BOX.items():
                if random.random() < 0.35:
                    q[kk] = min(hi, max(lo_, q[kk] + random.gauss(0, step*(hi-lo_))))
            m = ev(q, fps, n=12, lo=0)
            if ok(m) and (best is None or score(m) < best[0]):
                best = (score(m), dict(q)); p, e, since = q, score(m), 0
            elif ok(m) and score(m) < e:
                p, e, since = q, score(m), 0
            else:
                since += 1
                if since >= 14:
                    step *= 0.65; since = 0
                    if step < 0.008: break
        return best[1] if best else None

    results = {}
    print("  %-22s %7s %7s %6s %6s   %8s %8s %8s"
          % ("조건", "프레임ms", "데드ms", "오버슛", "반전pk", "획득40", "획득80", "획득120"))
    print("  " + "-" * 82)
    for lbl, fps, e2e in (("144fps FP16 (현재)", 144., 2.6),
                          ("144fps INT8", 144., 1.8),
                          ("240fps FP16", 240., 2.6),
                          ("240fps INT8", 240., 1.8),
                          ("360fps INT8", 360., 1.8)):
        T, dms, med = configure(fps, e2e)
        if fps == 144. and e2e == 2.6:
            p = base(**SHIP)
        else:
            p = refine(fps, e2e)
            if p is None:
                print("  %-22s %7.2f %7.2f   %s"
                      % (lbl, T, dms, "링잉·에러 무회귀 지점 없음"))
                continue
        m = ev(p, fps)
        f = lambda k: "%.1f (%+.0f%%)" % (m[k], 100*(m[k]-b[k])/b[k])
        flag = "" if (m['step_overshoot'] <= b['step_overshoot']*1.05
                      and m['rev_peak'] <= b['rev_peak']*1.03) else "  [확정에서 링잉 이탈]"
        print("  %-22s %7.2f %7.2f %6.2f %6.2f   %8s %8s %8s%s"
              % (lbl, T, dms, m['step_overshoot'], m['rev_peak'],
                 f('r40_ms'), f('r80_ms'), f('r120_ms'), flag))
        results[lbl] = p
    realjit.dt_sample = DT0
    for k, v in results.items():
        print("\n  %s: %s" % (k, json.dumps({j: round(v[j], 4) for j in BOX if j in v})))


if __name__ == "__main__":
    main()
