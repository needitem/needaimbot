#!/usr/bin/env python3
"""emit -> visible 데드타임을 '소수 프레임'까지 실측한다.

왜 필요한가: 컨트롤러의 `inflight_deadtime_frames`(현재 1.25)는 "내가 내보낸 이동이 몇
프레임 뒤에 검출에 나타나는가"를 보정하는 값이다. 이게 실제와 어긋나면 과보정/과소보정이
되어 **느려지는 동시에 링잉도 는다**. 반대로 맞추면 둘 다 좋아진다 - 이 시스템에서
드문 '트레이드 없는' 레버다. 그런데 지금 값은 부품 시간을 더한 추정치
(USB 1.0 + 렌더대기 3.5 + E2E 2.6)이고 emit->visible 을 직접 잰 적이 없다.

기존 calibrate.py 의 step-response 는 정수 프레임 해상도라 1.10 과 1.25 를 구분 못 한다.
여기서는 ms 단위로 연속 스캔해 소수 프레임까지 낸다.

원리: 앱이 aim OFF 상태에서 +-N 카운트를 주기적으로 주입한다(`calibration_step_px`).
마우스가 움직이면 시야가 움직이므로 표적의 검출 중심 cx 가 그만큼 반대로 밀린다. 주입
신호를 조금씩 지연시켜 가며 cx 와 가장 잘 맞는 지연을 찾으면 그게 데드타임이다.
1차 차분으로 비교해 표적 자체의 느린 드리프트를 제거한다.

  사용법:  python3 bench/deadtime_fit.py calib.csv
"""
import csv
import sys

import numpy as np


def load(path):
    t, cx, inj, aim, has = [], [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                t.append(float(r["t_us"]) * 1e-3)          # ms
                cx.append(float(r["cx"]))
                inj.append(float(r.get("inject_cum_x") or 0.0))
                aim.append(int(float(r.get("aiming") or 0)))
                has.append(int(float(r.get("hasTarget") or 0)))
            except (KeyError, ValueError):
                continue
    return (np.array(t), np.array(cx), np.array(inj),
            np.array(aim), np.array(has))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    t, cx, inj, aim, has = load(sys.argv[1])
    m = (aim == 0) & (has == 1)                 # 주입은 aim OFF 에서만 일어난다
    t, cx, inj = t[m], cx[m], inj[m]
    if len(t) < 200:
        print("  표본 부족: aim OFF + 검출 있는 프레임 %d 개 (200+ 필요)" % len(t))
        return 1
    if inj.max() - inj.min() < 1.0:
        print("  주입 신호 없음 — config 의 calibration_step_px 가 0 이 아닌지 확인")
        return 1

    dt = np.diff(t)
    T = float(np.median(dt))                    # 프레임 주기(ms)
    # 결측/정지 구간이 섞이면 차분이 오염되므로 정상 간격 구간만 쓴다
    ok = (dt > 0.3*T) & (dt < 3.0*T)
    d_cx = np.diff(cx)[ok]
    t_mid = (t[:-1] + T*0.5)[ok]

    # 주입 신호를 lag 만큼 지연시킨 뒤 그 1차 차분과 비교. 연속 지연은 선형보간으로.
    # 차분 창은 반드시 (t_mid - lag) 를 '중심'으로 잡아야 한다. [x-T, x] 로 잡으면
    # 창의 중심이 T/2 앞서므로 추정 lag 이 정확히 반 프레임 작게 나온다 - 합성
    # 정답 데이터로 확인된 실제 오차였다(1.10/1.37/2.05 전부 -0.48프레임).
    def dinj(lag):
        c = t_mid - lag
        return np.interp(c + T*0.5, t, inj) - np.interp(c - T*0.5, t, inj)

    best = None
    for lag in np.arange(0.0, 25.0, 0.05):      # ms
        di = dinj(lag)
        if di.std() < 1e-9:
            continue
        c = float(np.corrcoef(d_cx, di)[0, 1])
        if best is None or abs(c) > abs(best[1]):
            best = (lag, c)
    lag, corr = best

    # 스케일(카운트 -> 모델 px)도 같이 뽑아 물리적으로 말이 되는지 확인한다
    di = dinj(lag)
    gain = float(np.polyfit(di, d_cx, 1)[0])

    print("  표본 %d 프레임, 프레임 주기 %.2f ms (%.0f fps)" % (len(t), T, 1000.0/T))
    print("  주입 진폭 누적 %.0f 카운트\n" % (inj.max() - inj.min()))
    print("  %-30s %s" % ("emit -> visible 데드타임", "%.2f ms" % lag))
    print("  %-30s %s" % ("  = 프레임 단위", "%.2f 프레임" % (lag / T)))
    print("  %-30s %s" % ("상관계수", "%.2f" % abs(corr)))
    print("  %-30s %s" % ("주입 게인 (모델px/카운트)", "%.3f" % gain))
    print()
    if abs(corr) < 0.35:
        print("  ⚠ 상관이 약합니다. calibration_step_px 를 키우거나(예: 12),")
        print("    더 정지된 표적/장면에서 다시 수집하세요. 이 값은 신뢰하지 마세요.")
        return 1
    cur = 1.25
    print("  현재 설정 inflight_deadtime_frames = %.2f" % cur)
    d = lag/T - cur
    if abs(d) < 0.12:
        print("  -> 실측과 %.2f 프레임 차이. 이미 맞음 — 바꿔서 얻을 게 없습니다." % d)
    else:
        print("  -> 실측이 %.2f 프레임 %s. inflight_deadtime_frames 를 %.2f 로 바꾸면"
              % (abs(d), "더 큼(과소보정 중)" if d > 0 else "더 작음(과보정 중)", lag/T))
        print("     같은 게인에서 링잉과 지연이 함께 줄어듭니다 (트레이드 없음).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
