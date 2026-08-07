#!/usr/bin/env python3
"""주입 없이, 우리가 내보낸 이동만으로 emit->visible 데드타임을 추정할 수 있나.

왜: `inflight_deadtime_ms` 를 상수로 박아두면 결국 같은 문제가 남는다. 그 8.10ms 는
143fps 에서 잰 11.60ms 에서 캡처 샘플링분(T/2)만 뺀 값인데, 나머지에는 **게임 자신의
렌더 파이프라인**이 들어 있고 그건 게임 프레임시간을 따라 변한다. 부하·설정으로도 변한다.

`deadtime_fit.py` 는 aim OFF 로 +-12 카운트를 주입해 잰다. 플레이 중엔 못 쓴다 -
마우스가 튄다. 그런데 조준 중이면 **컨트롤러의 emit 자체가 가진 신호**다. 같은 상관을
주입 대신 emit 누적으로 걸면 되는지 본다.

주의: 이건 폐루프 식별이다. emit 이 표적 위치의 함수이므로 입력과 출력 잡음이 상관돼
추정이 편향될 수 있다. 그래서 여기서는 '되는지'부터 확인한다 - 상관 봉우리가 뾰족한지,
분할 재현이 되는지. 안 되면 안 되는 대로 결론이다.

  사용법:  python3 bench/deadtime_closedloop.py <calib.csv> [--fps N]
"""
import csv
import sys

import numpy as np


def load(path):
    t, cx, dx, ms, aim, has = [], [], [], [], [], []
    for r in csv.DictReader(open(path, newline="")):
        try:
            t.append(float(r["t_us"])*1e-3); cx.append(float(r["cx"]))
            dx.append(float(r["emit_dx"])); ms.append(float(r["mscale_x"]))
            aim.append(int(float(r["aiming"]))); has.append(int(float(r["hasTarget"])))
        except (KeyError, ValueError):
            continue
    return tuple(np.array(a) for a in (t, cx, dx, ms, aim, has))


def runs(mask, lo):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        elif not v and s is not None:
            if i-s >= lo: out.append((s, i))
            s = None
    if s is not None and len(mask)-s >= lo: out.append((s, len(mask)))
    return out


def fit(t, cx, dx, ms, seg, T, lags):
    """주입 대신 emit 누적을 쓰는 것 말고는 deadtime_fit.py 와 같은 구조.

    우리가 시야를 +X 모델px 움직이면 표적의 cx 는 그만큼 **줄어든다**. 따라서 에고가
    만든 검출 변위는 -(emit 누적의 증가분)이다. 차분 창은 반드시 중심을 맞춘다 -
    한쪽으로 치우치면 정확히 0.5 프레임 편향이 생긴다(실제로 겪었다)."""
    a, b = seg
    tt, xx = t[a:b], cx[a:b]
    ego = np.cumsum(dx[a:b] / np.maximum(ms[a:b], 1e-6))   # 모델px 누적
    mid = tt[1:-1]
    dobs = np.interp(mid + T*0.5, tt, xx) - np.interp(mid - T*0.5, tt, xx)
    out = []
    for L in lags:
        c = mid - L*T
        dego = np.interp(c + T*0.5, tt, ego) - np.interp(c - T*0.5, tt, ego)
        if dego.std() < 1e-9 or dobs.std() < 1e-9:
            out.append(0.0); continue
        out.append(float(np.corrcoef(dobs, -dego)[0, 1]))
    return np.array(out)


def main():
    path = sys.argv[1]
    fps = None
    if "--fps" in sys.argv: fps = float(sys.argv[sys.argv.index("--fps")+1])
    t, cx, dx, ms, aim, has = load(path)
    d = np.diff(t); d = d[(d > 0) & (d < 50)]
    T = 1000.0/fps if fps else d.mean()
    print("  %s\n  %d 프레임, 평균 간격 %.2fms (%.0f fps)\n" % (path, len(t), T, 1000/T))

    segs = runs((aim == 1) & (has == 1), 60)
    tot = sum(b-a for a, b in segs)
    print("  조준+검출 연속 60프레임 이상 구간 %d개, 합계 %d프레임 (%.1f초)"
          % (len(segs), tot, tot*T/1000))
    if not segs:
        print("  구간 없음 - 조준 중 데이터가 부족하다"); return

    lags = np.arange(0.0, 5.01, 0.05)
    # 구간 길이로 가중 평균: 짧은 구간의 잡음이 결과를 흔들지 않게.
    acc = np.zeros_like(lags); wsum = 0.0
    for seg in segs:
        c = fit(t, cx, dx, ms, seg, T, lags)
        w = seg[1]-seg[0]
        acc += w*c; wsum += w
    corr = acc/wsum
    k = int(np.argmax(corr))
    print("  최적 지연 %.2f 프레임 = %.2f ms   (상관 %.3f)" % (lags[k], lags[k]*T, corr[k]))

    print("\n  지연별 상관 (봉우리가 뾰족해야 신뢰할 수 있다)")
    for L in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        i = int(np.argmin(np.abs(lags-L)))
        bar = "#"*max(0, int(40*corr[i])) if corr[i] > 0 else ""
        print("    %4.1f 프레임  %+.3f  %s" % (L, corr[i], bar))

    # 절반씩 나눠 재현되는지
    if len(segs) >= 4:
        half = len(segs)//2
        for lbl, ss in (("전반", segs[:half]), ("후반", segs[half:])):
            acc = np.zeros_like(lags); wsum = 0.0
            for seg in ss:
                acc += (seg[1]-seg[0])*fit(t, cx, dx, ms, seg, T, lags); wsum += seg[1]-seg[0]
            c = acc/wsum; i = int(np.argmax(c))
            print("  %s %d구간: %.2f 프레임 (%.2f ms), 상관 %.3f"
                  % (lbl, len(ss), lags[i], lags[i]*T, c[i]))


if __name__ == "__main__":
    main()
