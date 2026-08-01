#!/usr/bin/env python3
"""박스의 '어느 점'이 가장 덜 흔들리는가 — 관측 지점 자체를 바꾸는 여지 측정.

지금까지의 개선은 전부 "같은 관측을 어떻게 처리할까"였다(필터·게인·리드). 관측 자체를
바꾼 건 크롭 크기 하나뿐이다. 그런데 조준점 식이 이렇다:
    aim_y = y1 + h * offset  =  (1-offset)*y1 + offset*y2      (h = y2 - y1)
즉 위 엣지와 아래 엣지의 노이즈를 '둘 다' 물려받는다. 두 엣지의 σ 가 크게 다르면
지저분한 쪽을 쓸데없이 빨아들이고 있는 것이다.

물리적으로 다를 이유가 있다: 위 엣지(머리 윤곽)는 배경과 고대비인 반면 아래 엣지
(발·다리)는 지형·엄폐·그림자에 묻힌다. 그렇다면 중심(=두 엣지의 평균)은 최악의
선택일 수 있다.

측정: 각 좌표의 프레임간 백색 노이즈를 1차차분/√2 로 추정(표적이 움직이므로 추세는
차분으로 제거된다). 그리고 현재 조준점 식과 대안들을 같은 척도로 비교한다.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from crop_ab import Engine, preprocess          # noqa: E402
from fusion_check import decode_per_class       # noqa: E402


def wn(series):
    """1차차분/√2 = 백색 노이즈 σ 추정. 결측(None)은 구간을 끊는다."""
    d = []
    for a, b in zip(series, series[1:]):
        if a is not None and b is not None:
            d.append(b - a)
    if len(d) < 30:
        return None
    return float(np.std(d) / np.sqrt(2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.npy")
    ap.add_argument("--engine",
                    default="/home/taeho/portfolio/inferencetool/engines/sunxds_0.8.3_320_fp16.engine")
    ap.add_argument("--crop", type=int, default=160,
                    help="중앙 크롭 크기(0=네이티브 320)")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    frames = np.load(a.frames, mmap_mode="r")
    eng = Engine(a.engine)
    M = eng.in_shape[-1]
    W = frames.shape[2]
    off = (W - a.crop)//2 if a.crop else 0
    size = a.crop if a.crop else W
    # 모델 px -> 화면 px 환산 (크롭이 작을수록 표적이 크게 잡히므로 화면 px 는 작아짐)
    to_screen = size / float(M)

    det = {0: [], 1: []}
    for f in frames:
        img = np.ascontiguousarray(f[off:off+size, off:off+size, :3])
        d = decode_per_class(eng.infer(preprocess(img, M)), a.conf)
        for c in (0, 1):
            det[c].append(d.get(c))

    print("  프레임 %d, 크롭 %d (화면px 환산 x%.2f), conf %.2f\n"
          % (len(frames), size, to_screen, a.conf))
    for c, name in ((0, "body"), (1, "head")):
        got = [x for x in det[c] if x is not None]
        if len(got) < 60:
            print("  %s: 검출 %d 프레임 — 표본 부족" % (name, len(got)))
            continue
        cx = [x[0] if x else None for x in det[c]]
        cy = [x[1] if x else None for x in det[c]]
        w = [x[2] if x else None for x in det[c]]
        h = [x[3] if x else None for x in det[c]]
        y1 = [(x[1]-x[3]*.5) if x else None for x in det[c]]
        y2 = [(x[1]+x[3]*.5) if x else None for x in det[c]]
        x1 = [(x[0]-x[2]*.5) if x else None for x in det[c]]
        x2 = [(x[0]+x[2]*.5) if x else None for x in det[c]]
        print("  [%s]  검출 %d/%d 프레임   (단위: 화면 px)" % (name, len(got), len(det[c])))
        print("    %-16s %8s" % ("관측 지점", "σ"))
        print("    " + "-"*26)
        for lbl, s in (("위 엣지 y1", y1), ("아래 엣지 y2", y2), ("중심 cy", cy),
                       ("좌 엣지 x1", x1), ("우 엣지 x2", x2), ("중심 cx", cx),
                       ("폭 w", w), ("높이 h", h)):
            v = wn(s)
            print("    %-16s %8s" % (lbl, "%.3f" % (v*to_screen) if v else "-"))
        # 조준점 식 비교: aim = (1-k)*y1 + k*y2  vs  y1 + c*w  vs  y1 + 고정px
        k = 0.601 if c == 1 else 0.15
        mw = float(np.median([x for x in w if x is not None]))
        mh = float(np.median([x for x in h if x is not None]))
        cur = [(None if (p is None or q is None) else (1-k)*p + k*q)
               for p, q in zip(y1, y2)]
        byw = [(None if (p is None or q is None) else p + (k*mh/mw)*q)
               for p, q in zip(y1, w)]
        fixed = [(None if p is None else p + k*mh) for p in y1]
        print("    %-16s %8s" % ("--- 조준점 식 ---", ""))
        for lbl, s in (("현재 y1+k·h", cur), ("폭 기준 y1+c·w", byw),
                       ("고정 y1+상수", fixed)):
            v = wn(s)
            print("    %-16s %8s" % (lbl, "%.3f" % (v*to_screen) if v else "-"))
        print()


if __name__ == "__main__":
    main()
