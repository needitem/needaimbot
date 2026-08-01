#!/usr/bin/env python3
"""조준 위치는 그대로 두고 관측식만 바꿔 σ 를 줄일 수 있는가.

anchor_noise.py 결과: 박스 노이즈는 '통째 이동(공통)' + '엣지별 독립' 으로 갈리고,
중심은 독립 성분을 절반으로 평균해 가장 조용하다(body 중심 3.111 vs 현재 조준점식
3.487). 현재 식 aim = y1 + k*h = (1-k)*y1 + k*y2 는 k=0.15 라 중심에서 멀리 떨어져
있어 그 이점을 못 받는다.

같은 점을 이렇게도 쓸 수 있다:
    aim = cy + (k - 0.5) * h
h 는 사람 박스 높이라 느리게 변하므로 무겁게 평활해도 편향이 거의 안 생긴다. 그러면
빠른 성분은 조용한 cy 가 담당하고 h 의 노이즈는 평활로 죽는다 - 조준 위치는 동일.

평활 계수를 훑어 σ 와 '편향(느린 추종 지연)'을 함께 본다. 공짜인지 확인해야 한다.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from crop_ab import Engine, preprocess          # noqa: E402
from fusion_check import decode_per_class       # noqa: E402


def wn(s):
    d = [b-a for a, b in zip(s, s[1:]) if a is not None and b is not None]
    return float(np.std(d)/np.sqrt(2.0)) if len(d) >= 30 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.npy")
    ap.add_argument("--engine",
                    default="/home/taeho/portfolio/inferencetool/engines/sunxds_0.8.3_320_fp16.engine")
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    frames = np.load(a.frames, mmap_mode="r")
    eng = Engine(a.engine); M = eng.in_shape[-1]
    W = frames.shape[2]; size = a.crop or W
    off = (W - size)//2
    ts = size/float(M)

    det = {0: [], 1: []}
    for f in frames:
        img = np.ascontiguousarray(f[off:off+size, off:off+size, :3])
        d = decode_per_class(eng.infer(preprocess(img, M)), a.conf)
        for c in (0, 1):
            det[c].append(d.get(c))

    for c, name, k in ((0, "body", 0.15), (1, "head", 0.601)):
        got = [x for x in det[c] if x is not None]
        if len(got) < 60:
            print("  %s: 표본 부족"); continue
        cy = [x[1] if x else None for x in det[c]]
        h = [x[3] if x else None for x in det[c]]
        cur = [None if x is None else (x[1]-x[3]*.5) + k*x[3] for x in det[c]]
        print("  [%s]  k=%.3f  검출 %d/%d   (화면 px)" % (name, k, len(got), len(det[c])))
        print("    %-24s %8s %10s" % ("관측식", "σ", "vs 현재"))
        print("    " + "-"*46)
        s0 = wn(cur)
        print("    %-24s %8.3f %10s" % ("현재  y1 + k·h", s0*ts, "-"))
        for alpha in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
            e = None; out = []
            for x in det[c]:
                if x is None:
                    out.append(None); continue
                e = x[3] if e is None else e + alpha*(x[3]-e)
                out.append(x[1] + (k-0.5)*e)
            s = wn(out)
            # 편향은 |평활h - 원시h| 로 재면 안 된다 - 그 차이의 대부분은 제거하려던
            # h 의 노이즈(σ~4px)라서 평활할수록 커지는 게 당연하다. 진짜 편향은 '지연'
            # 이므로 위상 지연이 없는 기준(전후방 이중 필터 = zero-phase)과 비교한다.
            raw = [x[3] for x in det[c] if x is not None]
            fw, e = [], None
            for v in raw:
                e = v if e is None else e + alpha*(v-e)
                fw.append(e)
            bw, e = [], None
            for v in reversed(raw):
                e = v if e is None else e + alpha*(v-e)
                bw.append(e)
            ref = 0.5*(np.array(fw) + np.array(list(reversed(bw))))   # zero-phase
            bias = float(np.mean(np.abs(np.array(fw)-ref)))*abs(k-0.5)
            print("    %-24s %8.3f %9.1f%%   편향 %.2f px"
                  % ("cy + (k-0.5)·h  a=%.2f" % alpha, s*ts,
                     100*(s-s0)/s0, bias*ts))
        print()


if __name__ == "__main__":
    main()
