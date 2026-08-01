#!/usr/bin/env python3
"""중심와(foveated) 이중 관측이 실제로 값을 하는가.

지금은 크롭 하나를 골라야 한다: 160 은 정밀(표적이 모델 좌표에서 2배 크게 잡힘)하지만
시야가 좁고, 320 은 시야가 넓지만 부정확하다. 프리셋으로 게임마다 고르는 이유가 이것.

그런데 게임PC 가 320 을 보내면 Jetson 에서 두 관측을 '동시에' 만들 수 있다 - 중앙 160 을
잘라 확대한 것과 320 원본. GPU 는 64% 놀고 있으니 추론 2회가 들어간다. 사람 눈과 같은
구조다(중심와 고해상도 + 주변시 광각).

값을 하려면 두 가지가 참이어야 한다:
  1) 조준 중인 표적에서 160 관측이 실제로 더 조용한가 (정밀도 이득이 실재하는가)
  2) 320 관측이 160 이 놓치는 표적을 실제로 잡는가 (시야 이득이 실재하는가)
둘 중 하나라도 아니면 추론 2회를 쓸 이유가 없다. 같은 프레임에서 나란히 잰다.
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
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    frames = np.load(a.frames, mmap_mode="r")
    eng = Engine(a.engine); M = eng.in_shape[-1]
    W = frames.shape[2]
    off = (W - 160)//2

    # 두 관측을 같은 프레임에서. 좌표는 둘 다 '320 캡처 화면 px' 로 통일해 비교한다.
    #   160 뷰: 모델 px -> x 0.5, 그리고 크롭 오프셋만큼 평행이동
    #   320 뷰: 모델 px -> x 1.0
    obs = {"fovea160": {0: [], 1: []}, "wide320": {0: [], 1: []}}
    for f in frames:
        wide = np.ascontiguousarray(f[:, :, :3])
        fov = np.ascontiguousarray(f[off:off+160, off:off+160, :3])
        dw = decode_per_class(eng.infer(preprocess(wide, M)), a.conf)
        df = decode_per_class(eng.infer(preprocess(fov, M)), a.conf)
        for c in (0, 1):
            w = dw.get(c)
            obs["wide320"][c].append(None if w is None else
                                     (w[0]*(W/M), w[1]*(W/M), w[2]*(W/M), w[3]*(W/M)))
            v = df.get(c)
            obs["fovea160"][c].append(None if v is None else
                                      (off + v[0]*0.5, off + v[1]*0.5, v[2]*0.5, v[3]*0.5))

    print("  프레임 %d, conf %.2f, 좌표계 = 320 캡처 화면 px\n" % (len(frames), a.conf))
    print("  [1] 정밀도 - 같은 표적을 두 관측이 얼마나 조용하게 보는가")
    print("    %-10s %-8s %8s %8s %8s" % ("클래스", "관측", "σ(cx)", "σ(cy)", "검출률"))
    print("    " + "-"*46)
    for c, name in ((0, "body"), (1, "head")):
        for k in ("wide320", "fovea160"):
            s = obs[k][c]
            n = sum(1 for x in s if x is not None)
            sx = wn([x[0] if x else None for x in s])
            sy = wn([x[1] if x else None for x in s])
            print("    %-10s %-8s %8s %8s %7.0f%%"
                  % (name, k, "%.3f" % sx if sx else "-", "%.3f" % sy if sy else "-",
                     100.0*n/len(s)))
        sxw = wn([x[0] if x else None for x in obs["wide320"][c]])
        syw = wn([x[1] if x else None for x in obs["wide320"][c]])
        sxf = wn([x[0] if x else None for x in obs["fovea160"][c]])
        syf = wn([x[1] if x else None for x in obs["fovea160"][c]])
        if sxw and sxf:
            print("      -> 160 이 X %+.0f%% / Y %+.0f%% (음수 = 더 조용)"
                  % (100*(sxf-sxw)/sxw, 100*(syf-syw)/syw))
    print()
    print("  [2] 시야 - 320 만 잡는 표적이 실제로 있는가")
    print("    %-10s %10s %10s %10s %10s"
          % ("클래스", "둘 다", "320만", "160만", "둘 다 없음"))
    print("    " + "-"*54)
    for c, name in ((0, "body"), (1, "head")):
        both = only_w = only_f = none = 0
        for w, v in zip(obs["wide320"][c], obs["fovea160"][c]):
            if w is not None and v is not None: both += 1
            elif w is not None: only_w += 1
            elif v is not None: only_f += 1
            else: none += 1
        print("    %-10s %10d %10d %10d %10d" % (name, both, only_w, only_f, none))
        if only_w:
            print("      -> 320 전용 검출 %.1f%% — 160 크롭 밖의 표적" % (100.0*only_w/len(frames)))


if __name__ == "__main__":
    main()
