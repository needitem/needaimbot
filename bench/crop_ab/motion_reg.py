#!/usr/bin/env python3
"""표적 속도를 '박스 중심 차분' 대신 '픽셀 정합'으로 구하면 더 정밀한가?

지금 리드 항이 쓰는 속도는 노이즈 낀 박스 중심 두 개의 차분이다. 중심 하나가
sigma~4px 면 차분은 sigma~5.7px/frame - 그래서 v_ema 0.2 로 무겁게 평활해야 하고,
그 평활이 곧 리드의 지연이 된다.

프레임을 '포개서' 표적 영역의 픽셀을 직접 정합하면 (phase correlation) 박스 추정을
거치지 않으므로 훨씬 정밀할 수 있다. 정밀해지면 평활을 줄일 수 있고 = 리드가 빨라진다.

두 추정기의 프레임간 노이즈를 같은 프레임에서 비교한다. 정답(ground truth)은 없지만,
부드럽게 움직이는 표적에서는 속도 추정이 매끄러워야 하므로 '추정의 2차 차분'
(= 가속도 노이즈) 이 작을수록 정밀하다.
"""
import argparse
import sys

import numpy as np
import cv2

sys.path.insert(0, ".")
from crop_ab import Engine, preprocess          # noqa: E402
from fusion_check import decode_per_class       # noqa: E402


def phase_shift(a, b):
    """a -> b 로의 (dx, dy) 서브픽셀 변위. 둘 다 float32 grayscale, 같은 크기."""
    if a.shape != b.shape or min(a.shape) < 8:
        return None
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx, dy, resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.npy")
    ap.add_argument("--engine",
                    default="/home/taeho/portfolio/inferencetool/engines/sunxds_0.8.3_320_fp16.engine")
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.15, help="박스 주변 여유 비율")
    a = ap.parse_args()

    frames = np.load(a.frames)
    eng = Engine(a.engine); M = eng.in_shape[-1]
    W = frames.shape[2]
    off = (W - a.crop) // 2 if a.crop else 0
    scale = (a.crop / float(M)) if a.crop else 1.0

    # 1) 각 프레임의 body 박스(모델 좌표) 검출
    boxes = []
    grays = []
    for f in frames:
        img = f[off:off+a.crop, off:off+a.crop, :3] if a.crop else f[:, :, :3]
        grays.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32))
        d = decode_per_class(eng.infer(preprocess(img, M)), a.conf)
        boxes.append(d.get(0))

    # 2) 두 추정기의 속도 (화면 px/frame)
    v_box, v_reg = [], []
    for i in range(1, len(frames)):
        b0, b1 = boxes[i-1], boxes[i]
        if b0 is None or b1 is None:
            v_box.append(None); v_reg.append(None); continue
        # (a) 박스 중심 차분 - 현재 방식
        v_box.append(((b1[0]-b0[0])*scale, (b1[1]-b0[1])*scale))
        # (b) 표적 영역 픽셀 정합 - 제안 방식
        cx, cy, bw, bh, _ = b0
        s = a.crop / float(M) if a.crop else 1.0     # 모델->이미지 좌표
        x0 = int((cx-bw*0.5*(1+a.pad))*s); x1 = int((cx+bw*0.5*(1+a.pad))*s)
        y0 = int((cy-bh*0.5*(1+a.pad))*s); y1 = int((cy+bh*0.5*(1+a.pad))*s)
        H = grays[i].shape[0]
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(H, x1), min(H, y1)
        if x1-x0 < 16 or y1-y0 < 16:
            v_reg.append(None); continue
        r = phase_shift(grays[i-1][y0:y1, x0:x1], grays[i][y0:y1, x0:x1])
        v_reg.append((r[0], r[1]) if r else None)

    def noise(vs, axis):
        """추정의 프레임간 변화(2차차분) = 속도추정 노이즈. 작을수록 정밀."""
        v = [x[axis] for x in vs if x is not None]
        if len(v) < 30: return None
        return float(np.std(np.diff(v)) / np.sqrt(2))

    n_ok = sum(1 for x in v_reg if x is not None)
    print("프레임 %d, 두 추정 모두 가능한 구간 %d\n" % (len(frames), n_ok))
    print("  %-28s %10s %10s" % ("속도 추정 방식", "X 노이즈", "Y 노이즈"))
    print("  " + "-" * 52)
    for lbl, vs in (("박스 중심 차분 (현재)", v_box), ("픽셀 정합 (제안)", v_reg)):
        nx, ny = noise(vs, 0), noise(vs, 1)
        print("  %-28s %10s %10s" % (
            lbl, "%.3f" % nx if nx else "-", "%.3f" % ny if ny else "-"))
    bx, by = noise(v_box, 0), noise(v_box, 1)
    rx, ry = noise(v_reg, 0), noise(v_reg, 1)
    if bx and rx:
        print("\n  -> 정합 방식이 X %+.0f%% / Y %+.0f%% (음수 = 더 정밀)"
              % (100*(rx-bx)/bx, 100*(ry-by)/by))
        print("     정밀해지면 v_ema 를 키울 수 있고(평활 감소) = 리드 반응이 빨라진다.")


if __name__ == "__main__":
    main()
