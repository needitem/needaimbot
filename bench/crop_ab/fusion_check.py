#!/usr/bin/env python3
"""Is fusing the head and body boxes worth it?

Today the selector picks ONE box and throws the other away. If head and body are
two INDEPENDENT measurements of the same aim point, inverse-variance fusion would
cut the noise. But they come from the same image through the same network, so
their errors are probably CORRELATED - and correlated errors do not average out.
That is the whole question, and it is measurable:

  sigma_fused^2 = (s_h^2 s_b^2 (1-r^2)) / (s_h^2 + s_b^2 - 2 r s_h s_b)

with r the correlation of the two errors. r -> 1 means fusion buys nothing.

Decodes BOTH the best head box and the best body box per frame, maps each to the
aim point the controller would use, and measures per-frame noise plus their
correlation.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from crop_ab import Engine, preprocess  # noqa: E402

import cv2  # noqa: E402,F401  (used via preprocess)


def decode_per_class(out, conf_thr, nclass=2):
    """Best box for each class. Returns {cls: (cx, cy, w, h, conf)}."""
    o = out[0] if out.ndim == 3 else out
    nc = o.shape[0] - 4
    res = {}
    for c in range(min(nc, nclass)):
        sc = o[4 + c]
        i = int(np.argmax(sc))
        if sc[i] > conf_thr:
            res[c] = (float(o[0, i]), float(o[1, i]), float(o[2, i]), float(o[3, i]), float(sc[i]))
    return res


def aim_y(box, cls, head_cls, head_ap, body_ap):
    """Same geometry as the kernel: y1 + h * aim_point."""
    _, cy, _, bh, _ = box
    y1 = cy - bh * 0.5
    return y1 + bh * (head_ap if cls == head_cls else body_ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.npy")
    ap.add_argument("--engine",
                    default="/home/taeho/portfolio/inferencetool/engines/sunxds_0.8.3_320_fp16.engine")
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--head-cls", type=int, default=1)
    ap.add_argument("--head-ap", type=float, default=1.0)
    ap.add_argument("--body-ap", type=float, default=0.15)
    a = ap.parse_args()

    frames = np.load(a.frames)
    eng = Engine(a.engine)
    M = eng.in_shape[-1]
    W = frames.shape[2]
    off = (W - a.crop) // 2 if a.crop else 0
    scale = (a.crop / float(M)) if a.crop else (W / float(M))

    hx, hy, bx, by, both = [], [], [], [], 0
    for f in frames:
        img = f[:, :, :3]
        if a.crop:
            img = img[off:off + a.crop, off:off + a.crop]
        d = decode_per_class(eng.infer(preprocess(img, M)), a.conf)
        h = d.get(a.head_cls); b = d.get(1 - a.head_cls)
        if h is None or b is None:
            hx.append(None); bx.append(None); continue
        both += 1
        hx.append(off + h[0] * scale)
        hy.append(off + aim_y(h, a.head_cls, a.head_cls, a.head_ap, a.body_ap) * scale)
        bx.append(off + b[0] * scale)
        by.append(off + aim_y(b, 1 - a.head_cls, a.head_cls, a.head_ap, a.body_ap) * scale)

    print("frames %d | head+body 동시 검출 %d (%.0f%%)" %
          (len(frames), both, 100.0 * both / len(frames)))
    if both < 40:
        print("\n동시 검출이 너무 적어 융합 판정 불가.")
        print("머리가 잘 잡히는 거리(가까운 교전)에서 프레임을 더 모아야 합니다.")
        return

    HX = np.array([v for v in hx if v is not None]); HY = np.array(hy)
    BX = np.array([v for v in bx if v is not None]); BY = np.array(by)

    def stats(H, B, axis):
        dh = np.diff(H); db = np.diff(B)          # first difference removes real motion
        sh = float(np.std(dh) / np.sqrt(2)); sb = float(np.std(db) / np.sqrt(2))
        r = float(np.corrcoef(dh, db)[0, 1])
        den = sh * sh + sb * sb - 2 * r * sh * sb
        sf = float(np.sqrt(max(1e-9, (sh * sh * sb * sb * (1 - r * r)) / den))) if den > 1e-9 else min(sh, sb)
        best = min(sh, sb)
        print("  %s  head σ=%.2f  body σ=%.2f  상관 r=%+.2f" % (axis, sh, sb, r))
        print("      최적융합 σ=%.2f   vs 지금(더 나은 쪽만 사용) %.2f  ->  %+.0f%%"
              % (sf, best, 100 * (sf - best) / best))
        # offset between the two anchors: fusion needs them to agree on average
        print("      두 조준점 평균 차이 %.2fpx (표준편차 %.2f) - 이만큼은 편향으로 남음"
              % (float(np.mean(H - B)), float(np.std(H - B))))
        return sf, best

    print("\n=== 축별 (화면 픽셀) ===")
    stats(HX, BX, "X")
    stats(HY, BY, "Y")
    print("\n  r 이 1에 가까우면 같은 오차를 공유 -> 융합해도 안 줄어듦.")
    print("  또 두 조준점의 평균 차이는 융합해도 사라지지 않는 편향입니다.")


if __name__ == "__main__":
    main()
