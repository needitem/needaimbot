#!/usr/bin/env python3
"""Find the body_aim_point that puts the BODY anchor on the same line as the HEAD
anchor, killing the vertical jump that every head<->body selection flip injects.

body_aim_point is a FRACTION of the body box height, so it can only cancel a bias
that scales with box height. If the measured bias is instead roughly constant in
pixels, no single fraction fixes it at all distances - that has to be checked, not
assumed, so the bias is regressed against box height here.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from crop_ab import Engine, preprocess  # noqa: E402
from fusion_check import decode_per_class, aim_y  # noqa: E402


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
    eng = Engine(a.engine); M = eng.in_shape[-1]
    W = frames.shape[2]
    off = (W - a.crop) // 2 if a.crop else 0

    dy_model, body_h, head_h = [], [], []
    for f in frames:
        img = f[:, :, :3]
        if a.crop:
            img = img[off:off + a.crop, off:off + a.crop]
        d = decode_per_class(eng.infer(preprocess(img, M)), a.conf)
        h = d.get(a.head_cls); b = d.get(1 - a.head_cls)
        if h is None or b is None:
            continue
        ay_h = aim_y(h, a.head_cls, a.head_cls, a.head_ap, a.body_ap)
        ay_b = aim_y(b, 1 - a.head_cls, a.head_cls, a.head_ap, a.body_ap)
        dy_model.append(ay_h - ay_b)      # >0: head anchor sits BELOW body anchor
        body_h.append(b[3]); head_h.append(h[3])

    dy = np.array(dy_model); bh = np.array(body_h)
    n = len(dy)
    print("동시 검출 %d 프레임 (model px 기준)" % n)
    if n < 40:
        sys.exit("샘플 부족")
    print("  body 박스 높이  평균 %.1f  (%.0f ~ %.0f)" % (bh.mean(), bh.min(), bh.max()))
    print("  head 박스 높이  평균 %.1f" % np.mean(head_h))
    print("  앵커 차이 dy    평균 %+.2f  표준편차 %.2f" % (dy.mean(), dy.std()))
    print()

    # Does the bias scale with box height? If yes a fraction can cancel it.
    r = float(np.corrcoef(bh, dy)[0, 1])
    k = float(np.sum(bh * dy) / np.sum(bh * bh))     # dy ~ k * bh  (через origin)
    resid_frac = dy - k * bh
    resid_const = dy - dy.mean()
    print("  === 편향이 박스 높이에 비례하나? ===")
    print("  상관 r(box_h, dy) = %+.2f" % r)
    print("  비례 모델  dy = %.4f * box_h   잔차 표준편차 %.2f" % (k, resid_frac.std()))
    print("  상수 모델  dy = %.2f          잔차 표준편차 %.2f" % (dy.mean(), resid_const.std()))
    scales = resid_frac.std() < resid_const.std()
    print("  -> %s" % ("높이에 비례 : body_aim_point(분수)로 제거 가능"
                       if scales else
                       "거의 상수 : 분수로는 완전히 못 지움 (거리별로 남음)"))
    print()

    new_ap = a.body_ap + k
    print("  === 권장값 ===")
    print("  body_aim_point  %.3f -> %.3f   (+%.4f = 평균 %.2f model px 하향)"
          % (a.body_ap, new_ap, k, k * bh.mean()))
    print("  적용 시 잔여 편향 %.2f -> ~0 model px (%.2f -> ~0 화면 px)"
          % (dy.mean(), dy.mean() * (a.crop / float(M)) if a.crop else dy.mean()))
    print("  전환 시 튐(표준편차) %.2f model px 는 남습니다 - 이건 검출 노이즈라 못 지움"
          % resid_frac.std())


if __name__ == "__main__":
    main()
