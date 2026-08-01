#!/usr/bin/env python3
"""body 박스로 머리 위치를 가장 정확히 짚는 방법은 무엇인가?

지금은 body_aim_point = 박스 '높이'의 비율이다. 하지만 높이는 다리가 엄폐/지형에
가려지면 크게 줄어드는 반면 머리는 그대로다. 그러면 비율 기준 조준점이 머리에서
떨어진다. 폭(width)은 자세·엄폐에 덜 민감하므로 더 안정적인 기준일 수 있다.

head 박스가 같이 잡힌 프레임에서 '진짜 머리 조준점'을 정답으로 두고, 여러 예측식의
잔차 산포를 비교한다. 산포가 작을수록 머리에 타이트하게 붙는다.

  A  top + k*h      현재 방식 (높이 비율)
  B  top + k*w      폭 기준 오프셋
  C  top + k        고정 픽셀 오프셋
  D  top + k*min(w*a, h)  폭 기준이되 높이로 상한
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from crop_ab import Engine, preprocess          # noqa: E402
from fusion_check import decode_per_class       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.npy")
    ap.add_argument("--engine",
                    default="/home/taeho/portfolio/inferencetool/engines/sunxds_0.8.3_320_fp16.engine")
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--head-cls", type=int, default=1)
    ap.add_argument("--head-ap", type=float, default=0.9)
    a = ap.parse_args()

    frames = np.load(a.frames)
    eng = Engine(a.engine); M = eng.in_shape[-1]
    W = frames.shape[2]
    off = (W - a.crop) // 2 if a.crop else 0

    tgt, btop, bw, bh = [], [], [], []
    for f in frames:
        img = f[:, :, :3]
        if a.crop:
            img = img[off:off + a.crop, off:off + a.crop]
        d = decode_per_class(eng.infer(preprocess(img, M)), a.conf)
        h = d.get(a.head_cls); b = d.get(1 - a.head_cls)
        if h is None or b is None:
            continue
        hcx, hcy, hw, hh, _ = h
        bcx, bcy, bwid, bhgt, _ = b
        tgt.append((hcy - hh*0.5) + hh*a.head_ap)   # 정답: head 앵커 y
        btop.append(bcy - bhgt*0.5)                 # body 박스 상단
        bw.append(bwid); bh.append(bhgt)

    y = np.array(tgt); top = np.array(btop); w = np.array(bw); hh = np.array(bh)
    n = len(y)
    print("동시 검출 %d 프레임 (model px)" % n)
    if n < 40:
        sys.exit("샘플 부족")
    d = y - top                                     # 상단에서 머리까지의 거리
    print("  body 박스  높이 %.0f (%.0f~%.0f)   폭 %.0f (%.0f~%.0f)"
          % (hh.mean(), hh.min(), hh.max(), w.mean(), w.min(), w.max()))
    print("  상단->머리 거리  평균 %.1f  표준편차 %.1f" % (d.mean(), d.std()))
    print()

    def fit(name, basis):
        k = float(np.sum(basis*d) / np.sum(basis*basis))
        res = d - k*basis
        print("  %-26s k=%8.4f   잔차 표준편차 %6.2f model px  (%5.2f 화면px)"
              % (name, k, res.std(), res.std() * (a.crop/float(M) if a.crop else 1.0)))
        return res.std(), k

    print("  예측식별 잔차 (작을수록 머리에 타이트):")
    r_h, k_h = fit("A  top + k*height (현재)", hh)
    r_w, k_w = fit("B  top + k*width", w)
    r_c, k_c = fit("C  top + k (고정 px)", np.ones_like(d))
    # D: 폭 기준이되 높이가 폭*ratio 보다 작으면 높이를 쓴다(가려진 경우 대비)
    ratio = float(np.median(hh/w))
    basis_d = np.minimum(w*ratio, hh)
    r_d, k_d = fit("D  top + k*min(w*%.1f, h)" % ratio, basis_d)

    best = min((r_h, "A"), (r_w, "B"), (r_c, "C"), (r_d, "D"))
    print()
    print("  -> 최선: %s  (현재 A 대비 %+.0f%%)" % (best[1], 100*(best[0]-r_h)/r_h))
    if best[1] != "A":
        print("     현재 방식은 높이 변동을 그대로 조준점 흔들림으로 넘긴다.")
    print()
    print("  참고: 현재 config 의 body_aim_point 는 A 방식의 k 이며,")
    print("        이 프레임 기준 최적 k = %.4f" % k_h)


if __name__ == "__main__":
    main()
