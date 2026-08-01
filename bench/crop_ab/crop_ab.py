#!/usr/bin/env python3
"""Offline A/B: does a smaller capture crop, upscaled to the model input, reduce
the detector's centre noise - or make it worse?

The question cannot be answered from theory. Upscaling adds NO information; any
gain comes purely from the detection head having more output-grid cells across
the target (a 17px head spans ~2 cells at stride 8, ~4 cells at 2x). Whether that
helps depends on the model's stride and the scale distribution it was trained on,
so it has to be measured on THIS engine with THESE frames.

Method: replay the same captured frames two ways through the same engine -
  NATIVE : the 320 frame as-is                 (movement_scale 1.0)
  CROPPED: centre WxW crop, bilinear to 320    (movement_scale W/320)
- convert every detection back to SCREEN pixels so the two are comparable,
- report per-frame centre noise (first-difference/sqrt2, which removes the
  target's real motion) plus detection rate and box size.

Lower sigma_screen wins. If CROPPED is not clearly better, the idea is dead.
"""
import argparse
import sys

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except Exception as e:  # pragma: no cover
    sys.exit("need tensorrt + pycuda: %s" % e)

import cv2


class Engine:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))
        self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape)) * np.dtype(self.in_dtype).itemsize)
        self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * np.dtype(self.out_dtype).itemsize)
        self.h_out = np.empty(self.out_shape, dtype=self.out_dtype)
        self.stream = cuda.Stream()

    def infer(self, chw):
        h_in = np.ascontiguousarray(chw, dtype=self.in_dtype)
        cuda.memcpy_htod_async(self.d_in, h_in, self.stream)
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out.copy()


def preprocess(img_rgb, size):
    """Match the shipped GPU path: bilinear resize -> CHW -> /255."""
    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    chw = img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[None]


def decode(out, conf_thr, head_cls):
    """YOLO head: (1, 4+C, N) -> best box per class set. Returns (cx, cy, w, h, cls, conf)."""
    o = out[0] if out.ndim == 3 else out
    nc = o.shape[0] - 4
    scores = o[4:4 + nc]                       # (C, N)
    cls = np.argmax(scores, axis=0)
    conf = scores[cls, np.arange(scores.shape[1])]
    i = int(np.argmax(conf))
    if conf[i] <= conf_thr:
        return None
    return float(o[0, i]), float(o[1, i]), float(o[2, i]), float(o[3, i]), int(cls[i]), float(conf[i])


def run(frames, eng, model_in, crop, conf_thr, head_cls):
    """crop=None -> native. Returns list of (screen_cx, screen_cy, box_h_model, cls, conf)."""
    res = []
    H, W = frames.shape[1], frames.shape[2]
    for f in frames:
        img = f[:, :, :3]
        if crop:
            o = (W - crop) // 2
            img = img[o:o + crop, o:o + crop]
            scale = crop / float(model_in)      # model px -> screen px
            origin = o
        else:
            scale = W / float(model_in)
            origin = 0
        det = decode(eng.infer(preprocess(img, model_in)), conf_thr, head_cls)
        if det is None:
            res.append(None); continue
        cx, cy, bw, bh, c, cf = det
        res.append((origin + cx * scale, origin + cy * scale, bh, c, cf))
    return res


def sigma(seq, cls_filter=None):
    """First-difference/sqrt2 on consecutive frames with a detection (and the same
    class, so head<->body flips are not counted as noise)."""
    dx, dy = [], []
    for a, b in zip(seq, seq[1:]):
        if a is None or b is None:
            continue
        if cls_filter is not None and (a[3] != cls_filter or b[3] != cls_filter):
            continue
        if a[3] != b[3]:
            continue
        dx.append(b[0] - a[0]); dy.append(b[1] - a[1])
    if len(dx) < 20:
        return None
    return (float(np.std(dx) / np.sqrt(2)), float(np.std(dy) / np.sqrt(2)), len(dx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.npy")
    ap.add_argument("--engine",
                    default="/home/taeho/portfolio/inferencetool/engines/sunxds_0.8.3_320_fp16.engine")
    ap.add_argument("--crops", default="224,160,128")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--head-cls", type=int, default=1)
    a = ap.parse_args()

    frames = np.load(a.frames)
    print("frames %s" % (frames.shape,))
    eng = Engine(a.engine)
    model_in = eng.in_shape[-1]
    print("engine input %s output %s\n" % (eng.in_shape, eng.out_shape))

    print("  %-22s %9s %9s %8s %8s %8s" % ("setting", "sigX(px)", "sigY(px)", "det%", "box_h", "conf"))
    print("  " + "-" * 70)
    base = None
    for label, crop in [("NATIVE 320 (current)", None)] + \
                       [("CROP %d -> %d" % (c, model_in), c) for c in
                        (int(x) for x in a.crops.split(","))]:
        seq = run(frames, eng, model_in, crop, a.conf, a.head_cls)
        s = sigma(seq)
        got = [d for d in seq if d]
        if not s or not got:
            print("  %-22s   (too few detections)" % label); continue
        bh = float(np.mean([d[2] for d in got])); cf = float(np.mean([d[4] for d in got]))
        tag = ""
        if base is None:
            base = s
        else:
            tag = "  X %+.0f%% / Y %+.0f%%" % (100 * (s[0] - base[0]) / base[0],
                                               100 * (s[1] - base[1]) / base[1])
        print("  %-22s %9.2f %9.2f %7.0f%% %8.1f %8.2f%s" %
              (label, s[0], s[1], 100 * len(got) / len(seq), bh, cf, tag))
    print("\n  sigma is in SCREEN px (comparable across settings).")
    print("  CROP must be clearly LOWER on both axes to be worth the lost FOV.")


if __name__ == "__main__":
    main()
