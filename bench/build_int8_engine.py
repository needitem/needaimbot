#!/usr/bin/env python3
"""Build an int8+fp16 TensorRT engine with entropy calibration from collected
game frames.

The frames are raw HWC RGB uint8 dumps produced by:
    simple_inference --collect-calib <dir> [N]

Preprocessing here MUST match the CUDA inference path exactly:
    float = uint8 * (1/255), channel order RGB, layout CHW, no BGR swap.
The model input is square (imgsz); frames are resized to it if needed.

Usage:
    python3 build_int8_engine.py --onnx model.onnx --frames <dir> \
            --out engine.engine [--imgsz 256] [--max-frames 500]
"""
import argparse, glob, os, sys, re
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_frame(path, imgsz):
    m = re.search(r"_(\d+)x(\d+)_rgb", os.path.basename(path))
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size != w * h * 3:
        return None
    img = buf.reshape(h, w, 3).astype(np.float32)
    if (w, h) != (imgsz, imgsz):
        # nearest-ish resize via numpy indexing (calibration tolerates this)
        ys = (np.linspace(0, h - 1, imgsz)).astype(np.int32)
        xs = (np.linspace(0, w - 1, imgsz)).astype(np.int32)
        img = img[ys][:, xs]
    img *= (1.0 / 255.0)                 # match scale_factor
    chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW, RGB order preserved
    return np.ascontiguousarray(chw)


class FrameCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, frames, imgsz, cache_path, batch=8):
        super().__init__()
        self.frames = frames
        self.imgsz = imgsz
        self.batch = batch
        self.cache_path = cache_path
        self.idx = 0
        self.dev = cuda.mem_alloc(batch * 3 * imgsz * imgsz * 4)  # fp32

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.idx + self.batch > len(self.frames):
            return None
        arrs = []
        for p in self.frames[self.idx:self.idx + self.batch]:
            a = load_frame(p, self.imgsz)
            if a is None:
                a = np.zeros((3, self.imgsz, self.imgsz), np.float32)
            arrs.append(a)
        self.idx += self.batch
        batch_np = np.ascontiguousarray(np.stack(arrs).astype(np.float32))
        cuda.memcpy_htod(self.dev, batch_np)
        if self.idx % (self.batch * 10) == 0:
            print(f"  calibrated {self.idx}/{len(self.frames)} frames")
        return [int(self.dev)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--frames", required=True, help="dir of *_WxH_rgb.bin frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "*_rgb.bin")))[: args.max_frames]
    if len(frames) < args.batch:
        print(f"ERROR: only {len(frames)} frames found (need >= {args.batch}). "
              f"Collect more with --collect-calib.")
        return 1
    print(f"[int8] {len(frames)} calibration frames, imgsz={args.imgsz}, batch={args.batch}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)          # int8 + fp16 mixed
    config.set_flag(trt.BuilderFlag.INT8)
    try:
        config.builder_optimization_level = 5      # max tactic search
    except Exception:
        pass
    cache_path = os.path.splitext(args.out)[0] + ".calib"
    config.int8_calibrator = FrameCalibrator(frames, args.imgsz, cache_path, args.batch)

    print("[int8] building engine (calibration + tactic search, takes a while)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: engine build failed")
        return 1
    with open(args.out, "wb") as f:
        f.write(serialized)
    print(f"[int8] saved -> {args.out}  ({len(serialized)/1e6:.1f} MB)")
    print(f"[int8] calibration cache -> {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
