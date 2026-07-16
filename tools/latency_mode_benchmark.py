#!/usr/bin/env python3
"""Compare frame transport strategies with a simple latency simulation.

The model is intentionally small: display frames are produced at a fixed
refresh cadence, capture/network/control/inference each add configurable
latency and jitter, and max in-flight inference is one frame.
"""

from __future__ import annotations

import argparse
import bisect
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable


MODES = ("continuous_latest", "credit_latest", "request_capture")
MODE_SEED_OFFSET = {
    "continuous_latest": 17,
    "credit_latest": 29,
    "request_capture": 43,
}


@dataclass(frozen=True)
class Params:
    refresh_fps: float
    continuous_send_fps: float
    width: int
    height: int
    bytes_per_pixel: int
    capture_ms: float
    capture_jitter_ms: float
    frame_tx_ms: float
    frame_tx_jitter_ms: float
    control_oneway_ms: float
    control_jitter_ms: float
    duration_ms: float
    warmup_ms: float
    events: int
    seed: int

    @property
    def period_ms(self) -> float:
        return 1000.0 / self.refresh_fps

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.bytes_per_pixel


def clipped_normal(rng: random.Random, mean: float, sd: float, lo: float, hi: float) -> float:
    if sd <= 0:
        return min(max(mean, lo), hi)
    for _ in range(8):
        value = rng.gauss(mean, sd)
        if lo <= value <= hi:
            return value
    return min(max(mean, lo), hi)


def percentile(values: Iterable[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def make_timeline(params: Params, seed: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    phase = random.Random(params.seed).random() * params.period_ms
    max_seq = int(params.duration_ms / params.period_ms) + 10000
    capture_times: list[float] = []
    arrival_times: list[float] = []

    for seq in range(max_seq):
        capture_delay = clipped_normal(
            rng,
            params.capture_ms,
            params.capture_jitter_ms,
            max(0.0, params.capture_ms - 4.0 * params.capture_jitter_ms),
            params.capture_ms + 6.0 * params.capture_jitter_ms,
        )
        tx_delay = clipped_normal(
            rng,
            params.frame_tx_ms,
            params.frame_tx_jitter_ms,
            max(0.01, params.frame_tx_ms - 4.0 * params.frame_tx_jitter_ms),
            params.frame_tx_ms + 6.0 * params.frame_tx_jitter_ms,
        )
        captured_at = phase + seq * params.period_ms + capture_delay
        capture_times.append(captured_at)
        arrival_times.append(captured_at + tx_delay)

    return capture_times, arrival_times


def infer_latency(rng: random.Random, mean_ms: float) -> float:
    return clipped_normal(
        rng,
        mean_ms,
        max(0.05, mean_ms * 0.10),
        max(0.01, mean_ms * 0.65),
        mean_ms * 1.45,
    )


def control_latency(rng: random.Random, params: Params) -> float:
    return clipped_normal(
        rng,
        params.control_oneway_ms,
        params.control_jitter_ms,
        max(0.0, params.control_oneway_ms - 4.0 * params.control_jitter_ms),
        params.control_oneway_ms + 6.0 * params.control_jitter_ms,
    )


def simulate(mode: str, infer_ms: float, params: Params) -> list[tuple[float, float, float, float]]:
    seed = params.seed + int(infer_ms * 100) + MODE_SEED_OFFSET[mode]
    rng = random.Random(seed)
    capture_times, arrival_times = make_timeline(params, seed + 2000)

    actions: list[tuple[float, float, float, float]] = []
    ready_at = 0.0
    last_seq = -1

    while ready_at < params.duration_ms:
        infer_ms_sample = infer_latency(rng, infer_ms)
        ctrl_ms_sample = control_latency(rng, params)

        if mode == "continuous_latest":
            latest_arrived_seq = bisect.bisect_right(arrival_times, ready_at) - 1
            if latest_arrived_seq > last_seq:
                seq = latest_arrived_seq
                start_at = ready_at
            else:
                seq = last_seq + 1
                start_at = max(ready_at, arrival_times[seq])
            captured_at = capture_times[seq]

        elif mode == "credit_latest":
            credit_arrives_at = ready_at + ctrl_ms_sample
            latest_captured_seq = bisect.bisect_right(capture_times, credit_arrives_at) - 1
            seq = max(latest_captured_seq, last_seq + 1, 0)
            captured_at = capture_times[seq]
            tx_ms = max(0.01, arrival_times[seq] - capture_times[seq])
            start_at = max(credit_arrives_at, captured_at) + tx_ms

        elif mode == "request_capture":
            request_arrives_at = ready_at + ctrl_ms_sample
            seq = max(0, bisect.bisect_left(capture_times, request_arrives_at))
            captured_at = capture_times[seq]
            start_at = arrival_times[seq]

        else:
            raise ValueError(f"unknown mode: {mode}")

        action_at = start_at + infer_ms_sample
        if action_at >= params.warmup_ms:
            actions.append((action_at, captured_at, start_at, ready_at))
        ready_at = action_at
        last_seq = seq

    return actions


def summarize(mode: str, infer_ms: float, params: Params) -> dict[str, float]:
    actions = simulate(mode, infer_ms, params)
    action_times = [row[0] for row in actions]
    capture_times = [row[1] for row in actions]
    start_times = [row[2] for row in actions]
    ready_times = [row[3] for row in actions]

    waits = [start - ready for start, ready in zip(start_times, ready_times)]
    ages = [action - captured for action, captured in zip(action_times, capture_times)]

    rng = random.Random(params.seed + 9000 + int(infer_ms * 100))
    event_latencies = []
    event_window_ms = max(1.0, params.duration_ms - params.warmup_ms - 1000.0)
    for _ in range(params.events):
        event_at = params.warmup_ms + rng.random() * event_window_ms
        idx = bisect.bisect_left(capture_times, event_at)
        if idx < len(actions):
            event_latencies.append(action_times[idx] - event_at)

    duration_s = (action_times[-1] - action_times[0]) / 1000.0 if len(action_times) > 1 else 0.0
    action_fps = (len(action_times) - 1) / duration_s if duration_s > 0 else 0.0
    send_fps = params.continuous_send_fps if mode == "continuous_latest" else action_fps

    return {
        "action_fps": action_fps,
        "send_fps": send_fps,
        "payload_mbps": send_fps * params.frame_bytes * 8 / 1_000_000.0,
        "wait_avg": statistics.mean(waits),
        "wait_p95": percentile(waits, 95),
        "age_avg": statistics.mean(ages),
        "age_p95": percentile(ages, 95),
        "event_avg": statistics.mean(event_latencies),
        "event_p50": percentile(event_latencies, 50),
        "event_p95": percentile(event_latencies, 95),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-fps", type=float, default=144.0)
    parser.add_argument("--continuous-send-fps", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--bytes-per-pixel", type=int, default=3)
    parser.add_argument("--capture-ms", type=float, default=0.35)
    parser.add_argument("--capture-jitter-ms", type=float, default=0.06)
    parser.add_argument("--frame-tx-ms", type=float, default=1.80)
    parser.add_argument("--frame-tx-jitter-ms", type=float, default=0.18)
    parser.add_argument("--control-oneway-ms", type=float, default=0.15)
    parser.add_argument("--control-jitter-ms", type=float, default=0.04)
    parser.add_argument("--infer-ms", type=float, nargs="+", default=[4.0, 6.0, 8.0, 12.0, 20.0])
    parser.add_argument("--duration-ms", type=float, default=180_000.0)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--events", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    continuous_send_fps = args.continuous_send_fps or args.refresh_fps
    params = Params(
        refresh_fps=args.refresh_fps,
        continuous_send_fps=continuous_send_fps,
        width=args.width,
        height=args.height,
        bytes_per_pixel=args.bytes_per_pixel,
        capture_ms=args.capture_ms,
        capture_jitter_ms=args.capture_jitter_ms,
        frame_tx_ms=args.frame_tx_ms,
        frame_tx_jitter_ms=args.frame_tx_jitter_ms,
        control_oneway_ms=args.control_oneway_ms,
        control_jitter_ms=args.control_jitter_ms,
        duration_ms=args.duration_ms,
        warmup_ms=args.warmup_ms,
        events=args.events,
        seed=args.seed,
    )

    print(
        "params "
        f"refresh_fps={params.refresh_fps:.0f} "
        f"period_ms={params.period_ms:.3f} "
        f"capture_ms~N({params.capture_ms:.2f},{params.capture_jitter_ms:.2f}) "
        f"frame_tx_ms~N({params.frame_tx_ms:.2f},{params.frame_tx_jitter_ms:.2f}) "
        f"control_oneway_ms~N({params.control_oneway_ms:.2f},{params.control_jitter_ms:.2f}) "
        f"payload_bytes={params.frame_bytes}"
    )
    print("latency columns are milliseconds; payload Mbps excludes UDP/IP/Ethernet overhead")

    for infer_ms in args.infer_ms:
        print(f"\n=== infer_ms_mean={infer_ms:.1f} ===")
        print(
            "mode                 act_fps send_fps payload_Mbps "
            "wait_avg wait_p95 age_avg age_p95 event_avg event_p50 event_p95"
        )
        for mode in MODES:
            row = summarize(mode, infer_ms, params)
            print(
                f"{mode:<20} "
                f"{row['action_fps']:7.1f} "
                f"{row['send_fps']:8.1f} "
                f"{row['payload_mbps']:12.1f} "
                f"{row['wait_avg']:8.2f} "
                f"{row['wait_p95']:8.2f} "
                f"{row['age_avg']:7.2f} "
                f"{row['age_p95']:7.2f} "
                f"{row['event_avg']:9.2f} "
                f"{row['event_p50']:9.2f} "
                f"{row['event_p95']:9.2f}"
            )


if __name__ == "__main__":
    main()
