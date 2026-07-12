#!/usr/bin/env python3
"""UDP frame-pattern load tester for NeedAimBot's two-PC transport.

Run the receiver on the inference PC and the sender on the game PC. The sender
uses the same V3 packet header shape as the app (game_pc/src/main.cpp) and
fragments synthetic frames with the configured payload size, so packet rate and
payload Mbps match the real stream without needing screen capture or TensorRT.
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time
from dataclasses import dataclass


# v3 wire format (game_pc/src/main.cpp): v2 layout plus a trailing uint64
# captureUnixMicros. Magic bumped so v2/v3 peers reject each other cleanly.
UDP_PACKET_V3_MAGIC = 0x33415047  # "GPA3" little-endian
UDP_PIXEL_FORMAT_RGB = 2
HEADER_V3 = struct.Struct("<IHHIIIHHIHHBBHQ")
HEADER_BYTES = HEADER_V3.size
assert HEADER_BYTES == 44, HEADER_BYTES


@dataclass
class FrameTrack:
    total_chunks: int
    first_seen: float
    received_count: int = 0
    received_mask: int = 0
    received_set: set[int] | None = None

    def mark(self, chunk_index: int) -> bool:
        if self.total_chunks <= 64:
            bit = 1 << chunk_index
            if self.received_mask & bit:
                return False
            self.received_mask |= bit
        else:
            if self.received_set is None:
                self.received_set = set()
            if chunk_index in self.received_set:
                return False
            self.received_set.add(chunk_index)
        self.received_count += 1
        return True

    def complete(self) -> bool:
        return self.received_count >= self.total_chunks


@dataclass
class WindowStats:
    packets: int = 0
    bytes: int = 0
    complete_frames: int = 0
    partial_frames: int = 0
    missing_whole_frames: int = 0
    duplicate_packets: int = 0
    invalid_packets: int = 0

    def reset(self) -> None:
        self.packets = 0
        self.bytes = 0
        self.complete_frames = 0
        self.partial_frames = 0
        self.missing_whole_frames = 0
        self.duplicate_packets = 0
        self.invalid_packets = 0


def mbps(byte_count: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return byte_count * 8.0 / elapsed / 1_000_000.0


def make_socket_buffer(sock: socket.socket, opt_name: int, requested_mb: int) -> None:
    if requested_mb <= 0:
        return
    requested_bytes = requested_mb * 1024 * 1024
    try:
        sock.setsockopt(socket.SOL_SOCKET, opt_name, requested_bytes)
    except OSError as exc:
        print(f"warning: failed to set socket buffer: {exc}", file=sys.stderr)


def run_sender(args: argparse.Namespace) -> int:
    if args.payload_bytes < 512 or args.payload_bytes > 60000:
        raise ValueError("--payload-bytes must be between 512 and 60000")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    bytes_per_pixel = 3
    frame_bytes = args.width * args.height * bytes_per_pixel
    total_chunks = math.ceil(frame_bytes / args.payload_bytes)
    if total_chunks <= 0 or total_chunks > 65535:
        raise ValueError("frame size/payload size produces invalid chunk count")

    payload_block = bytes(args.payload_bytes)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    make_socket_buffer(sock, socket.SO_SNDBUF, args.sndbuf_mb)
    if args.bind_ip:
        sock.bind((args.bind_ip, 0))

    destination = (args.host, args.port)
    interval = 1.0 / args.fps
    started_at = time.perf_counter()
    last_report = started_at
    next_frame_at = started_at
    frame_id = 0
    stats = WindowStats()

    print(
        "sender "
        f"dest={args.host}:{args.port} "
        f"frame={args.width}x{args.height}x{bytes_per_pixel} "
        f"bytes={frame_bytes} chunks={total_chunks} "
        f"fps={args.fps:.1f} payload={args.payload_bytes}"
    )

    try:
        while True:
            now = time.perf_counter()
            if args.duration > 0 and now - started_at >= args.duration:
                break
            if now < next_frame_at:
                time.sleep(min(next_frame_at - now, 0.005))
                continue
            if now - next_frame_at > interval * 4:
                next_frame_at = now

            capture_us = int(time.time() * 1_000_000)  # same in every chunk of a frame
            for chunk_index in range(total_chunks):
                offset = chunk_index * args.payload_bytes
                remaining = frame_bytes - offset
                chunk_size = min(remaining, args.payload_bytes)
                header = HEADER_V3.pack(
                    UDP_PACKET_V3_MAGIC,
                    HEADER_BYTES,
                    0,
                    frame_id & 0xFFFFFFFF,
                    offset,
                    frame_bytes,
                    chunk_index,
                    total_chunks,
                    chunk_size,
                    args.width,
                    args.height,
                    UDP_PIXEL_FORMAT_RGB,
                    bytes_per_pixel,
                    0,
                    capture_us,
                )
                packet = header + payload_block[:chunk_size]
                sent = sock.sendto(packet, destination)
                stats.packets += 1
                stats.bytes += sent

            stats.complete_frames += 1
            frame_id += 1
            next_frame_at += interval

            now = time.perf_counter()
            if now - last_report >= args.report_interval:
                elapsed = now - last_report
                print(
                    f"send fps={stats.complete_frames / elapsed:7.1f} "
                    f"pps={stats.packets / elapsed:8.1f} "
                    f"wire_payload={mbps(stats.bytes, elapsed):8.1f} Mbps"
                )
                stats.reset()
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    return 0


def cleanup_stale(
    frames: dict[int, FrameTrack],
    now: float,
    stale_after: float,
    stats: WindowStats,
) -> None:
    stale_ids = [frame_id for frame_id, track in frames.items() if now - track.first_seen >= stale_after]
    for frame_id in stale_ids:
        track = frames.pop(frame_id)
        if not track.complete():
            stats.partial_frames += 1


def run_receiver(args: argparse.Namespace) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    make_socket_buffer(sock, socket.SO_RCVBUF, args.rcvbuf_mb)
    sock.bind((args.bind_ip, args.port))
    sock.settimeout(0.05)

    frames: dict[int, FrameTrack] = {}
    max_frame_id_seen: int | None = None
    started_at = time.perf_counter()
    last_report = started_at
    stale_after = args.stale_ms / 1000.0
    stats = WindowStats()

    print(f"receiver bind={args.bind_ip}:{args.port} stale_ms={args.stale_ms}")

    try:
        while True:
            now = time.perf_counter()
            if args.duration > 0 and now - started_at >= args.duration:
                break
            if now - last_report >= args.report_interval:
                cleanup_stale(frames, now, stale_after, stats)
                elapsed = now - last_report
                total_frames = (
                    stats.complete_frames + stats.partial_frames + stats.missing_whole_frames
                )
                loss_pct = (
                    (stats.partial_frames + stats.missing_whole_frames) * 100.0 / total_frames
                    if total_frames > 0
                    else 0.0
                )
                print(
                    f"recv fps={stats.complete_frames / elapsed:7.1f} "
                    f"pps={stats.packets / elapsed:8.1f} "
                    f"wire_payload={mbps(stats.bytes, elapsed):8.1f} Mbps "
                    f"loss={loss_pct:5.2f}% "
                    f"partial={stats.partial_frames} "
                    f"missing={stats.missing_whole_frames} "
                    f"dup={stats.duplicate_packets} "
                    f"bad={stats.invalid_packets} "
                    f"active={len(frames)}"
                )
                stats.reset()
                last_report = now

            try:
                packet, _addr = sock.recvfrom(65536)
            except socket.timeout:
                cleanup_stale(frames, time.perf_counter(), stale_after, stats)
                continue

            stats.packets += 1
            stats.bytes += len(packet)
            if len(packet) < HEADER_BYTES:
                stats.invalid_packets += 1
                continue

            try:
                (
                    magic,
                    header_size,
                    _flags,
                    frame_id,
                    payload_offset,
                    frame_bytes,
                    chunk_index,
                    total_chunks,
                    chunk_size,
                    frame_width,
                    frame_height,
                    pixel_format,
                    bytes_per_pixel,
                    _reserved,
                    _capture_us,
                ) = HEADER_V3.unpack_from(packet)
            except struct.error:
                stats.invalid_packets += 1
                continue

            if (
                magic != UDP_PACKET_V3_MAGIC
                or header_size < HEADER_BYTES
                or header_size > len(packet)
                or chunk_index >= total_chunks
                or total_chunks <= 0
                or chunk_size != len(packet) - header_size
                or payload_offset + chunk_size > frame_bytes
                or frame_width <= 0
                or frame_height <= 0
                or bytes_per_pixel != 3
                or pixel_format != UDP_PIXEL_FORMAT_RGB
            ):
                stats.invalid_packets += 1
                continue

            if max_frame_id_seen is None:
                max_frame_id_seen = frame_id
            elif frame_id > max_frame_id_seen:
                gap = frame_id - max_frame_id_seen - 1
                if gap > 0:
                    stats.missing_whole_frames += gap
                max_frame_id_seen = frame_id

            track = frames.get(frame_id)
            if track is None:
                track = FrameTrack(total_chunks=total_chunks, first_seen=time.perf_counter())
                frames[frame_id] = track
            elif track.total_chunks != total_chunks:
                stats.invalid_packets += 1
                continue

            if not track.mark(chunk_index):
                stats.duplicate_packets += 1
                continue

            if track.complete():
                frames.pop(frame_id, None)
                stats.complete_frames += 1
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sender = subparsers.add_parser("sender", help="send synthetic fragmented frames")
    sender.add_argument("--host", required=True, help="receiver IP address")
    sender.add_argument("--port", type=int, default=5017)
    sender.add_argument("--bind-ip", default="", help="optional local NIC IP")
    sender.add_argument("--width", type=int, default=256)
    sender.add_argument("--height", type=int, default=256)
    sender.add_argument("--payload-bytes", type=int, default=60000)
    sender.add_argument("--fps", type=float, default=144.0)
    sender.add_argument("--duration", type=float, default=30.0, help="seconds; 0 means run until Ctrl+C")
    sender.add_argument("--sndbuf-mb", type=int, default=8)
    sender.add_argument("--report-interval", type=float, default=1.0)
    sender.set_defaults(func=run_sender)

    receiver = subparsers.add_parser("receiver", help="receive and measure synthetic frames")
    receiver.add_argument("--port", type=int, default=5017)
    receiver.add_argument("--bind-ip", default="0.0.0.0")
    receiver.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means run until Ctrl+C")
    receiver.add_argument("--rcvbuf-mb", type=int, default=32)
    receiver.add_argument("--stale-ms", type=float, default=200.0)
    receiver.add_argument("--report-interval", type=float, default=1.0)
    receiver.set_defaults(func=run_receiver)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
