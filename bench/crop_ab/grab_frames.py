#!/usr/bin/env python3
"""Capture raw frames off the game_pc UDP stream and save them as .npy.

Used to answer the crop question OFFLINE: does feeding the detector a smaller
crop upscaled to the model input actually reduce its centre noise, or does it
make things worse? Guessing is not good enough - upscaling adds no information,
so the whole benefit rests on the detection head having more output-grid cells
per target, which is model-specific and must be measured.

Reassembles the V2 chunk protocol (44-byte header, see game_pc/src/main.cpp).
The aim app must be stopped while this runs - it binds the same port.

  ./grab_frames.py --count 300 --out frames.npy
"""
import argparse
import socket
import struct
import sys

import numpy as np

HDR = "<IHHIIIHHIHHBBHQ"          # 44 bytes, UDPPacketHeaderV2
HDR_SIZE = struct.calcsize(HDR)
MAGIC = None                      # learned from the first packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--count", type=int, default=300, help="frames to keep")
    ap.add_argument("--out", default="frames.npy")
    ap.add_argument("--timeout", type=float, default=20.0)
    a = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 << 20)
    try:
        s.bind(("0.0.0.0", a.port))
    except OSError as e:
        sys.exit("bind failed (%s) - stop the aim app first" % e)
    s.settimeout(a.timeout)

    frames = []
    partial = {}                  # frameId -> [buffer, got_bytes, total, w, h, bpp]
    print("listening on :%d ..." % a.port)
    while len(frames) < a.count:
        try:
            pkt, _ = s.recvfrom(65535)
        except socket.timeout:
            print("timeout - is the streamer running?")
            break
        if len(pkt) < HDR_SIZE:
            continue
        (magic, hsz, flags, fid, off, fbytes, cidx, ctot,
         csz, w, h, pxfmt, bpp, _rsv, _ts) = struct.unpack(HDR, pkt[:HDR_SIZE])
        if hsz != HDR_SIZE or fbytes == 0 or w == 0 or h == 0:
            continue
        body = pkt[HDR_SIZE:HDR_SIZE + csz]
        st = partial.get(fid)
        if st is None:
            st = [bytearray(fbytes), 0, fbytes, w, h, bpp]
            partial[fid] = st
            if len(partial) > 8:                      # drop stale partials
                for k in sorted(partial)[:-8]:
                    partial.pop(k, None)
        if off + len(body) <= st[2]:
            st[0][off:off + len(body)] = body
            st[1] += len(body)
        if st[1] >= st[2]:
            arr = np.frombuffer(bytes(st[0]), dtype=np.uint8)
            try:
                frames.append(arr.reshape(st[4], st[3], st[5]))
            except ValueError:
                pass
            partial.pop(fid, None)
            if len(frames) % 50 == 0:
                print("  %d/%d" % (len(frames), a.count))
    s.close()
    if not frames:
        sys.exit("no complete frames captured")
    out = np.stack(frames)
    np.save(a.out, out)
    print("saved %s  shape=%s  dtype=%s" % (a.out, out.shape, out.dtype))


if __name__ == "__main__":
    main()
