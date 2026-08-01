#!/usr/bin/env python3
"""Draw a square with the MAKCU mouse - a standalone liveness test.

Mirrors MakcuConnection exactly: open at 115200, send the 9-byte baud-change
command, reopen at 4 Mbit/s, then stream ASCII "km.move(x,y)\\r\\n". Useful when
the aim app misbehaves and you need to know whether the device itself is alive
without starting the whole pipeline.

  ./makcu_square.py [--side PX] [--step PX] [--laps N] [--port DEV]
"""
import argparse
import os
import sys
import termios
import time

BAUD_CHANGE_CMD = bytes([0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00])
BOOT_BAUD = termios.B115200
WORK_BAUD = 0o10017          # B4000000 on Linux/aarch64 (not exposed by termios)


def open_serial(dev, speed):
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY)
    attrs = termios.tcgetattr(fd)
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs
    # 8N1, raw, no flow control, ignore modem lines
    cflag = termios.CS8 | termios.CREAD | termios.CLOCAL
    iflag = 0
    oflag = 0
    lflag = 0
    cc = list(cc)
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW,
                      [iflag, oflag, cflag, lflag, speed, speed, cc])
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--side", type=int, default=200, help="square side in mouse counts")
    ap.add_argument("--step", type=int, default=10, help="counts per move packet")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.006, help="seconds between packets")
    a = ap.parse_args()

    if not os.path.exists(a.port):
        sys.exit("no such device: %s" % a.port)

    # Step 1: boot baud, ask the device to switch to 4 Mbit/s
    fd = open_serial(a.port, BOOT_BAUD)
    os.write(fd, BAUD_CHANGE_CMD)
    time.sleep(0.05)
    os.close(fd)

    # Step 2: reopen at the working baud and drive it
    fd = open_serial(a.port, WORK_BAUD)
    print("opened %s @4Mbps; drawing %dx%d square, %d lap(s)" % (a.port, a.side, a.side, a.laps))

    n = max(1, a.side // a.step)
    sides = [(a.step, 0), (0, a.step), (-a.step, 0), (0, -a.step)]  # right, down, left, up
    names = ["→ right", "↓ down", "← left", "↑ up"]
    try:
        for lap in range(a.laps):
            for (dx, dy), nm in zip(sides, names):
                print("  lap %d  %s" % (lap + 1, nm))
                for _ in range(n):
                    os.write(fd, ("km.move(%d,%d)\r\n" % (dx, dy)).encode())
                    time.sleep(a.delay)
        print("done - if the cursor traced a square, MAKCU is fine")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
