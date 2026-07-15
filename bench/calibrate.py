#!/usr/bin/env python3
"""Turn a calibration CSV (from simple_inference's calibration_log_path) into the
real detector-noise / timing numbers to feed the sim (aim_tune.py, aim_sim_full.py).

Capture protocol
  1. Aim the game crosshair at a STATIONARY target (a wall pattern, a static dummy).
  2. Do NOT hold the aim key (force_aim_on=false) so the bot detects but does not
     move -> the detected centre only varies by detector noise.
  3. Set "calibration_log_path": "calib.csv" in the config, run ~30 s, Ctrl-C.
  4. python3 bench/calibrate.py calib.csv [head_class_id]

Reads: t_us,lat_us,aiming,hasTarget,classId,conf,cx,cy,w,h,emit_dx,emit_dy
"""
import sys, csv, math

def mean(a): return sum(a) / len(a) if a else 0.0
def std(a):
    if len(a) < 2: return 0.0
    m = mean(a); return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1))
def pct(a, p):
    if not a: return 0.0
    s = sorted(a); i = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]
def diffs(a): return [a[i] - a[i - 1] for i in range(1, len(a))]
def autocorr1(a):
    if len(a) < 3: return 0.0
    m = mean(a); num = sum((a[i] - m) * (a[i - 1] - m) for i in range(1, len(a)))
    den = sum((x - m) ** 2 for x in a)
    return num / den if den else 0.0

def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: r[k] for k in r})
    for r in rows:
        for k in ("t_us", "lat_us"): r[k] = int(float(r[k]))
        for k in ("aiming", "hasTarget", "classId"): r[k] = int(float(r[k]))
        for k in ("conf", "cx", "cy", "w", "h"): r[k] = float(r[k])
        for k in ("emit_dx", "emit_dy"): r[k] = int(float(r[k]))
        for k in ("mscale_x", "mscale_y"):
            r[k] = float(r.get(k, 1.0)) if r.get(k, "") not in ("", None) else 1.0
        r["inject_cum_x"] = float(r.get("inject_cum_x", 0.0)) if r.get("inject_cum_x", "") not in ("", None) else 0.0
    return rows

def deadtime_lag(cx, inj, maxlag):
    """Lagged cross-correlation: cx(t) tracks -inj(t - deadtime). Returns the lag
    (in frames) that best aligns them, and the correlation strength."""
    mc = mean(cx); mi = mean(inj)
    a = [x - mc for x in cx]; b = [x - mi for x in inj]
    best_lag, best = 0, 0.0
    for lag in range(0, maxlag):
        num = sum(a[i] * b[i - lag] for i in range(lag, len(a)))
        da = math.sqrt(sum(a[i] ** 2 for i in range(lag, len(a))))
        db = math.sqrt(sum(b[i - lag] ** 2 for i in range(lag, len(a))))
        c = num / (da * db) if da * db else 0.0
        if abs(c) > abs(best):
            best, best_lag = c, lag
    return best_lag, best

def detrend(a, win=9):
    """subtract a local moving average -> leaves the high-freq (noise) residual."""
    n = len(a); out = []
    for i in range(n):
        lo = max(0, i - win // 2); hi = min(n, i + win // 2 + 1)
        out.append(a[i] - sum(a[lo:hi]) / (hi - lo))
    return out

def noise_of(vals):
    """high-freq (detector) sigma + low-freq drift sigma + AR(1)."""
    tot = std(vals); res = detrend(vals); wht = std(res)
    drift = math.sqrt(max(0.0, tot * tot - wht * wht))
    return tot, wht, drift, autocorr1(res)

def reconstruct_world(det):
    """Undo the view rotation you applied (emit / mscale) so the target's
    position is in a fixed WORLD frame. Its high-freq jitter = real detector
    noise UNDER THE MOVING VIEW (blur included); its low-freq = real target
    motion. Independent of which controller ran."""
    twx, twy = [], []
    vx = vy = 0.0
    for r in det:
        sx = r["mscale_x"] if r["mscale_x"] > 1e-6 else 1.0
        sy = r["mscale_y"] if r["mscale_y"] > 1e-6 else 1.0
        twx.append(r["cx"] + vx); twy.append(r["cy"] + vy)
        vx += r["emit_dx"] / sx; vy += r["emit_dy"] / sy
    return twx, twy

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    head_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rows = load(path)
    if not rows:
        print("empty CSV"); return
    n = len(rows)
    span_s = (rows[-1]["t_us"] - rows[0]["t_us"]) / 1e6

    det = [r for r in rows if r["hasTarget"] == 1]
    dropout = 100.0 * (n - len(det)) / n
    head_rate = 100.0 * sum(1 for r in det if r["classId"] == head_id) / len(det) if det else 0.0

    still = [r for r in det if r["aiming"] == 0]      # static view (baseline)
    aim_on = [r for r in det if r["aiming"] == 1]     # moving view (real condition)
    conf = [r["conf"] for r in det]
    w = mean([r["w"] for r in det]); h = mean([r["h"] for r in det])
    dt_ms = [d / 1000.0 for d in diffs([r["t_us"] for r in rows])]
    lat = [r["lat_us"] / 1000.0 for r in rows if r["lat_us"] >= 0]
    fps = 1000.0 / mean(dt_ms) if mean(dt_ms) else 144.0

    p = print
    p("=" * 68)
    p("CALIBRATION  %s   (%d frames, %.1f s)" % (path, n, span_s))
    p("=" * 68)

    def report(tag, cx, cy):
        _, wx, dx_, rx = noise_of(cx); _, wy, dy_, ry = noise_of(cy)
        p("  %s" % tag)
        p("    white (high-freq) sigma      X %5.2f  Y %5.2f   <- sim white noise" % (wx, wy))
        p("    drift (low-freq)  sigma      X %5.2f  Y %5.2f   AR %.2f/%.2f" % (dx_, dy_, rx, ry))
        p("    Y/X noise ratio              %5.2f            (>1 = Y-heavy)" % (wy / wx if wx else 0))
        return wx, wy

    off = on = None
    if len(still) > 100:
        off = report("AIM-OFF baseline (static view, no blur), %d fr" % len(still),
                     [r["cx"] for r in still], [r["cy"] for r in still]); p("")
    if len(aim_on) > 100:
        # reconstruct WORLD frame (undo your view rotation) = the real condition
        twx, twy = reconstruct_world(aim_on)
        on = report("AIM-ON real (moving view + blur), reconstructed, %d fr" % len(aim_on), twx, twy)
        maf = detrend(twx, 9)                          # residual removed -> smooth motion
        smooth = [twx[i] - maf[i] for i in range(len(twx))]
        vel = mean([abs(smooth[i] - smooth[i - 1]) for i in range(1, len(smooth))])
        p("    real target speed            %5.2f px/frame (~%.0f px/s)" % (vel, vel * fps)); p("")
    if off and on and off[0]:
        p("  MOTION-BLUR factor  aim-on / aim-off :  X %.2f  Y %.2f" %
          (on[0] / off[0], (on[1] / off[1]) if off[1] else 0)); p("")

    # step-response dead-time (aim-OFF frames where the injection pulses)
    step = [r for r in det if r["aiming"] == 0]
    if step:
        inj = [r["inject_cum_x"] for r in step]
        if max(inj) - min(inj) > 1.0:
            lag, corr = deadtime_lag([r["cx"] for r in step], inj, 18)
            p("  STEP-RESPONSE DEAD-TIME  (emit -> visible in the detection)")
            p("    lag                          %d frames  (~%.1f ms)   corr %.2f" %
              (lag, lag / fps * 1000.0, abs(corr)))
            p("    full round trip: MAKCU + game input + render + capture + UDP + inference")
            if abs(corr) < 0.3:
                p("    (weak corr -> use a bigger calibration_step_px or a cleaner still target)")
            p("")

    p("  DETECTION RELIABILITY")
    p("    head-selection rate            %5.1f %%   <- sim p_head" % head_rate)
    p("    dropout (no target)            %5.1f %%   <- sim dropout" % dropout)
    p("    confidence mean/p10            %.2f / %.2f" % (mean(conf), pct(conf, 10)))
    p("    mean box w x h (px)            %.1f x %.1f" % (w, h)); p("")
    p("  TIMING")
    p("    frame dt mean/std (ms)         %.2f / %.2f  -> %.0f fps" % (mean(dt_ms), std(dt_ms), fps))
    if lat:
        p("    inference latency mean/p95     %.2f / %.2f ms  (pipeline part only)" % (mean(lat), pct(lat, 95)))
    p("")
    use = on or off or (0.0, 0.0)
    p("  PASTE INTO SIM (use the AIM-ON numbers - that is the real condition):")
    p("    white_x=%.2f  white_y=%.2f  p_head=%.2f  dropout=%.3f  box_h=%.1f" %
      (use[0], use[1], head_rate / 100, dropout / 100, h))
    p("=" * 68)

if __name__ == "__main__":
    main()
