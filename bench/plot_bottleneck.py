#!/usr/bin/env python3
"""Parse simple_inference perf status output and plot the pipeline bottleneck.

The app prints a carriage-return-updated status line that, with
stage_timing_enabled + perf_stats_enabled, includes:
  Aw:<avg>/<max>ms  (host frame-acquire wait)
  Su:<avg>/<max>us  (host submit cost)
  Cb:<avg>/<max>ms  (submit -> completion callback latency, ~end-to-end)
  St[h2d:a/m pre:a/m inf:a/m post:a/m d2h:a/m us]  (per-stage GPU timings)

Usage: python3 plot_bottleneck.py <status_log> [out_png]
"""
import re, sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "run.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bottleneck.png"

raw = open(LOG, "r", errors="replace").read()
# status updates are separated by '\r' (and the prelude by '\n')
chunks = re.split(r"[\r\n]+", raw)

st_re = re.compile(
    r"St\[h2d:([\d.]+)/(\d+) pre:([\d.]+)/(\d+) inf:([\d.]+)/(\d+) "
    r"post:([\d.]+)/(\d+) d2h:([\d.]+)/(\d+)us\]")
aw_re = re.compile(r"Aw:([\d.]+)/([\d.]+)ms")
cb_re = re.compile(r"Cb:([\d.]+)/([\d.]+)ms")
d_re  = re.compile(r" D:([\d.]+)")

samples = []
for c in chunks:
    m = st_re.search(c)
    if not m:
        continue
    g = [float(x) for x in m.groups()]
    s = dict(h2d=g[0], pre=g[2], inf=g[4], post=g[6], d2h=g[8],
             h2d_max=g[1], pre_max=g[3], inf_max=g[5], post_max=g[7], d2h_max=g[9])
    aw = aw_re.search(c); cb = cb_re.search(c); d = d_re.search(c)
    s["aw_ms"]  = float(aw.group(1)) if aw else 0.0
    s["cb_ms"]  = float(cb.group(1)) if cb else 0.0
    s["cb_max"] = float(cb.group(2)) if cb else 0.0
    s["fps"]    = float(d.group(1)) if d else 0.0
    samples.append(s)

if not samples:
    print("No stage-timing samples found. Was stage_timing_enabled=true and did frames flow?")
    sys.exit(1)

# Warm-up: drop the first 2 windows (graph capture / engine warmup)
warm = samples[2:] if len(samples) > 4 else samples
n = len(warm)
def avg(k): return sum(s[k] for s in warm) / n
def mx(k):  return max(s[k] for s in warm)

stages = ["h2d", "pre", "inf", "post", "d2h"]
labels = {"h2d": "H2D copy", "pre": "Preprocess", "inf": "Inference (TRT)",
          "post": "Postprocess", "d2h": "D2H copy"}
avg_us = {k: avg(k) for k in stages}
max_us = {k: mx(k + "_max") for k in stages}
gpu_total = sum(avg_us.values())

print(f"\n=== Pipeline stage breakdown ({n} windows, warmup dropped) ===")
print(f"{'stage':<18}{'avg us':>10}{'max us':>10}{'% of GPU':>10}")
for k in stages:
    print(f"{labels[k]:<18}{avg_us[k]:>10.1f}{max_us[k]:>10.0f}{100*avg_us[k]/gpu_total:>9.1f}%")
print(f"{'GPU total':<18}{gpu_total:>10.1f}")
print(f"\nHost acquire-wait Aw : {avg('aw_ms'):.3f} ms avg")
print(f"End-to-end Cb        : {avg('cb_ms'):.3f} ms avg / {mx('cb_max'):.3f} ms max")
print(f"Throughput D         : {avg('fps'):.1f} fps")
bottleneck = max(stages, key=lambda k: avg_us[k])
print(f"\n>>> Biggest GPU-pipeline bottleneck: {labels[bottleneck]} "
      f"({avg_us[bottleneck]:.1f} us, {100*avg_us[bottleneck]/gpu_total:.0f}% of GPU time)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: average stage breakdown bar chart (the bottleneck answer)
xs = list(range(len(stages)))
avals = [avg_us[k] for k in stages]
mvals = [max_us[k] for k in stages]
colors = ["#6c8ebf"] * len(stages)
colors[stages.index(bottleneck)] = "#d6604d"
ax1.bar(xs, avals, color=colors, label="avg")
ax1.plot(xs, mvals, "k.", markersize=9, label="window max")
for i, v in enumerate(avals):
    ax1.text(i, v, f"{v:.0f}us\n{100*v/gpu_total:.0f}%", ha="center", va="bottom", fontsize=9)
ax1.set_xticks(xs)
ax1.set_xticklabels([labels[k] for k in stages], rotation=20, ha="right")
ax1.set_ylabel("microseconds (us)")
ax1.set_title(f"Per-stage GPU latency  (GPU total ~{gpu_total:.0f}us, "
              f"end-to-end ~{avg('cb_ms'):.2f}ms)")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Panel 2: stacked time series of stages over the run
t = np.arange(n)
bottom = np.zeros(n)
series_colors = {"h2d": "#8dd3c7", "pre": "#ffffb3", "inf": "#fb8072",
                 "post": "#80b1d3", "d2h": "#bebada"}
for k in stages:
    vals = np.array([s[k] for s in warm])
    ax2.bar(t, vals, bottom=bottom, width=1.0, label=labels[k], color=series_colors[k])
    bottom += vals
ax2b = ax2.twinx()
ax2b.plot(t, [s["cb_ms"] * 1000 for s in warm], "k-", lw=1.4, label="Cb end-to-end (us)")
ax2.set_xlabel("perf window (#)")
ax2.set_ylabel("stacked stage latency (us)")
ax2b.set_ylabel("end-to-end Cb (us)")
ax2.set_title("Stage latency over time")
ax2.legend(loc="upper left", fontsize=8)
ax2b.legend(loc="upper right", fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"\nSaved graph -> {OUT}")
