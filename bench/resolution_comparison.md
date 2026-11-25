# Resolution Comparison Benchmark: 256 vs 320 vs 640 (FP16)

**Date**: 2026-11-25
**Device**: NVIDIA Jetson Orin (Compute Capability 8.7, 16 SMs, 30696 MiB)
**TensorRT**: 10.3.0
**Benchmark Config**: 1000 iterations, 500ms warmup, 100 avgRuns

## Engine Details

| Property | 256x256 | 320x320 | 640x640 |
|---|---|---|---|
| Engine File | sunxds_0.8.2_256_fp16.engine | sunxds_0.8.3_320_fp16.engine | sunxds_0.8.3_640_fp16.engine |
| Engine Size | 6.2 MB | 6.9 MB | 8.0 MB |
| Input Shape | 1x3x256x256 | 1x3x320x320 | 1x3x640x640 |
| Output Shape | 1x6x1344 | 1x6x2100 | 1x6x8400 |
| Pixel Count | 65,536 | 102,400 | 409,600 |
| Detection Anchors | 1,344 | 2,100 | 8,400 |
| Device Memory | 1.68 MiB | 2.18 MiB | 9.08 MiB |

## Throughput

| Resolution | Throughput (QPS) | Relative to 320 |
|---|---|---|
| **256x256** | **768.2** | +7.1% faster |
| **320x320** | **717.6** | baseline |
| **640x640** | **449.6** | -37.4% slower |

## End-to-End Latency (Host)

| Metric | 256x256 | 320x320 | 640x640 |
|---|---|---|---|
| Min | 1.249 ms | 1.429 ms | 2.374 ms |
| Max | 1.516 ms | 1.788 ms | 2.684 ms |
| **Mean** | **1.331 ms** | **1.448 ms** | **2.452 ms** |
| **Median** | **1.320 ms** | **1.448 ms** | **2.452 ms** |
| P90 | 1.371 ms | 1.453 ms | 2.461 ms |
| P95 | 1.392 ms | 1.455 ms | 2.466 ms |
| P99 | 1.507 ms | 1.460 ms | 2.474 ms |

## GPU Compute Time

| Metric | 256x256 | 320x320 | 640x640 |
|---|---|---|---|
| Min | 1.209 ms | 1.379 ms | 2.188 ms |
| Max | 1.469 ms | 1.727 ms | 2.482 ms |
| **Mean** | **1.293 ms** | **1.390 ms** | **2.220 ms** |
| **Median** | **1.283 ms** | **1.390 ms** | **2.219 ms** |

## Latency Stability

| Metric | 256x256 | 320x320 | 640x640 |
|---|---|---|---|
| Jitter (max - min) | 0.267 ms | 0.359 ms | 0.310 ms |
| P99 - Median | 0.187 ms | 0.012 ms | 0.022 ms |
| CoV Warning | **Yes (2.63%)** | No | No |
| Enqueue Bottleneck | Yes | Yes | No |

**Key observation**: 256x256 has the lowest mean latency but the **worst tail latency stability** -- the P99-to-median spread is 0.187ms (14.2% above median), compared to 320x320's 0.012ms (0.8% above median). The 320 and 640 engines are remarkably stable.

## Scaling Analysis

| Comparison | Pixel Ratio | Latency Ratio (mean) | Anchor Ratio |
|---|---|---|---|
| 320 vs 256 | 1.56x | 1.09x (+8.8%) | 1.56x |
| 640 vs 320 | 4.00x | 1.69x (+69.3%) | 4.00x |
| 640 vs 256 | 6.25x | 1.84x (+84.2%) | 6.25x |

The latency does **not** scale linearly with pixel count. 320x320 is only 8.8% slower than 256x256 despite processing 56% more pixels and anchors. This indicates the GPU is underutilized at 256x256 resolution (confirmed by the enqueue bottleneck warning). At 640x640, the GPU compute becomes the actual bottleneck, with better utilization.

## Detection Quality vs Latency Tradeoff

| Resolution | Mean Latency | Max FPS | Anchors | Small Object Detection | FOV Coverage |
|---|---|---|---|---|---|
| 256x256 | 1.33 ms | 768 | 1,344 | Poor | Narrow |
| 320x320 | 1.45 ms | 718 | 2,100 | Moderate | Standard |
| 640x640 | 2.45 ms | 450 | 8,400 | Excellent | Wide |

### Resolution impact on detection
- **256x256**: 56% fewer anchors than 320. Small or distant targets may be missed. Minimum detectable target size is roughly 2x larger (in pixels) compared to 640.
- **320x320**: The production baseline. 2,100 anchors provide reasonable coverage for mid-range targets.
- **640x640**: 4x the anchors of 320. Significantly better at detecting small or distant targets. The 8,400 anchors cover far more spatial positions.

## Recommendation

**For real-time aimbot on Jetson Orin, 320x320 is the optimal choice.**

Rationale:

1. **Latency delta is marginal**: 320 is only 0.12ms slower than 256 in mean GPU compute (1.390ms vs 1.293ms). In a full pipeline with capture, preprocessing, NMS, and rendering, this ~0.1ms difference is negligible.

2. **320 is far more stable**: The 256 engine has a 2.63% coefficient of variance and poor tail latency (P99 is 14% above median). For real-time aiming, **consistent latency matters more than peak latency**. A single frame spike of 1.5ms (256's P99) versus 320's rock-solid 1.46ms P99 means 320 delivers more predictable frame timing.

3. **Detection quality gap**: 256x256 has 37% fewer anchors (1,344 vs 2,100), resulting in meaningfully worse detection of small or distant targets. In an aimbot context, missing a target is worse than adding 0.1ms of latency.

4. **640 is overkill for this use case**: While 640 provides excellent detection, 2.45ms per inference means the detection alone consumes a significant portion of a 16.7ms frame budget (at 60 FPS). The extra 1ms over 320 is substantial when every millisecond counts for end-to-end pipeline latency.

5. **Pipeline headroom**: At 1.45ms inference, 320x320 leaves ~15.2ms per frame for capture + preprocess + NMS + mouse movement at 60 FPS, or ~5.2ms at 150 FPS. The 256 resolution would only recover ~0.1ms of that budget.

### When to consider alternatives
- **Use 256x256** only if the pipeline is extremely latency-constrained (sub-1ms budget for inference) and targets are always large/close.
- **Use 640x640** if switching to a lower frame rate (30 FPS) where the 2.45ms budget is affordable, or for offline analysis / screenshot-based detection where latency is not critical.
