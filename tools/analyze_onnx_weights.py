#!/usr/bin/env python3
"""
ONNX Weight Statistics Analyzer for Weight-Space Programming experiments.
Extracts detailed per-initializer statistics, anomaly detection,
detection head analysis, and quantization readiness assessment.
"""

import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper

MODEL_PATH = "/home/taeho/portfolio/inferencetool/engines/sunxds_0.7.8.onnx"
OUTPUT_PATH = "/tmp/claude-1000/-home-taeho-portfolio-inferencetool/abbbed58-5fd7-45dd-9c2b-2250e48754d2/scratchpad/weight_statistics.txt"


def compute_stats(name, tensor):
    """Compute comprehensive statistics for a single weight tensor."""
    flat = tensor.flatten().astype(np.float64)
    n = flat.size
    if n == 0:
        return None

    abs_flat = np.abs(flat)
    mn, mx = float(flat.min()), float(flat.max())
    mean = float(flat.mean())
    std = float(flat.std())
    median = float(np.median(flat))
    sparsity = float(np.sum(abs_flat < 1e-6) / n * 100)
    l1 = float(abs_flat.sum())
    l2 = float(np.sqrt((flat ** 2).sum()))
    hist_counts, hist_edges = np.histogram(flat, bins=10)
    dynamic_range = mx - mn

    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "elements": n,
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "median": median,
        "sparsity_pct": sparsity,
        "l1_norm": l1,
        "l2_norm": l2,
        "dynamic_range": dynamic_range,
        "hist_counts": hist_counts.tolist(),
        "hist_edges": hist_edges.tolist(),
    }


def find_detection_heads(model, initializer_map):
    """
    Find the final Conv layers that produce the [1,15,2100] output.
    Walk backward from the output node.
    """
    # Build a map from output name -> node
    output_map = {}
    for node in model.graph.node:
        for out in node.output:
            output_map[out] = node

    # The model output name
    model_output_names = [o.name for o in model.graph.output]

    detection_nodes = []
    visited = set()

    def trace_back(name, depth=0):
        if name in visited or depth > 10:
            return
        visited.add(name)
        if name in output_map:
            node = output_map[name]
            if node.op_type == "Conv":
                detection_nodes.append(node)
                return
            elif node.op_type == "Reshape":
                # Trace through reshape
                for inp in node.input:
                    trace_back(inp, depth + 1)
            elif node.op_type == "Transpose":
                for inp in node.input:
                    trace_back(inp, depth + 1)
            elif node.op_type == "Concat":
                for inp in node.input:
                    trace_back(inp, depth + 1)
            elif node.op_type == "Sigmoid":
                for inp in node.input:
                    trace_back(inp, depth + 1)
            else:
                # Generic: trace all inputs
                for inp in node.input:
                    trace_back(inp, depth + 1)

    for out_name in model_output_names:
        trace_back(out_name)

    return detection_nodes


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = onnx.load(MODEL_PATH)
    graph = model.graph

    # Extract all initializers
    initializer_map = {}
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        initializer_map[init.name] = tensor

    print(f"Found {len(initializer_map)} initializers")

    # Compute per-initializer statistics
    all_stats = []
    for name, tensor in initializer_map.items():
        s = compute_stats(name, tensor)
        if s:
            all_stats.append(s)

    # Sort by element count descending
    all_stats.sort(key=lambda x: x["elements"], reverse=True)

    # Anomaly detection
    high_sparsity = [s for s in all_stats if s["sparsity_pct"] > 50]
    low_std = [s for s in all_stats if s["std"] < 1e-4 and s["elements"] > 1]
    outlier_layers = []
    for s in all_stats:
        if s["std"] > 0:
            max_sigma = max(abs(s["max"] - s["mean"]), abs(s["min"] - s["mean"])) / s["std"]
            if max_sigma > 3:
                outlier_layers.append((s, max_sigma))

    # Bias anomalies: bias terms far from zero
    bias_anomalies = []
    for s in all_stats:
        name_lower = s["name"].lower()
        if "bias" in name_lower or (len(s["shape"]) == 1 and s["elements"] < 2048):
            if abs(s["mean"]) > 1.0 or s["dynamic_range"] > 5.0:
                bias_anomalies.append(s)

    # Find detection head Conv nodes
    det_nodes = find_detection_heads(model, initializer_map)

    # Build node input map for all Conv nodes
    all_conv_nodes = [n for n in graph.node if n.op_type == "Conv"]

    # Quantization readiness
    quant_stats = []
    for s in all_stats:
        if s["elements"] > 10:  # skip tiny tensors
            # Kurtosis for distribution shape
            tensor = initializer_map[s["name"]].flatten().astype(np.float64)
            if s["std"] > 0:
                kurt = float(np.mean(((tensor - s["mean"]) / s["std"]) ** 4) - 3)
            else:
                kurt = 0.0
            quant_stats.append({
                "name": s["name"],
                "dynamic_range": s["dynamic_range"],
                "std": s["std"],
                "kurtosis": kurt,
                "elements": s["elements"],
            })

    quant_stats.sort(key=lambda x: x["dynamic_range"], reverse=True)

    # === Generate report ===
    lines = []
    lines.append("=" * 100)
    lines.append("ONNX WEIGHT STATISTICS REPORT")
    lines.append(f"Model: {MODEL_PATH}")
    lines.append(f"Total initializers: {len(all_stats)}")
    total_params = sum(s["elements"] for s in all_stats)
    lines.append(f"Total parameters: {total_params:,}")
    lines.append("=" * 100)

    # Section 1: Per-initializer statistics
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 1: PER-INITIALIZER STATISTICS (sorted by element count)")
    lines.append("=" * 100)
    for i, s in enumerate(all_stats):
        lines.append(f"\n--- [{i+1}/{len(all_stats)}] {s['name']} ---")
        lines.append(f"  Shape: {s['shape']}  Dtype: {s['dtype']}  Elements: {s['elements']:,}")
        lines.append(f"  Min: {s['min']:.8f}  Max: {s['max']:.8f}  Range: {s['dynamic_range']:.8f}")
        lines.append(f"  Mean: {s['mean']:.8f}  Std: {s['std']:.8f}  Median: {s['median']:.8f}")
        lines.append(f"  Sparsity: {s['sparsity_pct']:.2f}%  L1: {s['l1_norm']:.4f}  L2: {s['l2_norm']:.4f}")
        lines.append(f"  Histogram (10 bins): {s['hist_counts']}")
        lines.append(f"  Bin edges: {[f'{e:.6f}' for e in s['hist_edges']]}")

    # Section 2: Anomaly detection
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 2: ANOMALY DETECTION")
    lines.append("=" * 100)

    lines.append(f"\n--- High Sparsity Layers (>50%): {len(high_sparsity)} found ---")
    for s in high_sparsity:
        lines.append(f"  {s['name']}: {s['sparsity_pct']:.1f}% sparse, shape={s['shape']}")

    lines.append(f"\n--- Very Low Std Layers (<1e-4): {len(low_std)} found ---")
    for s in low_std:
        lines.append(f"  {s['name']}: std={s['std']:.8f}, mean={s['mean']:.8f}, shape={s['shape']}")

    lines.append(f"\n--- Outlier Layers (max/min > 3*std from mean): {len(outlier_layers)} found ---")
    for s, sigma in sorted(outlier_layers, key=lambda x: x[1], reverse=True)[:30]:
        lines.append(f"  {s['name']}: {sigma:.1f} sigma, range=[{s['min']:.6f}, {s['max']:.6f}], std={s['std']:.6f}")

    lines.append(f"\n--- Bias Anomalies (mean>1.0 or range>5.0): {len(bias_anomalies)} found ---")
    for s in bias_anomalies:
        lines.append(f"  {s['name']}: mean={s['mean']:.6f}, range={s['dynamic_range']:.6f}, shape={s['shape']}")

    # Section 3: Detection head weights
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 3: DETECTION HEAD WEIGHTS")
    lines.append("=" * 100)

    lines.append(f"\nDetection head Conv nodes found by tracing back from output: {len(det_nodes)}")
    for node in det_nodes:
        lines.append(f"\n  Node: {node.name or '(unnamed)'}, Op: {node.op_type}")
        lines.append(f"  Inputs: {list(node.input)}")
        lines.append(f"  Outputs: {list(node.output)}")

        for inp_name in node.input:
            if inp_name in initializer_map:
                tensor = initializer_map[inp_name]
                lines.append(f"\n  Weight: {inp_name}")
                lines.append(f"    Shape: {tensor.shape}, Dtype: {tensor.dtype}")
                lines.append(f"    Min: {tensor.min():.8f}, Max: {tensor.max():.8f}")
                lines.append(f"    Mean: {tensor.mean():.8f}, Std: {tensor.std():.8f}")

                # Print full values if small enough
                if tensor.size <= 500:
                    lines.append(f"    FULL VALUES:")
                    if len(tensor.shape) == 1:
                        # Bias vector
                        lines.append(f"    {np.array2string(tensor, precision=8, separator=', ', max_line_width=120)}")
                    else:
                        # Weight tensor - print per output channel summary
                        for oc in range(tensor.shape[0]):
                            ch_data = tensor[oc].flatten()
                            lines.append(f"      Channel {oc}: min={ch_data.min():.8f} max={ch_data.max():.8f} mean={ch_data.mean():.8f}")
                            if ch_data.size <= 50:
                                lines.append(f"        {np.array2string(ch_data, precision=8, separator=', ', max_line_width=120)}")

    # Also search for Conv layers that output 15 channels (likely detection heads)
    lines.append("\n--- All Conv layers with 15 output channels ---")
    for node in all_conv_nodes:
        if len(node.input) >= 1 and node.input[0]:
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name and weight_name in initializer_map:
                w = initializer_map[weight_name]
                if w.shape[0] == 15:
                    lines.append(f"\n  Conv node: {node.name or '(unnamed)'}")
                    lines.append(f"    Weight: {weight_name}, shape={w.shape}")
                    lines.append(f"    Output: {list(node.output)}")

                    # Print full weight statistics per channel
                    for oc in range(15):
                        ch = w[oc].flatten()
                        lines.append(f"    Ch{oc:2d}: min={ch.min():.8f} max={ch.max():.8f} mean={ch.mean():.8f} std={ch.std():.8f}")

                    # Print bias if exists
                    if len(node.input) >= 3 and node.input[2] in initializer_map:
                        bias = initializer_map[node.input[2]]
                        lines.append(f"\n    BIAS ({node.input[2]}): shape={bias.shape}")
                        lines.append(f"    FULL BIAS VALUES:")
                        lines.append(f"    {np.array2string(bias, precision=10, separator=', ', max_line_width=120)}")
                        lines.append(f"    Interpretation (YOLO 15=3*(1+4+C) with 3 anchors):")
                        lines.append(f"      Anchors x 5+C per anchor. For 3 anchors, 5 values each = 15 (objectness + 4 bbox + 0 classes)")
                        lines.append(f"      OR 3 anchors x (4 bbox + 1 obj + 0 cls) = 15 if no extra classes")

                        # Parse as 3 anchors x 5 values
                        if bias.size == 15:
                            b = bias.flatten()
                            for a in range(3):
                                chunk = b[a*5:(a+1)*5]
                                lines.append(f"      Anchor {a}: bbox=[{chunk[0]:.8f}, {chunk[1]:.8f}, {chunk[2]:.8f}, {chunk[3]:.8f}] obj={chunk[4]:.8f}")

    # Section 4: Quantization readiness
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 4: QUANTIZATION READINESS")
    lines.append("=" * 100)

    lines.append("\n--- Top 20 layers by dynamic range (hardest to quantize) ---")
    for qs in quant_stats[:20]:
        lines.append(f"  {qs['name']}: range={qs['dynamic_range']:.6f}, std={qs['std']:.6f}, kurtosis={qs['kurtosis']:.2f}, elements={qs['elements']:,}")

    lines.append("\n--- Bottom 20 layers by dynamic range (easiest / best for INT8) ---")
    for qs in quant_stats[-20:]:
        lines.append(f"  {qs['name']}: range={qs['dynamic_range']:.6f}, std={qs['std']:.6f}, kurtosis={qs['kurtosis']:.2f}, elements={qs['elements']:,}")

    # Layers with high kurtosis (peaky distribution, harder for uniform quantization)
    high_kurt = sorted([qs for qs in quant_stats if abs(qs["kurtosis"]) > 10], key=lambda x: abs(x["kurtosis"]), reverse=True)
    lines.append(f"\n--- High kurtosis layers (|kurt|>10, non-uniform, harder to quantize): {len(high_kurt)} ---")
    for qs in high_kurt[:20]:
        lines.append(f"  {qs['name']}: kurtosis={qs['kurtosis']:.2f}, range={qs['dynamic_range']:.6f}")

    # Summary
    lines.append("\n" + "=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Total initializers analyzed: {len(all_stats)}")
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"High sparsity layers (>50%): {len(high_sparsity)}")
    lines.append(f"Near-constant layers (std<1e-4): {len(low_std)}")
    lines.append(f"Layers with outlier values: {len(outlier_layers)}")
    lines.append(f"Bias anomalies: {len(bias_anomalies)}")
    lines.append(f"Detection head Conv nodes: {len(det_nodes)}")

    # Weight-space modification candidates
    lines.append("\n--- WEIGHT-SPACE MODIFICATION CANDIDATES ---")
    lines.append("Priority targets for direct LLM weight modification:")
    lines.append("1. Detection head biases (15-element vectors): Directly control confidence/bbox prediction")
    lines.append("2. High-sparsity layers: Already partially zeroed, safe to prune further")
    lines.append("3. BatchNorm parameters: Scale/shift are low-dimensional and interpretable")
    lines.append("4. Final layer weights: Small 1x1 conv weights that map features to detections")

    report = "\n".join(lines)

    # Write to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {OUTPUT_PATH}")
    print(f"\n{'=' * 60}")
    print("KEY FINDINGS (console summary)")
    print(f"{'=' * 60}")
    print(f"Total params: {total_params:,}")
    print(f"High sparsity (>50%): {len(high_sparsity)} layers")
    print(f"Near-constant (std<1e-4): {len(low_std)} layers")
    print(f"Outlier layers: {len(outlier_layers)}")
    print(f"Detection head nodes: {len(det_nodes)}")

    # Print detection head biases to console
    print(f"\n--- DETECTION HEAD BIASES ---")
    for node in all_conv_nodes:
        if len(node.input) >= 2:
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name and weight_name in initializer_map:
                w = initializer_map[weight_name]
                if w.shape[0] == 15:
                    print(f"\nConv node outputting 15 channels: {node.name or '(unnamed)'}")
                    print(f"  Weight: {weight_name}, shape={w.shape}")
                    if len(node.input) >= 3 and node.input[2] in initializer_map:
                        bias = initializer_map[node.input[2]]
                        print(f"  Bias: {node.input[2]}")
                        print(f"  Values: {bias.flatten()}")
                    else:
                        print(f"  No bias term found")

    # Print top anomalies
    print(f"\n--- TOP ANOMALIES ---")
    for s in high_sparsity[:5]:
        print(f"  HIGH SPARSITY: {s['name']}: {s['sparsity_pct']:.1f}%")
    for s in low_std[:5]:
        print(f"  LOW STD: {s['name']}: std={s['std']:.2e}")
    for s, sigma in sorted(outlier_layers, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  OUTLIER: {s['name']}: {sigma:.1f} sigma")


if __name__ == "__main__":
    main()
