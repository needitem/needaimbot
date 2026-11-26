#!/usr/bin/env python3
"""
modify_onnx_weights.py - Weight-space modifications for YOLO ONNX models
Inspired by MIT CSAIL's Weight-Space Programming concept.

Applies conservative, gradient-free modifications to neural network weights
to improve speed, accuracy, or INT8 quantization readiness.

Model format assumed:
  Input:  [1, 3, H, W]
  Output: [1, (4+C), N]  where 4=bbox(cx,cy,w,h), C=num_classes, N=anchors
  Class scores are direct confidence (no separate objectness channel).

Usage:
  python3 tools/modify_onnx_weights.py --mode bias-tune
  python3 tools/modify_onnx_weights.py --mode prune-filters
  python3 tools/modify_onnx_weights.py --mode quant-prep
  python3 tools/modify_onnx_weights.py --mode analyze        # dry-run analysis only
  python3 tools/modify_onnx_weights.py --mode all
"""

import argparse
import os
import sys
import copy
from collections import OrderedDict

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_initializer_map(model):
    """Return {name: TensorProto} for all initializers."""
    return OrderedDict((init.name, init) for init in model.graph.initializer)


def get_weight_array(tensor_proto):
    """Convert ONNX TensorProto to numpy array."""
    return numpy_helper.to_array(tensor_proto)


def set_weight_array(tensor_proto, arr):
    """Replace data in an existing TensorProto with a numpy array."""
    new_tp = numpy_helper.from_array(arr, name=tensor_proto.name)
    tensor_proto.ClearField('raw_data')
    tensor_proto.ClearField('float_data')
    tensor_proto.ClearField('int32_data')
    tensor_proto.ClearField('int64_data')
    tensor_proto.ClearField('double_data')
    if new_tp.HasField('raw_data'):
        tensor_proto.raw_data = new_tp.raw_data
    elif len(new_tp.float_data) > 0:
        tensor_proto.float_data[:] = new_tp.float_data
    tensor_proto.data_type = new_tp.data_type
    tensor_proto.dims[:] = new_tp.dims


def find_conv_bias_pairs(model):
    """Find Conv nodes and their weight/bias initializer names."""
    init_names = set(i.name for i in model.graph.initializer)
    pairs = []
    for node in model.graph.node:
        if node.op_type == 'Conv':
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None
            if weight_name and weight_name in init_names:
                pairs.append({
                    'node_name': node.name or node.output[0],
                    'weight': weight_name,
                    'bias': bias_name if (bias_name and bias_name in init_names) else None,
                    'output': node.output[0] if node.output else None,
                })
    return pairs


def is_detection_head_conv(conv_info, model, num_output_channels):
    """Heuristic: a Conv whose output channels match the detection head width."""
    init_map = get_initializer_map(model)
    w = get_weight_array(init_map[conv_info['weight']])
    return w.shape[0] == num_output_channels


def find_class_head_convs(model, num_classes):
    """
    Find the class-score head Conv layers in a YOLOv8/v11 Detect head.

    YOLOv8/v11 structure: the detection head (model.21) has separate branches:
    - cv2.{scale}.2/Conv: bbox DFL heads (output_channels = 4 * reg_max)
    - cv3.{scale}.2/Conv: class score heads (output_channels = num_classes)

    These are concatenated after reshape+sigmoid to form the final output.
    The class bias directly controls the confidence distribution.
    """
    init_names = set(i.name for i in model.graph.initializer)
    class_heads = []

    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        name = node.name or node.output[0]
        # Match cv3.X.Y.2/Conv pattern (class head final 1x1 convs)
        if 'cv3' not in name:
            continue
        weight_name = node.input[1] if len(node.input) > 1 else None
        bias_name = node.input[2] if len(node.input) > 2 else None
        if not weight_name or weight_name not in init_names:
            continue

        for init in model.graph.initializer:
            if init.name == weight_name:
                w_shape = list(init.dims)
                # Class head: output channels == num_classes, kernel 1x1
                if w_shape[0] == num_classes and len(w_shape) == 4:
                    class_heads.append({
                        'node_name': name,
                        'weight': weight_name,
                        'bias': bias_name if (bias_name and bias_name in init_names) else None,
                        'output': node.output[0] if node.output else None,
                        'weight_shape': w_shape,
                    })
                break

    return class_heads


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Mode: analyze (dry-run)
# ---------------------------------------------------------------------------

def mode_analyze(model, args):
    """Analyze model weights without modification."""
    print_section("Model Analysis")
    init_map = get_initializer_map(model)
    conv_pairs = find_conv_bias_pairs(model)

    total_params = 0
    conv_weights = []
    for name, tp in init_map.items():
        arr = get_weight_array(tp)
        total_params += arr.size
    print(f"Total parameters: {total_params:,}")
    print(f"Total initializers: {len(init_map)}")
    print(f"Conv nodes: {len(conv_pairs)}")

    # Output shape analysis
    output_shape = None
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if len(dims) == 3:
            output_shape = dims
            print(f"Output shape: {dims}  =>  {dims[1]-4} classes, {dims[2]} anchors")

    num_det_channels = output_shape[1] if output_shape else None

    # Detection head identification
    print_section("Detection Head Analysis (YOLOv8/v11 Detect)")
    num_classes = output_shape[1] - 4 if output_shape else None
    if num_classes and num_classes > 0:
        class_heads = find_class_head_convs(model, num_classes)
        if class_heads:
            print(f"  Classes: {num_classes}")
            print(f"  Class head Conv layers found: {len(class_heads)}")
            for ci in class_heads:
                w = get_weight_array(init_map[ci['weight']])
                print(f"\n  {ci['node_name']}:")
                print(f"    Weight shape: {list(w.shape)}  (classes={w.shape[0]}, in_ch={w.shape[1]})")
                if ci['bias']:
                    b = get_weight_array(init_map[ci['bias']])
                    def sigmoid(x):
                        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
                    conf = sigmoid(b)
                    print(f"    Bias (class logits): {b}")
                    print(f"    Resting confidence (sigmoid of bias): "
                          f"mean={conf.mean():.4f}, range=[{conf.min():.4f}, {conf.max():.4f}]")
                else:
                    print(f"    No bias")
        else:
            print("  No class head Conv layers found.")
    else:
        print("  Could not determine number of classes from output shape.")

    # Filter pruning analysis
    print_section("Filter Pruning Analysis (L1-norm)")
    prune_candidates = 0
    total_filters = 0
    for ci in conv_pairs:
        w = get_weight_array(init_map[ci['weight']])
        if w.ndim != 4:
            continue
        n_filters = w.shape[0]
        total_filters += n_filters
        # L1 norm per filter (sum over in_ch, kH, kW)
        l1_norms = np.abs(w).reshape(n_filters, -1).sum(axis=1)
        mean_l1 = l1_norms.mean()
        threshold = mean_l1 * args.prune_threshold_ratio
        near_zero = (l1_norms < threshold).sum()
        prune_candidates += near_zero
        if near_zero > 0:
            print(f"  {ci['node_name']}: {near_zero}/{n_filters} filters below "
                  f"{args.prune_threshold_ratio:.0%} of mean L1 "
                  f"(threshold={threshold:.6f}, min={l1_norms.min():.6f})")

    print(f"\n  Total pruneable filters: {prune_candidates}/{total_filters} "
          f"({100*prune_candidates/max(total_filters,1):.1f}%)")

    # Quantization analysis
    print_section("INT8 Quantization Readiness")
    outlier_layers = 0
    for ci in conv_pairs:
        w = get_weight_array(init_map[ci['weight']])
        abs_max = np.abs(w).max()
        abs_mean = np.abs(w).mean()
        # Outlier ratio: if max is >> mean, quantization will lose precision
        ratio = abs_max / max(abs_mean, 1e-10)
        if ratio > 20:
            outlier_layers += 1
            if outlier_layers <= 10:
                print(f"  {ci['node_name']}: max/mean ratio = {ratio:.1f} "
                      f"(abs_max={abs_max:.6f}, abs_mean={abs_mean:.6f})")
    print(f"\n  Layers with high outlier ratio (>20x): {outlier_layers}/{len(conv_pairs)}")

    # Channel importance analysis
    print_section("Channel Importance Analysis")
    low_importance = 0
    total_channels = 0
    for ci in conv_pairs:
        w = get_weight_array(init_map[ci['weight']])
        if w.ndim != 4:
            continue
        # Input channel importance: sum of abs weights across all output filters
        in_ch = w.shape[1]
        total_channels += in_ch
        ch_importance = np.abs(w).sum(axis=(0, 2, 3))  # [in_channels]
        mean_imp = ch_importance.mean()
        threshold = mean_imp * 0.01
        low = (ch_importance < threshold).sum()
        low_importance += low
        if low > 0 and low <= 5:
            print(f"  {ci['node_name']}: {low}/{in_ch} input channels below "
                  f"1% of mean importance")

    print(f"\n  Total low-importance input channels: {low_importance}/{total_channels}")

    changes = 0  # analyze mode makes no changes
    return changes


# ---------------------------------------------------------------------------
# Mode: bias-tune
# ---------------------------------------------------------------------------

def mode_bias_tune(model, args):
    """
    Adjust detection head biases to shift the confidence distribution.

    YOLOv8/v11 Detect head structure:
    - cv3.{scale}.2/Conv: class score heads with [num_classes] output channels
    - These outputs pass through Sigmoid to become confidence scores.
    - Separate from bbox heads (cv2.X).

    Shifting the class bias:
    - Negative shift -> lower raw logits -> fewer detections (reduce FP)
    - Positive shift -> higher raw logits -> more detections (improve recall)

    At the sigmoid midpoint (logit=0), a bias shift of delta changes
    confidence by approximately delta * 0.25.

    Default: small negative shift (-0.1) to reduce false positives, which is
    the safer direction for a real-time tracking system.
    """
    print_section("Detection Head Bias Tuning")
    init_map = get_initializer_map(model)

    output_shape = None
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if len(dims) == 3:
            output_shape = dims

    if not output_shape:
        print("  Cannot determine output shape, skipping bias-tune.")
        return 0

    num_classes = output_shape[1] - 4  # 15 - 4 = 11 classes
    class_heads = find_class_head_convs(model, num_classes)
    shift = args.bias_shift
    changes = 0

    if not class_heads:
        print(f"  No class head Conv layers found (looking for {num_classes}-output Conv).")
        return 0

    print(f"  Found {len(class_heads)} class head Conv(s) with {num_classes} output channels.")
    print(f"  Bias shift: {shift:+.4f}")
    print()

    for ci in class_heads:
        if not ci['bias']:
            print(f"  {ci['node_name']}: no bias tensor, skipping")
            continue

        bias_tp = init_map[ci['bias']]
        bias = get_weight_array(bias_tp).copy()

        # All channels are class scores (no bbox in this Conv)
        bias_before = bias.copy()
        bias += shift

        print(f"  {ci['node_name']} (shape {ci['weight_shape']}):")
        print(f"    Before: mean={bias_before.mean():.4f}, "
              f"std={bias_before.std():.4f}, "
              f"range=[{bias_before.min():.4f}, {bias_before.max():.4f}]")
        print(f"    After:  mean={bias.mean():.4f}, "
              f"std={bias[...].std():.4f}, "
              f"range=[{bias.min():.4f}, {bias.max():.4f}]")

        # Show per-class sigmoid confidence at the bias value (logit=bias when
        # input activation is zero, which approximates the "resting" confidence)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        conf_before = sigmoid(bias_before)
        conf_after = sigmoid(bias)
        print(f"    Resting confidence: mean {conf_before.mean():.4f} -> {conf_after.mean():.4f}")

        set_weight_array(bias_tp, bias)
        changes += 1

    return changes


# ---------------------------------------------------------------------------
# Mode: prune-filters
# ---------------------------------------------------------------------------

def mode_prune_filters(model, args):
    """
    Zero out Conv filters with near-zero L1 norm.

    This is 'soft pruning': the filters remain in the model but their weights
    become exactly zero. Benefits:
    - Zero weights compress better in INT8/sparse formats
    - TensorRT's sparsity support can skip zero computations
    - No structural change needed (output shape stays the same)
    """
    print_section("Filter Pruning (soft zero-out)")
    init_map = get_initializer_map(model)
    conv_pairs = find_conv_bias_pairs(model)

    output_shape = None
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if len(dims) == 3:
            output_shape = dims

    num_det_channels = output_shape[1] if output_shape else None

    total_pruned = 0
    total_filters = 0
    changes = 0

    for ci in conv_pairs:
        # Skip detection head -- pruning the final layer directly removes classes
        if num_det_channels and is_detection_head_conv(ci, model, num_det_channels):
            continue

        weight_tp = init_map[ci['weight']]
        w = get_weight_array(weight_tp).copy()
        if w.ndim != 4:
            continue

        n_filters = w.shape[0]
        total_filters += n_filters
        l1_norms = np.abs(w).reshape(n_filters, -1).sum(axis=1)
        mean_l1 = l1_norms.mean()
        threshold = mean_l1 * args.prune_threshold_ratio
        prune_mask = l1_norms < threshold

        n_pruned = prune_mask.sum()
        if n_pruned == 0:
            continue

        # Safety: never prune more than 20% of filters in a single layer
        max_prune = max(1, int(n_filters * 0.20))
        if n_pruned > max_prune:
            # Only prune the smallest ones
            indices = np.argsort(l1_norms)[:max_prune]
            prune_mask = np.zeros(n_filters, dtype=bool)
            prune_mask[indices] = True
            n_pruned = max_prune

        w[prune_mask] = 0.0
        set_weight_array(weight_tp, w)

        # Also zero corresponding bias entries if present
        if ci['bias']:
            bias_tp = init_map[ci['bias']]
            bias = get_weight_array(bias_tp).copy()
            bias[prune_mask] = 0.0
            set_weight_array(bias_tp, bias)

        total_pruned += n_pruned
        changes += 1
        print(f"  {ci['node_name']}: pruned {n_pruned}/{n_filters} filters "
              f"(threshold L1={threshold:.6f})")

    print(f"\n  Total filters zeroed: {total_pruned}/{total_filters}")
    return changes


# ---------------------------------------------------------------------------
# Mode: quant-prep (INT8 pre-conditioning)
# ---------------------------------------------------------------------------

def mode_quant_prep(model, args):
    """
    Pre-condition weights for clean INT8 quantization.

    INT8 quantization maps FP32 range [-max, +max] to [-127, 127].
    Weights that fall between quantization levels get rounded, introducing error.
    By slightly adjusting weights toward the nearest quantization level BEFORE
    TensorRT's PTQ, we reduce the rounding error.

    This is NOT full quantization -- the model stays FP32/FP16 ONNX. It just
    makes the weight values friendlier for the subsequent TensorRT INT8 build.

    Also clips outlier weights that would waste dynamic range.
    """
    print_section("INT8 Quantization Pre-conditioning")
    init_map = get_initializer_map(model)
    conv_pairs = find_conv_bias_pairs(model)
    alpha = args.quant_smooth_alpha  # blend factor toward quantized value

    changes = 0
    total_adjusted = 0
    total_weights = 0

    for ci in conv_pairs:
        weight_tp = init_map[ci['weight']]
        w = get_weight_array(weight_tp).copy()
        original = w.copy()
        n_weights = w.size
        total_weights += n_weights

        # Step 1: Clip outliers to reduce wasted dynamic range
        # Use per-channel (per-filter) clipping for Conv weights
        if w.ndim == 4:
            for f in range(w.shape[0]):
                channel_w = w[f]
                abs_max = np.abs(channel_w).max()
                abs_p99 = np.percentile(np.abs(channel_w), 99.5)
                if abs_max > abs_p99 * 2 and abs_p99 > 0:
                    clip_val = abs_p99 * 1.5
                    w[f] = np.clip(channel_w, -clip_val, clip_val)

        # Step 2: Soft-round toward INT8 quantization levels
        # For each filter/channel, compute the quantization step size,
        # then blend weights toward the nearest quantization grid point.
        if w.ndim == 4:
            for f in range(w.shape[0]):
                channel_w = w[f]
                abs_max = np.abs(channel_w).max()
                if abs_max < 1e-8:
                    continue
                # INT8 scale: maps [-abs_max, abs_max] to [-127, 127]
                scale = abs_max / 127.0
                # Quantized then dequantized
                quantized = np.round(channel_w / scale) * scale
                # Blend: w = (1-alpha)*w + alpha*quantized
                w[f] = (1.0 - alpha) * channel_w + alpha * quantized
        else:
            abs_max = np.abs(w).max()
            if abs_max > 1e-8:
                scale = abs_max / 127.0
                quantized = np.round(w / scale) * scale
                w = (1.0 - alpha) * w + alpha * quantized

        # Count significantly adjusted weights
        diff = np.abs(w - original)
        adjusted = (diff > 1e-7).sum()
        total_adjusted += adjusted

        if adjusted > 0:
            set_weight_array(weight_tp, w)
            changes += 1
            max_diff = diff.max()
            mean_diff = diff[diff > 1e-7].mean() if adjusted > 0 else 0
            print(f"  {ci['node_name']}: adjusted {adjusted:,}/{n_weights:,} weights "
                  f"(max_delta={max_diff:.6f}, mean_delta={mean_diff:.6f})")

    print(f"\n  Total weights adjusted: {total_adjusted:,}/{total_weights:,} "
          f"({100*total_adjusted/max(total_weights,1):.2f}%)")
    print(f"  Smooth alpha: {alpha} (0=no change, 1=full snap to INT8 grid)")
    return changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Weight-space modifications for YOLO ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', type=str,
                        default='engines/sunxds_0.7.8.onnx',
                        help='Path to input ONNX model')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save modified model (default: <model>_modified.onnx)')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['analyze', 'bias-tune', 'prune-filters',
                                 'quant-prep', 'all'],
                        help='Modification mode')
    # bias-tune params
    parser.add_argument('--bias-shift', type=float, default=-0.1,
                        help='Class score bias shift (negative=fewer FP, positive=more recall)')
    # prune-filters params
    parser.add_argument('--prune-threshold-ratio', type=float, default=0.02,
                        help='Prune filters with L1 norm below this fraction of layer mean')
    # quant-prep params
    parser.add_argument('--quant-smooth-alpha', type=float, default=0.3,
                        help='Blend factor toward INT8 grid (0=none, 1=full)')

    args = parser.parse_args()

    # Resolve model path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_dir, model_path)

    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_dir, output_path)
    else:
        base, ext = os.path.splitext(model_path)
        suffix = f"_{args.mode.replace('-', '_')}" if args.mode != 'all' else '_modified'
        output_path = f"{base}{suffix}{ext}"

    # Load model
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    print(f"Opset version: {model.opset_import[0].version}")

    # Print model I/O
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"Input '{inp.name}': {dims}")
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"Output '{out.name}': {dims}")

    # Run selected mode(s)
    total_changes = 0

    if args.mode in ('analyze', 'all'):
        total_changes += mode_analyze(model, args)

    if args.mode == 'bias-tune' or args.mode == 'all':
        total_changes += mode_bias_tune(model, args)

    if args.mode == 'prune-filters' or args.mode == 'all':
        total_changes += mode_prune_filters(model, args)

    if args.mode == 'quant-prep' or args.mode == 'all':
        total_changes += mode_quant_prep(model, args)

    # Save if modifications were made
    if total_changes > 0 and args.mode != 'analyze':
        print_section("Validation & Save")
        print("Validating modified model...")
        try:
            onnx.checker.check_model(model)
            print("  Model passes ONNX validation.")
        except onnx.checker.ValidationError as e:
            print(f"  WARNING: Validation failed: {e}")
            print("  Saving anyway (TensorRT may still accept it).")

        print(f"Saving to: {output_path}")
        onnx.save(model, output_path)
        orig_size = os.path.getsize(model_path)
        new_size = os.path.getsize(output_path)
        print(f"  Original size: {orig_size/1024/1024:.2f} MB")
        print(f"  Modified size: {new_size/1024/1024:.2f} MB")
        print(f"  Size delta: {(new_size-orig_size)/1024:+.1f} KB")
    elif args.mode == 'analyze':
        print("\n[Analyze mode] No modifications made.")
    else:
        print("\nNo modifications were applied (no matching layers found).")

    print_section("Summary")
    print(f"Mode: {args.mode}")
    print(f"Layers modified: {total_changes}")
    if args.mode != 'analyze' and total_changes > 0:
        print(f"Output: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Build TensorRT engine from modified ONNX:")
        print(f"     trtexec --onnx={output_path} --fp16 --saveEngine=<output.engine>")
        print(f"  2. Benchmark against original:")
        print(f"     ./tools/benchmark_modified_engine.sh --original <orig.engine> "
              f"--modified <new.engine>")


if __name__ == '__main__':
    main()
