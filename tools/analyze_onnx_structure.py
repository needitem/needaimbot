#!/usr/bin/env python3
"""
Comprehensive ONNX Model Structure Analyzer
Designed for YOLO-family detection models.
Produces a full layer map, architecture breakdown, parameter distribution,
and detection head analysis.
"""

import sys
import onnx
import numpy as np
from collections import defaultdict, OrderedDict

def get_shape(tensor_type):
    """Extract shape from TensorProto type."""
    shape = []
    if tensor_type.HasField('tensor_type'):
        for d in tensor_type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append(d.dim_value)
    return shape

def analyze_model(model_path, output_path=None):
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Build lookup tables
    initializer_map = {}  # name -> numpy array
    for init in graph.initializer:
        initializer_map[init.name] = onnx.numpy_helper.to_array(init)

    # Build shape inference map
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        inferred_graph = inferred.graph
    except Exception as e:
        print(f"Shape inference failed: {e}, using partial info")
        inferred_graph = graph

    value_info_map = {}
    for vi in list(inferred_graph.value_info) + list(inferred_graph.input) + list(inferred_graph.output):
        value_info_map[vi.name] = get_shape(vi.type)

    # Build connectivity: output_name -> producing node, input_name -> consuming nodes
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node

    input_to_nodes = defaultdict(list)
    for node in graph.node:
        for i in node.input:
            input_to_nodes[i].append(node)

    # Node name helper
    def node_id(node):
        return node.name if node.name else f"{node.op_type}_{id(node)}"

    # =========================================================================
    # 1. Layer-by-layer Conv analysis
    # =========================================================================
    conv_layers = []
    all_nodes_by_name = {}
    for idx, node in enumerate(graph.node):
        nid = node.name if node.name else f"{node.op_type}_{idx}"
        all_nodes_by_name[nid] = node

        if node.op_type == 'Conv':
            info = {'index': idx, 'name': nid, 'op': 'Conv'}

            # Get attributes
            attrs = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INTS:
                    attrs[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.INT:
                    attrs[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs[attr.name] = attr.f

            info['kernel_shape'] = attrs.get('kernel_shape', [])
            info['strides'] = attrs.get('strides', [1, 1])
            info['pads'] = attrs.get('pads', [0, 0, 0, 0])
            info['dilations'] = attrs.get('dilations', [1, 1])
            info['group'] = attrs.get('group', 1)

            # Weights and bias
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None

            weight_params = 0
            bias_params = 0
            weight_shape = None
            if weight_name and weight_name in initializer_map:
                w = initializer_map[weight_name]
                weight_params = w.size
                weight_shape = w.shape
            if bias_name and bias_name in initializer_map:
                b = initializer_map[bias_name]
                bias_params = b.size

            info['weight_shape'] = weight_shape
            info['weight_params'] = weight_params
            info['bias_params'] = bias_params
            info['total_params'] = weight_params + bias_params

            # Input/output shapes
            if node.input[0] in value_info_map:
                info['input_shape'] = value_info_map[node.input[0]]
            else:
                info['input_shape'] = None
            if node.output[0] in value_info_map:
                info['output_shape'] = value_info_map[node.output[0]]
            else:
                info['output_shape'] = None

            # Connectivity
            predecessors = []
            for inp in node.input:
                if inp in output_to_node:
                    pred = output_to_node[inp]
                    predecessors.append(node_id(pred))
            info['predecessors'] = predecessors

            successors = []
            for out in node.output:
                for succ in input_to_nodes[out]:
                    successors.append(node_id(succ))
            info['successors'] = successors

            # Store raw node for later
            info['_node'] = node
            conv_layers.append(info)

    # =========================================================================
    # 2. Architecture identification - trace data flow
    # =========================================================================
    # Build full node list with index
    node_list = []
    for idx, node in enumerate(graph.node):
        nid = node.name if node.name else f"{node.op_type}_{idx}"
        node_list.append({'index': idx, 'name': nid, 'op': node.op_type, 'node': node})

    # Find output-producing nodes by tracing backward from graph outputs
    def trace_backward(output_name, depth=0, visited=None):
        if visited is None:
            visited = set()
        if output_name in visited:
            return []
        visited.add(output_name)
        path = []
        if output_name in output_to_node:
            node = output_to_node[output_name]
            nid = node_id(node)
            path.append((nid, node.op_type, depth))
            for inp in node.input:
                if inp not in initializer_map:  # skip weight inputs
                    path.extend(trace_backward(inp, depth + 1, visited))
        return path

    # Trace from model output
    output_traces = []
    for out in graph.output:
        trace = trace_backward(out.name)
        output_traces.append((out.name, trace))

    # Identify detection head Conv layers (close to output)
    head_conv_names = set()
    # Find Reshape/Concat near output that shapes the [1,15,2100]
    # Trace backward to find the final Conv layers before reshape

    def find_final_convs(output_name, max_depth=15, visited=None):
        """Find Conv nodes reachable from output within max_depth, before hitting another Conv."""
        if visited is None:
            visited = set()
        if output_name in visited:
            return []
        visited.add(output_name)
        convs = []
        if output_name in output_to_node:
            node = output_to_node[output_name]
            if node.op_type == 'Conv':
                convs.append(node_id(node))
            elif max_depth > 0:
                for inp in node.input:
                    if inp not in initializer_map:
                        convs.extend(find_final_convs(inp, max_depth - 1, visited))
        return convs

    final_convs = set()
    for out in graph.output:
        fc = find_final_convs(out.name)
        final_convs.update(fc)

    # Classify Conv layers into backbone/neck/head based on position and connectivity
    # Heuristic:
    #   - Head: Conv layers that feed into the output path (found above)
    #   - Backbone: Early Conv layers with progressive downsampling
    #   - Neck: Conv layers between backbone and head (feature fusion)

    conv_by_name = {c['name']: c for c in conv_layers}

    # Determine which convs are in head by checking if they're in final_convs or feed into them
    head_conv_set = set(final_convs)

    # Expand head: find Conv layers that feed into head convs (1 level)
    def get_conv_predecessors(conv_name):
        if conv_name not in conv_by_name:
            return []
        preds = []
        node = conv_by_name[conv_name]['_node']
        # Trace backward through non-conv nodes to find feeding convs
        visited = set()
        queue = list(node.input)
        depth = 0
        while queue and depth < 10:
            next_queue = []
            for inp in queue:
                if inp in visited or inp in initializer_map:
                    continue
                visited.add(inp)
                if inp in output_to_node:
                    pred_node = output_to_node[inp]
                    if pred_node.op_type == 'Conv':
                        preds.append(node_id(pred_node))
                    else:
                        for pi in pred_node.input:
                            next_queue.append(pi)
            queue = next_queue
            depth += 1
        return preds

    # Analyze stride patterns and output sizes to identify backbone
    # Backbone typically has stride-2 convolutions that progressively reduce spatial dims
    stride2_convs = [c for c in conv_layers if c['strides'] == [2, 2]]

    # Build a forward dependency graph for conv layers
    # First, map each conv output to that conv
    conv_output_map = {}
    for c in conv_layers:
        for o in c['_node'].output:
            conv_output_map[o] = c['name']

    # For each conv, find which conv layers its output eventually reaches
    def trace_forward_convs(node, max_depth=20, visited=None):
        if visited is None:
            visited = set()
        nid = node_id(node)
        if nid in visited:
            return []
        visited.add(nid)
        result = []
        for out in node.output:
            for succ in input_to_nodes[out]:
                sid = node_id(succ)
                if succ.op_type == 'Conv':
                    result.append(sid)
                elif max_depth > 0:
                    result.extend(trace_forward_convs(succ, max_depth - 1, visited))
        return result

    # Classify based on index position and connectivity patterns
    total_convs = len(conv_layers)

    # Find Concat nodes (indicators of feature fusion in neck)
    concat_nodes = [n for n in graph.node if n.op_type == 'Concat']
    concat_input_convs = set()
    for cn in concat_nodes:
        for inp in cn.input:
            if inp in output_to_node:
                pred = output_to_node[inp]
                if pred.op_type == 'Conv':
                    concat_input_convs.add(node_id(pred))

    # Find Resize/Upsample nodes (indicator of FPN)
    resize_nodes = [n for n in graph.node if n.op_type in ('Resize', 'Upsample')]

    # Classify architecture sections
    # Strategy: Use spatial dimension changes to identify backbone boundaries
    # Then use concat/resize proximity for neck, and output proximity for head

    section_map = {}

    # First pass: identify head convs (near output)
    for c in conv_layers:
        if c['name'] in final_convs:
            section_map[c['name']] = 'head'

    # Expand head: convs whose only forward path leads to head convs
    # and that have 1x1 kernels producing the detection channels
    for c in conv_layers:
        if c['name'] in section_map:
            continue
        if c['weight_shape'] is not None:
            out_channels = c['weight_shape'][0]
            # Detection head typically has specific channel counts (15, 18, 21, etc.)
            if out_channels in [15, 18, 21, 24, 255, 85*3] and c['kernel_shape'] == [1, 1]:
                section_map[c['name']] = 'head'
                head_conv_set.add(c['name'])

    # Identify backbone: sequential conv layers from input, with stride-2 downsampling
    # Find the longest chain from input
    input_name = graph.input[0].name

    def trace_conv_chain(start_output, visited=None):
        """Trace the main conv chain from a tensor."""
        if visited is None:
            visited = set()
        chain = []
        current = start_output
        while True:
            found_conv = False
            for succ in input_to_nodes.get(current, []):
                if node_id(succ) in visited:
                    continue
                if succ.op_type == 'Conv':
                    chain.append(node_id(succ))
                    visited.add(node_id(succ))
                    current = succ.output[0]
                    found_conv = True
                    break
                else:
                    # Follow through non-conv nodes
                    for o in succ.output:
                        for succ2 in input_to_nodes.get(o, []):
                            if succ2.op_type == 'Conv' and node_id(succ2) not in visited:
                                chain.append(node_id(succ2))
                                visited.add(node_id(succ2))
                                current = succ2.output[0]
                                found_conv = True
                                break
                        if found_conv:
                            break
                if found_conv:
                    break
            if not found_conv:
                break
        return chain

    # Alternative approach: classify by index ranges and spatial dimensions
    # Analyze spatial dimension progression
    spatial_dims = []
    for c in conv_layers:
        if c['output_shape'] and len(c['output_shape']) == 4:
            spatial_dims.append((c['name'], c['index'], c['output_shape'][2], c['output_shape'][3]))
        else:
            spatial_dims.append((c['name'], c['index'], None, None))

    # Find where spatial dims change (backbone downsampling points)
    # and where they increase (neck upsampling)
    prev_h = 320
    backbone_end_idx = 0
    neck_start_idx = 0
    min_spatial = 320
    min_spatial_idx = 0

    for name, idx, h, w in spatial_dims:
        if h is not None:
            if h < min_spatial:
                min_spatial = h
                min_spatial_idx = idx
            if h > prev_h and backbone_end_idx == 0:
                # First upsampling -> backbone ended before this
                backbone_end_idx = idx
            prev_h = h

    # If no upsampling found, use the point of minimum spatial dimension
    if backbone_end_idx == 0:
        backbone_end_idx = min_spatial_idx

    # Classify by index
    for c in conv_layers:
        if c['name'] in section_map:
            continue
        if c['index'] <= backbone_end_idx:
            # Check if it's really backbone or early neck
            section_map[c['name']] = 'backbone'
        elif c['name'] in final_convs or c['name'] in head_conv_set:
            section_map[c['name']] = 'head'
        else:
            section_map[c['name']] = 'neck'

    # Refine: any unclassified conv near concat/resize is likely neck
    for c in conv_layers:
        if c['name'] not in section_map:
            section_map[c['name']] = 'neck'

    # =========================================================================
    # 3. Parameter distribution
    # =========================================================================
    section_params = defaultdict(int)
    section_layer_count = defaultdict(int)
    for c in conv_layers:
        sec = section_map.get(c['name'], 'unknown')
        section_params[sec] += c['total_params']
        section_layer_count[sec] += 1

    # Also count non-conv parameters (BatchNorm, etc.)
    all_params = {}
    total_model_params = 0
    for name, arr in initializer_map.items():
        all_params[name] = arr.size
        total_model_params += arr.size

    # Top 10 layers by parameter count
    top_params = sorted(conv_layers, key=lambda x: x['total_params'], reverse=True)[:10]

    # =========================================================================
    # 4. Detection head analysis
    # =========================================================================
    # Output is [1, 15, 2100]
    # 15 channels: likely 5 * 3 anchors (4 bbox + 1 obj) or (4 + 1 + num_classes) * num_anchors
    # 2100 = sum of grid cells across scales
    # Common YOLO scales for 320x320:
    #   40x40 = 1600, 20x20 = 400, 10x10 = 100 -> total 2100!
    # So 3 scales: 40x40, 20x20, 10x10

    # Find the Concat that produces the final 2100
    final_concat = None
    for node in graph.node:
        if node.op_type == 'Concat':
            for out in node.output:
                if out in value_info_map:
                    shape = value_info_map[out]
                    if shape and len(shape) >= 2 and (2100 in shape or shape[-1] == 2100):
                        final_concat = node
                        break

    # Trace each branch of the final concat to find per-scale detection layers
    scale_info = []
    if final_concat:
        for inp_name in final_concat.input:
            shape = value_info_map.get(inp_name, None)
            # Trace back to find the Conv
            convs_in_branch = find_final_convs(inp_name)
            scale_info.append({
                'input_name': inp_name,
                'shape': shape,
                'conv_layers': convs_in_branch
            })

    # =========================================================================
    # Generate Report
    # =========================================================================
    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 80)
    w("ONNX MODEL STRUCTURE ANALYSIS")
    w(f"Model: {model_path}")
    w("=" * 80)
    w()

    w("## 1. MODEL OVERVIEW")
    w(f"  Opset version: {model.opset_import[0].version}")
    w(f"  Input:  {graph.input[0].name} -> {value_info_map.get(graph.input[0].name, 'unknown')}")
    w(f"  Output: {graph.output[0].name} -> {value_info_map.get(graph.output[0].name, 'unknown')}")
    w(f"  Total nodes: {len(graph.node)}")
    w(f"  Total initializers: {len(graph.initializer)}")
    w(f"  Total parameters: {total_model_params:,} ({total_model_params/1e6:.2f}M)")
    w()

    # Op type distribution
    op_counts = defaultdict(int)
    for n in graph.node:
        op_counts[n.op_type] += 1
    w("  Op type distribution:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        w(f"    {op:20s}: {cnt}")
    w()

    w("=" * 80)
    w("## 2. CONV LAYER BREAKDOWN")
    w("=" * 80)
    w()

    for c in conv_layers:
        sec = section_map.get(c['name'], '?')
        w(f"  [{c['index']:3d}] {c['name']}")
        w(f"        Section:     {sec.upper()}")
        w(f"        Kernel:      {c['kernel_shape']}, Stride: {c['strides']}, Pad: {c['pads']}, Group: {c['group']}")
        if c['weight_shape'] is not None:
            w(f"        Weight:      {c['weight_shape']} ({c['weight_params']:,} params)")
        if c['bias_params'] > 0:
            w(f"        Bias:        {c['bias_params']:,} params")
        w(f"        Total params: {c['total_params']:,}")
        w(f"        Input shape:  {c['input_shape']}")
        w(f"        Output shape: {c['output_shape']}")
        if c['predecessors']:
            w(f"        From:        {', '.join(c['predecessors'][:3])}")
        if c['successors']:
            w(f"        To:          {', '.join(c['successors'][:3])}")
        w()

    w("=" * 80)
    w("## 3. ARCHITECTURE MAP")
    w("=" * 80)
    w()

    # Group by section and show summary
    for sec in ['backbone', 'neck', 'head']:
        sec_convs = [c for c in conv_layers if section_map.get(c['name']) == sec]
        w(f"  ### {sec.upper()} ({len(sec_convs)} Conv layers, {section_params[sec]:,} params)")
        for c in sec_convs:
            ks = f"{c['kernel_shape'][0]}x{c['kernel_shape'][1]}" if len(c['kernel_shape']) == 2 else str(c['kernel_shape'])
            out_ch = c['weight_shape'][0] if c['weight_shape'] is not None else '?'
            in_ch = c['weight_shape'][1] if c['weight_shape'] is not None and len(c['weight_shape']) > 1 else '?'
            grp = c['group']
            if grp > 1:
                grp_str = f" (dw, g={grp})"
            else:
                grp_str = ""
            spatial = ""
            if c['output_shape'] and len(c['output_shape']) == 4:
                spatial = f" -> {c['output_shape'][2]}x{c['output_shape'][3]}"
            s_mark = " [stride=2]" if c['strides'] == [2, 2] else ""
            w(f"    [{c['index']:3d}] {ks} Conv {in_ch}->{out_ch}{grp_str}{s_mark}{spatial}  ({c['total_params']:,} params)")
        w()

    # ASCII architecture diagram
    w("  ### ASCII DATA FLOW")
    w()
    w("  Input [1,3,320,320]")
    w("    |")
    w("    v")
    w("  ============ BACKBONE (Feature Extraction) ============")

    # Show backbone stride-2 points
    backbone_convs = [c for c in conv_layers if section_map.get(c['name']) == 'backbone']
    current_spatial = 320
    for c in backbone_convs:
        if c['strides'] == [2, 2]:
            new_spatial = current_spatial // 2
            out_ch = c['weight_shape'][0] if c['weight_shape'] is not None else '?'
            w(f"    | Conv stride=2 ({current_spatial}x{current_spatial} -> {new_spatial}x{new_spatial}, ch={out_ch})")
            current_spatial = new_spatial
        elif c['output_shape'] and len(c['output_shape']) == 4:
            current_spatial = c['output_shape'][2]

    # Find distinct spatial resolutions in backbone outputs
    backbone_outputs = set()
    for c in backbone_convs:
        if c['output_shape'] and len(c['output_shape']) == 4:
            backbone_outputs.add(c['output_shape'][2])

    w("    |")
    w(f"    | Feature maps at: {sorted(backbone_outputs, reverse=True)}")
    w("    v")
    w("  ============ NECK (Feature Fusion / FPN+PAN) ============")

    neck_convs = [c for c in conv_layers if section_map.get(c['name']) == 'neck']
    neck_spatials = set()
    for c in neck_convs:
        if c['output_shape'] and len(c['output_shape']) == 4:
            neck_spatials.add(c['output_shape'][2])

    w(f"    | {len(resize_nodes)} Resize/Upsample nodes, {len(concat_nodes)} Concat nodes")
    w(f"    | Operating at spatial resolutions: {sorted(neck_spatials, reverse=True)}")
    w("    |")
    w("    v")
    w("  ============ DETECTION HEAD ============")

    head_convs = [c for c in conv_layers if section_map.get(c['name']) == 'head']
    for c in head_convs:
        out_ch = c['weight_shape'][0] if c['weight_shape'] is not None else '?'
        spatial = f"{c['output_shape'][2]}x{c['output_shape'][3]}" if c['output_shape'] and len(c['output_shape']) == 4 else '?'
        w(f"    | Conv {c['kernel_shape']} -> ch={out_ch}, spatial={spatial}")

    w("    |")
    w("    | Reshape + Transpose + Concat")
    w("    v")
    w("  Output [1, 15, 2100]")
    w()

    w("=" * 80)
    w("## 4. PARAMETER DISTRIBUTION")
    w("=" * 80)
    w()

    w("  ### Section Parameter Summary")
    for sec in ['backbone', 'neck', 'head']:
        pct = section_params[sec] / total_model_params * 100 if total_model_params > 0 else 0
        w(f"    {sec.upper():10s}: {section_params[sec]:>10,} params ({pct:5.1f}%) in {section_layer_count[sec]} Conv layers")

    # Non-conv params (BN, etc.)
    conv_total = sum(c['total_params'] for c in conv_layers)
    non_conv = total_model_params - conv_total
    w(f"    {'OTHER':10s}: {non_conv:>10,} params ({non_conv/total_model_params*100:5.1f}%) (BatchNorm, etc.)")
    w(f"    {'TOTAL':10s}: {total_model_params:>10,} params (100.0%)")
    w()

    w("  ### Top 10 Conv Layers by Parameter Count")
    for i, c in enumerate(top_params):
        sec = section_map.get(c['name'], '?')
        w(f"    {i+1:2d}. {c['name']:40s} {c['total_params']:>10,} params  [{sec}]")
        if c['weight_shape'] is not None:
            w(f"        weight: {c['weight_shape']}")
    w()

    w("=" * 80)
    w("## 5. DETECTION HEAD ANALYSIS")
    w("=" * 80)
    w()

    w("  ### Output Tensor Decoding: [1, 15, 2100]")
    w()
    w("  Anchor grid decomposition (input 320x320):")
    w("    Scale 1: 40x40 = 1600 cells")
    w("    Scale 2: 20x20 =  400 cells")
    w("    Scale 3: 10x10 =  100 cells")
    w("    Total:          2100 cells  (matches output dim)")
    w()

    # Determine what the 15 channels encode
    # 15 = 5 * 3 (3 anchors per scale, each with 4 bbox + 1 objectness)
    # or 15 = (4 + 1 + 10) * 1 (1 anchor, 10 classes)
    # or 15 = (4 + 1 + K) for some K
    w("  Channel interpretation (15 channels):")
    w("    Hypothesis A: 15 = (4 bbox + 1 obj + 10 classes) x 1 anchor/cell")
    w("    Hypothesis B: 15 = (4 bbox + 1 obj) x 3 anchors/cell")
    w()

    # Check which is correct by looking at the Conv layers producing the output
    head_out_channels = set()
    for c in head_convs:
        if c['weight_shape'] is not None:
            head_out_channels.add(c['weight_shape'][0])

    w(f"  Head Conv output channels: {sorted(head_out_channels)}")

    if 15 in head_out_channels:
        w("  -> Found Conv with 15 output channels (1x1 Conv, detection output)")
        # Count how many 15-channel convs
        det_convs_15 = [c for c in head_convs if c['weight_shape'] is not None and c['weight_shape'][0] == 15]
        w(f"  -> {len(det_convs_15)} detection output Conv(s) found")
        if len(det_convs_15) == 3:
            w("  -> 3 output convs = 3 detection scales (confirms multi-scale detection)")
            w("  -> Each Conv outputs 15 channels at its scale")
            w("  -> Most likely: 15 = (4 + 1 + 10) x 1 anchor = 10-class detector")
            w("     OR 15 = 5 x 3 anchors = anchor-based with no class output")
        for dc in det_convs_15:
            spatial = f"{dc['output_shape'][2]}x{dc['output_shape'][3]}" if dc['output_shape'] and len(dc['output_shape']) == 4 else '?'
            in_ch = dc['weight_shape'][1] if dc['weight_shape'] is not None else '?'
            w(f"     - {dc['name']}: {in_ch} -> 15 ch, spatial {spatial}")
    w()

    # Analyze the concat structure producing 2100
    if final_concat:
        w("  ### Final Concat producing 2100 anchors:")
        w(f"    Node: {node_id(final_concat)}")
        for inp_name in final_concat.input:
            shape = value_info_map.get(inp_name, 'unknown')
            w(f"    Branch: {inp_name[:50]:50s} shape={shape}")
    w()

    if scale_info:
        w("  ### Per-Scale Detection Branches:")
        for i, si in enumerate(scale_info):
            w(f"    Scale {i+1}:")
            w(f"      Shape: {si['shape']}")
            w(f"      Conv layers: {si['conv_layers']}")
    w()

    # Detection head structure
    w("  ### Detection Head Conv Layer Details:")
    for c in head_convs:
        out_ch = c['weight_shape'][0] if c['weight_shape'] is not None else '?'
        in_ch = c['weight_shape'][1] if c['weight_shape'] is not None else '?'
        ks = c['kernel_shape']
        spatial = f"{c['output_shape'][2]}x{c['output_shape'][3]}" if c['output_shape'] and len(c['output_shape']) == 4 else '?'
        w(f"    [{c['index']:3d}] {c['name']}")
        w(f"          {in_ch} -> {out_ch} ch, kernel {ks}, spatial {spatial}, params: {c['total_params']:,}")
    w()

    w("=" * 80)
    w("## 6. WEIGHT-SPACE MODIFICATION OPPORTUNITIES")
    w("=" * 80)
    w()
    w("  ### High-Impact Modification Targets:")
    w()
    w("  A. Detection Head (1x1 Conv, output channels = 15):")
    w("     - These directly control bbox/objectness/class predictions")
    w("     - Modifying weights here changes detection sensitivity per class")
    w("     - Low parameter count = easy to modify, high impact on output")
    w()
    w("  B. Final Neck Convolutions (before detection head):")
    w("     - Feature mixing before prediction")
    w("     - Modifying these can shift feature emphasis between scales")
    w()
    w("  C. Backbone Stride-2 Convolutions:")
    w("     - Control what features survive downsampling")
    w("     - Higher parameter count, broader impact")
    w()

    # Identify specific modification candidates
    w("  ### Specific Candidates for Weight-Space Programming:")
    w()

    # 1x1 convs in head
    head_1x1 = [c for c in head_convs if c['kernel_shape'] == [1, 1]]
    w(f"  1x1 Head Convs ({len(head_1x1)} layers):")
    for c in head_1x1:
        w(f"    - {c['name']}: {c['total_params']:,} params, weight shape {c['weight_shape']}")
    w()

    # Depthwise convs (group == in_channels)
    dw_convs = [c for c in conv_layers if c['group'] > 1]
    w(f"  Depthwise Convolutions ({len(dw_convs)} layers):")
    for c in dw_convs[:5]:
        w(f"    - {c['name']}: group={c['group']}, {c['total_params']:,} params")
    if len(dw_convs) > 5:
        w(f"    ... and {len(dw_convs) - 5} more")
    w()

    # Largest convs (most parameters)
    w("  Largest Conv Layers (highest parameter count):")
    for c in top_params[:5]:
        sec = section_map.get(c['name'], '?')
        w(f"    - {c['name']}: {c['total_params']:,} params [{sec}], weight {c['weight_shape']}")
    w()

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nAnalysis saved to: {output_path}")

    print(report)
    return report


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else '/home/taeho/portfolio/inferencetool/engines/sunxds_0.7.8.onnx'
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_model(model_path, output_path)
