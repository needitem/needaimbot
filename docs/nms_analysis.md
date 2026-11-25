# NMS Architecture Analysis: Current Pipeline vs TensorRT EfficientNMS

## 1. What the Current Pipeline Actually Does

The postprocessing in `inference_pc/needaimbot/cuda/simple_postprocess.cu` is **not NMS**
in any traditional sense. It is a **fused decode-select-aim pipeline** implemented as
two back-to-back CUDA kernels that together find the single best target and compute the
mouse movement in one shot.

### Stage 1: `stage1DecodeAndSelectKernel`

Each thread decodes one YOLO anchor box from the raw model output (FP16 or FP32),
applies confidence filtering and class-mask filtering, and then participates in two
parallel warp-level reductions:

| Track            | Reduction          | Purpose                                            |
|------------------|--------------------|-----------------------------------------------------|
| **Nearest**      | `warpReduceDistMin`| Find the detection whose aim point is closest to the screen crosshair, with body-priority ranking (body always outranks head, head is fallback only). |
| **Sticky (IoU)** | `warpReduceIouMax` | Find the detection with the highest stickiness score against the previous frame's tracked target (IoU + distance-based continuity). |

Each thread block writes its per-track winner to global memory (one candidate per
block, per track). With typical configs this is 4-16 blocks, so the intermediate buffer
is tiny.

### Stage 2: `stage2FinalizeKernel`

A single block reduces the stage-1 candidates across blocks (same two reductions), then
runs the full target-selection state machine:

1. **IoU stickiness** -- if the best sticky candidate exceeds the threshold, keep
   tracking it (hysteresis prevents flicker).
2. **Nearest fallback** -- otherwise, acquire the nearest-to-crosshair candidate.
3. **Track persistence** -- if no detection is found this frame but one existed
   recently, hold the crosshair still for `track_persistence_frames` to bridge brief
   detection gaps.
4. **Fresh-acquire reset** -- a target with zero overlap to the previous track is
   treated as a new enemy; filters, velocity, and derivative state are wiped to prevent
   the aim from dragging across the gap between two objects.

Once the single chosen target is determined, Stage 2 computes the aim point (with
`aim_h_ema` for quieter body observations) and calls `computeAimMovement()` in
`pd_controller.cuh`, which runs:

- One Euro adaptive low-pass on the target center (jitter suppression)
- Ego-corrected velocity estimation via the in-flight ring
- Dead-time compensation (subtract in-flight moves the measurement hasn't seen yet)
- Target-side extrapolation (`predict_frames`)
- Nonlinear P + D convergence
- Output feedforward (`ff_gain`)
- Max-step clamp
- Vertical strength trim (`aim_y_scale`)
- Sub-pixel residual accumulation and integer mouse delta emission

The entire pipeline produces a single `InferenceResult` struct (48 bytes, one cache
line) that is DMA'd to the host in a single D2H copy. There is no intermediate
detection list, no box array, no sorting -- just one target and one mouse delta.

### Key Architectural Insight

Traditional NMS answers: "which of these N overlapping boxes should survive?" It outputs
a variable-length list of non-overlapping detections.

This pipeline answers: "which ONE box is the best thing to aim at right now, and how
many mouse counts should I move?" It outputs a scalar result. The IoU computation is
used for **temporal tracking continuity** (same target across frames), not for spatial
overlap suppression within a single frame.

## 2. Would TensorRT EfficientNMS Help?

**Short answer: No.**

TensorRT's EfficientNMS plugin (`EfficientNMS_TRT`) is designed for the standard
multi-class NMS problem:

- Input: N decoded boxes + N x C scores
- Output: variable-length list of surviving boxes after overlap suppression
- Purpose: remove redundant overlapping detections of the SAME object

What we need is fundamentally different:

| Requirement                     | EfficientNMS | Current Pipeline |
|---------------------------------|:------------:|:----------------:|
| Suppress overlapping boxes      | Yes          | Not needed       |
| Find single nearest-to-crosshair target | No   | Yes (fused)      |
| Body-priority class ranking     | No           | Yes              |
| IoU-based temporal stickiness   | No           | Yes              |
| Track persistence across gaps   | No           | Yes              |
| PD controller + mouse delta     | No           | Yes (fused)      |
| In-flight dead-time compensation| No           | Yes              |
| Output: variable box list       | Yes          | No (single target)|
| Output: integer mouse delta     | No           | Yes              |

The EfficientNMS plugin would give us a cleaned-up box list, but we would still need
to:
1. Iterate that list to find the nearest-to-crosshair target
2. Compute IoU against the previous frame's target for stickiness
3. Run the entire PD controller pipeline

This would be strictly **more** work than the current approach, which skips the box
list entirely and fuses everything into two kernel launches.

## 3. What Would Be Required to Use EfficientNMS

Integrating EfficientNMS would require **un-fusing** the pipeline:

### Step 1: Add EfficientNMS as a TensorRT plugin layer
- Append the NMS plugin to the TensorRT network after the detection head.
- Configure `score_threshold`, `iou_threshold`, `max_output_boxes`, `background_class`.
- The plugin would output: `num_detections`, `detection_boxes`, `detection_scores`,
  `detection_classes`.

### Step 2: Write a new post-NMS selection kernel
- Read the variable-length NMS output (up to `max_output_boxes` entries).
- Compute distance to crosshair for each surviving box, with body-priority ranking.
- Compute IoU stickiness against the previous frame's tracked target.
- Select the single best candidate.
- Handle track persistence for detection gaps.
- Compute the aim point with `aim_h_ema`.

### Step 3: Run the PD controller separately
- Call `computeAimMovement()` on the selected target.
- Pack the `InferenceResult`.

### Performance Impact

The current pipeline launches **2 kernels** total. The EfficientNMS approach would
require:

1. EfficientNMS plugin execution (~0.1-0.3ms on typical hardware)
2. Post-NMS selection kernel (new)
3. PD controller kernel (extracted from current Stage 2)

This is **3 kernel launches** instead of 2, plus the NMS plugin does work we don't
need (sorting boxes, computing pairwise IoU for spatial suppression). The current
Stage 1 already discards >99% of candidates via confidence filtering before the
warp reduction, so there is no redundant-box problem to solve.

**Estimated latency impact: +0.1 to +0.4ms**, which at 230fps (4.3ms/frame) is a
5-10% regression in the postprocess stage for zero functional benefit.

### Additional Costs

- **Complexity**: the fused pipeline is ~880 lines in one file with no intermediate
  allocations. Un-fusing it splits the logic across the TensorRT graph and multiple
  kernels, with a variable-length intermediate buffer.
- **Memory**: EfficientNMS allocates workspace for sorting; the current pipeline uses
  only `max_candidate_blocks` entries of fixed-size shared memory.
- **Maintainability**: the AimConfig/AimState contract is currently self-contained
  in `simple_postprocess.h` + `pd_controller.cuh`. Splitting it across a TensorRT
  plugin boundary adds a versioning surface.

## 4. Conclusion and Recommendation

**Do not replace the current pipeline with TensorRT EfficientNMS.**

The current architecture is purpose-built for the actual task (single-target selection
with fused aim computation) and is already optimal for it:

- **Zero unnecessary work**: no pairwise IoU for spatial suppression (not needed),
  no sorting, no variable-length output allocation.
- **Minimal kernel launches**: 2 total, with warp-level reductions that fully utilize
  the GPU's SIMT model.
- **Fused output**: the mouse delta comes out of the same kernel that selects the
  target, eliminating a round-trip through global memory.
- **Latency-optimal**: the pipeline is on the critical path of a real-time control
  loop running at 230+ fps; every microsecond matters.

TensorRT EfficientNMS solves a different problem (multi-box spatial suppression) and
would add latency, complexity, and memory overhead without improving any metric that
matters to this application.

### When EfficientNMS Would Be Appropriate

EfficientNMS would be the right choice if the application needed to:
- Output a full list of all visible enemies (e.g., for a minimap or UI overlay)
- Feed detections to a downstream tracker that handles multiple objects
- Serve as a general-purpose detection API where the consumer is not known at build time

None of these apply here. The pipeline's only consumer is the PD aim controller, and it
needs exactly one target per frame.

---

## 5. Prediction Parameter Tuning Notes

### Background: Why prediction was previously disabled

The PD controller's `AimConfig` struct defines two prediction-related parameters:

- **`predict_frames`** (target-side dead-time extrapolation): shifts the aim SETPOINT
  forward by `predict_frames * velocity`, compensating for the target's own motion
  during the emit-to-visible dead time. This passes through the nonlinear P and
  inherits its softness/max-step behavior.

- **`ff_gain`** (output feedforward): adds `ff_gain * velocity` directly to the OUTPUT
  (after the P term), canceling the structural steady-state ramp lag of the
  proportional controller against a moving target.

Both terms ride the **ego-corrected target velocity** estimate. Earlier versions of the
pipeline computed velocity from raw screen-space drift, which is polluted by the
crosshair's own motion (d_raw = dTarget - dCrosshair). This made the velocity estimate
nearly useless at steady tracking (where d_raw approaches 0 regardless of target
speed), so both terms were disabled (set to 0.0).

### What changed: the in-flight ring

The in-flight ring (`AimState::inflight_x/y[]`) was added for dead-time compensation.
It records every emitted mouse move, so the crosshair's own contribution to the drift
is known exactly:

```
true_target_velocity = d_raw + ring[ff_ego_lag]
```

This recovers the target's TRUE screen velocity and makes the prediction terms sound.
The velocity is further protected by two gates (see `leadGate()` in `pd_controller.cuh`):

- **Speed gate** (`lead_vgate`): suppresses prediction at low speeds so detector noise
  cannot shake a held aim.
- **Error gate** (`lead_err_gate`): suppresses prediction during acquisition jumps so
  the huge one-frame velocity of a target switch does not overshoot.

### Chosen default values

| Parameter        | Value | Rationale                                                |
|------------------|-------|----------------------------------------------------------|
| `predict_frames` | 0.5   | Well under the measured ~1.13-frame physical dead time (at the tuning rate of 143fps). Conservative: extrapolates only ~44% of the dead time, avoiding over-extrapolation that would cause oscillation. Even partial compensation reduces ramp lag significantly. |
| `ff_gain`        | 0.3   | Mild output feedforward. Adds 0.3 * velocity to each frame's move, partially canceling the proportional controller's structural lag against moving targets. At typical tracking velocities (2-8 px/frame model-space), this contributes 0.6-2.4 px/frame -- enough to visibly reduce tracking lag without risking instability. |

Both values are intentionally conservative as a starting point. The system tolerates
higher values (the in-flight ring and dual gates make it safe), but starting low allows
validation on hardware before increasing. The `aim_rate_adaptive` system automatically
rescales these for the actual capture rate, so the physical meaning is preserved across
frame rate variations.
