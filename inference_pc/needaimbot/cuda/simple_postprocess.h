#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gpa {

// One Euro derivative cutoff. Was a config knob (oneeuro_dcutoff); removed because
// sweeping 0.1..5.0 moved total error <0.2% - it only shapes the derivative that
// drives the adaptive cutoff, so it was tuning surface with no lever behind it.
constexpr float kOneEuroDCutoff = 1.0f;

struct Detection;  // Forward declaration

// GPU State Structures

// P controller state (GPU persistent)
struct AimState {
    float residual_x = 0.0f;
    float residual_y = 0.0f;
    // Number of consecutive frames the current track has been unmatched.
    // Used together with AimConfig::track_persistence_frames to keep a
    // committed target alive across brief detection gaps.
    int frames_since_seen = 0;

    // Set once a target has been acted on; gates One Euro seeding and the
    // fresh-acquire derivative reset (has_track == 0 => first frame of a track).
    int   has_track = 0;

    // --- Ego-corrected target velocity (feeds the lead / feedforward term) ---
    // raw = T - C + SC, so d_raw = dT - dC: the measured drift is polluted by our
    // OWN motion (that is why a plain screen-space velocity is useless here). But
    // dC is known exactly - it is the emit sitting in the in-flight ring - so
    //     dT = d_raw + ring[ff_ego_lag]
    // recovers the target's TRUE screen velocity. vel_* is the EMA of that.
    float prev_raw_x = 0.0f;
    float prev_raw_y = 0.0f;
    float vel_x = 0.0f;
    float vel_y = 0.0f;
    // Class the aim point came from last frame (-1 = none). head and body are two
    // DIFFERENT anchors on the same enemy, so when selection flips between them
    // the measured centre jumps by the anchor offset - an artifact, not target
    // motion. Knowing the previous class lets that frame be excluded from the
    // VELOCITY estimate (the damping term deliberately still sees it - after a
    // flip the aim point really has moved, so it is a setpoint change; measured,
    // suppressing it is worse). See class_changed in pd_controller.cuh.
    int prev_class = -1;

    // Slow EMA of the chosen box's height, for the body aim point (see
    // aim_h_ema in AimConfig). <= 0 means "not seeded yet".
    float h_ema = 0.0f;

    // --- One Euro position filter state ---
    // Adaptive low-pass of the target center. filt_* is the filtered position;
    // dfilt_* is the low-passed per-frame derivative driving the adaptive cutoff.
    // Seeded on fresh acquire (has_track == 0) to avoid a jump from a stale value.
    float filt_x = 0.0f;
    float filt_y = 0.0f;
    float dfilt_x = 0.0f;
    float dfilt_y = 0.0f;

    // --- Derivative (damping) term state ---
    // prev_err_* is last frame's error; derr_* is the smoothed error rate that
    // the D term acts on to brake the approach and suppress overshoot ringing.
    float prev_err_x = 0.0f;
    float prev_err_y = 0.0f;
    float derr_x = 0.0f;
    float derr_y = 0.0f;

    // --- In-flight move history (dead-time compensation / Smith predictor) ---
    // The detection we act on is ~deadtime_frames old, so moves emitted since
    // that snapshot have NOT yet moved the target in the measurement. Without
    // this, the controller re-corrects an error it has already answered ->
    // double-correction -> overshoot/ringing. Ring of the last emitted moves in
    // OUTPUT px, subtracted from the measured error. The ring also supplies the
    // ego correction for the target-velocity estimate.
    static constexpr int kInflightMax = 4;
    float inflight_x[kInflightMax] = {0.0f, 0.0f, 0.0f, 0.0f};
    float inflight_y[kInflightMax] = {0.0f, 0.0f, 0.0f, 0.0f};
    int   inflight_head = 0;   // next write slot
};

// Nonlinear P controller configuration.
struct AimConfig {
    float kp_x = 0.55f;
    float kp_y = 0.6f;
    float p_softness_x = 11.0f;
    float p_softness_y = 10.0f;
    // Derivative (damping) gain. Brakes the approach in proportion to the error
    // rate so kp can stay high (snappy) without oscillating. 0 = pure P. kd_y
    // runs higher than kd_x because the vertical axis (higher kp_y) rings more.
    float kd_x = 0.18f;
    float kd_y = 0.22f;
    // Per-frame max move (output px). Caps the slew so a large error is crossed
    // in bounded smooth steps instead of one delayed leap that overshoots/rings.
    // 0 = disabled (unbounded).
    float max_step = 30.0f;
    // Same-target stickiness when IoU fails: a box whose center is within
    // (prev_diag * factor) of the previous target's center is treated as the
    // same target. 0 = disabled, ~0.5 typical for fast close targets.
    float distance_stickiness_factor = 0.5f;
    // Track persistence: consecutive missed frames to bridge - the crosshair
    // holds still, the target kept alive for a clean re-acquire - before
    // dropping the target. 0 = disabled.
    int track_persistence_frames = 5;

    // HISTORY / WHY THE LEAD TERM IS BACK. An earlier velocity feedforward and a
    // coast gap-glide were removed because they rode a RAW screen-drift estimate
    // that is ego-polluted: the crosshair's own motion shifts the scene, so the
    // measured drift is d(error)/dt, ~0 at steady tracking. Feedforward on that
    // cannot cancel ramp lag and only amplifies detector noise. The removal note
    // said a lead term needs a LAG-ALIGNED EGO-CORRECTED velocity first.
    //
    // The in-flight ring (added later, for dead-time comp) is exactly that
    // missing piece: it stores the emits we know landed, so the target's true
    // screen velocity is recoverable as d_raw + ring[ff_ego_lag]. The lead term
    // below rides THAT signal, and is double-gated (speed + error) so it cannot
    // amplify rest jitter or fire during an acquisition jump. Sim on measured rig
    // noise: fast-target lag -12%/-15% (4 and 8 px/frame) at unchanged step
    // overshoot and oscillation. Coast stays removed - it had no such fix.

    // --- Dead-time compensation (in-flight move subtraction) ---
    // Subtract inflight_comp * (moves emitted in the last deadtime_frames) from
    // the measured error, so the controller does not re-answer an error its own
    // in-flight moves have already addressed. This is what removes the
    // overshoot/ringing that dead-time causes; with it on, kp can be raised for
    // faster convergence (measured sweet spot ~0.75 at 1 frame of dead-time).
    //   inflight_comp 0 = off (legacy behavior), 1 = full compensation.
    //   deadtime_frames must match the rig's measured emit->visible lag
    //   (bench/calibrate.py STEP-RESPONSE section). Over-estimating it
    //   over-subtracts and can destabilize, so measure before changing.
    float inflight_comp = 0.0f;
    // FRACTIONAL frame count. The real emit->visible lag is not an integer
    // (~1.13 frames here: USB ~1ms + render wait 0..6.94ms + E2E 3.3ms @144Hz),
    // so rounding it to 1 or 2 either under- or over-compensates. A fractional w
    // subtracts the whole last emit plus frac*(the one before it).
    // 1.0 == the old integer deadtime_frames=1 behaviour, exactly.
    float deadtime_frames = 1.0f;

    // --- Lead / feedforward on the EGO-CORRECTED target velocity ---
    // Cancels the nonlinear-P controller's steady-state ramp lag against a moving
    // target - the dominant error term (sim: 18px at 8 px/frame), which no gain
    // tuning can remove because it is structural to a proportional law.
    // ff_gain 0 = off (pure P+D, the previous behaviour).
    // See the history note above for why this is safe now and was not before.
    float ff_gain = 0.0f;
    // Ring lag (frames) at which our own motion between the last two
    // measurements sits. Two consecutive detections are 1 frame apart in capture
    // time, so this is ~dead time + 1. MIS-ALIGNING THIS IS WHAT BROKE THE
    // EARLIER EGO ATTEMPT - keep it consistent with deadtime_frames.
    float ff_ego_lag = 2.25f;
    float ff_v_ema = 0.2f;      // EMA weight on the velocity estimate (0..1)
    // Speed gate: g *= |v|/(|v|+lead_vgate). ~0 at rest, so a noisy velocity
    // estimate cannot inject jitter into a stationary hold.
    float lead_vgate = 9.0f;
    // Error gate: g *= e0^2/(e0^2 + |err|^2). The lead must be OFF while
    // acquiring - a position jump (fresh lock / target switch) produces a huge
    // one-frame velocity that would overshoot. A real moving target instead shows
    // SMALL error with sustained velocity. 0 = no error gating.
    //
    // ONE pair of gates serves BOTH lead terms (ff_gain and predict_frames).
    // Separate per-term gates were tried and removed: they gate the same velocity
    // against the same error, and giving each its own pair bought at most 0.5%
    // while doubling the tuning surface. A single shared pair tuned properly
    // beat the four-knob version outright (measured -0.8% error at equal ringing).
    float lead_err_gate = 18.0f;

    // --- Symmetric dead-time compensation (target-side extrapolation) ---
    // inflight_comp removes OUR motion during the dead time. But the TARGET moved
    // during that same dead time and nothing accounted for it - the compensation
    // was half-done. Extrapolating the target forward by predict_frames * v
    // completes it. Unlike ff_gain (which acts on the output) this shifts the
    // SETPOINT, so it passes through the nonlinear P and inherits its softness /
    // max-step behaviour. The two compose: sim shows predict+ff together beat
    // either alone. predict_frames 0 = off. Values above the physical dead time
    // act as a tuned lead rather than a strict predictor - that is intended, the
    // gates keep it safe.
    float predict_frames = 0.0f;

    // REMOVED: ego-free-frame filtering. Lifting the measurement into an ego-free
    // frame before the One Euro update cut step overshoot -68% but made
    // ACQUISITION 40-80% slower - fast reach and overshoot are the same mechanism,
    // so it was a trade, not a win, and this rig's binding constraint is speed.
    // Superseded 2026-08-01 by per-frame adaptive dead time (deadtime_adaptive in
    // simple_main.cpp), which cuts overshoot -34% at -1.7% reach: strictly better
    // on both axes, so the branch had no remaining use and was deleted.
    // --- Class-switch artifact rejection ---
    // The rig CSVs show head<->body anchor flips on 0.6-2.4% of frames, each
    // moving the aim point 11-21px (about a third of the vertical variance). The
    // selected classId is already known here, so those frames can be identified
    // exactly - no inference, no threshold that could fire on noise (a magnitude
    // gate cannot tell an anchor flip from a real fast move; the class can).
    // On a flip: the frame contributes ZERO drift to the velocity estimate and the
    // damping term is cleared afterwards so the artifact cannot ring on for the
    // length of the EMA memory.
    //   sim (80 held-out seeds, realistic noise): total error -0.9%, reversal peak
    //   -1.5%, hold -1.2%, tracking -0.4..-1.4%, step overshoot/osc/reach UNCHANGED
    //   - nothing regresses. 0 = off.
    float class_switch_reject = 1.0f;

    // --- Body-priority target selection ---
    // head and body are two DIFFERENT anchors on the same enemy. Letting the
    // nearest-to-crosshair rule alternate between them adds variance the aim
    // cannot filter out: measured aim-point sigma was 4.22px mixed vs 4.00
    // body-only and 3.80 head-only - the MIX is worse than either source alone.
    // With this on, a body detection always outranks a head detection, so the aim
    // point is produced the same way every frame. head is kept as a FALLBACK for
    // the case that needs it (only the head visible over cover), which dropping
    // the class outright would lose. 0 = off (pure nearest-to-crosshair).
    // SCOPE: this orders the nearest-candidate rule only. IoU/distance
    // stickiness still wins over it - a head that is already the tracked target
    // stays the target even while a body is visible, because switching anchors
    // mid-track is exactly the variance this setting exists to avoid.
    float head_deprioritized = 1.0f;

    // --- Quieter body aim point (same point, less noise) ---
    // The aim point is y1 + k*h. Since h = y2 - y1 that is (1-k)*y1 + k*y2, so it
    // inherits BOTH edges' noise. Measured on 400 rig frames (crop 160, screen px):
    // the box jitter splits into a common whole-box translation (sigma ~2.5) and an
    // independent per-edge part (sigma ~2.8), so the CENTRE - which averages the
    // independent part down - is the quietest point on the box (3.11) while the
    // shipped body formula sits far from it (3.49, k=0.15).
    // Rewriting the same point as  cy + (k - 0.5)*h  is algebraically identical,
    // but now the fast component rides on the quiet centre and h carries its own
    // slow EMA. h is a person's box height: it changes with distance, i.e. slowly,
    // so heavy smoothing costs almost no tracking lag.
    //   measured: sigma_y 3.49 -> 3.12 (-10.4%) at alpha 0.2, lag bias 0.92px
    //   (bias vs a zero-phase reference; a quarter of sigma, so it stays buried)
    // head gains NOTHING (its k = 0.601 is already near the centre; measured +1%),
    // so this applies to the body anchor only.
    // 0 = off (use the raw height, i.e. the original formula exactly).
    float aim_h_ema = 0.2f;

    // --- Vertical strength trim ---
    // Multiplies the whole Y output (P + D + lead). 1.0 = the controller exactly
    // as tuned; lower weakens vertical assist only; 0 = horizontal-only. Applied
    // before the max-step clamp and the emit, so the clamp bounds real motion and
    // the in-flight ring records what actually went out (scaling afterwards would
    // desync the dead-time compensation). This is a taste/feel knob, not a tuned
    // one - the gains stay where the optimiser put them.
    float aim_y_scale = 1.0f;

    // --- One Euro adaptive low-pass on the target center ---
    // Removes detector jitter at the source: heavy smoothing when the target is
    // near-stationary (kills settle-shake), light smoothing when it moves fast
    // (no added lag on flicks). Cutoffs are in cycles/frame (sample period Te=1,
    // frames assumed near-constant rate). oneeuro_enabled != 0 to activate.
    float oneeuro_enabled = 1.0f;
    float oneeuro_min_cutoff = 0.1f; // base cutoff at rest (lower = smoother/more lag)
    float oneeuro_beta = 0.02f;      // speed coefficient (higher = less lag when fast)
    // NOTE: oneeuro_dcutoff was removed as a config knob. Sweeping it 0.1..5.0
    // moves total error by less than 0.2% (it only shapes the derivative used for
    // the adaptive cutoff), so it was pure tuning surface with no lever behind it.
    // Fixed at kOneEuroDCutoff below (1.0 measured marginally best).

    // --- Output gate (does this frame's move actually reach the mouse?) ---
    // The GPU records every emitted move into AimState's in-flight ring, and the
    // dead-time compensation then subtracts those moves from the next
    // measurement. That is only correct if the move REALLY went out. Inference,
    // however, also runs while the aim key is up (inference_keepwarm_ms < 0 keeps
    // it always on), and the host sends nothing on those frames - so the ring was
    // filling with moves that never happened and the compensation was subtracting
    // phantom motion, carrying that state into the next real aim.
    // The host knows at submit time whether this frame's move will be sent, so it
    // says so here. 0 = compute and keep the tracking state warm, but emit 0 and
    // push 0 into the ring. The decision is echoed back in
    // InferenceResult::outputEnabled so the host emits on exactly the frames the
    // ring recorded - the two can no longer disagree.
    float aim_output_enabled = 1.0f;

    // --- Static shoot-offset aim-shift (in OUTPUT/screen px) ---
    // Shifts the aim REFERENCE POINT away from screen center by this vector so
    // the controller converges with the target resting at center + offset
    // (e.g. some weapons' shots land above the crosshair, so the aim point is
    // above center). This is a SETPOINT shift folded into the error - the aim
    // settles at the offset and holds, unlike a per-frame additive nudge which
    // would drift/jerk. Set per-frame by the host: the configured value while
    // shooting, 0 otherwise. Negative Y = reference above center.
    float shoot_offset_x = 0.0f;
    float shoot_offset_y = 0.0f;
};

struct MouseMovement {
    int dx = 0;
    int dy = 0;
};

// All inference outputs packed into one struct for a single D2H copy
// (cuts the transfer from 3 copies to 1).
struct InferenceResult {
    MouseMovement movement;     // 8 bytes: dx, dy
    int hasTarget;              // 4 bytes: 1 if target found, 0 otherwise
    float targetX1, targetY1;   // 8 bytes: best target bbox (if hasTarget)
    float targetX2, targetY2;   // 8 bytes
    float targetConf;           // 4 bytes: confidence
    int targetClassId;          // 4 bytes: class ID
    // Model-input px -> output px scale that pd_controller's own movement_scale
    // used this frame. Only the calibration logger reads it (calib.csv records
    // moves in both spaces); the controller itself needs nothing from here.
    float movementScaleX, movementScaleY;  // 8 bytes
    // Echo of AimConfig::aim_output_enabled for the frame that produced this
    // result. The host emits the move if and only if this is non-zero, so the
    // moves it sends are exactly the ones the in-flight ring recorded. Reading
    // the live aim key in the completion callback instead would let the two
    // disagree whenever the key moved while the frame was in flight.
    int outputEnabled;          // 4 bytes
    // Total: 48 bytes - still within a single 64B cache line
};

// One-pass fused postprocess:
// 1) Decode YOLO output
// 2) Target selection with IoU stickiness
// 3) Nonlinear P movement calculation
// 4) Pack final InferenceResult (single D2H copy)
cudaError_t postprocessYoloFusedGpu(
    const void* d_raw_output,      // Raw model output (FP32/FP16)
    bool is_fp16,                  // Output tensor is FP16
    int num_boxes,                 // Number of anchor boxes
    int num_classes,               // Number of classes
    float conf_threshold,          // Confidence threshold
    uint32_t allowedClassMask,     // Bitmask of allowed classes
    int max_candidate_blocks,      // Upper bound of stage-1 candidate blocks
    float max_box_extent,          // Max valid width/height for decoded boxes
    float screen_center_x,         // Crosshair X
    float screen_center_y,         // Crosshair Y
    float movement_scale_x,        // Model-space movement -> source/screen-space scale
    float movement_scale_y,
    int head_class_id,             // Head class ID for priority
    const AimConfig* d_aim_config, // Movement parameters (device pointer)
    float iou_stickiness_threshold, // IoU threshold for target stickiness (0.3 typical)
    float head_y_offset,           // Aim point offset for head (0.0-1.0)
    float body_y_offset,           // Aim point offset for body (0.0-1.0)
    Detection* d_selected_target,  // Persistent selected target (for IoU tracking)
    AimState* d_aim_state,         // Persistent movement state (device)
    InferenceResult* d_inference_result, // Packed output result (required)
    Detection* d_stage1_best_dist, // [max_candidate_blocks] best candidate by distance per block
    float* d_stage1_dist_score,    // [max_candidate_blocks] distance score per block
    Detection* d_stage1_best_iou,  // [max_candidate_blocks] best candidate by IoU per block
    float* d_stage1_iou_score,     // [max_candidate_blocks] IoU score per block
    cudaStream_t stream = 0
);

} // namespace gpa
