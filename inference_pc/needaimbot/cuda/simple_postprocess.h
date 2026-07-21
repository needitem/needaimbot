#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gpa {

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

    // --- Coast state ---
    // During a detection gap, follow only the TARGET's screen drift (its
    // per-frame center velocity), decayed - NOT the P-convergence move (which
    // would keep closing toward a point and overshoot). prev_center_* is the
    // last detected aim point; vel_* is its smoothed per-frame drift.
    int   has_track = 0;
    float prev_center_x = 0.0f;
    float prev_center_y = 0.0f;
    float vel_x = 0.0f;
    float vel_y = 0.0f;

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
    // Track persistence / coast window: how many consecutive missed frames to
    // bridge before dropping the target. 0 = disabled.
    int track_persistence_frames = 5;

    // --- Coast (smooth bridging of detection gaps) ---
    // coast_enabled != 0: during the persistence window, keep emitting the last
    // movement scaled by coast_decay^frames (glide) instead of holding still.
    // Decays to zero so a vanished target does not cause shake or freeze.
    //
    // NOTE: velocity feedforward / dead-time prediction were removed. The
    // screen-space velocity estimate (AimState::vel_*) is EGO-POLLUTED: the
    // crosshair's own movement shifts the whole scene, so the measured drift is
    // d(error)/dt, which is ~0 at steady tracking - feedforward on it cannot
    // cancel ramp lag and only amplifies detector noise (sim-confirmed). Do not
    // reintroduce lead terms on this signal; they need a LAG-ALIGNED ego-motion
    // corrected velocity (add back the applied moves from ~dead-time frames ago)
    // to work. vel_* itself stays: coast uses it to glide across detection gaps.
    float coast_enabled = 1.0f;
    float coast_decay = 0.85f;

    // --- One Euro adaptive low-pass on the target center ---
    // Removes detector jitter at the source: heavy smoothing when the target is
    // near-stationary (kills settle-shake), light smoothing when it moves fast
    // (no added lag on flicks). Cutoffs are in cycles/frame (sample period Te=1,
    // frames assumed near-constant rate). oneeuro_enabled != 0 to activate.
    float oneeuro_enabled = 1.0f;
    float oneeuro_min_cutoff = 0.1f; // base cutoff at rest (lower = smoother/more lag)
    float oneeuro_beta = 0.02f;      // speed coefficient (higher = less lag when fast)
    float oneeuro_dcutoff = 0.5f;    // derivative cutoff for the speed estimate

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
    // 1 if has_track transitioned 0->1 this frame (fresh target lock, not a
    // continued/coasted track) - lets the host play a one-shot humanized
    // acquisition flick (needaimbot/mouse/warped_replay.hpp) instead of the
    // raw PD movement for this target's first few frames.
    int freshAcquire;           // 4 bytes
    float targetX1, targetY1;   // 8 bytes: best target bbox (if hasTarget)
    float targetX2, targetY2;   // 8 bytes
    float targetConf;           // 4 bytes: confidence
    int targetClassId;          // 4 bytes: class ID
    // Pre-movement-scale error vector (raw_center - screen_center, model-input
    // space) and the scale that converts it to output px - same inputs
    // pd_controller's own error_x/error_y + movement_scale use. Only
    // meaningful when freshAcquire is set; lets the host seed a
    // warped_replay flick without needing its own screen-center/scale state.
    float errorX, errorY;             // 8 bytes
    float movementScaleX, movementScaleY;  // 8 bytes
    // Total: 56 bytes - still within a single 64B cache line
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
