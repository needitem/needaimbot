#pragma once

// GPU-side nonlinear P(D) aim controller: the actual per-frame movement math,
// split out of stage2FinalizeKernel (simple_postprocess.cu) for readability.
// Config assembly (gains/coast/oneeuro -> AimConfig) lives host-side in
// needaimbot/mouse/pd_controller.hpp; this file is what actually turns a
// target center + AimConfig + AimState into an emitted (dx, dy) each frame.

#include "simple_postprocess.h"

namespace gpa {

// alpha = 1 / (1 + tau/Te), tau = 1 / (2*pi*cutoff).
__device__ __forceinline__ float oneEuroAlpha(float cutoff) {
    const float tau = 1.0f / (2.0f * 3.14159265f * fmaxf(cutoff, 1e-4f));
    return 1.0f / (1.0f + tau);
}

// Cap a per-frame move vector to maxStep px, preserving direction. maxStep <= 0
// disables. Applied to every movement path (P+D+ff and coast glide) so the slew
// bound is uniform and a large leap cannot overshoot/ring regardless of source.
__device__ __forceinline__ void clampMaxStep(float& mx, float& my, float maxStep) {
    if (maxStep > 0.0f) {
        const float step = sqrtf(mx * mx + my * my);
        if (step > maxStep) {
            const float s = maxStep / step;
            mx *= s;
            my *= s;
        }
    }
}

__device__ __forceinline__ float nonlinearPMove(float error, float kp, float softness) {
    const float abs_error = fabsf(error);
    const float safe_softness = fmaxf(softness, 1.0f);
    const float gain = fmaxf(kp, 0.0f) * (abs_error / (abs_error + safe_softness));
    return error * gain;
}

__device__ __forceinline__ int emitMouseDelta(float movement, float* residual) {
    // Clear carry if it opposes the new direction: stale residual from a
    // previous frame must not fight a freshly reversed target motion.
    float carried = *residual;
    if (movement * carried < 0.0f) {
        carried = 0.0f;
    }
    const float value = movement + carried;
    int emit = __float2int_rz(value);
    if (emit > 127) {
        // Saturated: keep the overflow so the next frame can keep catching
        // up, but cap the carry so runaway accumulation can't outlive the
        // motion that caused it.
        *residual = fminf(256.0f, value - 127.0f);
        return 127;
    }
    if (emit < -127) {
        *residual = fmaxf(-256.0f, value + 127.0f);
        return -127;
    }
    *residual = value - static_cast<float>(emit);
    return emit;
}

// Detection-gap "coast": glide from the target's last known screen drift
// (decayed per missed frame) instead of freezing or snapping, bridging brief
// occlusions/misses. Mutates aim_state's residual carry; does not touch
// velocity/filter/derivative state (those only advance on an actual detection).
__device__ __forceinline__ void computeCoastMovement(
    const AimConfig& aim_config,
    AimState* aim_state,
    float movement_scale_x, float movement_scale_y,
    int missed_frames,
    int& out_dx, int& out_dy) {
    float factor = 1.0f;
    for (int i = 0; i < missed_frames; ++i) factor *= aim_config.coast_decay;
    // Follow only the target's drift (vel * scale), decayed. No convergence
    // term, so a stationary target does not drift off.
    float mx = aim_state->vel_x * movement_scale_x * factor;
    float my = aim_state->vel_y * movement_scale_y * factor;
    clampMaxStep(mx, my, aim_config.max_step);
    out_dx = emitMouseDelta(mx, &aim_state->residual_x);
    out_dy = emitMouseDelta(my, &aim_state->residual_y);
}

// Full PD controller for a target detected THIS frame: One Euro pre-filter on
// the measured center -> per-frame drift tracking (feeds coast/feedforward) ->
// nonlinear P + D convergence + velocity feedforward -> max-step clamp ->
// integer mouse delta. Mutates aim_state in place (filter/velocity/derivative/
// residual carry, and has_track/prev_center for the next frame).
__device__ __forceinline__ void computeAimMovement(
    float raw_center_x, float raw_center_y,
    float screen_center_x, float screen_center_y,
    float movement_scale_x, float movement_scale_y,
    const AimConfig& aim_config,
    AimState* aim_state,
    int& out_dx, int& out_dy) {
    // Was this target already being tracked last frame? Captured before
    // has_track is overwritten below; used to seed the One Euro filter and
    // reset the error derivative on a fresh acquire (avoids a derivative kick).
    const bool fresh_track = (aim_state->has_track == 0);

    // One Euro adaptive low-pass on the measured center, applied BEFORE
    // velocity and error so the whole controller (P move, feedforward, coast)
    // runs on the de-noised signal. Seed on fresh acquire to avoid a jump from
    // a stale value.
    float target_center_x = raw_center_x;
    float target_center_y = raw_center_y;
    if (aim_config.oneeuro_enabled != 0.0f) {
        if (aim_state->has_track) {
            const float ad = oneEuroAlpha(aim_config.oneeuro_dcutoff);
            const float de_x = raw_center_x - aim_state->filt_x;  // per-frame derivative
            aim_state->dfilt_x = ad * de_x + (1.0f - ad) * aim_state->dfilt_x;
            const float cutoff_x =
                aim_config.oneeuro_min_cutoff + aim_config.oneeuro_beta * fabsf(aim_state->dfilt_x);
            const float ax = oneEuroAlpha(cutoff_x);
            aim_state->filt_x = ax * raw_center_x + (1.0f - ax) * aim_state->filt_x;

            const float de_y = raw_center_y - aim_state->filt_y;
            aim_state->dfilt_y = ad * de_y + (1.0f - ad) * aim_state->dfilt_y;
            const float cutoff_y =
                aim_config.oneeuro_min_cutoff + aim_config.oneeuro_beta * fabsf(aim_state->dfilt_y);
            const float ay = oneEuroAlpha(cutoff_y);
            aim_state->filt_y = ay * raw_center_y + (1.0f - ay) * aim_state->filt_y;
        } else {
            aim_state->filt_x = raw_center_x;
            aim_state->filt_y = raw_center_y;
            aim_state->dfilt_x = 0.0f;
            aim_state->dfilt_y = 0.0f;
        }
        target_center_x = aim_state->filt_x;
        target_center_y = aim_state->filt_y;
    }

    // Update the target's per-frame screen drift (EMA, clamped) FIRST so both
    // the feedforward term below and a following detection gap's coast use the
    // freshest velocity. Use the RAW center deltas here, NOT the One Euro
    // filtered center: the filter delays position, and feeding a lagged
    // velocity into feedforward under-leads a moving target (aim trails its
    // tail). The P/error term below still uses the filtered center for a
    // stable aim point, so smoothing stabilizes WHERE we point without eating
    // the lead signal. (When One Euro is off, raw_center == target_center, so
    // this is a no-op.)
    if (aim_state->has_track) {
        const float maxDrift = 60.0f;  // model px/frame sanity clamp
        float nvx = raw_center_x - aim_state->prev_center_x;
        float nvy = raw_center_y - aim_state->prev_center_y;
        nvx = fminf(fmaxf(nvx, -maxDrift), maxDrift);
        nvy = fminf(fmaxf(nvy, -maxDrift), maxDrift);
        aim_state->vel_x = 0.6f * aim_state->vel_x + 0.4f * nvx;
        aim_state->vel_y = 0.6f * aim_state->vel_y + 0.4f * nvy;
    } else {
        aim_state->vel_x = 0.0f;
        aim_state->vel_y = 0.0f;
    }
    aim_state->prev_center_x = raw_center_x;  // raw (un-lagged) for next velocity
    aim_state->prev_center_y = raw_center_y;
    aim_state->has_track = 1;

    // Aim at the measured target center, shifted by the static shoot-offset
    // reference. The offset is in output/screen px; convert to the model-space
    // error here by dividing out movement_scale (error is later multiplied by
    // it to produce the screen-space move). Folding it into the error - rather
    // than adding it to the output every frame - makes it a true setpoint: the
    // aim converges with the target resting at center + offset and HOLDS there,
    // instead of drifting/jerking as an unconditional per-frame nudge would.
    // Velocity feedforward keeps pace with a moving target (cancels P
    // steady-state lag) without leading/overshooting.
    const float shoot_off_x =
        (movement_scale_x != 0.0f) ? aim_config.shoot_offset_x / movement_scale_x : 0.0f;
    const float shoot_off_y =
        (movement_scale_y != 0.0f) ? aim_config.shoot_offset_y / movement_scale_y : 0.0f;
    const float error_x = target_center_x - screen_center_x - shoot_off_x;
    const float error_y = target_center_y - screen_center_y - shoot_off_y;
    const float ff = aim_config.feedforward_gain;

    // Derivative (damping) term: react to how fast the error is shrinking and
    // push back, so a high-kp approach decelerates BEFORE it overshoots. This
    // is pure damping of OUR convergence - target motion is handled by
    // feedforward, so D stays quiet (de ~ 0) while tracking well and only
    // bites on transients. Error rides the One Euro-filtered center, so the
    // derivative is clean; it is still clamped and reset on fresh acquire to
    // avoid a derivative kick. The clamp is generous (a full-frame initial
    // slew can change error by >60px in one frame); a tighter clamp would
    // starve the damping exactly in the large-error regime that overshoots
    // most. Fresh-acquire reset, not this clamp, guards against the real
    // derivative kick.
    const float maxDErr = 150.0f;
    float de_x = fminf(fmaxf(error_x - aim_state->prev_err_x, -maxDErr), maxDErr);
    float de_y = fminf(fmaxf(error_y - aim_state->prev_err_y, -maxDErr), maxDErr);
    if (fresh_track) {
        aim_state->derr_x = 0.0f;
        aim_state->derr_y = 0.0f;
    } else {
        aim_state->derr_x = 0.6f * de_x + 0.4f * aim_state->derr_x;
        aim_state->derr_y = 0.6f * de_y + 0.4f * aim_state->derr_y;
    }
    aim_state->prev_err_x = error_x;
    aim_state->prev_err_y = error_y;

    float movement_x =
        (nonlinearPMove(error_x, aim_config.kp_x, aim_config.p_softness_x)
         + aim_config.kd_x * aim_state->derr_x
         + ff * aim_state->vel_x) * movement_scale_x;
    float movement_y =
        (nonlinearPMove(error_y, aim_config.kp_y, aim_config.p_softness_y)
         + aim_config.kd_y * aim_state->derr_y
         + ff * aim_state->vel_y) * movement_scale_y;

    // Per-frame max-step clamp (output px). Bounds the slew so a large initial
    // error is crossed in several smooth steps instead of one delayed leap
    // that overshoots and rings - overshoot D cannot prevent reactively.
    clampMaxStep(movement_x, movement_y, aim_config.max_step);

    out_dx = emitMouseDelta(movement_x, &aim_state->residual_x);
    out_dy = emitMouseDelta(movement_y, &aim_state->residual_y);
}

}  // namespace gpa
