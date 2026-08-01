#pragma once

// GPU-side nonlinear P(D) aim controller: the actual per-frame movement math,
// split out of stage2FinalizeKernel (simple_postprocess.cu) for readability.
// Config assembly (gains/oneeuro -> AimConfig) lives host-side in
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
// disables. Applied to the P+D movement path so the slew bound is uniform and a
// large leap cannot overshoot/ring regardless of source.
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

// Record an emitted move into the in-flight ring (OUTPUT px). Every path that
// moves the mouse must call this, or the dead-time compensation under-counts.
__device__ __forceinline__ void pushInflight(AimState* s, float dx, float dy) {
    const int i = s->inflight_head;
    s->inflight_x[i] = dx;
    s->inflight_y[i] = dy;
    s->inflight_head = (i + 1) % AimState::kInflightMax;
}

// Sum of the moves emitted in the last `frames` frames (OUTPUT px) - i.e. the
// moves the current (stale) measurement cannot have seen yet. `frames` is
// FRACTIONAL: whole entries are summed, then the boundary entry is weighted by
// the remainder, so a real dead time of e.g. 1.13 frames is represented exactly
// instead of being rounded to 1 (under-compensate) or 2 (over-compensate).
__device__ __forceinline__ void inflightSum(const AimState* s, float frames,
                                            float& sx, float& sy) {
    sx = 0.0f; sy = 0.0f;
    const float w = fminf(fmaxf(frames, 0.0f), static_cast<float>(AimState::kInflightMax));
    const int n = static_cast<int>(floorf(w));
    for (int k = 1; k <= n; ++k) {
        const int i = (s->inflight_head - k + AimState::kInflightMax) % AimState::kInflightMax;
        sx += s->inflight_x[i];
        sy += s->inflight_y[i];
    }
    const float frac = w - static_cast<float>(n);
    if (frac > 0.0f && (n + 1) <= AimState::kInflightMax) {
        const int i = (s->inflight_head - (n + 1) + AimState::kInflightMax) % AimState::kInflightMax;
        sx += frac * s->inflight_x[i];
        sy += frac * s->inflight_y[i];
    }
}

// The single emit sitting `lag` frames back (lag 1 = most recent), fractional lag
// blending the two neighbours. Used to recover our own view motion between the
// last two detections so the target's true velocity can be reconstructed.
__device__ __forceinline__ void inflightAt(const AimState* s, float lag,
                                           float& ax, float& ay) {
    const float w = fminf(fmaxf(lag, 1.0f), static_cast<float>(AimState::kInflightMax));
    const int n = static_cast<int>(floorf(w));
    const float frac = w - static_cast<float>(n);
    const int i = (s->inflight_head - n + AimState::kInflightMax) % AimState::kInflightMax;
    ax = s->inflight_x[i];
    ay = s->inflight_y[i];
    if (frac > 0.0f && (n + 1) <= AimState::kInflightMax) {
        const int j = (s->inflight_head - (n + 1) + AimState::kInflightMax) % AimState::kInflightMax;
        ax = (1.0f - frac) * ax + frac * s->inflight_x[j];
        ay = (1.0f - frac) * ay + frac * s->inflight_y[j];
    }
}

// Confidence gate shared by BOTH lead terms (ff_gain on the output and
// predict_frames on the setpoint). They ride the same velocity estimate and must
// be suppressed in the same two situations, so the gate lives in one place:
//   speed gate  |v|/(|v|+vgate)      -> ~0 at rest, so the (noisy) velocity
//                                       estimate cannot shake a held aim.
//   error gate  e0^2/(e0^2+|err|^2)  -> ~0 while acquiring, so the huge one-frame
//                                       velocity of a position jump is never led
//                                       on (that is what would overshoot).
__device__ __forceinline__ float leadGate(const AimConfig& cfg, const AimState* s,
                                          float error_x, float error_y) {
    float g = 1.0f;
    if (cfg.lead_vgate > 0.0f) {
        const float sp = sqrtf(s->vel_x * s->vel_x + s->vel_y * s->vel_y);
        g *= sp / (sp + cfg.lead_vgate);
    }
    if (cfg.lead_err_gate > 0.0f) {
        const float e0 = cfg.lead_err_gate;
        const float e2 = error_x * error_x + error_y * error_y;
        g *= (e0 * e0) / (e0 * e0 + e2);
    }
    return g;
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

// Full PD controller for a target detected THIS frame: One Euro pre-filter on
// the measured center -> nonlinear P + D convergence -> max-step clamp ->
// integer mouse delta. Mutates aim_state in place (filter/derivative/residual
// carry and has_track for the next frame).
__device__ __forceinline__ void computeAimMovement(
    float raw_center_x, float raw_center_y,
    float screen_center_x, float screen_center_y,
    float movement_scale_x, float movement_scale_y,
    const AimConfig& aim_config,
    AimState* aim_state,
    int target_class,
    int& out_dx, int& out_dy) {
    // NOTE: confidence-weighted filtering was tried and removed. The idea was to
    // down-weight low-confidence detections (outliers) in the One Euro update. It
    // works in sim but NOT on hardware: a real MOVING target has low confidence too
    // (motion blur drops it to ~0.4-0.5, overlapping the ~0.38 outlier level), so
    // weighting by confidence suppresses real moving-target frames -> the aim tracks
    // a moving target in visible steps ("툭툭툭") instead of smoothly. Confidence
    // cannot distinguish a blurred real target from an outlier. Do not reintroduce a
    // plain conf weight; if outlier rejection is wanted, gate it on LOW target
    // velocity (only suppress while locked/stationary, never while tracking).
    // Was this target already being tracked last frame? Captured before
    // has_track is overwritten below; used to seed the One Euro filter and
    // reset the error derivative on a fresh acquire (avoids a derivative kick).
    const bool fresh_track = (aim_state->has_track == 0);

    // Anchor flip (head<->body on the same enemy): the centre jumps by the offset
    // between the two aim points.
    // For the VELOCITY estimate that jump is a pure artifact - the target did not
    // move - and the lead term would fling on it and keep flinging for the length
    // of the EMA. So the frame is excluded there (below), and that exclusion is
    // where this feature's measured benefit comes from.
    // The DAMPING term deliberately still sees it. After a flip the aim point has
    // genuinely moved to the other anchor, so the error step is a real SETPOINT
    // CHANGE, not a measurement artifact, and letting D respond once helps cross
    // it. Measured (sim, 3 blocks x60, both gain profiles): suppressing that
    // response - by holding the EMA or by zeroing it before the move - is
    // 0.03-0.22% WORSE. Do not "fix" it; the earlier comment here claimed the
    // damping term was excluded too, which the code never did.
    const bool class_changed =
        (aim_config.class_switch_reject != 0.0f && !fresh_track &&
         aim_state->prev_class >= 0 && target_class != aim_state->prev_class);
    if (class_changed) {
        // zero drift for this frame: velocity sees no jump at all
        aim_state->prev_raw_x = raw_center_x;
        aim_state->prev_raw_y = raw_center_y;
    }
    aim_state->prev_class = target_class;

    // One Euro adaptive low-pass on the measured center, applied BEFORE the
    // error so the whole controller (P move) runs on the de-noised signal. Seed
    // on fresh acquire to avoid a jump from a stale value.
    // In-flight sum is needed by BOTH paths: the ego-free path folds it into the
    // filter input, the classic path subtracts it from the error further down.
    float inf_x = 0.0f, inf_y = 0.0f;
    const bool use_comp =
        (aim_config.inflight_comp > 0.0f && aim_config.deadtime_frames > 0.0f);
    if (use_comp) {
        inflightSum(aim_state, aim_config.deadtime_frames, inf_x, inf_y);
    }
    const bool ego_frame =
        (aim_config.ego_frame_filter != 0.0f && aim_config.oneeuro_enabled != 0.0f);

    float target_center_x = raw_center_x;
    float target_center_y = raw_center_y;
    if (aim_config.oneeuro_enabled != 0.0f) {
        // Ego-free path: filter the measurement with our own motion removed, and
        // carry the previous estimate corrected by our own last emit. Then the
        // filter smooths only the TARGET instead of also smoothing our corrections
        // (which is what makes the classic path report a stale error and overshoot).
        float in_x = raw_center_x, in_y = raw_center_y;
        float carry_x = aim_state->filt_x, carry_y = aim_state->filt_y;
        if (ego_frame) {
            if (use_comp) {
                if (movement_scale_x != 0.0f) in_x -= aim_config.inflight_comp * inf_x / movement_scale_x;
                if (movement_scale_y != 0.0f) in_y -= aim_config.inflight_comp * inf_y / movement_scale_y;
            }
            float last_x = 0.0f, last_y = 0.0f;
            inflightAt(aim_state, 1.0f, last_x, last_y);   // the emit applied since last frame
            if (movement_scale_x != 0.0f) carry_x -= last_x / movement_scale_x;
            if (movement_scale_y != 0.0f) carry_y -= last_y / movement_scale_y;
        }
        if (aim_state->has_track) {
            const float ad = oneEuroAlpha(kOneEuroDCutoff);
            // Derivative for the adaptive cutoff is (measurement - estimate) in
            // whichever frame we are working in: with the ego-free frame both
            // terms carry the same +M, so it is in_x - filt_x, NOT in_x - carry_x
            // (carry already has our own motion removed - using it here would add
            // a spurious +lastEmit to the speed estimate).
            const float de_x = in_x - aim_state->filt_x;
            aim_state->dfilt_x = ad * de_x + (1.0f - ad) * aim_state->dfilt_x;
            const float cutoff_x =
                aim_config.oneeuro_min_cutoff + aim_config.oneeuro_beta * fabsf(aim_state->dfilt_x);
            const float ax = oneEuroAlpha(cutoff_x);
            aim_state->filt_x = ax * in_x + (1.0f - ax) * carry_x;

            const float de_y = in_y - aim_state->filt_y;
            aim_state->dfilt_y = ad * de_y + (1.0f - ad) * aim_state->dfilt_y;
            const float cutoff_y =
                aim_config.oneeuro_min_cutoff + aim_config.oneeuro_beta * fabsf(aim_state->dfilt_y);
            const float ay = oneEuroAlpha(cutoff_y);
            aim_state->filt_y = ay * in_y + (1.0f - ay) * carry_y;
        } else {
            aim_state->filt_x = in_x;
            aim_state->filt_y = in_y;
            aim_state->dfilt_x = 0.0f;
            aim_state->dfilt_y = 0.0f;
        }
        target_center_x = aim_state->filt_x;
        target_center_y = aim_state->filt_y;
    }

    // Ego-corrected target velocity. Feeds BOTH lead terms, so it must be updated
    // whenever EITHER is enabled - gating it on ff_gain alone would make
    // predict_frames silently do nothing. Uses the RAW center - not the One Euro
    // output - so the velocity is not position-lagged. d_raw = dTarget -
    // dCrosshair; our own dCrosshair between the last two detections is the emit
    // at ring lag ff_ego_lag, so adding it back recovers the target's TRUE screen
    // drift. Updated before has_track flips so a fresh acquire starts from zero
    // velocity instead of a spurious jump.
    const bool lead_active =
        (aim_config.ff_gain > 0.0f || aim_config.predict_frames > 0.0f);
    if (lead_active) {
        if (aim_state->has_track) {
            float ax = 0.0f, ay = 0.0f;
            inflightAt(aim_state, aim_config.ff_ego_lag, ax, ay);
            const float maxDrift = 60.0f;   // model px/frame sanity clamp
            float dvx = (raw_center_x - aim_state->prev_raw_x)
                      + ((movement_scale_x != 0.0f) ? ax / movement_scale_x : 0.0f);
            float dvy = (raw_center_y - aim_state->prev_raw_y)
                      + ((movement_scale_y != 0.0f) ? ay / movement_scale_y : 0.0f);
            dvx = fminf(fmaxf(dvx, -maxDrift), maxDrift);
            dvy = fminf(fmaxf(dvy, -maxDrift), maxDrift);
            const float a = fminf(fmaxf(aim_config.ff_v_ema, 0.0f), 1.0f);
            aim_state->vel_x = (1.0f - a) * aim_state->vel_x + a * dvx;
            aim_state->vel_y = (1.0f - a) * aim_state->vel_y + a * dvy;
        } else {
            aim_state->vel_x = 0.0f;
            aim_state->vel_y = 0.0f;
        }
        aim_state->prev_raw_x = raw_center_x;
        aim_state->prev_raw_y = raw_center_y;
    }

    aim_state->has_track = 1;

    // Aim at the measured target center, shifted by the static shoot-offset
    // reference. The offset is in output/screen px; convert to the model-space
    // error here by dividing out movement_scale (error is later multiplied by
    // it to produce the screen-space move). Folding it into the error - rather
    // than adding it to the output every frame - makes it a true setpoint: the
    // aim converges with the target resting at center + offset and HOLDS there,
    // instead of drifting/jerking as an unconditional per-frame nudge would.
    const float shoot_off_x =
        (movement_scale_x != 0.0f) ? aim_config.shoot_offset_x / movement_scale_x : 0.0f;
    const float shoot_off_y =
        (movement_scale_y != 0.0f) ? aim_config.shoot_offset_y / movement_scale_y : 0.0f;
    float error_x = target_center_x - screen_center_x - shoot_off_x;
    float error_y = target_center_y - screen_center_y - shoot_off_y;

    // Dead-time compensation: this measurement is ~deadtime_frames old, so the
    // moves emitted since it was captured have not shifted the target in it
    // yet. Subtract them (converted OUTPUT px -> model px) so we answer only
    // the error our in-flight moves have NOT already addressed. Without this
    // the loop double-corrects and rings; with it, kp can go higher.
    // Classic path only: the ego-free path already folded this into the filter
    // input, so subtracting it again here would double-compensate.
    if (use_comp && !ego_frame) {
        if (movement_scale_x != 0.0f) error_x -= aim_config.inflight_comp * inf_x / movement_scale_x;
        if (movement_scale_y != 0.0f) error_y -= aim_config.inflight_comp * inf_y / movement_scale_y;
    }

    // Symmetric dead-time compensation: the in-flight subtraction above removed
    // OUR motion during the dead time; this extrapolates the TARGET forward over
    // the same dead time, which nothing did before. Shifting the SETPOINT (not
    // the output) means it passes through the nonlinear P and the max-step clamp.
    // Gated exactly like the lead term: ~0 at rest (noise) and ~0 while acquiring
    // (a position jump's one-frame velocity must never be extrapolated).
    if (aim_config.predict_frames > 0.0f) {
        const float g = leadGate(aim_config, aim_state, error_x, error_y);
        error_x += aim_config.predict_frames * g * aim_state->vel_x;
        error_y += aim_config.predict_frames * g * aim_state->vel_y;
    }

    // Derivative (damping) term: react to how fast the error is shrinking and
    // push back, so a high-kp approach decelerates BEFORE it overshoots. The
    // error rate includes target drift, so D stays quiet (de ~ 0) at steady
    // tracking and only bites on transients. Error rides the One Euro-filtered center, so the
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

    // Second lead term, on the OUTPUT this time (predict_frames above shifts the
    // setpoint). Same gate - see leadGate(). Both are kept because they compose:
    // the setpoint form passes through the nonlinear P, the output form does not,
    // and together they beat either alone (measured).
    float ff_x = 0.0f, ff_y = 0.0f;
    if (aim_config.ff_gain > 0.0f) {
        const float g = leadGate(aim_config, aim_state, error_x, error_y);
        ff_x = aim_config.ff_gain * g * aim_state->vel_x;
        ff_y = aim_config.ff_gain * g * aim_state->vel_y;
    }

    float movement_x =
        (nonlinearPMove(error_x, aim_config.kp_x, aim_config.p_softness_x)
         + aim_config.kd_x * aim_state->derr_x + ff_x) * movement_scale_x;
    float movement_y =
        (nonlinearPMove(error_y, aim_config.kp_y, aim_config.p_softness_y)
         + aim_config.kd_y * aim_state->derr_y + ff_y) * movement_scale_y;

    // Per-frame max-step clamp (output px). Bounds the slew so a large initial
    // error is crossed in several smooth steps instead of one delayed leap
    // that overshoots and rings - overshoot D cannot prevent reactively.
    clampMaxStep(movement_x, movement_y, aim_config.max_step);

    out_dx = emitMouseDelta(movement_x, &aim_state->residual_x);
    out_dy = emitMouseDelta(movement_y, &aim_state->residual_y);
    pushInflight(aim_state, static_cast<float>(out_dx), static_cast<float>(out_dy));

    // Bound the echo, not the response. The flip frame's single D response above
    // is wanted (see class_changed), but the derivative EMA would keep replaying
    // that one step for the length of its memory. Clearing after the move keeps
    // the response and drops the tail. At the shipped kd this is worth ~0.01%
    // (i.e. nothing measurable) - it is a bound for large kd, not a win here.
    if (class_changed) {
        aim_state->derr_x = 0.0f;
        aim_state->derr_y = 0.0f;
    }
}

}  // namespace gpa
