#pragma once

// GPU-side nonlinear P(D) aim controller: the actual per-frame movement math,
// split out of stage2FinalizeKernel (simple_postprocess.cu) for readability.
// Config assembly (gains/oneeuro -> AimConfig) lives host-side in
// needaimbot/mouse/pd_controller.hpp; this file is what actually turns a
// target center + AimConfig + AimState into an emitted (dx, dy) each frame.

#include "simple_postprocess.h"

namespace gpa {

// --- Hardcoded internal constants (tuned, not user-configurable) ---
// kHeadDeprioritized and kAimHEma are in simple_postprocess.h (used by both files).
// Lead term gates
__device__ constexpr float kLeadVgate       = 8.6f;
__device__ constexpr float kLeadErrGate     = 22.25f;
__device__ constexpr float kFfVEma          = 0.137f;
__device__ constexpr float kFfEgoLag        = 2.85f;
// SEM internals
__device__ constexpr float kSemRate         = 0.02f;
__device__ constexpr float kSemMax          = 3.0f;
__device__ constexpr float kSemDecay        = 0.95f;
// CGA internals
__device__ constexpr float kCgaConfLow      = 0.4f;
__device__ constexpr float kCgaConfHigh     = 0.8f;
__device__ constexpr float kCgaSpeedThreshold = 3.0f;
// JCAD internals
__device__ constexpr float kJcadAlpha       = 0.15f;
__device__ constexpr float kJcadBaseDb      = 0.5f;
__device__ constexpr float kJcadJitterDb    = 3.0f;
__device__ constexpr float kJcadJitterThreshold = 2.0f;
// PAGS internals
__device__ constexpr float kPagsAcquireThreshold = 15.0f;
__device__ constexpr float kPagsOscThreshold = 0.5f;
__device__ constexpr float kPagsAcquireMult  = 1.1f;
__device__ constexpr float kPagsOscMult      = 0.6f;
__device__ constexpr float kPagsAlpha        = 0.2f;

// alpha = 1 / (1 + tau/Te), tau = 1 / (2*pi*cutoff).
__device__ __forceinline__ float oneEuroAlpha(float cutoff) {
    const float w = 2.0f * 3.14159265f * fmaxf(cutoff, 1e-4f);
    return w / (w + 1.0f);
}

// Precomputed derivative alpha for kOneEuroDCutoff = 1.0:
// oneEuroAlpha(1.0) = 2*pi / (2*pi + 1) ≈ 0.8626802...
__device__ constexpr float kDerivAlpha =
    (2.0f * 3.14159265f) / (2.0f * 3.14159265f + 1.0f);

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

__device__ __forceinline__ float nonlinearPMove(float error, float kp, float softness,
                                                 float fine_boost = 1.0f,
                                                 float fine_range = 0.0f) {
    const float abs_error = fabsf(error);
    const float safe_softness = fmaxf(softness, 1.0f);
    float effective_kp = fmaxf(kp, 0.0f);

    // Fine-tracking boost: when close to target, increase kp to improve
    // steady-state tracking accuracy. The boost tapers linearly to 1.0
    // at the range boundary (softness * fine_range) so the gain curve
    // stays continuous — no step that could cause limit-cycle oscillation.
    if (fine_boost > 1.0f && fine_range > 0.0f) {
        const float threshold = safe_softness * fine_range;
        if (abs_error < threshold) {
            const float t = abs_error / threshold;  // 0 at center, 1 at boundary
            effective_kp *= fine_boost * (1.0f - t) + t;  // lerp(fine_boost, 1.0, t)
        }
    }

    const float gain = effective_kp * (abs_error / (abs_error + safe_softness));
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
__device__ __forceinline__ float leadGate(const AimState* s,
                                          float error_x, float error_y) {
    float g = 1.0f;
    {
        const float sp = sqrtf(s->vel_x * s->vel_x + s->vel_y * s->vel_y);
        g *= sp / (sp + kLeadVgate);
    }
    {
        const float e0 = kLeadErrGate;
        const float e2 = error_x * error_x + error_y * error_y;
        g *= (e0 * e0) / (e0 * e0 + e2);
    }
    return g;
}

// Branchless median of three floats (uses fminf/fmaxf only).
__device__ __forceinline__ float median3(float a, float b, float c) {
    return fmaxf(fminf(a, b), fminf(fmaxf(a, b), c));
}

// Apply 3-frame median to raw center. Returns the filtered value and
// shifts the history. On the first two frames (med_initialized < 2),
// passes through unchanged to avoid startup artifacts.
__device__ __forceinline__ void applyMedianPrefilter(
    float raw_x, float raw_y,
    AimState* s,
    float& out_x, float& out_y)
{
    if (s->med_initialized < 2) {
        s->med_raw_x[s->med_initialized] = raw_x;
        s->med_raw_y[s->med_initialized] = raw_y;
        s->med_initialized++;
        out_x = raw_x;
        out_y = raw_y;
        return;
    }
    out_x = median3(s->med_raw_x[0], s->med_raw_x[1], raw_x);
    out_y = median3(s->med_raw_y[0], s->med_raw_y[1], raw_y);
    // Shift history
    s->med_raw_x[0] = s->med_raw_x[1];
    s->med_raw_x[1] = raw_x;
    s->med_raw_y[0] = s->med_raw_y[1];
    s->med_raw_y[1] = raw_y;
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

// CUDA smoothstep: 0 at edge0, 1 at edge1, smooth hermite interpolation between.
// Used by CGA for the confidence gate curve.
__device__ __forceinline__ float gpaSmoothstep(float edge0, float edge1, float x) {
    const float t = fminf(fmaxf((x - edge0) / (edge1 - edge0 + 1e-7f), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
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
    float detection_confidence,
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
    // 0.03-0.22% WORSE, on both gain profiles. Do not "fix" it.
    const bool class_changed =
        (!fresh_track &&
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
    // The in-flight sum is computed here and subtracted from the error further
    // down.
    float inf_x = 0.0f, inf_y = 0.0f;
    const bool use_comp = (aim_config.deadtime_frames > 0.0f);
    if (use_comp) {
        inflightSum(aim_state, aim_config.deadtime_frames, inf_x, inf_y);
    }

    float target_center_x = raw_center_x;
    float target_center_y = raw_center_y;
    if (kOneEuroEnabled != 0.0f) {
        const float in_x = raw_center_x, in_y = raw_center_y;
        const float carry_x = aim_state->filt_x, carry_y = aim_state->filt_y;
        if (aim_state->has_track) {
            const float ad = kDerivAlpha;
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

    // --- JCAD: capture inter-frame displacement BEFORE the velocity estimator
    // updates prev_raw. The displacement feeds the online jitter classifier
    // that decides the adaptive deadband width further down.
    if (aim_config.jcad_enabled != 0.0f) {
        if (fresh_track) {
            aim_state->jcad_ew_mean = 0.0f;
            aim_state->jcad_ew_var = 0.0f;
        } else {
            const float jdx = raw_center_x - aim_state->prev_raw_x;
            const float jdy = raw_center_y - aim_state->prev_raw_y;
            const float jcad_disp = sqrtf(jdx * jdx + jdy * jdy);
            const float ja = kJcadAlpha;
            aim_state->jcad_ew_mean = ja * jcad_disp
                                    + (1.0f - ja) * aim_state->jcad_ew_mean;
            const float jdiff = jcad_disp - aim_state->jcad_ew_mean;
            aim_state->jcad_ew_var = ja * jdiff * jdiff
                                   + (1.0f - ja) * aim_state->jcad_ew_var;
        }
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
        // Median-filter the raw center for velocity estimation only. The position
        // path keeps the unfiltered center so there is no aim-point delay; the
        // velocity path gets outlier rejection so a one-frame detector glitch
        // does not spike ff_gain * v and predict_frames * v.
        float med_x, med_y;
        applyMedianPrefilter(raw_center_x, raw_center_y, aim_state, med_x, med_y);

        if (aim_state->has_track) {
            float ax = 0.0f, ay = 0.0f;
            inflightAt(aim_state, kFfEgoLag, ax, ay);
            const float maxDrift = 60.0f;   // model px/frame sanity clamp
            float dvx = (med_x - aim_state->prev_raw_x)
                      + ((movement_scale_x != 0.0f) ? ax / movement_scale_x : 0.0f);
            float dvy = (med_y - aim_state->prev_raw_y)
                      + ((movement_scale_y != 0.0f) ? ay / movement_scale_y : 0.0f);
            dvx = fminf(fmaxf(dvx, -maxDrift), maxDrift);
            dvy = fminf(fmaxf(dvy, -maxDrift), maxDrift);
            const float a = kFfVEma;
            aim_state->vel_x = (1.0f - a) * aim_state->vel_x + a * dvx;
            aim_state->vel_y = (1.0f - a) * aim_state->vel_y + a * dvy;
        } else {
            aim_state->vel_x = 0.0f;
            aim_state->vel_y = 0.0f;
            aim_state->med_initialized = 0;
            aim_state->accel_x = 0.0f;
            aim_state->accel_y = 0.0f;
            aim_state->prev_vel_x = 0.0f;
            aim_state->prev_vel_y = 0.0f;
        }

        // Acceleration estimation: EMA of velocity change per frame.
        {
            const float a_ema = kAccelEma;
            const float dax = aim_state->vel_x - aim_state->prev_vel_x;
            const float day = aim_state->vel_y - aim_state->prev_vel_y;
            const float maxAccel = 15.0f;
            aim_state->accel_x = (1.0f - a_ema) * aim_state->accel_x
                + a_ema * fminf(fmaxf(dax, -maxAccel), maxAccel);
            aim_state->accel_y = (1.0f - a_ema) * aim_state->accel_y
                + a_ema * fminf(fmaxf(day, -maxAccel), maxAccel);
        }
        aim_state->prev_vel_x = aim_state->vel_x;
        aim_state->prev_vel_y = aim_state->vel_y;

        aim_state->prev_raw_x = med_x;
        aim_state->prev_raw_y = med_y;
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
    if (use_comp) {
        if (movement_scale_x != 0.0f) error_x -= inf_x / movement_scale_x;
        if (movement_scale_y != 0.0f) error_y -= inf_y / movement_scale_y;
    }

    // --- JCAD: adaptive deadband. J = var / (mean^2 + eps): high when variance
    // dominates mean (jitter on a stationary target), low when mean dominates
    // (real movement). The deadband widens smoothly from base_db to
    // base_db + jitter_db as J rises. Error inside the deadband is zeroed so
    // the controller does not chase detector noise.
    if (aim_config.jcad_enabled != 0.0f) {
        const float mean_sq = aim_state->jcad_ew_mean * aim_state->jcad_ew_mean;
        const float J = aim_state->jcad_ew_var / (mean_sq + 1e-6f);
        const float sat_J = fminf(J / kJcadJitterThreshold, 1.0f);
        const float db = kJcadBaseDb + kJcadJitterDb * sat_J;
        const float err_mag = sqrtf(error_x * error_x + error_y * error_y);
        if (err_mag < db) {
            error_x = 0.0f;
            error_y = 0.0f;
        }
    }

    // Symmetric dead-time compensation: the in-flight subtraction above removed
    // OUR motion during the dead time; this extrapolates the TARGET forward over
    // the same dead time, which nothing did before. Shifting the SETPOINT (not
    // the output) means it passes through the nonlinear P and the max-step clamp.
    // Gated exactly like the lead term: ~0 at rest (noise) and ~0 while acquiring
    // (a position jump's one-frame velocity must never be extrapolated).
    if (aim_config.predict_frames > 0.0f) {
        const float g = leadGate(aim_state, error_x, error_y);
        const float t = aim_config.predict_frames;
        error_x += g * (t * aim_state->vel_x
                         + 0.5f * t * t * aim_state->accel_x);
        error_y += g * (t * aim_state->vel_y
                         + 0.5f * t * t * aim_state->accel_y);
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
        const float g = leadGate(aim_state, error_x, error_y);
        ff_x = aim_config.ff_gain * g * aim_state->vel_x;
        ff_y = aim_config.ff_gain * g * aim_state->vel_y;
    }

    float movement_x =
        (nonlinearPMove(error_x, aim_config.kp_x, aim_config.p_softness_x,
                        kFineTrackBoost, kFineTrackRange)
         + aim_config.kd_x * aim_state->derr_x + ff_x) * movement_scale_x;
    float movement_y =
        (nonlinearPMove(error_y, aim_config.kp_y, aim_config.p_softness_y,
                        kFineTrackBoost, kFineTrackRange)
         + aim_config.kd_y * aim_state->derr_y + ff_y) * movement_scale_y;

    // --- SEM: Signed-Error Momentum. Accumulates momentum while the error sign
    // is consistent on each axis; instant kill on sign flip (overshoot detected).
    // Unlike an integral term: no windup, no sign-agnostic accumulation, no
    // anti-windup needed. The SEM term naturally accelerates approach on sustained
    // error and instantly brakes on overshoot.
    if (aim_config.sem_enabled != 0.0f) {
        // Reset on fresh acquire so a previous target's momentum cannot persist
        if (fresh_track) {
            aim_state->sem_x = 0.0f;
            aim_state->sem_y = 0.0f;
        }
        // Decay existing momentum
        aim_state->sem_x *= kSemDecay;
        aim_state->sem_y *= kSemDecay;
        // X axis: accumulate or kill
        if (error_x * aim_state->sem_x < 0.0f) {
            aim_state->sem_x = 0.0f;  // sign flip: overshoot detected
        }
        aim_state->sem_x += kSemRate * error_x;
        aim_state->sem_x = fminf(fmaxf(aim_state->sem_x, -kSemMax), kSemMax);
        // Y axis: accumulate or kill
        if (error_y * aim_state->sem_y < 0.0f) {
            aim_state->sem_y = 0.0f;
        }
        aim_state->sem_y += kSemRate * error_y;
        aim_state->sem_y = fminf(fmaxf(aim_state->sem_y, -kSemMax), kSemMax);
        // Add momentum to output movement
        movement_x += aim_state->sem_x * movement_scale_x;
        movement_y += aim_state->sem_y * movement_scale_y;
    }

    // --- PAGS: Phase-Aware Gain Scheduling. Detects which phase the controller
    // is in by tracking the HISTORY of error sign changes (not just current error
    // magnitude). Three phases: ACQUIRE (large error, sign consistent) -> slightly
    // boosted gains for fast reach. TRACK (small error, stable) -> baseline.
    // OSCILLATE (sign alternating rapidly) -> reduced gains to damp ringing.
    // This is distinct from nonlinear P (error-proportional gain): PAGS detects
    // the temporal PATTERN of sign reversals that characterize oscillation.
    if (aim_config.pags_enabled != 0.0f) {
        if (fresh_track) {
            aim_state->pags_flip_rate_x = 0.0f;
            aim_state->pags_flip_rate_y = 0.0f;
            aim_state->pags_prev_sign_x = 0;
            aim_state->pags_prev_sign_y = 0;
            aim_state->pags_gain_mult = 1.0f;
        }
        // Track sign-flip frequency per axis
        const int sign_x = (error_x > 0.0f) ? 1 : ((error_x < 0.0f) ? -1 : 0);
        const int sign_y = (error_y > 0.0f) ? 1 : ((error_y < 0.0f) ? -1 : 0);
        const float flip_x = (aim_state->pags_prev_sign_x != 0 && sign_x != 0 &&
                              sign_x != aim_state->pags_prev_sign_x) ? 1.0f : 0.0f;
        const float flip_y = (aim_state->pags_prev_sign_y != 0 && sign_y != 0 &&
                              sign_y != aim_state->pags_prev_sign_y) ? 1.0f : 0.0f;
        aim_state->pags_flip_rate_x = kPagsAlpha * flip_x
                                    + (1.0f - kPagsAlpha) * aim_state->pags_flip_rate_x;
        aim_state->pags_flip_rate_y = kPagsAlpha * flip_y
                                    + (1.0f - kPagsAlpha) * aim_state->pags_flip_rate_y;
        if (sign_x != 0) aim_state->pags_prev_sign_x = sign_x;
        if (sign_y != 0) aim_state->pags_prev_sign_y = sign_y;
        // Phase detection: use max flip rate across both axes
        const float err_mag = sqrtf(error_x * error_x + error_y * error_y);
        const float flip_rate = fmaxf(aim_state->pags_flip_rate_x,
                                      aim_state->pags_flip_rate_y);
        float target_mult;
        if (err_mag > kPagsAcquireThreshold) {
            target_mult = kPagsAcquireMult;   // ACQUIRE
        } else if (flip_rate > kPagsOscThreshold) {
            target_mult = kPagsOscMult;       // OSCILLATE
        } else {
            target_mult = 1.0f;                           // TRACK
        }
        // Smooth transition to avoid gain discontinuities
        aim_state->pags_gain_mult = 0.3f * target_mult
                                  + 0.7f * aim_state->pags_gain_mult;
        movement_x *= aim_state->pags_gain_mult;
        movement_y *= aim_state->pags_gain_mult;
    }

    // Per-frame max-step clamp (output px). Bounds the slew so a large initial
    // error is crossed in several smooth steps instead of one delayed leap
    // that overshoots and rings - overshoot D cannot prevent reactively.
    clampMaxStep(movement_x, movement_y, aim_config.max_step);

    // Vertical strength trim, applied AFTER the clamp. Scales the whole Y output
    // (P + D + lead) so the assist can be weakened on that axis alone without
    // touching the tuned gains - 1.0 is the controller exactly as tuned, 0 is
    // horizontal-only.
    // The order matters and is not the obvious one. clampMaxStep scales X and Y
    // TOGETHER to preserve direction, so shrinking Y first shrinks the vector and
    // lets the clamp bind less - which makes X move FURTHER than it would have.
    // With movement (20, 20) and max_step 19.56: clamped first gives x 13.8;
    // scaling Y first gives x 19.5, a 41% horizontal change from a "vertical"
    // knob. Clamping first leaves X bit-identical to aim_y_scale = 1.
    // Still before the emit, so pushInflight records the Y that actually went out
    // and the dead-time compensation stays consistent.
    movement_y *= aim_config.aim_y_scale;

    // Re-clamp: aim_y_scale above 1 would otherwise push the emit past max_step,
    // and a per-frame vertical spike is exactly what that bound exists to stop.
    // Below 1 this call cannot fire - the first clamp left |v| <= max_step and
    // shrinking Y cannot grow it (0 hits in 80k random vectors; above 1 it fires
    // on 98%) - so the attenuating case, which is what the rig runs, gets no new
    // X coupling from it. X still differs from aim_y_scale = 1 there, but through
    // the loop, not through this clamp: a weaker Y leaves more Y error standing,
    // so movement_y is larger and the FIRST clamp binds more often. That is the
    // controller reacting correctly. Measured closed-loop on an X-only target,
    // horizontal error moves 0.05% across y_scale 1.0 -> 0.
    clampMaxStep(movement_x, movement_y, aim_config.max_step);

    // --- CGA: Confidence-Gated Authority. Modulates the whole output by
    // detection confidence, with a speed bypass so MOVING targets get full
    // authority regardless of confidence (motion blur drops conf to ~0.4-0.5,
    // overlapping the ~0.38 outlier level - see the removed conf-weight note
    // above). At rest (speed near 0), low confidence suppresses the output
    // through a smoothstep gate; at speed the bypass saturates to ~1.
    if (aim_config.cga_enabled != 0.0f) {
        const float speed = sqrtf(aim_state->vel_x * aim_state->vel_x +
                                  aim_state->vel_y * aim_state->vel_y);
        const float speed_bypass = speed / (speed + kCgaSpeedThreshold + 1e-7f);
        const float conf_gate = gpaSmoothstep(kCgaConfLow, kCgaConfHigh,
                                              detection_confidence);
        const float gate = fmaxf(conf_gate, speed_bypass);
        movement_x *= gate;
        movement_y *= gate;
    }

    // Output gate. The ring must record what the mouse ACTUALLY receives, so when
    // the host has told us this frame's move will not be sent (aim key up while
    // inference keeps running warm), emit nothing and push a zero. Everything
    // above still ran, so the filters, the velocity estimate and the track stay
    // warm for the next real aim - only the physical output and the record of it
    // are suppressed. The sub-pixel carry is dropped too: a residual accumulated
    // over an idle stretch would otherwise discharge as a jump on the first
    // emitted frame.
    if (aim_config.aim_output_enabled == 0.0f) {
        aim_state->residual_x = 0.0f;
        aim_state->residual_y = 0.0f;
        out_dx = 0;
        out_dy = 0;
        pushInflight(aim_state, 0.0f, 0.0f);
        return;
    }

    out_dx = emitMouseDelta(movement_x, &aim_state->residual_x);
    out_dy = emitMouseDelta(movement_y, &aim_state->residual_y);
    pushInflight(aim_state, static_cast<float>(out_dx), static_cast<float>(out_dy));
}

}  // namespace gpa
