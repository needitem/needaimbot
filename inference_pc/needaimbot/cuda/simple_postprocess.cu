// GPU-based postprocessing for simple inference
// Full optimizations from unified_graph_pipeline:
// - Warp-level primitives for fast reduction
// - Head-in-body priority selection
// - IoU-based target stickiness (hysteresis)
// - Fused target selection + nonlinear P movement calculation
// - Strict garbage value filtering
#include "simple_postprocess.h"
#include "simple_inference.h"
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <cmath>

namespace gpa {

// =============================================================================
// Constants
// =============================================================================
constexpr int MAX_SELECTION_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_SELECTION_WARPS = MAX_SELECTION_THREADS / WARP_SIZE;
static_assert((MAX_SELECTION_THREADS % WARP_SIZE) == 0,
              "MAX_SELECTION_THREADS must be multiple of warp size");

// =============================================================================
// Helper Functions
// =============================================================================

template<bool kIsFp16>
__device__ __forceinline__ float readValue(const void* buffer, size_t idx);

template<>
__device__ __forceinline__ float readValue<true>(const void* buffer, size_t idx) {
    return __half2float(reinterpret_cast<const __half*>(buffer)[idx]);
}

template<>
__device__ __forceinline__ float readValue<false>(const void* buffer, size_t idx) {
    return reinterpret_cast<const float*>(buffer)[idx];
}

__device__ __forceinline__ Detection shuffleDownDetection(
    const Detection& det, unsigned mask, int offset) {
    Detection out;
    out.x1 = __shfl_down_sync(mask, det.x1, offset);
    out.y1 = __shfl_down_sync(mask, det.y1, offset);
    out.x2 = __shfl_down_sync(mask, det.x2, offset);
    out.y2 = __shfl_down_sync(mask, det.y2, offset);
    out.confidence = __shfl_down_sync(mask, det.confidence, offset);
    out.classId = __shfl_down_sync(mask, det.classId, offset);
    return out;
}

__device__ __forceinline__ void warpReduceDistMin(
    Detection& det, float& score, int& valid) {
    constexpr unsigned kFullMask = 0xFFFFFFFFu;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        const float otherScore = __shfl_down_sync(kFullMask, score, offset);
        const int otherValid = __shfl_down_sync(kFullMask, valid, offset);
        const Detection otherDet = shuffleDownDetection(det, kFullMask, offset);
        if (otherValid && (!valid || otherScore < score)) {
            det = otherDet;
            score = otherScore;
            valid = 1;
        }
    }
}

__device__ __forceinline__ void warpReduceIouMax(
    Detection& det, float& score, int& valid) {
    constexpr unsigned kFullMask = 0xFFFFFFFFu;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        const float otherScore = __shfl_down_sync(kFullMask, score, offset);
        const int otherValid = __shfl_down_sync(kFullMask, valid, offset);
        const Detection otherDet = shuffleDownDetection(det, kFullMask, offset);
        if (otherValid && (!valid || otherScore > score)) {
            det = otherDet;
            score = otherScore;
            valid = 1;
        }
    }
}

// =============================================================================
// IoU Calculation (from unified_graph_pipeline.cu)
// =============================================================================

// Compute Intersection over Union for two bounding boxes
__device__ __forceinline__ float computeBoundingBoxIoU(const Detection& a, const Detection& b) {
    if (a.classId < 0 || b.classId < 0) {
        return 0.0f;
    }

    float a_w = a.x2 - a.x1;
    float a_h = a.y2 - a.y1;
    float b_w = b.x2 - b.x1;
    float b_h = b.y2 - b.y1;

    if (a_w <= 0 || a_h <= 0 || b_w <= 0 || b_h <= 0) {
        return 0.0f;
    }

    // Intersection coordinates
    float inter_x1 = fmaxf(a.x1, b.x1);
    float inter_y1 = fmaxf(a.y1, b.y1);
    float inter_x2 = fminf(a.x2, b.x2);
    float inter_y2 = fminf(a.y2, b.y2);

    float inter_w = inter_x2 - inter_x1;
    float inter_h = inter_y2 - inter_y1;

    if (inter_w <= 0 || inter_h <= 0) {
        return 0.0f;
    }

    float inter_area = inter_w * inter_h;
    float area_a = a_w * a_h;
    float area_b = b_w * b_h;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0) {
        return 0.0f;
    }

    return inter_area / union_area;
}

// =============================================================================
// One-pass Fused Decode + Target Selection + Movement
// =============================================================================

__device__ __forceinline__ void writeEmptyInferenceResult(InferenceResult* result) {
    result->movement.dx = 0;
    result->movement.dy = 0;
    result->hasTarget = 0;
    result->reserved = 0;
    result->targetX1 = 0;
    result->targetY1 = 0;
    result->targetX2 = 0;
    result->targetY2 = 0;
    result->targetConf = 0;
    result->targetClassId = -1;
}

// Combined stickiness score for tracking the same target across frames.
// Returns max of (IoU, distance-based score) when both prev and current
// detections share the same class. Distance score falls off linearly to 0
// at (prev_diag * distance_factor) center separation.
__device__ __forceinline__ float computeStickinessScore(
    const Detection& det, const Detection& prev, float distance_factor) {
    if (prev.classId < 0 || det.classId < 0) return 0.0f;
    const float iou = computeBoundingBoxIoU(det, prev);
    if (distance_factor <= 0.0f || det.classId != prev.classId) {
        return iou;
    }
    const float prev_w = prev.x2 - prev.x1;
    const float prev_h = prev.y2 - prev.y1;
    if (prev_w <= 0.0f || prev_h <= 0.0f) return iou;
    // Squared comparisons keep the common "outside window" path sqrt-free.
    // sqrt is only paid once when the candidate actually scores.
    const float prev_diag_sq = prev_w * prev_w + prev_h * prev_h;
    const float window_sq = fmaxf(prev_diag_sq * distance_factor * distance_factor, 1.0f);
    const float det_cx = (det.x1 + det.x2) * 0.5f;
    const float det_cy = (det.y1 + det.y2) * 0.5f;
    const float prev_cx = (prev.x1 + prev.x2) * 0.5f;
    const float prev_cy = (prev.y1 + prev.y2) * 0.5f;
    const float dx = det_cx - prev_cx;
    const float dy = det_cy - prev_cy;
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq >= window_sq) {
        return iou;  // outside distance window - no positive distance score
    }
    const float dist_score = 1.0f - sqrtf(dist_sq / window_sq);
    return fmaxf(iou, dist_score);
}

// One Euro low-pass smoothing factor for a given cutoff frequency.
// Sample period Te is normalized to 1 (per-frame), so cutoff is in cycles/frame.
// alpha = 1 / (1 + tau/Te), tau = 1 / (2*pi*cutoff).
// One Euro smoothing factor for a cutoff (cycles per nominal frame) over an
// actual sample period te (in nominal frames). te = 1 reproduces the original
// per-frame form exactly; te != 1 keeps the time constant consistent when the
// frame interval drifts.
__device__ __forceinline__ float oneEuroAlpha(float cutoff, float te) {
    const float tau = 1.0f / (2.0f * 3.14159265f * fmaxf(cutoff, 1e-4f));
    return 1.0f / (1.0f + tau / fmaxf(te, 1e-4f));
}

// Re-express a fixed per-frame EMA weight (tuned at te = 1) for an actual sample
// period te, so the smoother's time constant is invariant to frame rate. Returns
// alpha for te = 1 unchanged; increases weight on the new sample for longer te.
__device__ __forceinline__ float emaAlphaDt(float alpha_per_frame, float te) {
    const float keep = 1.0f - fminf(fmaxf(alpha_per_frame, 0.0f), 1.0f);
    return 1.0f - powf(keep, fmaxf(te, 1e-4f));
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

template<bool kIsFp16>
__device__ __forceinline__ bool decodeDetectionIfValid(
    const void* d_raw_output,
    int box_idx,
    int num_boxes,
    int num_classes,
    float conf_threshold,
    uint32_t allowedClassMask,
    float max_box_extent,
    Detection& out_det)
{
    const int classLimit = (num_classes < 32) ? num_classes : 32;
    if (classLimit <= 0) return false;

    uint32_t classMask = allowedClassMask;
    if (classLimit < 32) {
        classMask &= ((1u << classLimit) - 1u);
    }
    if (classMask == 0u) return false;

    float bestScore = -1.0f;
    int bestClass = -1;
    if ((classMask & (classMask - 1u)) == 0u) {
        // Common fast path: only one allowed class.
        bestClass = __ffs(classMask) - 1;
        const int score_idx = (4 + bestClass) * num_boxes + box_idx;
        bestScore = readValue<kIsFp16>(d_raw_output, static_cast<size_t>(score_idx));
    } else {
        while (classMask) {
            const int c = __ffs(classMask) - 1;
            classMask &= (classMask - 1u);
            const int score_idx = (4 + c) * num_boxes + box_idx;
            const float score = readValue<kIsFp16>(d_raw_output, static_cast<size_t>(score_idx));
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
    }

    if (!(bestScore > conf_threshold) || bestClass < 0) {
        return false;
    }

    const float cx = readValue<kIsFp16>(d_raw_output, static_cast<size_t>(box_idx));
    const float cy = readValue<kIsFp16>(d_raw_output, static_cast<size_t>(num_boxes + box_idx));
    const float w  = readValue<kIsFp16>(d_raw_output, static_cast<size_t>(2 * num_boxes + box_idx));
    const float h  = readValue<kIsFp16>(d_raw_output, static_cast<size_t>(3 * num_boxes + box_idx));

    constexpr float kMaxCoord = 10000.0f;
    if (!isfinite(cx) || !isfinite(cy) || !isfinite(w) || !isfinite(h)) return false;
    if (!(cx >= 0.0f && cx <= kMaxCoord &&
          cy >= 0.0f && cy <= kMaxCoord &&
          w > 0.0f && w <= kMaxCoord &&
          h > 0.0f && h <= kMaxCoord)) {
        return false;
    }

    const float x1 = cx - w * 0.5f;
    const float y1 = cy - h * 0.5f;
    const float x2 = cx + w * 0.5f;
    const float y2 = cy + h * 0.5f;
    if (!((x1 >= -1000.0f && x1 <= kMaxCoord) &&
          (y1 >= -1000.0f && y1 <= kMaxCoord) &&
          (x2 >= -1000.0f && x2 <= kMaxCoord) &&
          (y2 >= -1000.0f && y2 <= kMaxCoord) &&
          (w <= max_box_extent && h <= max_box_extent))) {
        return false;
    }

    out_det.x1 = x1;
    out_det.y1 = y1;
    out_det.x2 = x2;
    out_det.y2 = y2;
    out_det.confidence = bestScore;
    out_det.classId = bestClass;
    return true;
}

template<bool kIsFp16>
__global__ void stage1DecodeAndSelectKernel(
    const void* __restrict__ d_raw_output,
    int num_boxes,
    int num_classes,
    float conf_threshold,
    uint32_t allowedClassMask,
    float max_box_extent,
    float screen_center_x,
    float screen_center_y,
    int head_class_id,
    float head_y_offset,
    float body_y_offset,
    const Detection* __restrict__ d_selected_target,
    const AimConfig* __restrict__ d_aim_config,
    Detection* __restrict__ d_stage1_best_dist,
    float* __restrict__ d_stage1_dist_score,
    Detection* __restrict__ d_stage1_best_iou,
    float* __restrict__ d_stage1_iou_score)
{
    const float distance_stickiness_factor =
        d_aim_config ? d_aim_config->distance_stickiness_factor : 0.0f;
    __shared__ Detection s_prevTarget;
    __shared__ bool s_prevValid;
    __shared__ Detection s_warpBestDistDet[MAX_SELECTION_WARPS];
    __shared__ Detection s_warpBestIouDet[MAX_SELECTION_WARPS];
    __shared__ float s_warpBestDist[MAX_SELECTION_WARPS];
    __shared__ float s_warpBestIou[MAX_SELECTION_WARPS];
    __shared__ uint8_t s_warpHasDist[MAX_SELECTION_WARPS];
    __shared__ uint8_t s_warpHasIou[MAX_SELECTION_WARPS];

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int warpCount = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if (tid == 0) {
        Detection emptyTarget = {};
        emptyTarget.classId = -1;
        s_prevTarget = emptyTarget;
        s_prevValid = false;
        if (d_selected_target) {
            const Detection cached = *d_selected_target;
            const float w = cached.x2 - cached.x1;
            const float h = cached.y2 - cached.y1;
            if (cached.classId >= 0 && cached.confidence > 0.0f && w > 0.0f && h > 0.0f) {
                s_prevTarget = cached;
                s_prevValid = true;
            }
        }
    }
    __syncthreads();

    Detection localBestByDist = {};
    localBestByDist.classId = -1;
    float localBestDist = FLT_MAX;
    bool localHasDist = false;

    Detection localBestByIou = {};
    localBestByIou.classId = -1;
    float localBestIou = -1.0f;
    bool localHasIou = false;

    const bool prevValid = s_prevValid;
    const Detection prevTarget = s_prevTarget;

    const int globalStart = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    for (int i = globalStart; i < num_boxes; i += stride) {
        Detection det = {};
        if (!decodeDetectionIfValid<kIsFp16>(
                d_raw_output, i, num_boxes, num_classes, conf_threshold, allowedClassMask,
                max_box_extent, det)) {
            continue;
        }

        const float w = det.x2 - det.x1;
        const float h = det.y2 - det.y1;
        const float centerX = det.x1 + w * 0.5f;
        const float aimY = (det.classId == head_class_id)
            ? (det.y1 + h * head_y_offset)
            : (det.y1 + h * body_y_offset);
        const float dx = centerX - screen_center_x;
        const float dy = aimY - screen_center_y;
        const float effectiveDist = dx * dx + dy * dy;
        if (effectiveDist < localBestDist) {
            localBestDist = effectiveDist;
            localBestByDist = det;
            localHasDist = true;
        }

        if (prevValid) {
            const float stickyScore = computeStickinessScore(
                det, prevTarget, distance_stickiness_factor);
            if (stickyScore > localBestIou) {
                localBestIou = stickyScore;
                localBestByIou = det;
                localHasIou = true;
            }
        }
    }

    int hasDist = localHasDist ? 1 : 0;
    int hasIou = localHasIou ? 1 : 0;
    warpReduceDistMin(localBestByDist, localBestDist, hasDist);
    warpReduceIouMax(localBestByIou, localBestIou, hasIou);

    if (lane == 0) {
        s_warpBestDistDet[warp] = localBestByDist;
        s_warpBestIouDet[warp] = localBestByIou;
        s_warpBestDist[warp] = localBestDist;
        s_warpBestIou[warp] = localBestIou;
        s_warpHasDist[warp] = hasDist ? 1u : 0u;
        s_warpHasIou[warp] = hasIou ? 1u : 0u;
    }
    __syncthreads();

    if (warp != 0) {
        return;
    }

    Detection bestByDist = {};
    bestByDist.classId = -1;
    float bestDist = FLT_MAX;
    int hasBestByDist = 0;
    if (lane < warpCount) {
        hasBestByDist = (s_warpHasDist[lane] != 0u) ? 1 : 0;
        if (hasBestByDist) {
            bestByDist = s_warpBestDistDet[lane];
            bestDist = s_warpBestDist[lane];
        }
    }
    warpReduceDistMin(bestByDist, bestDist, hasBestByDist);

    Detection bestByIou = {};
    bestByIou.classId = -1;
    float bestIou = -1.0f;
    int hasBestByIou = 0;
    if (lane < warpCount) {
        hasBestByIou = (s_warpHasIou[lane] != 0u) ? 1 : 0;
        if (hasBestByIou) {
            bestByIou = s_warpBestIouDet[lane];
            bestIou = s_warpBestIou[lane];
        }
    }
    warpReduceIouMax(bestByIou, bestIou, hasBestByIou);

    if (lane != 0) {
        return;
    }

    const bool hasDistResult = (hasBestByDist != 0);
    const bool hasIouResult = (hasBestByIou != 0);
    if (!hasDistResult) {
        bestByDist.classId = -1;
        bestDist = FLT_MAX;
    }
    if (!hasIouResult) {
        bestByIou.classId = -1;
        bestIou = -1.0f;
    }

    const int outIdx = blockIdx.x;
    if (hasDistResult) {
        d_stage1_best_dist[outIdx] = bestByDist;
        d_stage1_dist_score[outIdx] = bestDist;
    } else {
        Detection invalid = {};
        invalid.classId = -1;
        d_stage1_best_dist[outIdx] = invalid;
        d_stage1_dist_score[outIdx] = FLT_MAX;
    }

    if (hasIouResult) {
        d_stage1_best_iou[outIdx] = bestByIou;
        d_stage1_iou_score[outIdx] = bestIou;
    } else {
        Detection invalid = {};
        invalid.classId = -1;
        d_stage1_best_iou[outIdx] = invalid;
        d_stage1_iou_score[outIdx] = -1.0f;
    }
}

__global__ void stage2FinalizeKernel(
    int num_candidates,
    const Detection* __restrict__ d_stage1_best_dist,
    const float* __restrict__ d_stage1_dist_score,
    const Detection* __restrict__ d_stage1_best_iou,
    const float* __restrict__ d_stage1_iou_score,
    float screen_center_x,
    float screen_center_y,
    float movement_scale_x,
    float movement_scale_y,
    int head_class_id,
    const AimConfig* __restrict__ d_aim_config,
    float iou_stickiness_threshold,
    float head_y_offset,
    float body_y_offset,
    Detection* __restrict__ d_selected_target,
    AimState* __restrict__ d_aim_state,
    const FrameTiming* __restrict__ d_frame_timing,
    InferenceResult* __restrict__ d_inference_result)
{
    __shared__ Detection s_warpBestDistDet[MAX_SELECTION_WARPS];
    __shared__ Detection s_warpBestIouDet[MAX_SELECTION_WARPS];
    __shared__ float s_warpBestDist[MAX_SELECTION_WARPS];
    __shared__ float s_warpBestIou[MAX_SELECTION_WARPS];
    __shared__ uint8_t s_warpHasDist[MAX_SELECTION_WARPS];
    __shared__ uint8_t s_warpHasIou[MAX_SELECTION_WARPS];

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int warpCount = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    Detection localBestByDist = {};
    localBestByDist.classId = -1;
    float localBestDist = FLT_MAX;
    bool localHasDist = false;

    Detection localBestByIou = {};
    localBestByIou.classId = -1;
    float localBestIou = -1.0f;
    bool localHasIou = false;

    for (int i = tid; i < num_candidates; i += blockDim.x) {
        Detection candDist = d_stage1_best_dist[i];
        float distScore = d_stage1_dist_score[i];
        if (candDist.classId >= 0 && distScore < localBestDist) {
            localBestDist = distScore;
            localBestByDist = candDist;
            localHasDist = true;
        }

        Detection candIou = d_stage1_best_iou[i];
        float iouScore = d_stage1_iou_score[i];
        if (candIou.classId >= 0 && iouScore > localBestIou) {
            localBestIou = iouScore;
            localBestByIou = candIou;
            localHasIou = true;
        }
    }

    int hasDist = localHasDist ? 1 : 0;
    int hasIou = localHasIou ? 1 : 0;
    warpReduceDistMin(localBestByDist, localBestDist, hasDist);
    warpReduceIouMax(localBestByIou, localBestIou, hasIou);

    if (lane == 0) {
        s_warpBestDistDet[warp] = localBestByDist;
        s_warpBestIouDet[warp] = localBestByIou;
        s_warpBestDist[warp] = localBestDist;
        s_warpBestIou[warp] = localBestIou;
        s_warpHasDist[warp] = hasDist ? 1u : 0u;
        s_warpHasIou[warp] = hasIou ? 1u : 0u;
    }
    __syncthreads();

    if (warp != 0) {
        return;
    }

    Detection bestByDist = {};
    bestByDist.classId = -1;
    float bestDist = FLT_MAX;
    int hasBestByDist = 0;
    if (lane < warpCount) {
        hasBestByDist = (s_warpHasDist[lane] != 0u) ? 1 : 0;
        if (hasBestByDist) {
            bestByDist = s_warpBestDistDet[lane];
            bestDist = s_warpBestDist[lane];
        }
    }
    warpReduceDistMin(bestByDist, bestDist, hasBestByDist);

    Detection bestByIou = {};
    bestByIou.classId = -1;
    float bestIou = -1.0f;
    int hasBestByIou = 0;
    if (lane < warpCount) {
        hasBestByIou = (s_warpHasIou[lane] != 0u) ? 1 : 0;
        if (hasBestByIou) {
            bestByIou = s_warpBestIouDet[lane];
            bestIou = s_warpBestIou[lane];
        }
    }
    warpReduceIouMax(bestByIou, bestIou, hasBestByIou);

    if (lane != 0) {
        return;
    }

    const bool hasDistResult = (hasBestByDist != 0);
    const bool hasIouResult = (hasBestByIou != 0);
    (void)bestDist;
    if (!hasDistResult) {
        bestByDist.classId = -1;
    }
    if (!hasIouResult) {
        bestByIou.classId = -1;
        bestIou = -1.0f;
    }

    // Snapshot prev tracked target BEFORE we may overwrite it. The persistence
    // path keeps prev alive across detection gaps so stickiness can re-acquire.
    Detection prevTarget = {};
    prevTarget.classId = -1;
    if (d_selected_target) {
        prevTarget = *d_selected_target;
    }
    const int prevFramesSinceSeen = d_aim_state->frames_since_seen;
    const int trackPersistenceFrames = (d_aim_config)
        ? d_aim_config->track_persistence_frames : 0;

    // Per-frame timing. Defaults reproduce the original per-frame math:
    // dt_norm = 1 (smoothers stay identical) and lead_frames = 0 (no prediction).
    const float dt_norm = (d_frame_timing) ? fmaxf(d_frame_timing->dt_norm, 1e-3f) : 1.0f;
    const float lead_frames = (d_frame_timing) ? fmaxf(d_frame_timing->lead_frames, 0.0f) : 0.0f;

    const bool stickyMatch = hasIouResult && bestIou > iou_stickiness_threshold;
    const bool prevValid = (prevTarget.classId >= 0);

    Detection chosenTarget = {};
    chosenTarget.classId = -1;
    bool hasTarget = false;

    if (stickyMatch) {
        // Tracked target re-acquired this frame (IoU stickiness).
        chosenTarget = bestByIou;
        hasTarget = true;
    } else if (hasDistResult) {
        // A target is visible this frame - track the nearest candidate.
        chosenTarget = bestByDist;
        hasTarget = true;
    } else if (prevValid && prevFramesSinceSeen < trackPersistenceFrames) {
        // True detection gap (no candidate this frame) within the bridge
        // window. With coast enabled, follow the target's last drift (decayed)
        // so a momentarily-lost target neither freezes nor jumps; otherwise
        // hold still. d_selected_target is kept alive for clean re-acquire.
        const int gap = prevFramesSinceSeen + 1;
        d_aim_state->frames_since_seen = gap;
        const AimConfig coastCfg = *d_aim_config;
        if (coastCfg.coast_enabled != 0.0f && d_aim_state->has_track) {
            // Decay by elapsed TIME, not raw frame count, so the glide fades at a
            // fixed rate regardless of frame interval. gap*dt_norm is the
            // elapsed time in nominal frames; equals `gap` at the tuned rate.
            const float factor = powf(coastCfg.coast_decay, static_cast<float>(gap) * dt_norm);
            // Follow only the target's drift (vel * scale), decayed. No
            // convergence term, so a stationary target does not drift off.
            float mx = d_aim_state->vel_x * movement_scale_x * factor;
            float my = d_aim_state->vel_y * movement_scale_y * factor;
            clampMaxStep(mx, my, coastCfg.max_step);
            const int dx = emitMouseDelta(mx, &d_aim_state->residual_x);
            const int dy = emitMouseDelta(my, &d_aim_state->residual_y);
            d_inference_result->movement.dx = dx;
            d_inference_result->movement.dy = dy;
            d_inference_result->hasTarget = 1;
            d_inference_result->reserved = 0;
            d_inference_result->targetX1 = prevTarget.x1;
            d_inference_result->targetY1 = prevTarget.y1;
            d_inference_result->targetX2 = prevTarget.x2;
            d_inference_result->targetY2 = prevTarget.y2;
            d_inference_result->targetConf = prevTarget.confidence;
            d_inference_result->targetClassId = prevTarget.classId;
            return;
        }
        d_aim_state->residual_x = 0.0f;
        d_aim_state->residual_y = 0.0f;
        writeEmptyInferenceResult(d_inference_result);
        return;
    }

    if (!hasTarget) {
        if (d_selected_target) {
            Detection emptyTarget = {};
            emptyTarget.classId = -1;
            *d_selected_target = emptyTarget;
        }
        d_aim_state->residual_x = 0.0f;
        d_aim_state->residual_y = 0.0f;
        d_aim_state->frames_since_seen = 0;
        d_aim_state->has_track = 0;  // target fully lost -> stop coasting
        d_aim_state->vel_x = 0.0f;
        d_aim_state->vel_y = 0.0f;
        writeEmptyInferenceResult(d_inference_result);
        return;
    }

    d_aim_state->frames_since_seen = 0;
    if (d_selected_target) {
        *d_selected_target = chosenTarget;
    }

    const float raw_center_x = (chosenTarget.x1 + chosenTarget.x2) * 0.5f;
    const float target_h = chosenTarget.y2 - chosenTarget.y1;
    const float raw_center_y =
        (chosenTarget.classId == head_class_id)
            ? (chosenTarget.y1 + target_h * head_y_offset)
            : (chosenTarget.y1 + target_h * body_y_offset);

    const AimConfig aim_config = *d_aim_config;

    // Was this target already being tracked last frame? Captured before has_track
    // is overwritten below; used to seed the One Euro filter and reset the error
    // derivative on a fresh acquire (avoids a derivative kick).
    const bool fresh_track = (d_aim_state->has_track == 0);

    // One Euro adaptive low-pass on the measured center, applied BEFORE velocity
    // and error so the whole controller (P move, feedforward, coast) runs on the
    // de-noised signal. Seed on fresh acquire to avoid a jump from a stale value.
    float target_center_x = raw_center_x;
    float target_center_y = raw_center_y;
    if (aim_config.oneeuro_enabled != 0.0f) {
        if (d_aim_state->has_track) {
            // Sample period in nominal frames. At the tuned rate dt_norm==1
            // and every alpha below collapses to the original per-frame value.
            const float ad = oneEuroAlpha(aim_config.oneeuro_dcutoff, dt_norm);
            const float de_x = raw_center_x - d_aim_state->filt_x;  // per-frame derivative
            d_aim_state->dfilt_x = ad * de_x + (1.0f - ad) * d_aim_state->dfilt_x;
            const float cutoff_x =
                aim_config.oneeuro_min_cutoff + aim_config.oneeuro_beta * fabsf(d_aim_state->dfilt_x);
            const float ax = oneEuroAlpha(cutoff_x, dt_norm);
            d_aim_state->filt_x = ax * raw_center_x + (1.0f - ax) * d_aim_state->filt_x;

            const float de_y = raw_center_y - d_aim_state->filt_y;
            d_aim_state->dfilt_y = ad * de_y + (1.0f - ad) * d_aim_state->dfilt_y;
            const float cutoff_y =
                aim_config.oneeuro_min_cutoff + aim_config.oneeuro_beta * fabsf(d_aim_state->dfilt_y);
            const float ay = oneEuroAlpha(cutoff_y, dt_norm);
            d_aim_state->filt_y = ay * raw_center_y + (1.0f - ay) * d_aim_state->filt_y;
        } else {
            d_aim_state->filt_x = raw_center_x;
            d_aim_state->filt_y = raw_center_y;
            d_aim_state->dfilt_x = 0.0f;
            d_aim_state->dfilt_y = 0.0f;
        }
        target_center_x = d_aim_state->filt_x;
        target_center_y = d_aim_state->filt_y;
    }

    // Update the target's per-frame screen drift (EMA, clamped) FIRST so both
    // the feedforward term below and a following detection gap's coast use the
    // freshest velocity. Use the RAW center deltas here, NOT the One Euro
    // filtered center: the filter delays position, and feeding a lagged velocity
    // into feedforward under-leads a moving target (aim trails its tail). The
    // P/error term below still uses the filtered center for a stable aim point,
    // so smoothing stabilizes WHERE we point without eating the lead signal.
    // (When One Euro is off, raw_center == target_center, so this is a no-op.)
    if (d_aim_state->has_track) {
        // prev_center is only refreshed on detection frames, not during a coast
        // gap, so after bridging G frames it is (G+1) frames stale. Divide the
        // displacement by the elapsed interval so a multi-frame jump is not read
        // as one-frame drift and over-leads on re-acquire. elapsed_te is
        // that interval in nominal frames, so vel becomes px-per-nominal-frame,
        // consistent under frame-rate jitter. Both reduce to the original
        // (divide by 1) when tracking is continuous at the tuned rate.
        const float elapsed_te = (static_cast<float>(prevFramesSinceSeen) + 1.0f) * dt_norm;
        const float inv_elapsed = 1.0f / fmaxf(elapsed_te, 1e-3f);
        const float maxDrift = 60.0f;  // model px per nominal frame sanity clamp
        float nvx = (raw_center_x - d_aim_state->prev_center_x) * inv_elapsed;
        float nvy = (raw_center_y - d_aim_state->prev_center_y) * inv_elapsed;
        nvx = fminf(fmaxf(nvx, -maxDrift), maxDrift);
        nvy = fminf(fmaxf(nvy, -maxDrift), maxDrift);
        const float av = emaAlphaDt(0.4f, elapsed_te);
        d_aim_state->vel_x = (1.0f - av) * d_aim_state->vel_x + av * nvx;
        d_aim_state->vel_y = (1.0f - av) * d_aim_state->vel_y + av * nvy;
    } else {
        d_aim_state->vel_x = 0.0f;
        d_aim_state->vel_y = 0.0f;
    }
    d_aim_state->prev_center_x = raw_center_x;  // raw (un-lagged) for next velocity
    d_aim_state->prev_center_y = raw_center_y;
    d_aim_state->has_track = 1;

    // Forward latency lead: push the aim point ahead by the target's
    // per-nominal-frame velocity times the pipeline latency (lead_frames), so we
    // point where it WILL be when the move lands. Distinct from feedforward, which
    // only cancels steady-state tracking lag. Clamped so a noisy velocity spike
    // cannot fling the aim point; disabled entirely when lead_gain == 0.
    float aim_x = target_center_x;
    float aim_y = target_center_y;
    if (aim_config.lead_gain != 0.0f && lead_frames > 0.0f) {
        float lead_x = aim_config.lead_gain * d_aim_state->vel_x * lead_frames;
        float lead_y = aim_config.lead_gain * d_aim_state->vel_y * lead_frames;
        clampMaxStep(lead_x, lead_y, aim_config.lead_max_px);
        aim_x += lead_x;
        aim_y += lead_y;
    }

    // Aim at the (lead-adjusted) target center. Velocity feedforward keeps pace
    // with a moving target (cancels P steady-state lag) without overshooting.
    const float error_x = aim_x - screen_center_x;
    const float error_y = aim_y - screen_center_y;
    const float ff = aim_config.feedforward_gain;

    // Derivative (damping) term: react to how fast the error is shrinking and
    // push back, so a high-kp approach decelerates BEFORE it overshoots. This is
    // pure damping of OUR convergence - target motion is handled by feedforward,
    // so D stays quiet (de ~ 0) while tracking well and only bites on transients.
    // Error rides the One Euro-filtered center, so the derivative is clean; it is
    // still clamped and reset on fresh acquire to avoid a derivative kick. The
    // clamp is generous (a full-frame initial slew can change error by >60px in
    // one frame); a tighter clamp would starve the damping exactly in the
    // large-error regime that overshoots most. Fresh-acquire reset, not this
    // clamp, guards against the real derivative kick.
    const float maxDErr = 150.0f;
    float de_x = fminf(fmaxf(error_x - d_aim_state->prev_err_x, -maxDErr), maxDErr);
    float de_y = fminf(fmaxf(error_y - d_aim_state->prev_err_y, -maxDErr), maxDErr);
    if (fresh_track) {
        d_aim_state->derr_x = 0.0f;
        d_aim_state->derr_y = 0.0f;
    } else {
        // dt-aware EMA: weight collapses to 0.6 at the tuned rate.
        const float ade = emaAlphaDt(0.6f, dt_norm);
        d_aim_state->derr_x = ade * de_x + (1.0f - ade) * d_aim_state->derr_x;
        d_aim_state->derr_y = ade * de_y + (1.0f - ade) * d_aim_state->derr_y;
    }
    d_aim_state->prev_err_x = error_x;
    d_aim_state->prev_err_y = error_y;

    float movement_x =
        (nonlinearPMove(error_x, aim_config.kp_x, aim_config.p_softness_x)
         + aim_config.kd_x * d_aim_state->derr_x
         + ff * d_aim_state->vel_x) * movement_scale_x;
    float movement_y =
        (nonlinearPMove(error_y, aim_config.kp_y, aim_config.p_softness_y)
         + aim_config.kd_y * d_aim_state->derr_y
         + ff * d_aim_state->vel_y) * movement_scale_y;

    // Per-frame max-step clamp (output px). Bounds the slew so a large initial
    // error is crossed in several smooth steps instead of one delayed leap that
    // overshoots and rings - overshoot D cannot prevent reactively.
    clampMaxStep(movement_x, movement_y, aim_config.max_step);

    int emit_dx = emitMouseDelta(movement_x, &d_aim_state->residual_x);
    int emit_dy = emitMouseDelta(movement_y, &d_aim_state->residual_y);

    d_inference_result->movement.dx = emit_dx;
    d_inference_result->movement.dy = emit_dy;
    d_inference_result->hasTarget = 1;
    d_inference_result->reserved = 0;
    d_inference_result->targetX1 = chosenTarget.x1;
    d_inference_result->targetY1 = chosenTarget.y1;
    d_inference_result->targetX2 = chosenTarget.x2;
    d_inference_result->targetY2 = chosenTarget.y2;
    d_inference_result->targetConf = chosenTarget.confidence;
    d_inference_result->targetClassId = chosenTarget.classId;
}

cudaError_t postprocessYoloFusedGpu(
    const void* d_raw_output,
    bool is_fp16,
    int num_boxes,
    int num_classes,
    float conf_threshold,
    uint32_t allowedClassMask,
    int max_candidate_blocks,
    float max_box_extent,
    float screen_center_x,
    float screen_center_y,
    float movement_scale_x,
    float movement_scale_y,
    int head_class_id,
    const AimConfig* d_aim_config,
    float iou_stickiness_threshold,
    float head_y_offset,
    float body_y_offset,
    Detection* d_selected_target,
    AimState* d_aim_state,
    const FrameTiming* d_frame_timing,
    InferenceResult* d_inference_result,
    Detection* d_stage1_best_dist,
    float* d_stage1_dist_score,
    Detection* d_stage1_best_iou,
    float* d_stage1_iou_score,
    cudaStream_t stream)
{
    if (!d_raw_output || !d_aim_config || !d_aim_state || !d_inference_result ||
        !d_stage1_best_dist || !d_stage1_dist_score ||
        !d_stage1_best_iou || !d_stage1_iou_score) {
        return cudaErrorInvalidValue;
    }
    if (num_boxes <= 0 || num_classes <= 0 || max_candidate_blocks <= 0) {
        return cudaErrorInvalidValue;
    }
    if (!isfinite(conf_threshold) || conf_threshold < 0.0f ||
        !isfinite(max_box_extent) || max_box_extent <= 0.0f ||
        !isfinite(movement_scale_x) || movement_scale_x <= 0.0f ||
        !isfinite(movement_scale_y) || movement_scale_y <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    int stage1Threads = 256;
    if (num_boxes <= 128) stage1Threads = 128;
    if (num_boxes <= 64) stage1Threads = 64;
    if (num_boxes <= 32) stage1Threads = 32;
    if (stage1Threads > MAX_SELECTION_THREADS) stage1Threads = MAX_SELECTION_THREADS;

    int stage1Blocks = (num_boxes + stage1Threads - 1) / stage1Threads;
    if (stage1Blocks < 1) stage1Blocks = 1;
    if (stage1Blocks > max_candidate_blocks) stage1Blocks = max_candidate_blocks;

    if (is_fp16) {
        stage1DecodeAndSelectKernel<true><<<stage1Blocks, stage1Threads, 0, stream>>>(
            d_raw_output,
            num_boxes,
            num_classes,
            conf_threshold,
            allowedClassMask,
            max_box_extent,
            screen_center_x,
            screen_center_y,
            head_class_id,
            head_y_offset,
            body_y_offset,
            d_selected_target,
            d_aim_config,
            d_stage1_best_dist,
            d_stage1_dist_score,
            d_stage1_best_iou,
            d_stage1_iou_score
        );
    } else {
        stage1DecodeAndSelectKernel<false><<<stage1Blocks, stage1Threads, 0, stream>>>(
            d_raw_output,
            num_boxes,
            num_classes,
            conf_threshold,
            allowedClassMask,
            max_box_extent,
            screen_center_x,
            screen_center_y,
            head_class_id,
            head_y_offset,
            body_y_offset,
            d_selected_target,
            d_aim_config,
            d_stage1_best_dist,
            d_stage1_dist_score,
            d_stage1_best_iou,
            d_stage1_iou_score
        );
    }

    int stage2Threads = 256;
    if (stage1Blocks <= 128) stage2Threads = 128;
    if (stage1Blocks <= 64) stage2Threads = 64;
    if (stage1Blocks <= 32) stage2Threads = 32;
    if (stage2Threads > MAX_SELECTION_THREADS) stage2Threads = MAX_SELECTION_THREADS;

    stage2FinalizeKernel<<<1, stage2Threads, 0, stream>>>(
        stage1Blocks,
        d_stage1_best_dist,
        d_stage1_dist_score,
        d_stage1_best_iou,
        d_stage1_iou_score,
        screen_center_x,
        screen_center_y,
        movement_scale_x,
        movement_scale_y,
        head_class_id,
        d_aim_config,
        iou_stickiness_threshold,
        head_y_offset,
        body_y_offset,
        d_selected_target,
        d_aim_state,
        d_frame_timing,
        d_inference_result
    );

    return cudaGetLastError();
}

} // namespace gpa
