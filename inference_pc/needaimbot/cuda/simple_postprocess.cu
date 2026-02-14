// GPU-based postprocessing for simple inference
// Full optimizations from unified_graph_pipeline:
// - Warp-level primitives for fast reduction
// - Head-in-body priority selection
// - IoU-based target stickiness (hysteresis)
// - Fused target selection + PID movement calculation
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
constexpr float DEADZONE_THRESHOLD = 5.0f;  // pixels
constexpr int MAX_SELECTION_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_SELECTION_WARPS = MAX_SELECTION_THREADS / WARP_SIZE;
static_assert((MAX_SELECTION_THREADS % WARP_SIZE) == 0,
              "MAX_SELECTION_THREADS must be multiple of warp size");

// =============================================================================
// Helper Functions
// =============================================================================

template<bool kIsFp16>
__device__ __forceinline__ float readValue(const void* buffer, size_t idx) {
    if constexpr (kIsFp16) {
        return __half2float(reinterpret_cast<const __half*>(buffer)[idx]);
    } else {
        return reinterpret_cast<const float*>(buffer)[idx];
    }
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
// One-pass Fused Decode + Target Selection + PID
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
    int head_class_id,
    float head_conf_bonus,
    const Detection* __restrict__ d_selected_target,
    Detection* __restrict__ d_stage1_best_dist,
    float* __restrict__ d_stage1_dist_score,
    Detection* __restrict__ d_stage1_best_iou,
    float* __restrict__ d_stage1_iou_score)
{
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
        const float centerX = det.x1 + w * 0.5f;
        float effectiveDist = fabsf(centerX - screen_center_x);
        if (det.classId == head_class_id && head_conf_bonus > 0.0f) {
            effectiveDist -= head_conf_bonus * 100.0f;
        }
        if (effectiveDist < localBestDist) {
            localBestDist = effectiveDist;
            localBestByDist = det;
            localHasDist = true;
        }

        if (prevValid) {
            const float iou = computeBoundingBoxIoU(det, prevTarget);
            if (iou > localBestIou) {
                localBestIou = iou;
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
    int head_class_id,
    float kp_x, float kp_y,
    float ki_x, float ki_y,
    float kd_x, float kd_y,
    float integral_max,
    float derivative_max,
    float iou_stickiness_threshold,
    float head_y_offset,
    float body_y_offset,
    Detection* __restrict__ d_selected_target,
    PIDState* __restrict__ d_pid_state,
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

    bool hasTarget = hasDistResult;
    Detection chosenTarget = bestByDist;
    if (hasIouResult && bestIou > iou_stickiness_threshold) {
        chosenTarget = bestByIou;
        hasTarget = true;
    }

    if (!hasTarget) {
        if (d_selected_target) {
            Detection emptyTarget = {};
            emptyTarget.classId = -1;
            *d_selected_target = emptyTarget;
        }
        d_pid_state->prev_error_x = 0.0f;
        d_pid_state->prev_error_y = 0.0f;
        d_pid_state->integral_x = 0.0f;
        d_pid_state->integral_y = 0.0f;
        writeEmptyInferenceResult(d_inference_result);
        return;
    }

    if (d_selected_target) {
        *d_selected_target = chosenTarget;
    }

    const float target_center_x = (chosenTarget.x1 + chosenTarget.x2) * 0.5f;
    const float target_h = chosenTarget.y2 - chosenTarget.y1;
    const float target_center_y =
        (chosenTarget.classId == head_class_id)
            ? (chosenTarget.y1 + target_h * head_y_offset)
            : (chosenTarget.y1 + target_h * body_y_offset);

    const float error_x = target_center_x - screen_center_x;
    const float error_y = target_center_y - screen_center_y;

    float prev_error_x = d_pid_state->prev_error_x;
    float prev_error_y = d_pid_state->prev_error_y;
    float integral_x = d_pid_state->integral_x;
    float integral_y = d_pid_state->integral_y;

    if (fabsf(error_x) < DEADZONE_THRESHOLD) integral_x = 0.0f;
    if (fabsf(error_y) < DEADZONE_THRESHOLD) integral_y = 0.0f;

    integral_x += error_x;
    integral_y += error_y;
    if (integral_x > integral_max) integral_x = integral_max;
    if (integral_x < -integral_max) integral_x = -integral_max;
    if (integral_y > integral_max) integral_y = integral_max;
    if (integral_y < -integral_max) integral_y = -integral_max;

    float derivative_x = error_x - prev_error_x;
    float derivative_y = error_y - prev_error_y;
    if (derivative_x > derivative_max) derivative_x = derivative_max;
    if (derivative_x < -derivative_max) derivative_x = -derivative_max;
    if (derivative_y > derivative_max) derivative_y = derivative_max;
    if (derivative_y < -derivative_max) derivative_y = -derivative_max;

    const float movement_x = kp_x * error_x + ki_x * integral_x + kd_x * derivative_x;
    const float movement_y = kp_y * error_y + ki_y * integral_y + kd_y * derivative_y;

    d_pid_state->prev_error_x = error_x;
    d_pid_state->prev_error_y = error_y;
    d_pid_state->integral_x = integral_x;
    d_pid_state->integral_y = integral_y;

    int emit_dx = __float2int_rn(movement_x);
    int emit_dy = __float2int_rn(movement_y);
    if (emit_dx > 127) emit_dx = 127;
    if (emit_dx < -127) emit_dx = -127;
    if (emit_dy > 127) emit_dy = 127;
    if (emit_dy < -127) emit_dy = -127;

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
    int head_class_id,
    float head_conf_bonus,
    const PIDConfig& pid_config,
    float iou_stickiness_threshold,
    float head_y_offset,
    float body_y_offset,
    Detection* d_selected_target,
    PIDState* d_pid_state,
    InferenceResult* d_inference_result,
    Detection* d_stage1_best_dist,
    float* d_stage1_dist_score,
    Detection* d_stage1_best_iou,
    float* d_stage1_iou_score,
    cudaStream_t stream)
{
    if (!d_raw_output || !d_pid_state || !d_inference_result ||
        !d_stage1_best_dist || !d_stage1_dist_score ||
        !d_stage1_best_iou || !d_stage1_iou_score) {
        return cudaErrorInvalidValue;
    }
    if (num_boxes <= 0 || num_classes <= 0 || max_candidate_blocks <= 0) {
        return cudaErrorInvalidValue;
    }
    if (!isfinite(conf_threshold) || conf_threshold < 0.0f ||
        !isfinite(max_box_extent) || max_box_extent <= 0.0f) {
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
            head_class_id,
            head_conf_bonus,
            d_selected_target,
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
            head_class_id,
            head_conf_bonus,
            d_selected_target,
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
        head_class_id,
        pid_config.kp_x, pid_config.kp_y,
        pid_config.ki_x, pid_config.ki_y,
        pid_config.kd_x, pid_config.kd_y,
        pid_config.integral_max,
        pid_config.derivative_max,
        iou_stickiness_threshold,
        head_y_offset,
        body_y_offset,
        d_selected_target,
        d_pid_state,
        d_inference_result
    );

    return cudaGetLastError();
}

} // namespace gpa
