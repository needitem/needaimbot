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
#define WARP_SIZE 32
constexpr float DEFAULT_IOU_STICKINESS_THRESHOLD = 0.3f;
constexpr float DEADZONE_THRESHOLD = 5.0f;  // pixels

// =============================================================================
// Helper Functions
// =============================================================================

// Helper to read FP16 or FP32 value
__device__ __forceinline__ float readValue(const void* buffer, bool is_fp16, size_t idx) {
    if (is_fp16) {
        return __half2float(reinterpret_cast<const __half*>(buffer)[idx]);
    }
    return reinterpret_cast<const float*>(buffer)[idx];
}

// Warp-level reduction for finding minimum (used in target selection)
__device__ __forceinline__ float warpReduceMin(float val, int& idx, int myIdx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float otherVal = __shfl_down_sync(0xffffffff, val, offset);
        int otherIdx = __shfl_down_sync(0xffffffff, myIdx, offset);
        if (otherVal < val) {
            val = otherVal;
            myIdx = otherIdx;
        }
    }
    idx = myIdx;
    return val;
}

// Warp-level reduction for finding maximum score
__device__ __forceinline__ float warpReduceMax(float val, int& idx, int myIdx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float otherVal = __shfl_down_sync(0xffffffff, val, offset);
        int otherIdx = __shfl_down_sync(0xffffffff, myIdx, offset);
        if (otherVal > val) {
            val = otherVal;
            myIdx = otherIdx;
        }
    }
    idx = myIdx;
    return val;
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
// Decode Kernel with Strict Validation
// =============================================================================

// YOLO11 decode kernel: [1, 4+num_classes, num_boxes] -> Detection[]
// With strict validation to prevent garbage values
// allowedClassMask: bitmask where bit N = 1 means class N is allowed
__global__ void decodeYoloKernel(
    const void* __restrict__ d_raw_output,
    bool is_fp16,
    int num_boxes,
    int num_classes,
    float conf_threshold,
    uint32_t allowedClassMask,
    Detection* d_decoded,
    int* d_decoded_count,
    int max_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    // Find best class score among ALLOWED classes only
    float max_score = -1.0f;
    int best_class = -1;

    for (int c = 0; c < num_classes && c < 32; c++) {
        // Skip if class is not allowed
        if (!(allowedClassMask & (1u << c))) continue;

        size_t score_idx = (4 + c) * num_boxes + idx;
        float score = readValue(d_raw_output, is_fp16, score_idx);
        if (score > max_score) {
            max_score = score;
            best_class = c;
        }
    }

    // Early exit for low confidence or no allowed class found
    if (max_score <= conf_threshold || best_class < 0) return;

    // Read bbox: cx, cy, w, h
    float cx = readValue(d_raw_output, is_fp16, 0 * num_boxes + idx);
    float cy = readValue(d_raw_output, is_fp16, 1 * num_boxes + idx);
    float w  = readValue(d_raw_output, is_fp16, 2 * num_boxes + idx);
    float h  = readValue(d_raw_output, is_fp16, 3 * num_boxes + idx);

    // STRICT validation to prevent garbage values (from postProcessGpu.cu)
    // Check for NaN, infinity
    if (!isfinite(cx) || !isfinite(cy) || !isfinite(w) || !isfinite(h)) {
        return;
    }

    // Reasonable bounds check (model output should be within input resolution)
    const float MAX_COORD = 10000.0f;
    if (cx < 0 || cx > MAX_COORD || cy < 0 || cy > MAX_COORD ||
        w <= 0 || w > MAX_COORD || h <= 0 || h > MAX_COORD) {
        return;
    }

    // Convert to x1,y1,x2,y2
    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    // Additional validation after conversion
    if (x1 < -1000 || x1 > MAX_COORD || y1 < -1000 || y1 > MAX_COORD ||
        x2 < -1000 || x2 > MAX_COORD || y2 < -1000 || y2 > MAX_COORD) {
        return;
    }

    // Validate dimensions (reasonable range: 1-640 pixels for typical YOLO)
    if (w <= 0 || h <= 0 || w > 640 || h > 640) {
        return;
    }

    // Atomic add to get write index
    int write_idx = atomicAdd(d_decoded_count, 1);
    if (write_idx >= max_detections) return;

    // Write detection
    Detection& det = d_decoded[write_idx];
    det.x1 = x1;
    det.y1 = y1;
    det.x2 = x2;
    det.y2 = y2;
    det.confidence = max_score;
    det.classId = best_class;
}

cudaError_t decodeYoloGpu(
    const void* d_raw_output,
    bool is_fp16,
    int num_boxes,
    int num_classes,
    float conf_threshold,
    uint32_t allowedClassMask,
    Detection* d_decoded,
    int* d_decoded_count,
    int max_detections,
    cudaStream_t stream)
{
    if (!d_raw_output || !d_decoded || !d_decoded_count) {
        return cudaErrorInvalidValue;
    }

    // Parameter validation
    if (num_boxes <= 0 || num_classes <= 0 || max_detections <= 0) {
        cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    if (!isfinite(conf_threshold) || conf_threshold < 0.0f) {
        return cudaErrorInvalidValue;
    }

    // Reset count
    cudaMemsetAsync(d_decoded_count, 0, sizeof(int), stream);

    const int block_size = 256;
    const int grid_size = (num_boxes + block_size - 1) / block_size;

    decodeYoloKernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_output, is_fp16, num_boxes, num_classes,
        conf_threshold, allowedClassMask, d_decoded, d_decoded_count, max_detections
    );

    return cudaGetLastError();
}

// =============================================================================
// Target Selection with Head-in-Body Priority (from postProcessGpu.cu)
// =============================================================================

// Check if head bbox is inside body bbox
__device__ __forceinline__ bool isHeadInsideBody(
    const Detection& head, const Detection& body)
{
    return (head.x1 >= body.x1 &&
            head.y1 >= body.y1 &&
            head.x2 <= body.x2 &&
            head.y2 <= body.y2);
}

// Validate detection values
__device__ __forceinline__ bool isValidDetection(const Detection& det) {
    // Skip invalid detections and extreme values
    float w = det.x2 - det.x1;
    float h = det.y2 - det.y1;

    if (w <= 0 || h <= 0 || det.confidence <= 0 || det.classId < 0) {
        return false;
    }

    // Check for extreme values
    if (det.x1 < -100 || det.x1 > 2000 || det.y1 < -100 || det.y1 > 2000 ||
        det.x2 < -100 || det.x2 > 2000 || det.y2 < -100 || det.y2 > 2000) {
        return false;
    }

    if (w > 1000 || h > 1000) {
        return false;
    }

    // Check for garbage values (10억대)
    if (fabsf(det.x1) > 1000000 || fabsf(det.y1) > 1000000 ||
        fabsf(det.x2) > 1000000 || fabsf(det.y2) > 1000000) {
        return false;
    }

    return true;
}



// Validate single best target before host copy
__global__ void validateBestTargetKernel(Detection* d_best_target, int* d_has_target)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*d_has_target == 0) return;

        Detection& target = *d_best_target;

        float w = target.x2 - target.x1;
        float h = target.y2 - target.y1;

        // Combined validation
        bool invalid_position = (target.x1 < -100) | (target.x1 > 2000) |
                                (target.y1 < -100) | (target.y1 > 2000) |
                                (target.x2 < -100) | (target.x2 > 2000) |
                                (target.y2 < -100) | (target.y2 > 2000);
        bool invalid_size = (w <= 0) | (w > 1000) | (h <= 0) | (h > 1000);
        bool invalid_meta = (target.classId < 0) | (target.confidence <= 0.0f) |
                            (target.confidence > 1.0f);

        if (invalid_position | invalid_size | invalid_meta) {
            // Clear invalid target
            target.classId = -1;
            target.confidence = 0.0f;
            target.x1 = -1;
            target.y1 = -1;
            target.x2 = -1;
            target.y2 = -1;
            *d_has_target = 0;
        }
    }
}

void validateBestTargetGpu(
    Detection* d_best_target,
    int* d_has_target,
    cudaStream_t stream)
{
    if (!d_best_target || !d_has_target) return;

    validateBestTargetKernel<<<1, 1, 0, stream>>>(d_best_target, d_has_target);
}

// =============================================================================
// Fused Target Selection + PID Movement Kernel (from unified_graph_pipeline.cu)
// =============================================================================

// Fused kernel: target selection with IoU stickiness + PID movement calculation
// All on GPU, minimal D2H transfer (only MouseMovement result)
__global__ void fusedTargetSelectionAndMovementKernel(
    const Detection* __restrict__ d_detections,
    const int* __restrict__ d_num_detections,
    int max_detections,
    float screen_center_x,
    float screen_center_y,
    int head_class_id,
    float head_conf_bonus,
    float kp_x, float kp_y,
    float ki_x, float ki_y,
    float kd_x, float kd_y,
    float integral_max,
    float derivative_max,
    float iou_stickiness_threshold,
    float head_y_offset,
    float body_y_offset,
    Detection* __restrict__ d_selected_target,  // Previous/current selected target (persistent)
    Detection* __restrict__ d_best_target,      // Output: best target this frame
    int* __restrict__ d_has_target,             // Output: 1 if target found
    MouseMovement* __restrict__ d_output_movement,
    PIDState* __restrict__ d_pid_state)
{
    // Using warp shuffle for reduction - no shared memory arrays needed
    __shared__ Detection s_prevTarget;
    __shared__ bool s_prevValid;

    if (threadIdx.x == 0) {
        Detection emptyTarget = {};
        emptyTarget.classId = -1;
        s_prevTarget = emptyTarget;
        s_prevValid = false;

        if (d_selected_target) {
            Detection cached = *d_selected_target;
            s_prevTarget = cached;
            float w = cached.x2 - cached.x1;
            float h = cached.y2 - cached.y1;
            s_prevValid = (cached.classId >= 0) && (cached.confidence > 0.0f) &&
                          (w > 0) && (h > 0);
        }

        *d_has_target = 0;
        d_output_movement->dx = 0;
        d_output_movement->dy = 0;
    }
    __syncthreads();

    int count = *d_num_detections;
    if (count <= 0 || count > max_detections) {
        if (threadIdx.x == 0 && d_selected_target) {
            Detection emptyTarget = {};
            emptyTarget.classId = -1;
            *d_selected_target = emptyTarget;

            // Reset PID state when no target
            d_pid_state->prev_error_x = 0.0f;
            d_pid_state->prev_error_y = 0.0f;
            d_pid_state->integral_x = 0.0f;
            d_pid_state->integral_y = 0.0f;
        }
        return;
    }

    Detection prevTarget = s_prevTarget;
    bool prevValid = s_prevValid;

    int localBestIdx = -1;
    float localBestDistX = 1e9f;
    float localBestIoU = -1.0f;
    int localBestIoUIdx = -1;

    // Each thread processes detections in strided fashion
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const Detection& t = d_detections[i];

        // Skip invalid targets
        float w = t.x2 - t.x1;
        float h = t.y2 - t.y1;
        if (t.x1 < -1000.0f || t.x1 > 10000.0f ||
            t.y1 < -1000.0f || t.y1 > 10000.0f ||
            w <= 0 || h <= 0 ||
            t.confidence <= 0.0f || t.confidence > 1.0f) {
            continue;
        }

        float centerX = t.x1 + w * 0.5f;
        float dx = fabsf(centerX - screen_center_x);

        // Apply head bonus to distance
        float effective_dist = dx;
        if (t.classId == head_class_id && head_conf_bonus > 0) {
            effective_dist -= head_conf_bonus * 100.0f;
        }

        if (effective_dist < localBestDistX) {
            localBestDistX = effective_dist;
            localBestIdx = i;
        }

        // Check IoU with previous target for stickiness
        if (prevValid) {
            float iou = computeBoundingBoxIoU(t, prevTarget);
            if (iou > localBestIoU) {
                localBestIoU = iou;
                localBestIoUIdx = i;
            }
        }
    }

    // Warp-level reduction using shuffle
    const unsigned int FULL_MASK = 0xffffffff;

    float bestDistX = localBestDistX;
    int bestIdx = localBestIdx;
    float bestIoU = localBestIoU;
    int bestIoUIdx = localBestIoUIdx;

    // Warp shuffle reduction - find min distance and max IoU
    for (int offset = 16; offset > 0; offset >>= 1) {
        float otherDistX = __shfl_down_sync(FULL_MASK, bestDistX, offset);
        int otherIdx = __shfl_down_sync(FULL_MASK, bestIdx, offset);
        float otherIoU = __shfl_down_sync(FULL_MASK, bestIoU, offset);
        int otherIoUIdx = __shfl_down_sync(FULL_MASK, bestIoUIdx, offset);

        // Min distance reduction
        if (otherDistX < bestDistX) {
            bestDistX = otherDistX;
            bestIdx = otherIdx;
        }
        // Max IoU reduction
        if (otherIoU > bestIoU) {
            bestIoU = otherIoU;
            bestIoUIdx = otherIoUIdx;
        }
    }

    // Only thread 0 has the final results
    if (threadIdx.x == 0) {
        int candidateIndex = bestIdx;
        bool candidateValid = candidateIndex >= 0;
        Detection candidateTarget = candidateValid ? d_detections[candidateIndex] : Detection{};

        // Hysteresis: prefer previous target if IoU stays above threshold
        int chosenIndex = candidateIndex;
        Detection chosenTarget = candidateTarget;
        bool haveTarget = candidateValid;

        if (prevValid && bestIoUIdx >= 0 && bestIoU > iou_stickiness_threshold) {
            chosenIndex = bestIoUIdx;
            chosenTarget = d_detections[bestIoUIdx];
            haveTarget = true;
        }

        if (haveTarget) {
            *d_has_target = 1;
            *d_best_target = chosenTarget;
            if (d_selected_target) {
                *d_selected_target = chosenTarget;
            }

            // Calculate aim point
            float target_center_x = (chosenTarget.x1 + chosenTarget.x2) * 0.5f;
            float target_h = chosenTarget.y2 - chosenTarget.y1;
            float target_center_y;

            if (chosenTarget.classId == head_class_id) {
                target_center_y = chosenTarget.y1 + target_h * head_y_offset;
            } else {
                target_center_y = chosenTarget.y1 + target_h * body_y_offset;
            }

            // Current error
            float error_x = target_center_x - screen_center_x;
            float error_y = target_center_y - screen_center_y;

            // Load previous PID state
            float prev_error_x = d_pid_state->prev_error_x;
            float prev_error_y = d_pid_state->prev_error_y;
            float integral_x = d_pid_state->integral_x;
            float integral_y = d_pid_state->integral_y;

            // Reset integral when very close to target (deadzone)
            if (fabsf(error_x) < DEADZONE_THRESHOLD) {
                integral_x = 0.0f;
            }
            if (fabsf(error_y) < DEADZONE_THRESHOLD) {
                integral_y = 0.0f;
            }

            // Update integral (with anti-windup clamping)
            integral_x += error_x;
            integral_y += error_y;

            // Clamp integral to prevent windup
            if (integral_x > integral_max) integral_x = integral_max;
            if (integral_x < -integral_max) integral_x = -integral_max;
            if (integral_y > integral_max) integral_y = integral_max;
            if (integral_y < -integral_max) integral_y = -integral_max;

            // Calculate derivative (error change)
            float derivative_x = error_x - prev_error_x;
            float derivative_y = error_y - prev_error_y;

            // Clamp derivative to prevent excessive oscillation
            if (derivative_x > derivative_max) derivative_x = derivative_max;
            if (derivative_x < -derivative_max) derivative_x = -derivative_max;
            if (derivative_y > derivative_max) derivative_y = derivative_max;
            if (derivative_y < -derivative_max) derivative_y = -derivative_max;

            // PID controller: P + I + D
            float movement_x = kp_x * error_x + ki_x * integral_x + kd_x * derivative_x;
            float movement_y = kp_y * error_y + ki_y * integral_y + kd_y * derivative_y;

            // Save current state for next iteration
            d_pid_state->prev_error_x = error_x;
            d_pid_state->prev_error_y = error_y;
            d_pid_state->integral_x = integral_x;
            d_pid_state->integral_y = integral_y;

            // Round to nearest int and clamp
            int emit_dx = static_cast<int>(lroundf(movement_x));
            int emit_dy = static_cast<int>(lroundf(movement_y));

            // Clamp to valid range
            if (emit_dx > 127) emit_dx = 127;
            if (emit_dx < -127) emit_dx = -127;
            if (emit_dy > 127) emit_dy = 127;
            if (emit_dy < -127) emit_dy = -127;

            d_output_movement->dx = emit_dx;
            d_output_movement->dy = emit_dy;
        } else {
            Detection emptyTarget = {};
            emptyTarget.classId = -1;
            *d_has_target = 0;
            *d_best_target = emptyTarget;
            if (d_selected_target) {
                *d_selected_target = emptyTarget;
            }
            d_output_movement->dx = 0;
            d_output_movement->dy = 0;

            // Reset PID state when no target
            d_pid_state->prev_error_x = 0.0f;
            d_pid_state->prev_error_y = 0.0f;
            d_pid_state->integral_x = 0.0f;
            d_pid_state->integral_y = 0.0f;
        }
    }
}

// =============================================================================
// Host API for Fused Kernel
// =============================================================================

cudaError_t fusedTargetSelectionAndMovementGpu(
    const Detection* d_detections,
    const int* d_num_detections,
    int max_detections,
    float screen_center_x,
    float screen_center_y,
    int head_class_id,
    float head_conf_bonus,
    const PIDConfig& pid_config,
    float iou_stickiness_threshold,
    float head_y_offset,
    float body_y_offset,
    Detection* d_selected_target,
    Detection* d_best_target,
    int* d_has_target,
    MouseMovement* d_output_movement,
    PIDState* d_pid_state,
    cudaStream_t stream)
{
    if (!d_detections || !d_num_detections || !d_best_target ||
        !d_has_target || !d_output_movement || !d_pid_state) {
        return cudaErrorInvalidValue;
    }

    // Launch with single warp (32 threads) for efficient warp shuffle
    fusedTargetSelectionAndMovementKernel<<<1, 32, 0, stream>>>(
        d_detections,
        d_num_detections,
        max_detections,
        screen_center_x,
        screen_center_y,
        head_class_id,
        head_conf_bonus,
        pid_config.kp_x, pid_config.kp_y,
        pid_config.ki_x, pid_config.ki_y,
        pid_config.kd_x, pid_config.kd_y,
        pid_config.integral_max,
        pid_config.derivative_max,
        iou_stickiness_threshold,
        head_y_offset,
        body_y_offset,
        d_selected_target,
        d_best_target,
        d_has_target,
        d_output_movement,
        d_pid_state
    );

    return cudaGetLastError();
}

} // namespace gpa
