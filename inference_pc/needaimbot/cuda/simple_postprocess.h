#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gpa {

struct Detection;  // Forward declaration

// =============================================================================
// GPU State Structures
// =============================================================================

// P controller state (GPU persistent)
struct AimState {
    float residual_x = 0.0f;
    float residual_y = 0.0f;
};

// Nonlinear P controller configuration.
struct AimConfig {
    float kp_x = 0.7f;
    float kp_y = 0.62f;
    float p_softness_x = 28.0f;
    float p_softness_y = 30.0f;
};

// Mouse movement output
struct MouseMovement {
    int dx = 0;
    int dy = 0;
};

// =============================================================================
// Combined Inference Result (for single D2H transfer optimization)
// =============================================================================

// All inference outputs packed into single struct for one cudaMemcpy
// This reduces D2H transfer overhead from 3 copies to 1
struct InferenceResult {
    MouseMovement movement;     // 8 bytes: dx, dy
    int hasTarget;              // 4 bytes: 1 if target found, 0 otherwise
    int reserved;               // 4 bytes: padding for alignment
    float targetX1, targetY1;   // 8 bytes: best target bbox (if hasTarget)
    float targetX2, targetY2;   // 8 bytes
    float targetConf;           // 4 bytes: confidence
    int targetClassId;          // 4 bytes: class ID
    // Total: 40 bytes - fits in single cache line
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
    float head_conf_bonus,         // Head bonus for target selection
    const AimConfig& aim_config,   // Movement parameters
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
