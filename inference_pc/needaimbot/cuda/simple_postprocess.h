#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gpa {

struct Detection;  // Forward declaration

// =============================================================================
// GPU State Structures
// =============================================================================

// PID controller state (GPU persistent)
struct PIDState {
    float prev_error_x = 0.0f;
    float prev_error_y = 0.0f;
    float integral_x = 0.0f;
    float integral_y = 0.0f;
};

// PID controller configuration
struct PIDConfig {
    float kp_x = 0.5f;
    float kp_y = 0.5f;
    float ki_x = 0.0f;
    float ki_y = 0.0f;
    float kd_x = 0.3f;
    float kd_y = 0.3f;
    float integral_max = 50.0f;
    float derivative_max = 30.0f;
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

// =============================================================================
// GPU-based YOLO Decoding
// =============================================================================

// Decodes YOLO11 format: [batch, 4+num_classes, num_boxes] -> Detection array
// Handles both FP32 and FP16 output tensors
// Includes strict validation to filter garbage values
// Returns decoded count via d_decoded_count (device pointer)
// allowedClassMask: bitmask where bit N = 1 means class N is allowed (0xFFFFFFFF = all allowed)
cudaError_t decodeYoloGpu(
    const void* d_raw_output,      // Raw model output (FP32 or FP16)
    bool is_fp16,                   // Output tensor is FP16
    int num_boxes,                  // Number of anchor boxes
    int num_classes,                // Number of classes
    float conf_threshold,           // Confidence threshold
    uint32_t allowedClassMask,      // Bitmask of allowed classes
    Detection* d_decoded,           // Output: decoded detections (device)
    int* d_decoded_count,           // Output: number of decoded detections (device)
    int max_detections,             // Maximum detections to output
    cudaStream_t stream = 0
);

// =============================================================================
// Fused Target Selection + PID Movement
// =============================================================================

// All-in-one GPU kernel:
// 1. Target selection with IoU-based stickiness (hysteresis)
// 2. PID controller calculation
// 3. Mouse movement output
// Minimal CPU involvement - only final mouse command needs D2H transfer
cudaError_t fusedTargetSelectionAndMovementGpu(
    const Detection* d_detections,  // Input detections (device)
    const int* d_num_detections,    // Number of detections (device)
    int max_detections,             // Maximum detections
    float screen_center_x,          // Crosshair X
    float screen_center_y,          // Crosshair Y
    int head_class_id,              // Head class ID for priority
    float head_conf_bonus,          // Head bonus for target selection
    const PIDConfig& pid_config,    // PID parameters
    float iou_stickiness_threshold, // IoU threshold for target stickiness (0.3 typical)
    float head_y_offset,            // Aim point offset for head (0.0-1.0)
    float body_y_offset,            // Aim point offset for body (0.0-1.0)
    Detection* d_selected_target,   // Persistent selected target (for IoU tracking)
    Detection* d_best_target,       // Output: best target this frame
    int* d_has_target,              // Output: 1 if target found
    MouseMovement* d_output_movement, // Output: mouse dx, dy
    PIDState* d_pid_state,          // Persistent PID state (device)
    cudaStream_t stream = 0
);

// =============================================================================
// Validation Kernel
// =============================================================================

// Validate single best target before host copy
// Sets d_has_target to 0 if target is invalid
void validateBestTargetGpu(
    Detection* d_best_target,
    int* d_has_target,
    cudaStream_t stream = 0
);

} // namespace gpa
