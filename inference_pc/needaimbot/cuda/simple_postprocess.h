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
// GPU-based Target Selection
// =============================================================================

// Finds the best target with head-in-body priority:
// 1. First checks if any head bbox is inside a body bbox -> select that head
// 2. Otherwise, selects the closest detection to crosshair
// Includes head bonus for preferring head targets
cudaError_t findBestTargetGpu(
    const Detection* d_detections,  // Input detections (device)
    const int* d_num_detections,    // Number of detections (device)
    float crosshairX,               // Center X (typically width/2)
    float crosshairY,               // Center Y (typically height/2)
    int head_class_id,              // Class ID for head (-1 to disable priority)
    float head_conf_bonus,          // Bonus confidence for head class
    Detection* d_best_target,       // Output: best target (device)
    int* d_has_target,              // Output: 1 if target found, 0 otherwise (device)
    cudaStream_t stream = 0
);

// =============================================================================
// Fused Target Selection + PID Movement (from unified_graph_pipeline.cu)
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
// Validation Kernels
// =============================================================================

// Validate all detections and mark invalid ones (classId = -1)
// Call after decoding to ensure clean data
void validateDetectionsGpu(
    Detection* d_detections,
    int* d_count,
    int max_detections,
    cudaStream_t stream = 0
);

// Validate single best target before host copy
// Sets d_has_target to 0 if target is invalid
void validateBestTargetGpu(
    Detection* d_best_target,
    int* d_has_target,
    cudaStream_t stream = 0
);

} // namespace gpa
