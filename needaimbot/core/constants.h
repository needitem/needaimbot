#ifndef NEEDAIMBOT_CORE_CONSTANTS_H
#define NEEDAIMBOT_CORE_CONSTANTS_H

/**
 * @file constants.h
 * @brief System-wide constants and configuration values
 * 
 * Contains all numerical constants, thresholds, and default values
 * used throughout the aimbot system. Centralized constant management
 * improves maintainability and performance tuning.
 */

#define NOMINMAX
#include <windows.h>

namespace Constants {
    // ============================================================================
    // AI Detection Constants
    // ============================================================================
    
    /// Default input resolution for YOLO models (width/height)
    constexpr int DEFAULT_YOLO_INPUT_SIZE = 640;
    
    /// Minimum confidence score for object detection (0.0-1.0)
    constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.25f;
    
    /// Non-Maximum Suppression threshold for duplicate removal
    constexpr float DEFAULT_NMS_THRESHOLD = 0.45f;
    
    /// Image normalization factor for preprocessing (converts 0-255 to 0-1 range)
    constexpr float IMAGE_NORMALIZATION_FACTOR = 1.0f / 255.0f;
    
    // ============================================================================
    // Mouse Movement & Control Constants
    // ============================================================================
    
    /// Emergency brake system - minimum brake force multiplier
    constexpr float EMERGENCY_BRAKE_MIN = 0.2f;
    
    /// Emergency brake system - maximum brake force multiplier
    constexpr float EMERGENCY_BRAKE_MAX = 0.5f;
    
    /// Emergency brake system - scaling factor for brake force calculation
    constexpr float EMERGENCY_BRAKE_SCALE = 0.3f;
    
    /// Angle threshold (degrees) for detecting rapid direction changes
    constexpr float DIRECTION_CHANGE_THRESHOLD = 20.0f;
    
    /// Damping factor applied during direction changes to reduce oscillation
    constexpr float DIRECTION_CHANGE_DAMPING = 0.3f;
    
    /// Deceleration multiplier when approaching target
    constexpr float DECELERATION_FACTOR = 0.5f;
    
    // Target Detection Constants
    constexpr float TARGET_CENTER_OFFSET = 0.5f;
    constexpr float DEFAULT_SCOPE_MARGIN = 0.1f;
    
    // Timing Constants
    constexpr float MS_TO_SECONDS = 0.001f;
    constexpr float US_TO_MS = 0.001f;
    constexpr float BUTTON_RELEASE_TIMEOUT_MS = 50.0f;
    constexpr float BUTTON_PRESS_TIMEOUT_MS = 100.0f;
    
    // Prediction Constants
    constexpr float VELOCITY_NORMALIZATION_FACTOR = 1000.0f;
    constexpr float ACCELERATION_NORMALIZATION_FACTOR = 5000.0f;
    constexpr float PREDICTION_ACCELERATION_WEIGHT = 0.5f;
    constexpr float MAX_PREDICTION_DISTANCE_PIXELS = 100.0f;
    
    // Latency Constants
    constexpr float MIN_LATENCY_MS = 5.0f;
    constexpr float MAX_LATENCY_MS = 100.0f;
    constexpr float DEFAULT_LATENCY_MS = 20.0f;
    
    // Performance Constants
    constexpr int DEFAULT_CAPTURE_FPS = 240;
    constexpr int MIN_CAPTURE_FPS = 30;
    constexpr int MAX_CAPTURE_FPS = 500;
    
    // Thread Pool Constants
    constexpr int DEFAULT_THREAD_POOL_SIZE = 4;
    constexpr int MAX_THREAD_POOL_SIZE = 16;
    
    // Buffer Sizes
    constexpr size_t DEFAULT_DETECTION_RESERVE = 512;
    constexpr size_t THREAD_LOCAL_BUFFER_RESERVE = 1024;
    constexpr int DEFAULT_HISTORY_SIZE = 100;
    constexpr int VELOCITY_HISTORY_SIZE = 5;
    constexpr int LATENCY_HISTORY_SIZE = 10;
    
    // Thread Priority Constants
    constexpr int MOUSE_THREAD_PRIORITY = THREAD_PRIORITY_ABOVE_NORMAL;
    constexpr int DETECTOR_THREAD_PRIORITY = THREAD_PRIORITY_TIME_CRITICAL;
    
    // Timing Intervals
    constexpr int ACTIVE_WAIT_TIMEOUT_MS = 10;
    constexpr int IDLE_WAIT_TIMEOUT_MS = 30;
    constexpr int DETECTION_WAIT_TIMEOUT_MS = 10;
    constexpr int FRAME_WAIT_TIMEOUT_MS = 33;
    
    // Serial Connection Timeouts
    constexpr int SERIAL_READ_TIMEOUT_MS = 100;
    constexpr int SERIAL_WRITE_TIMEOUT_MS = 100;
    constexpr int THREAD_JOIN_TIMEOUT_MS = 1000;
    
    // Capture constants
    constexpr int DEFAULT_CAPTURE_WIDTH = 320;
    constexpr int DEFAULT_CAPTURE_HEIGHT = 320;
    constexpr int MAX_CAPTURE_DIMENSION = 1920;
    constexpr int MIN_CAPTURE_DIMENSION = 50;
    
    // Detection constants
    constexpr int DEFAULT_DETECTION_RESOLUTION = 320;  
    constexpr int MIN_DETECTION_RESOLUTION = 50;
    constexpr int MAX_DETECTION_RESOLUTION = 1280;
    constexpr int MAX_DETECTIONS = 100;
    constexpr int MAX_CLASSES_FOR_FILTERING = 64;  ///< Maximum number of classes for filtering control
    
    // Memory pool sizes
    constexpr size_t DETECTION_POOL_SIZE = 1000;
    constexpr size_t RECT_POOL_SIZE = 1000;
    
    // Distance thresholds for target detection
    constexpr float MIN_SQUARED_DISTANCE = 100.0f;
    constexpr float MAX_SQUARED_DISTANCE = 10000.0f;
    
    // ============================================================================
    // Utility Constants (merged from utils/constants.h)
    // ============================================================================
    
    // Utility-specific constants
    constexpr float EPSILON = 1e-6f;
    
    // Mouse movement constants (utility-specific)
    constexpr int MOUSE_UPDATE_RATE_MS = 1;
    constexpr float DEFAULT_SMOOTHING_FACTOR = 0.5f;
    constexpr float MIN_MOVEMENT_THRESHOLD = 0.01f;
    
    // UI constants (overlay-specific)
    constexpr int BASE_OVERLAY_WIDTH = 800;
    constexpr int BASE_OVERLAY_HEIGHT = 600;
    constexpr int MIN_OVERLAY_WIDTH = 600;
    constexpr int MIN_OVERLAY_HEIGHT = 400;
    constexpr int MIN_OVERLAY_OPACITY = 20;
    constexpr int MAX_OVERLAY_OPACITY = 255;
    constexpr int OVERLAY_TARGET_FPS = 30;
    
    // CUDA constants (utility-specific)
    constexpr int CUDA_BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    
    // UI Slider/Input Limits
    constexpr int MAX_DETECTIONS_LIMIT = 100;
    constexpr int MIN_COLOR_PIXELS_LIMIT = 100;
    constexpr float OFFSET_DRAG_MIN = -100.0f;
    constexpr float OFFSET_DRAG_MAX = 100.0f;
    constexpr float OFFSET_DRAG_SPEED = 0.1f;
    constexpr int SCREENSHOT_DELAY_STEP = 50;
    constexpr int SCREENSHOT_DELAY_STEP_FAST = 500;
    constexpr int RCS_DELAY_MAX = 500;
    constexpr float CROUCH_REDUCTION_MIN = -100.0f;
    constexpr float CROUCH_REDUCTION_MAX = 100.0f;
    
    // UI Widget Sizes
    constexpr int SLIDER_WIDTH_SMALL = 100;
    constexpr int SLIDER_WIDTH_MEDIUM = 200;
    constexpr int COMBO_WIDTH_SMALL = 100;
    constexpr int COMBO_WIDTH_MEDIUM = 200;
    
    // Thread Sleep Durations - Optimized for lower latency
    constexpr int OVERLAY_OCCLUDED_SLEEP_MS = 30;  // Reduced from 100ms for better responsiveness
    constexpr int OVERLAY_HIDDEN_SLEEP_MS = 20;    // Reduced from 100ms for faster wake-up
    constexpr int OVERLAY_INIT_RETRY_SLEEP_MS = 50; // Reduced from 200ms for faster init
    constexpr int CONFIG_SAVE_INTERVAL_MS = 500;    // Keep as-is (removed in latest commit)
    
    // Performance Thresholds
    constexpr int DETECTION_RESOLUTION_HIGH_PERF = 400;
    
    // String Limits
    constexpr size_t MAX_DEBUG_LABEL_LENGTH = 100;
    
    // Error codes
    enum class ErrorCode {
        SUCCESS = 0,
        CUDA_ERROR = -1,
        DIRECTX_ERROR = -2,
        CAPTURE_ERROR = -3,
        MODEL_LOAD_ERROR = -4,
        CONFIG_ERROR = -5,
        THREAD_ERROR = -6
    };
}

#endif // NEEDAIMBOT_CORE_CONSTANTS_H