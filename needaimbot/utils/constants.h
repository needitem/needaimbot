#pragma once

// This file contains utility-specific constants that don't belong in core/constants.h
// Core constants have been moved to core/constants.h to avoid duplication

namespace UtilConstants {
    // Utility-specific constants (detection thresholds moved to core/constants.h)
    constexpr float EPSILON = 1e-6f;
    
    // Mouse movement constants (utility-specific)
    constexpr int MOUSE_UPDATE_RATE_MS = 1;
    constexpr float DEFAULT_SMOOTHING_FACTOR = 0.5f;
    constexpr float MIN_MOVEMENT_THRESHOLD = 0.01f;
    
    // UI constants (overlay-specific)
    constexpr float BASE_OVERLAY_WIDTH = 900.0f;
    constexpr float BASE_OVERLAY_HEIGHT = 700.0f;
    constexpr int MIN_OVERLAY_OPACITY = 20;
    constexpr int MAX_OVERLAY_OPACITY = 255;
    constexpr int OVERLAY_TARGET_FPS = 30;
    
    // CUDA constants (utility-specific)
    constexpr int CUDA_BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    
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