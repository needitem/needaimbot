#pragma once

namespace Constants {
    // Detection thresholds
    constexpr float MIN_SQUARED_DISTANCE = 100.0f;
    constexpr float MAX_SQUARED_DISTANCE = 500000.0f;
    constexpr float EPSILON = 1e-6f;
    
    // Mouse movement constants
    constexpr int MOUSE_UPDATE_RATE_MS = 1;
    constexpr float DEFAULT_SMOOTHING_FACTOR = 0.5f;
    constexpr float MIN_MOVEMENT_THRESHOLD = 0.01f;
    
    // Capture constants
    constexpr int DEFAULT_CAPTURE_WIDTH = 320;
    constexpr int DEFAULT_CAPTURE_HEIGHT = 320;
    constexpr int MAX_CAPTURE_DIMENSION = 1920;
    constexpr int MIN_CAPTURE_DIMENSION = 50;
    
    // Performance constants moved to core/constants.h to avoid duplication
    constexpr int OVERLAY_TARGET_FPS = 30;
    
    // UI constants
    constexpr float BASE_OVERLAY_WIDTH = 900.0f;
    constexpr float BASE_OVERLAY_HEIGHT = 700.0f;
    constexpr int MIN_OVERLAY_OPACITY = 20;
    constexpr int MAX_OVERLAY_OPACITY = 255;
    
    // Detection constants
    constexpr int DEFAULT_DETECTION_RESOLUTION = 320;
    constexpr int MIN_DETECTION_RESOLUTION = 50;
    constexpr int MAX_DETECTION_RESOLUTION = 1280;
    constexpr int MAX_DETECTIONS = 100;
    
    // Thread synchronization
    constexpr int DETECTION_WAIT_TIMEOUT_MS = 10;
    constexpr int FRAME_WAIT_TIMEOUT_MS = 33;
    
    // Memory pool sizes
    constexpr size_t DETECTION_POOL_SIZE = 1000;
    constexpr size_t RECT_POOL_SIZE = 1000;
    
    // CUDA constants
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