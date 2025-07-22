#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NOMINMAX

namespace Constants {
    // Detection Constants
    constexpr int DEFAULT_YOLO_INPUT_SIZE = 640;
    constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.25f;
    constexpr float DEFAULT_NMS_THRESHOLD = 0.45f;
    
    // Mouse Movement Constants
    constexpr float EMERGENCY_BRAKE_MIN = 0.2f;
    constexpr float EMERGENCY_BRAKE_MAX = 0.5f;
    constexpr float EMERGENCY_BRAKE_SCALE = 0.3f;
    constexpr float DIRECTION_CHANGE_THRESHOLD = 20.0f;
    constexpr float DIRECTION_CHANGE_DAMPING = 0.3f;
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
}

#endif // CONSTANTS_H