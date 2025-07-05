#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace constants {
    constexpr int PREDICTION_BUFFER_SIZE = 8;
    constexpr float SCOPE_MARGIN = 0.15f;
    constexpr float SENSITIVITY_FACTOR = 0.05f;
    constexpr float DEAD_ZONE = 0.5f;
    constexpr float MICRO_MOVEMENT_THRESHOLD = 0.3f;
    constexpr int ACTIVE_TIMEOUT_MS = 5;
    constexpr int IDLE_TIMEOUT_MS = 1;
}

#endif // CONSTANTS_H
