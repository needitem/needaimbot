#pragma once

// Capture method enum
enum class CaptureMethod {
    DESKTOP_DUPLICATION,  // Full screen capture with cropping (current method)
    REGION_CAPTURE       // Direct region capture (new method)
};

// GPU capture thread function
void gpuOnlyCaptureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);

// Region capture class forward declaration
class RegionCapture;