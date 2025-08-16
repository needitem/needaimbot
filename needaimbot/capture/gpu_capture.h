#pragma once

#include "../core/windows_headers.h"
#include <string>
#include <d3d11.h>

// Capture method enum
enum class CaptureMethod {
    DESKTOP_DUPLICATION  // Full screen capture with cropping
};

// GPU capture thread function
void gpuOnlyCaptureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);

#ifdef _MSC_VER
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib") 
#endif
