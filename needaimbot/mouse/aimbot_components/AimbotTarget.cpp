#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
// OpenCV removed - using standard C++ types

#include "needaimbot.h"
#include "AimbotTarget.h"
#include "config.h"

AimbotTarget::AimbotTarget(int x, int y, int w, int h, int cls) : x(x), y(y), w(w), h(h), classId(cls), detection_timestamp(std::chrono::high_resolution_clock::now()) {}


static AimbotTarget s_targetInstance(0, 0, 0, 0, 0);
