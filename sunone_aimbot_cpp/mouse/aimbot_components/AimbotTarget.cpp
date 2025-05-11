#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "sunone_aimbot_cpp.h"
#include "AimbotTarget.h"
#include "config.h"

AimbotTarget::AimbotTarget(int x, int y, int w, int h, int cls) : x(x), y(y), w(w), h(h), classId(cls) {}

// Static object declaration - Object reuse pattern to avoid dynamic allocation every time
static AimbotTarget s_targetInstance(0, 0, 0, 0, 0);